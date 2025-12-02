import os
import numpy as np
from typing import List, Dict, Optional, Tuple
import json
from datetime import datetime
import threading
import queue
import time

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDING_AVAILABLE = True
except ImportError:
    EMBEDDING_AVAILABLE = False


class SimpleVectorStore:
    def __init__(self):
        self.embeddings = []
        self.documents = []
        self.metadata = []

    def add(self, embedding: np.ndarray, document: str, metadata: dict):
        self.embeddings.append(embedding)
        self.documents.append(document)
        self.metadata.append(metadata)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[float, str, dict]]:
        if len(self.embeddings) == 0:
            return []

        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)

        similarities = []
        for i, emb in enumerate(self.embeddings):
            emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
            similarity = np.dot(query_norm, emb_norm)
            similarities.append((similarity, i))

        similarities.sort(reverse=True, key=lambda x: x[0])

        results = []
        for sim, idx in similarities[:top_k]:
            results.append((sim, self.documents[idx], self.metadata[idx]))

        return results

    def clear(self):
        self.embeddings = []
        self.documents = []
        self.metadata = []


class RAGSystem:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        if not EMBEDDING_AVAILABLE:
            raise ImportError("Install sentence-transformers: pip install sentence-transformers")

        self.model = SentenceTransformer(model_name)
        self.conversation_store = SimpleVectorStore()
        self.menu_store = SimpleVectorStore()
        self.profile_store = SimpleVectorStore()

    def _get_embedding(self, text: str) -> np.ndarray:
        return self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True)

    def index_conversation(self, user_id: str, messages: List[dict]):
        for msg in messages:
            text = f"{msg['role']}: {msg['content']}"
            embedding = self._get_embedding(text)

            metadata = {
                'user_id': user_id,
                'role': msg['role'],
                'timestamp': msg.get('timestamp', ''),
                'type': 'conversation'
            }

            self.conversation_store.add(embedding, text, metadata)

    def index_menu_items(self, user_id: str, menu_items: List[dict]):
        for item in menu_items:
            ingredients = ", ".join(item.get('ingredients', []))
            text = (
                f"Dish: {item['dish_name']}\n"
                f"Protein: {item.get('primary_protein', 'unknown')}\n"
                f"Ingredients: {ingredients}\n"
                f"Calories: {item.get('calories', 'unknown')}, "
                f"Protein: {item.get('protein', 'unknown')}g"
            )

            embedding = self._get_embedding(text)

            metadata = {
                'user_id': user_id,
                'dish_name': item['dish_name'],
                'primary_protein': item.get('primary_protein'),
                'date_served': item.get('date_served', ''),
                'type': 'menu'
            }

            self.menu_store.add(embedding, text, metadata)

    def index_user_profile(self, user_id: str, profile):
        text = profile.format_for_prompt()
        embedding = self._get_embedding(text)

        metadata = {
            'user_id': user_id,
            'type': 'profile'
        }

        self.profile_store.add(embedding, text, metadata)

    def search_conversations(self, query: str, user_id: Optional[str] = None, top_k: int = 5) -> List[dict]:
        query_embedding = self._get_embedding(query)
        results = self.conversation_store.search(query_embedding, top_k=top_k * 2)

        if user_id:
            results = [(sim, doc, meta) for sim, doc, meta in results
                      if meta['user_id'] == user_id]

        formatted = []
        for similarity, document, metadata in results[:top_k]:
            formatted.append({
                'content': document,
                'similarity': float(similarity),
                'timestamp': metadata.get('timestamp', ''),
                'role': metadata.get('role', 'unknown')
            })

        return formatted

    def search_menu_history(self, query: str, user_id: Optional[str] = None, top_k: int = 5) -> List[dict]:
        query_embedding = self._get_embedding(query)
        results = self.menu_store.search(query_embedding, top_k=top_k * 2)

        if user_id:
            results = [(sim, doc, meta) for sim, doc, meta in results
                      if meta['user_id'] == user_id]

        formatted = []
        for similarity, document, metadata in results[:top_k]:
            formatted.append({
                'content': document,
                'similarity': float(similarity),
                'dish_name': metadata.get('dish_name', ''),
                'protein': metadata.get('primary_protein', ''),
                'date': metadata.get('date_served', '')
            })

        return formatted

    def get_relevant_context(self, query: str, user_id: str,
                            num_conversations: int = 3,
                            num_menu_items: int = 3) -> dict:
        return {
            'relevant_conversations': self.search_conversations(
                query, user_id, top_k=num_conversations
            ),
            'relevant_menu_items': self.search_menu_history(
                query, user_id, top_k=num_menu_items
            )
        }


class BackgroundIndexer:
    def __init__(self, memory_manager, rag_system: RAGSystem):
        self.memory = memory_manager
        self.rag = rag_system
        self.index_queue = queue.Queue()
        self.running = False
        self.worker_thread = None
        self.indexed_users = set()

    def start(self):
        if self.running:
            return
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()

    def stop(self):
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)

    def queue_user_indexing(self, user_id: str):
        if user_id not in self.indexed_users:
            self.index_queue.put(('full_index', user_id))

    def queue_message_indexing(self, user_id: str, message: dict):
        self.index_queue.put(('index_message', user_id, message))

    def queue_menu_indexing(self, user_id: str, menu_item: dict):
        self.index_queue.put(('index_menu', user_id, menu_item))

    def _worker(self):
        while self.running:
            try:
                task = self.index_queue.get(timeout=1)
                self._process_task(task)
            except queue.Empty:
                continue
            except Exception:
                pass

    def _process_task(self, task):
        task_type = task[0]

        if task_type == 'full_index':
            user_id = task[1]
            self._index_user_full(user_id)
            self.indexed_users.add(user_id)

        elif task_type == 'index_message':
            user_id = task[1]
            message = task[2]
            self.rag.index_conversation(user_id, [message])

        elif task_type == 'index_menu':
            user_id = task[1]
            menu_item = task[2]
            self.rag.index_menu_items(user_id, [menu_item])

    def _index_user_full(self, user_id: str):
        try:
            profile = self.memory.get_user_profile(user_id)
            self.rag.index_user_profile(user_id, profile)

            messages = self.memory.get_recent_messages(user_id, limit=100)
            if messages:
                self.rag.index_conversation(user_id, messages)

            menu_items = self.memory.get_menu_history(user_id, days=30)
            if menu_items:
                self.rag.index_menu_items(user_id, menu_items)
        except Exception:
            pass


class EnhancedMemoryManager:
    def __init__(self, memory_manager, rag_system: Optional[RAGSystem] = None, async_indexing: bool = True):
        self.memory = memory_manager
        self.rag = rag_system or RAGSystem()
        self.async_indexing = async_indexing

        if async_indexing:
            self.indexer = BackgroundIndexer(memory_manager, self.rag)
            self.indexer.start()
        else:
            self.indexer = None
            self._indexed_users = set()

    def _ensure_indexed_sync(self, user_id: str):
        if user_id in self._indexed_users:
            return

        profile = self.memory.get_user_profile(user_id)
        self.rag.index_user_profile(user_id, profile)

        messages = self.memory.get_recent_messages(user_id, limit=100)
        if messages:
            self.rag.index_conversation(user_id, messages)

        menu_items = self.memory.get_menu_history(user_id, days=30)
        if menu_items:
            self.rag.index_menu_items(user_id, menu_items)

        self._indexed_users.add(user_id)

    def ensure_indexed(self, user_id: str):
        if self.async_indexing:
            self.indexer.queue_user_indexing(user_id)
        else:
            self._ensure_indexed_sync(user_id)

    def build_enhanced_context(self, user_id: str, current_query: str, use_rag: bool = True) -> dict:
        base_context = self.memory.build_context_for_llm(
            user_id=user_id,
            current_query=current_query
        )

        if not use_rag:
            return base_context

        self.ensure_indexed(user_id)

        try:
            rag_context = self.rag.get_relevant_context(
                query=current_query,
                user_id=user_id,
                num_conversations=3,
                num_menu_items=3
            )

            base_context['rag_conversations'] = rag_context['relevant_conversations']
            base_context['rag_menu_items'] = rag_context['relevant_menu_items']
        except Exception:
            pass

        return base_context

    def index_new_message(self, user_id: str, message: dict):
        if self.async_indexing:
            self.indexer.queue_message_indexing(user_id, message)
        else:
            self.rag.index_conversation(user_id, [message])

    def index_new_menu(self, user_id: str, menu_item: dict):
        if self.async_indexing:
            self.indexer.queue_menu_indexing(user_id, menu_item)
        else:
            self.rag.index_menu_items(user_id, [menu_item])

    def format_rag_context_for_prompt(self, context: dict) -> str:
        parts = []

        if context.get('rag_conversations'):
            parts.append("=== RELEVANT PAST CONVERSATIONS ===")
            for conv in context['rag_conversations']:
                sim = conv['similarity']
                parts.append(f"[Similarity: {sim:.2f}] {conv['content']}")
            parts.append("")

        if context.get('rag_menu_items'):
            parts.append("=== SIMILAR DISHES FROM HISTORY ===")
            for item in context['rag_menu_items']:
                sim = item['similarity']
                parts.append(f"[Similarity: {sim:.2f}] {item['content']}")
            parts.append("")

        return "\n".join(parts)

    def shutdown(self):
        if self.async_indexing and self.indexer:
            self.indexer.stop()
