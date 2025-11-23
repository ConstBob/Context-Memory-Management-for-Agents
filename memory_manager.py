"""
Hybrid Memory Management System for AI Agents
Combines structured profiles, conversation history, and menu tracking
"""

import os
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import re


@dataclass
class UserProfile:
    """Structured user preferences and constraints"""
    user_id: str
    calories_target: Optional[int] = None
    protein_min: Optional[int] = None
    fiber_min: Optional[int] = None
    allergies: List[str] = None
    dietary_restrictions: List[str] = None  # vegetarian, pescatarian, halal, etc.
    equipment: List[str] = None  # microwave, stove, oven, grill, etc.
    cuisine_preferences: List[str] = None
    unit_preference: str = "US"  # US or metric
    fasting_window: Optional[Dict[str, str]] = None  # {"start": "12:00", "end": "20:00"}
    dislikes: List[str] = None
    other_constraints: Dict[str, Any] = None

    def __post_init__(self):
        """Initialize empty lists for None values"""
        if self.allergies is None:
            self.allergies = []
        if self.dietary_restrictions is None:
            self.dietary_restrictions = []
        if self.equipment is None:
            self.equipment = []
        if self.cuisine_preferences is None:
            self.cuisine_preferences = []
        if self.dislikes is None:
            self.dislikes = []
        if self.other_constraints is None:
            self.other_constraints = {}

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'UserProfile':
        """Create from dictionary"""
        return cls(**data)

    def merge(self, updates: dict) -> 'UserProfile':
        """Merge updates into profile, combining lists"""
        for key, value in updates.items():
            if value is None:
                continue

            current = getattr(self, key, None)

            # For lists, merge unique values
            if isinstance(current, list) and isinstance(value, list):
                setattr(self, key, list(set(current + value)))
            # For dicts, merge
            elif isinstance(current, dict) and isinstance(value, dict):
                current.update(value)
                setattr(self, key, current)
            # For other types, replace
            else:
                setattr(self, key, value)

        return self

    def format_for_prompt(self) -> str:
        """Format profile as readable text for LLM context"""
        lines = []

        if self.calories_target:
            lines.append(f"- Daily Calories: {self.calories_target} kcal")
        if self.protein_min:
            lines.append(f"- Minimum Protein: {self.protein_min}g")
        if self.fiber_min:
            lines.append(f"- Minimum Fiber: {self.fiber_min}g")
        if self.allergies:
            lines.append(f"- Allergies: {', '.join(self.allergies)}")
        if self.dietary_restrictions:
            lines.append(f"- Dietary Restrictions: {', '.join(self.dietary_restrictions)}")
        if self.equipment:
            lines.append(f"- Available Equipment: {', '.join(self.equipment)}")
        if self.cuisine_preferences:
            lines.append(f"- Cuisine Preferences: {', '.join(self.cuisine_preferences)}")
        if self.unit_preference:
            lines.append(f"- Unit Preference: {self.unit_preference}")
        if self.fasting_window:
            lines.append(f"- Fasting Window: {self.fasting_window.get('start')} - {self.fasting_window.get('end')}")
        if self.dislikes:
            lines.append(f"- Dislikes: {', '.join(self.dislikes)}")
        if self.other_constraints:
            for key, value in self.other_constraints.items():
                lines.append(f"- {key.replace('_', ' ').title()}: {value}")

        return "\n".join(lines) if lines else "No profile information yet."


class HybridMemoryManager:
    """Manages user context with database persistence and smart retrieval"""

    def __init__(self, db_path: str = "data/users.db"):
        self.db_path = db_path
        self._ensure_db_exists()
        self.conn = None

    def _ensure_db_exists(self):
        """Create database and tables if they don't exist"""
        os.makedirs(os.path.dirname(self.db_path) if os.path.dirname(self.db_path) else ".", exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # User profiles table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id TEXT PRIMARY KEY,
                profile_json TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Conversation history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversation_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                session_id TEXT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                metadata_json TEXT,
                FOREIGN KEY (user_id) REFERENCES user_profiles(user_id)
            )
        """)

        # Create index for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_conversation_user_time
            ON conversation_history(user_id, timestamp DESC)
        """)

        # Menu history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS menu_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                session_id TEXT,
                meal_type TEXT,
                dish_name TEXT NOT NULL,
                primary_protein TEXT,
                ingredients TEXT,
                calories REAL,
                protein REAL,
                fiber REAL,
                date_served DATE NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES user_profiles(user_id)
            )
        """)

        # Create index for menu queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_menu_user_date
            ON menu_history(user_id, date_served DESC)
        """)

        conn.commit()
        conn.close()

    def _get_conn(self):
        """Get database connection"""
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row
        return self.conn

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            self.conn = None

    # ==================== User Profile Management ====================

    def get_user_profile(self, user_id: str) -> UserProfile:
        """Get user profile, creating default if doesn't exist"""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT profile_json FROM user_profiles WHERE user_id = ?",
            (user_id,)
        )
        row = cursor.fetchone()

        if row:
            profile_data = json.loads(row['profile_json'])
            return UserProfile.from_dict(profile_data)
        else:
            # Create default profile
            return UserProfile(user_id=user_id)

    def update_user_profile(self, user_id: str, updates: dict):
        """Update user profile with new information"""
        conn = self._get_conn()
        cursor = conn.cursor()

        # Get existing profile
        profile = self.get_user_profile(user_id)

        # Merge updates
        profile.merge(updates)

        # Save back to database
        cursor.execute("""
            INSERT OR REPLACE INTO user_profiles (user_id, profile_json, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
        """, (user_id, json.dumps(profile.to_dict())))

        conn.commit()

    def extract_profile_from_conversation(self, user_id: str, conversation_text: str, gemini_client) -> dict:
        """Use LLM to extract profile information from conversation"""
        prompt = {
            'system': """Extract user dietary profile information from the conversation.
Return ONLY a valid JSON object with these keys (omit any that aren't mentioned):
- calories_target (int)
- protein_min (int)
- fiber_min (int)
- allergies (list of strings)
- dietary_restrictions (list: vegetarian, pescatarian, halal, lactose-free, etc.)
- equipment (list: microwave, stove, oven, grill, air fryer, rice cooker, blender, etc.)
- cuisine_preferences (list: Mediterranean, Mexican, Japanese, Thai, etc.)
- unit_preference ("US" or "metric")
- fasting_window (dict with "start" and "end" times like "12:00")
- dislikes (list of foods)
- other_constraints (dict of any other relevant info)

Example: {"calories_target": 1800, "protein_min": 140, "allergies": ["peanuts"], "equipment": ["stove", "oven"]}""",
            'user': f"Conversation:\n{conversation_text}\n\nExtract profile information as JSON:"
        }

        try:
            response = gemini_client.infer(prompt)
            # Clean up response - remove markdown code blocks if present
            response = response.strip()
            if response.startswith('```'):
                response = re.sub(r'^```(?:json)?\s*', '', response)
                response = re.sub(r'\s*```$', '', response)

            extracted = json.loads(response)
            return extracted
        except Exception as e:
            print(f"Warning: Failed to extract profile: {e}")
            return {}

    # ==================== Conversation History Management ====================

    def add_message(self, user_id: str, role: str, content: str,
                   session_id: Optional[str] = None,
                   timestamp: Optional[datetime] = None,
                   metadata: Optional[dict] = None):
        """Add a message to conversation history"""
        conn = self._get_conn()
        cursor = conn.cursor()

        if timestamp is None:
            timestamp = datetime.now()

        cursor.execute("""
            INSERT INTO conversation_history
            (user_id, session_id, role, content, timestamp, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            user_id,
            session_id,
            role,
            content,
            timestamp,
            json.dumps(metadata) if metadata else None
        ))

        conn.commit()

    def get_recent_messages(self, user_id: str, limit: int = 20) -> List[dict]:
        """Get recent conversation messages"""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT role, content, timestamp, session_id, metadata_json
            FROM conversation_history
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (user_id, limit))

        rows = cursor.fetchall()

        # Reverse to get chronological order
        messages = []
        for row in reversed(rows):
            messages.append({
                'role': row['role'],
                'content': row['content'],
                'timestamp': row['timestamp'],
                'session_id': row['session_id'],
                'metadata': json.loads(row['metadata_json']) if row['metadata_json'] else None
            })

        return messages

    def get_session_messages(self, user_id: str, session_id: str) -> List[dict]:
        """Get all messages from a specific session"""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT role, content, timestamp, metadata_json
            FROM conversation_history
            WHERE user_id = ? AND session_id = ?
            ORDER BY timestamp ASC
        """, (user_id, session_id))

        rows = cursor.fetchall()

        messages = []
        for row in rows:
            messages.append({
                'role': row['role'],
                'content': row['content'],
                'timestamp': row['timestamp'],
                'metadata': json.loads(row['metadata_json']) if row['metadata_json'] else None
            })

        return messages

    def format_conversation_history(self, messages: List[dict], max_length: int = 10) -> str:
        """Format conversation history for LLM context"""
        if not messages:
            return "No previous conversation."

        # Take last max_length messages
        recent = messages[-max_length:]

        lines = []
        for msg in recent:
            timestamp = msg.get('timestamp', '')
            role = msg['role'].capitalize()
            content = msg['content'][:500]  # Truncate very long messages
            lines.append(f"[{timestamp}] {role}: {content}")

        return "\n".join(lines)

    # ==================== Menu History Management ====================

    def add_menu_item(self, user_id: str, dish_name: str, date_served: datetime,
                     meal_type: Optional[str] = None,
                     primary_protein: Optional[str] = None,
                     ingredients: Optional[List[str]] = None,
                     calories: Optional[float] = None,
                     protein: Optional[float] = None,
                     fiber: Optional[float] = None,
                     session_id: Optional[str] = None):
        """Add a menu item to history"""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO menu_history
            (user_id, session_id, meal_type, dish_name, primary_protein,
             ingredients, calories, protein, fiber, date_served)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id,
            session_id,
            meal_type,
            dish_name,
            primary_protein,
            json.dumps(ingredients) if ingredients else None,
            calories,
            protein,
            fiber,
            date_served
        ))

        conn.commit()

    def get_menu_history(self, user_id: str, days: int = 7) -> List[dict]:
        """Get menu history for the last N days"""
        conn = self._get_conn()
        cursor = conn.cursor()

        cutoff_date = datetime.now() - timedelta(days=days)

        cursor.execute("""
            SELECT meal_type, dish_name, primary_protein, ingredients,
                   calories, protein, fiber, date_served
            FROM menu_history
            WHERE user_id = ? AND date_served >= ?
            ORDER BY date_served DESC
        """, (user_id, cutoff_date))

        rows = cursor.fetchall()

        menu_items = []
        for row in rows:
            menu_items.append({
                'meal_type': row['meal_type'],
                'dish_name': row['dish_name'],
                'primary_protein': row['primary_protein'],
                'ingredients': json.loads(row['ingredients']) if row['ingredients'] else [],
                'calories': row['calories'],
                'protein': row['protein'],
                'fiber': row['fiber'],
                'date_served': row['date_served']
            })

        return menu_items

    def format_menu_history(self, menu_items: List[dict]) -> str:
        """Format menu history for LLM context"""
        if not menu_items:
            return "No recent menu history."

        lines = []
        for item in menu_items:
            date = item['date_served']
            dish = item['dish_name']
            protein = item['primary_protein']
            line = f"- {date}: {dish}"
            if protein:
                line += f" (protein: {protein})"
            lines.append(line)

        return "\n".join(lines)

    # ==================== Context Building ====================

    def build_context_for_llm(self, user_id: str, current_query: str,
                             include_menu_days: int = 7,
                             include_message_limit: int = 10) -> dict:
        """Build comprehensive context for LLM inference"""

        # Get all context pieces
        profile = self.get_user_profile(user_id)
        recent_messages = self.get_recent_messages(user_id, limit=include_message_limit)
        menu_history = self.get_menu_history(user_id, days=include_menu_days)

        # Format for LLM
        context = {
            'user_id': user_id,
            'profile': profile,
            'profile_text': profile.format_for_prompt(),
            'recent_conversation': self.format_conversation_history(recent_messages),
            'menu_history': self.format_menu_history(menu_history),
            'menu_items': menu_history,  # Raw data for variety checking
            'current_query': current_query
        }

        return context

    def format_full_context_prompt(self, context: dict) -> str:
        """Format complete context as a prompt section"""
        lines = [
            "=== USER CONTEXT ===",
            "",
            "User Profile:",
            context['profile_text'],
            "",
            "Recent Conversation:",
            context['recent_conversation'],
            "",
            "Recent Menu History (avoid repeating these dishes):",
            context['menu_history'],
            "",
            "Current Query:",
            context['current_query']
        ]

        return "\n".join(lines)
