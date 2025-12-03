from dotenv import load_dotenv
load_dotenv()

import os
import datetime
from agent import get_agent_message, Agent, GeminiClient, OpenRouterClient, RateLimiter, SimpleLogger
from memory_manager import HybridMemoryManager
from rag_system import RAGSystem, EnhancedMemoryManager
from tavily import TavilyClient


def get_agent_message_with_rag(username: str, inquiry: str, timestamp: datetime.datetime,
                               memory_manager=None, rag_system=None, return_metadata: bool = False):
    user_log_path = os.path.join("logs", f"{username}.log")
    logger = SimpleLogger(user_log_path, truncate=False)

    provider = os.getenv("LLM_PROVIDER", "gemini").lower()

    if provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        model_name = os.getenv("OPENROUTER_MODEL", "anthropic/claude-3.5-sonnet")
        llm_client = OpenRouterClient(
            api_key=api_key,
            model_name=model_name,
            max_new_tokens=1024,
            limiter=RateLimiter(0.3),
            logger=logger,
        )
    else:
        api_key = os.getenv("GEMINI_API_KEY")
        model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
        llm_client = GeminiClient(
            api_key=api_key,
            model_name=model_name,
            max_new_tokens=1024,
            limiter=RateLimiter(0.3),
            logger=logger,
        )

    tavily_api_key = os.getenv("TAVILY_API_KEY")
    tavily_client = TavilyClient(api_key=tavily_api_key)
    agent = Agent(gemini_client=llm_client, tavily_client=tavily_client, logger=logger)

    logger.log(f"[user] {username}", ts=timestamp)
    logger.log(f"[channel] discord", ts=timestamp)

    user_context = None
    if memory_manager and rag_system:
        try:
            enhanced_memory = EnhancedMemoryManager(memory_manager, rag_system)

            user_context = enhanced_memory.build_enhanced_context(
                user_id=username,
                current_query=inquiry,
                use_rag=True
            )

            memory_manager.add_message(
                user_id=username,
                role='user',
                content=inquiry,
                timestamp=timestamp
            )

            logger.log(f"[context] Loaded RAG-enhanced context for {username}")

        except Exception as e:
            logger.log(f"[warning] Failed to load RAG context: {repr(e)}")
            user_context = memory_manager.build_context_for_llm(
                user_id=username,
                current_query=inquiry
            ) if memory_manager else None

    try:
        result = agent.chat_with_tools(inquiry, max_steps=5, ts_start=timestamp, user_context=user_context)
        answer = result["response"]
        tools_used = result["tools_used"]
        logger.log(f"[final] answer(len={len(answer)}): {answer[:500]}", ts=timestamp)
        logger.log(f"[tools] used: {tools_used}")

        if memory_manager:
            try:
                memory_manager.add_message(
                    user_id=username,
                    role='assistant',
                    content=answer,
                    timestamp=datetime.datetime.now()
                )

                conversation_text = f"User: {inquiry}\nAssistant: {answer}"
                extracted_profile = memory_manager.extract_profile_from_conversation(
                    user_id=username,
                    conversation_text=conversation_text,
                    gemini_client=llm_client
                )

                if extracted_profile:
                    memory_manager.update_user_profile(username, extracted_profile)
                    logger.log(f"[profile] Updated profile with: {list(extracted_profile.keys())}")

                try:
                    from menu_extractor import integrate_menu_tracking
                    meals = integrate_menu_tracking(
                        user_id=username,
                        response=answer,
                        gemini_client=llm_client,
                        memory_manager=memory_manager
                    )
                    if meals:
                        logger.log(f"[menu] Saved {len(meals)} meals to history")

                        if rag_system:
                            rag_system.index_menu_items(username, meals)
                            logger.log(f"[rag] Indexed {len(meals)} new meals")

                except Exception as e:
                    logger.log(f"[warning] Failed to extract menu items: {repr(e)}")

            except Exception as e:
                logger.log(f"[warning] Failed to save conversation: {repr(e)}")

        if return_metadata:
            return {"response": answer, "tools_used": tools_used}
        return answer

    except Exception as e:
        logger.log(f"[error] {repr(e)}", ts=timestamp)
        error_response = "Sorry, something went wrong while generating the response."
        if return_metadata:
            return {"response": error_response, "tools_used": []}
        return error_response


def main():
    memory = HybridMemoryManager("data/users.db")
    rag = RAGSystem(model_name="BAAI/bge-small-en-v1.5")

    print("Testing agent with RAG enhancement...")
    print("=" * 60)

    response = get_agent_message_with_rag(
        username="test_rag_user",
        inquiry="I want a high-protein breakfast similar to what I had before",
        timestamp=datetime.datetime.now(),
        memory_manager=memory,
        rag_system=rag
    )

    print(f"\nResponse: {response}")
    print("=" * 60)


if __name__ == "__main__":
    main()
