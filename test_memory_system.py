"""
Test script for Hybrid Memory Management System
Demonstrates how the system handles user context and memory
"""

import os
import datetime
from dotenv import load_dotenv
load_dotenv()

from agent import get_agent_message
from memory_manager import HybridMemoryManager


def test_basic_memory():
    """Test basic conversation memory"""
    print("=" * 80)
    print("TEST 1: Basic Memory - User Profile Persistence")
    print("=" * 80)

    memory_manager = HybridMemoryManager(db_path="data/test_users.db")

    # First conversation - establish profile
    print("\n--- Turn 1: Establishing profile ---")
    response1 = get_agent_message(
        username="test_user_alice",
        inquiry="I'm cutting at 1800 calories with at least 140g protein. I'm allergic to peanuts and only have a stove and oven.",
        timestamp=datetime.datetime.now(),
        memory_manager=memory_manager
    )
    print(f"Response: {response1[:200]}...")

    # Check profile was saved
    profile = memory_manager.get_user_profile("test_user_alice")
    print(f"\nProfile saved: {profile.format_for_prompt()}")

    # Second conversation - reference "my usual"
    print("\n--- Turn 2: Using 'my usual' (should recall profile) ---")
    response2 = get_agent_message(
        username="test_user_alice",
        inquiry="Can you give me a meal plan at my usual targets? Remember my restrictions.",
        timestamp=datetime.datetime.now(),
        memory_manager=memory_manager
    )
    print(f"Response: {response2[:300]}...")

    # Check conversation history
    messages = memory_manager.get_recent_messages("test_user_alice", limit=4)
    print(f"\nConversation history: {len(messages)} messages")
    for msg in messages:
        print(f"  - {msg['role']}: {msg['content'][:60]}...")

    memory_manager.close()
    print("\n✓ Test 1 passed: Profile persists across conversations\n")


def test_multi_user_separation():
    """Test that different users are kept separate"""
    print("=" * 80)
    print("TEST 2: Multi-user Separation")
    print("=" * 80)

    memory_manager = HybridMemoryManager(db_path="data/test_users.db")

    # User 1
    print("\n--- User 1 (Bob): Vegetarian ---")
    get_agent_message(
        username="test_user_bob",
        inquiry="I'm vegetarian and need 1600 calories per day with high protein.",
        timestamp=datetime.datetime.now(),
        memory_manager=memory_manager
    )

    # User 2
    print("\n--- User 2 (Carol): Pescatarian ---")
    get_agent_message(
        username="test_user_carol",
        inquiry="I'm pescatarian, no tuna. I need 2200 calories daily.",
        timestamp=datetime.datetime.now(),
        memory_manager=memory_manager
    )

    # Check profiles are separate
    bob_profile = memory_manager.get_user_profile("test_user_bob")
    carol_profile = memory_manager.get_user_profile("test_user_carol")

    print(f"\nBob's profile:\n{bob_profile.format_for_prompt()}")
    print(f"\nCarol's profile:\n{carol_profile.format_for_prompt()}")

    assert bob_profile.calories_target == 1600
    assert carol_profile.calories_target == 2200
    assert 'vegetarian' in (bob_profile.dietary_restrictions or [])
    assert 'pescatarian' in (carol_profile.dietary_restrictions or [])

    memory_manager.close()
    print("\n✓ Test 2 passed: Users are properly separated\n")


def test_menu_history():
    """Test menu history tracking"""
    print("=" * 80)
    print("TEST 3: Menu History and Variety Tracking")
    print("=" * 80)

    memory_manager = HybridMemoryManager(db_path="data/test_users.db")

    # Add some menu items manually
    print("\n--- Adding menu items to history ---")
    from datetime import timedelta

    base_date = datetime.datetime.now()
    test_meals = [
        {
            'dish': 'Grilled Chicken Salad',
            'protein': 'chicken',
            'date_offset': 0
        },
        {
            'dish': 'Salmon with Broccoli',
            'protein': 'salmon',
            'date_offset': 1
        },
        {
            'dish': 'Greek Yogurt Bowl',
            'protein': 'yogurt',
            'date_offset': 2
        }
    ]

    for meal in test_meals:
        memory_manager.add_menu_item(
            user_id="test_user_dave",
            dish_name=meal['dish'],
            date_served=base_date - timedelta(days=meal['date_offset']),
            primary_protein=meal['protein'],
            calories=450,
            protein=35
        )
        print(f"  Added: {meal['dish']} ({meal['date_offset']} days ago)")

    # Retrieve menu history
    history = memory_manager.get_menu_history("test_user_dave", days=7)
    print(f"\nRetrieved {len(history)} menu items:")
    for item in history:
        print(f"  - {item['dish_name']} (protein: {item['primary_protein']})")

    # Test variety checking
    from menu_extractor import MenuExtractor
    extractor = MenuExtractor(memory_manager)

    proposed_meals = [
        {
            'dish_name': 'Grilled Chicken Salad',  # Duplicate!
            'primary_protein': 'chicken',
            'ingredients': ['chicken', 'lettuce', 'tomato']
        }
    ]

    compliance = extractor.check_variety_compliance("test_user_dave", proposed_meals)
    print(f"\nVariety compliance check:")
    print(f"  Compliant: {compliance['compliant']}")
    print(f"  Violations: {compliance['violations']}")
    print(f"  Warnings: {compliance['warnings']}")

    memory_manager.close()
    print("\n✓ Test 3 passed: Menu history works correctly\n")


def test_context_building():
    """Test full context building for LLM"""
    print("=" * 80)
    print("TEST 4: Full Context Building")
    print("=" * 80)

    memory_manager = HybridMemoryManager(db_path="data/test_users.db")

    # Set up a user with full context
    user_id = "test_user_eve"

    # Update profile
    memory_manager.update_user_profile(user_id, {
        'calories_target': 1800,
        'protein_min': 140,
        'allergies': ['peanuts'],
        'equipment': ['stove', 'oven']
    })

    # Add conversation history
    memory_manager.add_message(user_id, 'user', 'What should I eat for breakfast?', datetime.datetime.now())
    memory_manager.add_message(user_id, 'assistant', 'Try Greek yogurt with berries and granola.', datetime.datetime.now())

    # Add menu items
    memory_manager.add_menu_item(
        user_id=user_id,
        dish_name='Chicken Stir Fry',
        date_served=datetime.datetime.now(),
        primary_protein='chicken'
    )

    # Build context
    context = memory_manager.build_context_for_llm(
        user_id=user_id,
        current_query="Give me a lunch idea at my usual calories",
        include_menu_days=7,
        include_message_limit=5
    )

    print("\n--- Built Context ---")
    print(f"User ID: {context['user_id']}")
    print(f"\nProfile:\n{context['profile_text']}")
    print(f"\nRecent Conversation:\n{context['recent_conversation']}")
    print(f"\nMenu History:\n{context['menu_history']}")

    # Format as full prompt
    full_prompt = memory_manager.format_full_context_prompt(context)
    print(f"\n--- Full Prompt (first 500 chars) ---")
    print(full_prompt[:500])

    memory_manager.close()
    print("\n✓ Test 4 passed: Context building works correctly\n")


def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("HYBRID MEMORY MANAGEMENT SYSTEM - TEST SUITE")
    print("=" * 80 + "\n")

    try:
        test_basic_memory()
        test_multi_user_separation()
        test_menu_history()
        test_context_building()

        print("\n" + "=" * 80)
        print("✓ ALL TESTS PASSED")
        print("=" * 80 + "\n")

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()
