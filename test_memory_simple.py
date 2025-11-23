"""
Simple memory system test without requiring Tavily API
Tests core database and profile functionality
"""

import os
import datetime
from memory_manager import HybridMemoryManager, UserProfile


def test_profile_storage():
    """Test basic profile storage and retrieval"""
    print("=" * 80)
    print("TEST 1: Profile Storage and Retrieval")
    print("=" * 80)

    memory = HybridMemoryManager(db_path="data/test_simple.db")

    # Create and save a profile
    print("\n--- Creating user profile ---")
    memory.update_user_profile("alice", {
        'calories_target': 1800,
        'protein_min': 140,
        'fiber_min': 30,
        'allergies': ['peanuts', 'shellfish'],
        'dietary_restrictions': ['vegetarian'],
        'equipment': ['stove', 'oven'],
        'unit_preference': 'US'
    })

    # Retrieve profile
    profile = memory.get_user_profile("alice")
    print(f"\nRetrieved profile:")
    print(profile.format_for_prompt())

    # Verify
    assert profile.calories_target == 1800
    assert profile.protein_min == 140
    assert 'peanuts' in profile.allergies
    assert 'vegetarian' in profile.dietary_restrictions

    memory.close()
    print("\n✓ Test 1 passed: Profile storage works correctly\n")


def test_profile_merging():
    """Test that profile updates merge correctly"""
    print("=" * 80)
    print("TEST 2: Profile Merging")
    print("=" * 80)

    memory = HybridMemoryManager(db_path="data/test_simple.db")

    # Initial profile
    print("\n--- Creating initial profile ---")
    memory.update_user_profile("bob", {
        'calories_target': 2000,
        'allergies': ['peanuts']
    })

    # Update with new info
    print("\n--- Adding more information ---")
    memory.update_user_profile("bob", {
        'protein_min': 150,
        'allergies': ['shellfish']  # Should add to existing
    })

    # Retrieve
    profile = memory.get_user_profile("bob")
    print(f"\nMerged profile:")
    print(profile.format_for_prompt())

    # Verify merging
    assert profile.calories_target == 2000
    assert profile.protein_min == 150
    assert 'peanuts' in profile.allergies
    assert 'shellfish' in profile.allergies

    memory.close()
    print("\n✓ Test 2 passed: Profile merging works correctly\n")


def test_conversation_history():
    """Test conversation history storage"""
    print("=" * 80)
    print("TEST 3: Conversation History")
    print("=" * 80)

    memory = HybridMemoryManager(db_path="data/test_simple.db")

    # Add conversation messages
    print("\n--- Adding conversation messages ---")
    user_id = "charlie"

    memory.add_message(user_id, 'user', 'I need a meal plan', datetime.datetime.now())
    memory.add_message(user_id, 'assistant', 'Here is your meal plan...', datetime.datetime.now())
    memory.add_message(user_id, 'user', 'Can you add more protein?', datetime.datetime.now())
    memory.add_message(user_id, 'assistant', 'Updated with more protein...', datetime.datetime.now())

    # Retrieve history
    messages = memory.get_recent_messages(user_id, limit=10)
    print(f"\nRetrieved {len(messages)} messages:")
    for msg in messages:
        print(f"  {msg['role']}: {msg['content'][:50]}...")

    assert len(messages) == 4
    assert messages[0]['role'] == 'user'
    assert messages[1]['role'] == 'assistant'

    memory.close()
    print("\n✓ Test 3 passed: Conversation history works correctly\n")


def test_menu_history():
    """Test menu history storage"""
    print("=" * 80)
    print("TEST 4: Menu History")
    print("=" * 80)

    memory = HybridMemoryManager(db_path="data/test_simple.db")

    # Add menu items
    print("\n--- Adding menu items ---")
    user_id = "dave"

    from datetime import timedelta
    base_date = datetime.datetime.now()

    meals = [
        {'dish': 'Grilled Chicken Salad', 'protein': 'chicken', 'days_ago': 0},
        {'dish': 'Salmon Bowl', 'protein': 'salmon', 'days_ago': 1},
        {'dish': 'Tofu Stir Fry', 'protein': 'tofu', 'days_ago': 2},
    ]

    for meal in meals:
        memory.add_menu_item(
            user_id=user_id,
            dish_name=meal['dish'],
            date_served=base_date - timedelta(days=meal['days_ago']),
            primary_protein=meal['protein'],
            calories=450,
            protein=35
        )
        print(f"  Added: {meal['dish']}")

    # Retrieve history
    history = memory.get_menu_history(user_id, days=7)
    print(f"\nRetrieved {len(history)} menu items:")
    for item in history:
        print(f"  - {item['dish_name']} (protein: {item['primary_protein']})")

    assert len(history) == 3

    memory.close()
    print("\n✓ Test 4 passed: Menu history works correctly\n")


def test_multi_user_separation():
    """Test that users are kept separate"""
    print("=" * 80)
    print("TEST 5: Multi-user Separation")
    print("=" * 80)

    memory = HybridMemoryManager(db_path="data/test_simple.db")

    # Create profiles for two users
    print("\n--- Creating profiles for Eve and Frank ---")
    memory.update_user_profile("eve", {
        'calories_target': 1600,
        'dietary_restrictions': ['vegan']
    })

    memory.update_user_profile("frank", {
        'calories_target': 2400,
        'dietary_restrictions': ['keto']
    })

    # Retrieve and verify separation
    eve_profile = memory.get_user_profile("eve")
    frank_profile = memory.get_user_profile("frank")

    print(f"\nEve's profile:")
    print(f"  Calories: {eve_profile.calories_target}")
    print(f"  Diet: {eve_profile.dietary_restrictions}")

    print(f"\nFrank's profile:")
    print(f"  Calories: {frank_profile.calories_target}")
    print(f"  Diet: {frank_profile.dietary_restrictions}")

    assert eve_profile.calories_target == 1600
    assert frank_profile.calories_target == 2400
    assert 'vegan' in eve_profile.dietary_restrictions
    assert 'keto' in frank_profile.dietary_restrictions

    memory.close()
    print("\n✓ Test 5 passed: Users are properly separated\n")


def test_context_building():
    """Test full context building"""
    print("=" * 80)
    print("TEST 6: Context Building")
    print("=" * 80)

    memory = HybridMemoryManager(db_path="data/test_simple.db")

    user_id = "grace"

    # Set up profile
    print("\n--- Setting up complete context ---")
    memory.update_user_profile(user_id, {
        'calories_target': 1800,
        'protein_min': 120,
        'allergies': ['nuts']
    })

    # Add conversation
    memory.add_message(user_id, 'user', 'I need breakfast ideas', datetime.datetime.now())
    memory.add_message(user_id, 'assistant', 'Here are some options...', datetime.datetime.now())

    # Add menu
    memory.add_menu_item(
        user_id=user_id,
        dish_name='Oatmeal Bowl',
        date_served=datetime.datetime.now(),
        primary_protein='oats'
    )

    # Build context
    context = memory.build_context_for_llm(
        user_id=user_id,
        current_query="What should I have for lunch?",
        include_menu_days=7,
        include_message_limit=5
    )

    print(f"\nBuilt context contains:")
    print(f"  - Profile: {len(context['profile_text'])} chars")
    print(f"  - Conversation: {len(context['recent_conversation'])} chars")
    print(f"  - Menu: {len(context['menu_history'])} chars")

    print(f"\nFormatted context (first 300 chars):")
    full_context = memory.format_full_context_prompt(context)
    print(full_context[:300] + "...")

    assert 'calories' in context['profile_text'].lower()
    assert 'user' in context['recent_conversation'].lower()

    memory.close()
    print("\n✓ Test 6 passed: Context building works correctly\n")


def run_all_tests():
    """Run all simple tests"""
    print("\n" + "=" * 80)
    print("MEMORY SYSTEM - SIMPLE TEST SUITE (No API keys needed)")
    print("=" * 80 + "\n")

    try:
        test_profile_storage()
        test_profile_merging()
        test_conversation_history()
        test_menu_history()
        test_multi_user_separation()
        test_context_building()

        print("\n" + "=" * 80)
        print("✓ ALL TESTS PASSED")
        print("=" * 80 + "\n")

        print("The memory system is working correctly!")
        print("Next step: Add your Tavily API key to test with the full agent.\n")

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()
