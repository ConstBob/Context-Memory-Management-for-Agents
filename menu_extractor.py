"""
Menu extraction utilities for tracking diet plans
Extracts meals from agent responses for variety tracking
"""

import re
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from memory_manager import HybridMemoryManager


class MenuExtractor:
    """Extract structured menu information from agent responses"""

    def __init__(self, memory_manager: HybridMemoryManager):
        self.memory_manager = memory_manager

    def extract_meals_from_response(self, response: str, gemini_client) -> List[Dict]:
        """
        Use LLM to extract structured meal information from agent response
        Returns list of meals with dish names, proteins, calories, etc.
        """
        prompt = {
            'system': """Extract all meals/dishes mentioned in the response.
For each meal, return a JSON object with:
- dish_name (string): The name of the dish
- meal_type (string): breakfast, lunch, dinner, or snack
- primary_protein (string): main protein source (chicken, salmon, tofu, etc.)
- ingredients (list of strings): key ingredients
- calories (number): estimated calories if mentioned
- protein (number): protein grams if mentioned
- fiber (number): fiber grams if mentioned

Return a JSON array of meal objects. If no meals are mentioned, return an empty array [].

Example:
[
  {
    "dish_name": "Grilled Chicken Salad",
    "meal_type": "lunch",
    "primary_protein": "chicken",
    "ingredients": ["chicken breast", "mixed greens", "olive oil", "lemon"],
    "calories": 450,
    "protein": 35,
    "fiber": 5
  }
]""",
            'user': f"Response:\n{response}\n\nExtract meals as JSON array:"
        }

        try:
            result = gemini_client.infer(prompt)

            # Clean up response - remove markdown code blocks
            result = result.strip()
            if result.startswith('```'):
                result = re.sub(r'^```(?:json)?\s*', '', result)
                result = re.sub(r'\s*```$', '', result)

            meals = json.loads(result)

            # Validate it's a list
            if not isinstance(meals, list):
                return []

            return meals

        except Exception as e:
            print(f"Warning: Failed to extract meals: {e}")
            return []

    def save_meals_to_history(self, user_id: str, meals: List[Dict],
                             base_date: Optional[datetime] = None,
                             session_id: Optional[str] = None):
        """
        Save extracted meals to menu history
        If response contains multi-day plan, distribute across dates
        """
        if not meals:
            return

        if base_date is None:
            base_date = datetime.now()

        # Group meals by day if response contains "Day 1", "Day 2", etc.
        # For simplicity, assume sequential days
        current_date = base_date

        for meal in meals:
            try:
                self.memory_manager.add_menu_item(
                    user_id=user_id,
                    dish_name=meal.get('dish_name', 'Unknown'),
                    date_served=current_date,
                    meal_type=meal.get('meal_type'),
                    primary_protein=meal.get('primary_protein'),
                    ingredients=meal.get('ingredients', []),
                    calories=meal.get('calories'),
                    protein=meal.get('protein'),
                    fiber=meal.get('fiber'),
                    session_id=session_id
                )
            except Exception as e:
                print(f"Warning: Failed to save meal {meal.get('dish_name')}: {e}")

    def check_variety_compliance(self, user_id: str, proposed_meals: List[Dict],
                                 rotation_policy: Optional[Dict] = None) -> Dict:
        """
        Check if proposed meals comply with variety rules
        Returns dict with compliance status and violations
        """
        if rotation_policy is None:
            # Default policy from benchmark
            rotation_policy = {
                'no_repeat_days': 7,
                'max_same_primary_protein_per_week': 2,
                'ingredient_jaccard_max': 0.6
            }

        # Get recent menu history
        days_to_check = rotation_policy.get('no_repeat_days', 7)
        recent_history = self.memory_manager.get_menu_history(user_id, days=days_to_check)

        violations = []
        warnings = []

        # Check for exact dish repeats
        recent_dishes = {item['dish_name'].lower() for item in recent_history}
        for meal in proposed_meals:
            dish_name = meal.get('dish_name', '').lower()
            if dish_name in recent_dishes:
                violations.append(f"Dish '{meal.get('dish_name')}' was already served in the last {days_to_check} days")

        # Check protein rotation
        max_protein = rotation_policy.get('max_same_primary_protein_per_week', 2)
        protein_counts = {}

        for item in recent_history:
            protein = item.get('primary_protein', '').lower()
            if protein:
                protein_counts[protein] = protein_counts.get(protein, 0) + 1

        for meal in proposed_meals:
            protein = meal.get('primary_protein', '').lower()
            if protein:
                current_count = protein_counts.get(protein, 0)
                if current_count >= max_protein:
                    violations.append(f"Protein '{protein}' already used {current_count} times (limit: {max_protein})")

        # Check ingredient overlap (simplified - could be enhanced)
        max_jaccard = rotation_policy.get('ingredient_jaccard_max', 0.6)
        for meal in proposed_meals:
            meal_ingredients = set(ing.lower() for ing in meal.get('ingredients', []))
            if not meal_ingredients:
                continue

            for hist_item in recent_history[-3:]:  # Check last 3 days
                hist_ingredients = set(ing.lower() for ing in hist_item.get('ingredients', []))
                if not hist_ingredients:
                    continue

                intersection = meal_ingredients & hist_ingredients
                union = meal_ingredients | hist_ingredients
                jaccard = len(intersection) / len(union) if union else 0

                if jaccard > max_jaccard:
                    warnings.append(
                        f"Meal '{meal.get('dish_name')}' has high ingredient overlap "
                        f"({jaccard:.1%}) with '{hist_item['dish_name']}'"
                    )

        return {
            'compliant': len(violations) == 0,
            'violations': violations,
            'warnings': warnings
        }


def integrate_menu_tracking(user_id: str, response: str, gemini_client,
                           memory_manager: HybridMemoryManager,
                           session_id: Optional[str] = None):
    """
    Convenience function to extract and save menu items from a response
    Call this after generating a meal plan response
    """
    extractor = MenuExtractor(memory_manager)

    # Extract meals from response
    meals = extractor.extract_meals_from_response(response, gemini_client)

    # Save to history
    if meals:
        extractor.save_meals_to_history(
            user_id=user_id,
            meals=meals,
            base_date=datetime.now(),
            session_id=session_id
        )
        print(f"Saved {len(meals)} meals to menu history for {user_id}")

    return meals
