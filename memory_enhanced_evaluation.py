# memory_enhanced_evaluation.py
# Memory-Enhanced Evaluation for Context/Memory Management in Agents
# This script evaluates the agent's performance WITH memory management and RAG
# using the healthy diet benchmark dataset and grading helpers.

from dotenv import load_dotenv
load_dotenv()

import json
import os
import datetime
from typing import List, Dict, Any, Tuple
from memory_manager import HybridMemoryManager
from rag_system import RAGSystem
from agent_with_rag import get_agent_message_with_rag
from grading_helpers import (
    evaluate_day, extract_numbers_from_text, ANCHORS,
    check_no_repeats, check_protein_rotation, check_fasting_window,
    check_netcarb_caps, check_sodium, check_tool_usage,
    check_daily_targets, check_ingredient_diversity, jaccard
)

class MemoryEnhancedEvaluator:
    """Memory-enhanced evaluation with RAG for agent performance."""

    def __init__(self, benchmark_file: str = 'healthy_diet_benchmark.jsonl'):
        """Initialize the evaluator with memory management."""
        self.benchmark_file = benchmark_file
        self.tests = self._load_benchmark_data()
        self.results = []

        # Initialize memory manager and RAG system
        self.memory_manager = HybridMemoryManager(db_path="data/benchmark_eval.db")
        self.rag_system = RAGSystem(model_name="BAAI/bge-small-en-v1.5")

        # Create output directories
        os.makedirs('evaluation_logs', exist_ok=True)
        os.makedirs('evaluation_results', exist_ok=True)

        # Initialize log file
        self.log_file = f"evaluation_logs/memory_enhanced_evaluation_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self._log("Memory-Enhanced Evaluation Started (with RAG)")
        
    def _load_benchmark_data(self) -> List[Dict]:
        """Load benchmark test cases from JSONL file."""
        tests = []
        try:
            with open(self.benchmark_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        tests.append(json.loads(line))
            print(f"Loaded {len(tests)} test cases from {self.benchmark_file}")
        except FileNotFoundError:
            print(f"Error: {self.benchmark_file} not found")
            return []
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            return []
        return tests
    
    def _log(self, message: str):
        """Log message to file and console."""
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] {message}"
        print(log_entry)
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry + '\n')
    
    def _save_test_result(self, test_id: str, result: Dict[str, Any]):
        """Save individual test result to file."""
        result_file = f"evaluation_results/{test_id}_result.json"
        
        # Convert datetime objects to strings for JSON serialization
        serializable_result = {}
        for key, value in result.items():
            if isinstance(value, datetime.datetime):
                serializable_result[key] = value.isoformat()
            else:
                serializable_result[key] = value
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_result, f, indent=2, ensure_ascii=False)
        
        self._log(f"Saved result for {test_id} to {result_file}")
    
    def _extract_user_requirements(self, user_id: str, response: str) -> Dict[str, bool]:
        """Extract and check user-specific requirements from response."""
        import re
        requirements = {}

        if user_id == "u01":  # Peanut allergy, US units, Mediterranean/Mexican
            # Check for peanut ingredients (but allow "peanut-free" descriptions)
            response_lower = response.lower()
            peanut_ingredients = bool(re.search(r'\bpeanut(?!-free|\s+free|free)\b', response_lower))
            requirements['peanut_allergy'] = not peanut_ingredients
            requirements['us_units'] = any(unit in response_lower for unit in ["cup", "oz", "pound", "inch"])
            requirements['med_mex'] = any(cuisine in response_lower for cuisine in ["mediterranean", "mexican", "olive", "taco"])

        elif user_id == "u02":  # Vegetarian, lactose-free, metric units, fasting window
            response_lower = response.lower()
            requirements['vegetarian'] = "meat" not in response_lower and "chicken" not in response_lower
            requirements['lactose_free'] = "dairy" not in response_lower and "cheese" not in response_lower
            requirements['metric_units'] = any(unit in response_lower for unit in ["gram", "ml", "liter", "kg"])
            requirements['fasting_window'] = any(time in response_lower for time in ["12:", "15:", "19:", "lunch", "dinner"])

        elif user_id == "u03":  # Halal, low-glycemic, US units, grill/air fryer
            response_lower = response.lower()
            requirements['halal'] = "pork" not in response_lower and "alcohol" not in response_lower
            requirements['low_glycemic'] = "low-carb" in response_lower or "glycemic" in response_lower
            requirements['us_units'] = any(unit in response_lower for unit in ["cup", "oz", "pound"])
            requirements['grill_airfryer'] = any(method in response_lower for method in ["grill", "air fry", "bake"])

        elif user_id == "u04":  # Pescatarian, no tuna, US units, Japanese/Thai
            response_lower = response.lower()
            requirements['pescatarian'] = "fish" in response_lower or "seafood" in response_lower
            requirements['no_tuna'] = "tuna" not in response_lower
            requirements['us_units'] = any(unit in response.lower() for unit in ["cup", "oz", "pound"])
            requirements['japanese_thai'] = any(cuisine in response.lower() for cuisine in ["japanese", "thai", "miso", "curry"])
            
        return requirements
    
    def _check_context_dependencies(self, test: Dict, response: str) -> Dict[str, bool]:
        """Check if agent properly handles context dependencies."""
        dependencies = {}
        
        # Check inter-session memory dependencies
        inter_session = test.get('memory_dependencies', {}).get('inter_session', [])
        if inter_session:
            dependencies['inter_session_recall'] = any(
                ref in response.lower() for ref in ["previous", "last time", "before", "earlier"]
            )
        
        # Check for ambiguous references that require context
        ambiguous_phrases = ["my usual", "same targets", "like before", "don't repeat"]
        dependencies['ambiguous_handling'] = any(
            phrase in test['turns'][0]['text'].lower() for phrase in ambiguous_phrases
        )
        
        # Check for variety requirements
        if 'rotation_policy' in test:
            dependencies['variety_awareness'] = "variety" in response.lower() or "different" in response.lower()
        
        return dependencies
    
    def _extract_meal_times(self, response: str) -> List[str]:
        """Extract meal times from response text."""
        import re
        time_patterns = [
            r'(\d{1,2}:\d{2})',  # HH:MM format
            r'(\d{1,2}\s*:\s*\d{2})',  # H:MM format with spaces
            r'(\d{1,2}\s*am|\d{1,2}\s*pm)',  # 12-hour format
        ]
        
        times = []
        for pattern in time_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            times.extend(matches)
        
        return times
    
    def _extract_proposed_dishes(self, response: str) -> List[str]:
        """Extract proposed dishes from response text."""
        import re
        
        # Look for dish names in various formats
        dish_patterns = [
            r'\*\*([^*]+)\*\*',  # **Dish Name**
            r'•\s*([^•\n]+)',    # • Dish Name
            r'-\s*([^-\n]+)',    # - Dish Name
            r'(\d+\.\s*[^\n]+)', # 1. Dish Name
        ]
        
        dishes = []
        for pattern in dish_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            dishes.extend([match.strip() for match in matches])
        
        return dishes
    
    def _extract_primary_proteins(self, response: str) -> List[str]:
        """Extract primary protein sources from response text."""
        protein_keywords = [
            'chicken', 'beef', 'pork', 'lamb', 'turkey', 'duck',
            'fish', 'salmon', 'tuna', 'cod', 'shrimp', 'crab',
            'eggs', 'tofu', 'tempeh', 'beans', 'lentils', 'chickpeas',
            'cheese', 'yogurt', 'milk', 'whey', 'casein'
        ]
        
        proteins = []
        response_lower = response.lower()
        for protein in protein_keywords:
            if protein in response_lower:
                proteins.append(protein)
        
        return proteins
    
    def _extract_ingredients(self, response: str) -> List[str]:
        """Extract ingredients from response text."""
        import re
        
        # Common ingredient patterns
        ingredient_patterns = [
            r'(\w+\s+oil)',  # olive oil, coconut oil
            r'(\w+\s+cheese)',  # cheddar cheese, mozzarella
            r'(\w+\s+vegetables?)',  # mixed vegetables
            r'(\w+\s+spices?)',  # Italian spices
            r'(\w+\s+herbs?)',  # fresh herbs
        ]
        
        ingredients = []
        for pattern in ingredient_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            ingredients.extend([match.strip() for match in matches])
        
        return ingredients
    
    def _extract_netcarbs_by_meal(self, response: str) -> Dict[str, float]:
        """Extract net carbs by meal from response text."""
        import re
        
        meal_netcarbs = {}
        meal_keywords = ['breakfast', 'lunch', 'dinner', 'snack']
        
        for meal in meal_keywords:
            # Look for net carbs in context of each meal
            pattern = rf'{meal}[^.]*?(\d+)\s*g\s*net\s*carb'
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                meal_netcarbs[meal] = float(match.group(1))
        
        return meal_netcarbs
    
    def _extract_sodium_content(self, response: str) -> float:
        """Extract sodium content from response text."""
        import re
        
        # Look for sodium content in mg
        pattern = r'(\d+)\s*mg\s*sodium'
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return float(match.group(1))
        
        return 0.0
    
    def _extract_tool_usage(self, response: str) -> List[str]:
        """Extract tool usage from response text."""
        tools_used = []
        
        # Check for calculator usage
        if any(word in response.lower() for word in ['calculate', 'calculation', 'total', 'sum', 'add up']):
            tools_used.append('calculator')
        
        # Check for search usage
        if any(word in response.lower() for word in ['search', 'found', 'according to', 'source', 'link']):
            tools_used.append('search')
        
        return tools_used
    
    def _evaluate_single_test(self, test: Dict) -> Dict[str, Any]:
        """Evaluate a single test case comprehensively."""
        test_id = test['id']
        user_id = test['user_id']
        turns = test['turns']
        ground_truth = test['ground_truth']
        
        self._log(f"Evaluating test {test_id}: {user_id}")
        self._log(f"  Total turns: {len(turns)}")
        self._log(f"  Ground Truth: {ground_truth.get('final_answer_criteria', 'N/A')}")
        
        try:
            # Process all turns in the conversation
            all_responses = []
            conversation_context = ""
            
            all_tools_used = []
            
            total_latency = 0.0
            total_input_tokens = 0
            total_output_tokens = 0

            for i, turn in enumerate(turns):
                if turn['role'] == 'user':
                    user_message = turn['text']
                    self._log(f"  Turn {i+1} - User: {user_message}")
                    from agent import get_agent_message
                    # Get agent response with memory and RAG
                    result = get_agent_message_with_rag(
                        user_id, user_message, datetime.datetime.now(),
                        memory_manager=self.memory_manager,
                        rag_system=self.rag_system,
                        return_metadata=True
                    )
                    # result = get_agent_message(user_id, user_message, datetime.datetime.now(), return_metadata=True)

                    response = result["response"]
                    tools_used = result["tools_used"]
                    all_tools_used.extend(tools_used)
                    
                    latency = result.get("latency", 0.0)
                    usage = result.get("usage", {}) or result.get("token_usage", {})

                    # 这里假设 usage = {"input_tokens": ..., "output_tokens": ...}
                    total_latency += latency
                    total_input_tokens += usage.get("input_tokens", 0)
                    total_output_tokens += usage.get("output_tokens", 0)

                    # Log the full agent response
                    self._log(f"  Turn {i+1} - Agent Response:")
                    self._log(f"  {'='*50}")
                    self._log(f"  {response}")
                    self._log(f"  {'='*50}")

                    all_responses.append(response)
                    conversation_context += f"User: {user_message}\nAgent: {response}\n"
            
            # Use the last response for evaluation (final answer)
            # final_response = all_responses[-1] if all_responses else ""
            
            # # Basic nutrition evaluation (using final response)
            # nutrition_valid = evaluate_day(user_id, final_response, 0.1)
            # nutrition_numbers = extract_numbers_from_text(final_response)
            
                        # Use the last response for evaluation (final answer)
            final_response = all_responses[-1] if all_responses else ""
            
            nutrition_numbers = extract_numbers_from_text(final_response)
            nutrition_valid = self._evaluate_nutrition(
                test,
                final_response,
                nutrition_numbers,
            )

            
            # User-specific requirements
            user_requirements = self._extract_user_requirements(user_id, final_response)
            
            # Context dependency checks
            context_deps = self._check_context_dependencies(test, final_response)
            
            # Extract additional data for comprehensive evaluation
            meal_times = self._extract_meal_times(final_response)
            proposed_dishes = self._extract_proposed_dishes(final_response)
            primary_proteins = self._extract_primary_proteins(final_response)
            ingredients = self._extract_ingredients(final_response)
            netcarbs_by_meal = self._extract_netcarbs_by_meal(final_response)
            sodium_content = self._extract_sodium_content(final_response)
            tools_used = list(set(all_tools_used))
            
            # Advanced evaluation using grading_helpers methods
            no_repeats_valid = True
            protein_rotation_valid = True
            fasting_window_valid = True
            netcarb_caps_valid = True
            sodium_valid = True
            tool_usage_valid = True
            
            # Check for no repeats (if menu history is available)
            if 'menu_history' in test and proposed_dishes:
                menu_history = test.get('menu_history', [])
                no_repeats_valid = check_no_repeats(menu_history, proposed_dishes)
            
            # Check protein rotation (if rotation policy is available)
            if 'rotation_policy' in test and primary_proteins:
                rotation_policy = test.get('rotation_policy', {})
                max_same_protein = rotation_policy.get('max_same_primary_protein_per_week', 2)
                protein_rotation_valid = check_protein_rotation(primary_proteins, max_same_protein)
            
            # Check fasting window (if fasting window is specified)
            if meal_times and user_id in ANCHORS:
                user_anchor = ANCHORS[user_id]
                if 'fasting_window' in user_anchor:
                    window_start, window_end = user_anchor['fasting_window']
                    fasting_window_valid = check_fasting_window(meal_times, window_start, window_end)
            
            # Check net carb caps (if caps are specified)
            if netcarbs_by_meal:
                # Default caps - can be customized based on test requirements
                default_caps = {'lunch': 40.0, 'dinner': 30.0}
                netcarb_caps_valid = check_netcarb_caps(netcarbs_by_meal, default_caps)
            
            # Check sodium content (if limit is specified)
            if sodium_content > 0:
                sodium_limit = 2300.0  # Default sodium limit in mg
                sodium_valid = check_sodium(sodium_content, sodium_limit)
            
            # Check tool usage (if required tools are specified)
            if 'ground_truth' in test and 'must_use' in test['ground_truth']:
                must_use_tools = test['ground_truth']['must_use']
                tool_usage_valid = check_tool_usage(tools_used, must_use_tools)
            
            # Check ingredient diversity (if ingredients are available)
            ingredient_diversity_valid = True
            if ingredients and len(ingredients) > 1:
                # Group ingredients by day (simplified - assume single day for now)
                day_ingredients = [ingredients]  # Single day ingredients
                max_jaccard = 0.6  # Default max similarity threshold
                ingredient_diversity_valid = check_ingredient_diversity(day_ingredients, max_jaccard)
            
            # Enhanced nutrition evaluation using check_daily_targets
            enhanced_nutrition_valid = True
            if nutrition_numbers and user_id in ANCHORS:
                user_anchor = ANCHORS[user_id]
                target_kcal = user_anchor['kcal']
                protein_min = user_anchor.get('protein_min_g', 0)
                fiber_min = user_anchor.get('fiber_min_g', None)
                enhanced_nutrition_valid = check_daily_targets(
                    nutrition_numbers, target_kcal, 0.1, protein_min, fiber_min
                )
            
            # Calculate overall scores
            user_req_score = sum(user_requirements.values()) / len(user_requirements) if user_requirements else 0
            context_score = sum(context_deps.values()) / len(context_deps) if context_deps else 0
            
            # Calculate advanced evaluation scores
            advanced_scores = {
                'no_repeats': no_repeats_valid,
                'protein_rotation': protein_rotation_valid,
                'fasting_window': fasting_window_valid,
                'netcarb_caps': netcarb_caps_valid,
                'sodium': sodium_valid,
                'tool_usage': tool_usage_valid,
                'ingredient_diversity': ingredient_diversity_valid,
                'enhanced_nutrition': enhanced_nutrition_valid
            }
            advanced_score = sum(advanced_scores.values()) / len(advanced_scores)
            
            # Determine if test passed
            passed = nutrition_valid and user_req_score > 0.5
            
            result = {
                'test_id': test_id,
                'user_id': user_id,
                'question': turns[0]['text'] if turns else "",  # First user message
                'response': final_response,  # Final agent response
                'all_responses': all_responses,  # All responses in conversation
                'conversation_context': conversation_context,  # Full conversation
                'passed': passed,
                'nutrition_valid': nutrition_valid,
                'nutrition_numbers': nutrition_numbers,
                'user_requirements': user_requirements,
                'user_req_score': user_req_score,
                'context_dependencies': context_deps,
                'context_score': context_score,
                'tool_usage_valid': tool_usage_valid,
                'ground_truth': ground_truth,
                'memory_dependencies': test.get('memory_dependencies', {}),
                'length': test.get('length', 'unknown'),
                'total_turns': len(turns),
                'evaluation_timestamp': datetime.datetime.now(),
                
                'latency_total': total_latency,
                'input_tokens_total': total_input_tokens,
                'output_tokens_total': total_output_tokens,
                
                # Advanced evaluation metrics
                'advanced_scores': advanced_scores,
                'advanced_score': advanced_score,
                'no_repeats_valid': no_repeats_valid,
                'protein_rotation_valid': protein_rotation_valid,
                'fasting_window_valid': fasting_window_valid,
                'netcarb_caps_valid': netcarb_caps_valid,
                'sodium_valid': sodium_valid,
                'ingredient_diversity_valid': ingredient_diversity_valid,
                'enhanced_nutrition_valid': enhanced_nutrition_valid,
                
                # Extracted data for analysis
                'meal_times': meal_times,
                'proposed_dishes': proposed_dishes,
                'primary_proteins': primary_proteins,
                'ingredients': ingredients,
                'netcarbs_by_meal': netcarbs_by_meal,
                'sodium_content': sodium_content,
                'tools_used': tools_used
            }
            
            # Save individual test result
            self._save_test_result(test_id, result)
            
            # Log evaluation results
            self._log(f"  Evaluation Results for {test_id}:")
            self._log(f"    Passed: {passed}")
            self._log(f"    Nutrition Valid: {nutrition_valid}")
            self._log(f"    Nutrition Numbers: {nutrition_numbers}")
            self._log(f"    User Requirements Score: {user_req_score:.2f}")
            self._log(f"    Context Score: {context_score:.2f}")
            self._log(f"    Advanced Score: {advanced_score:.2f}")
            self._log(f"    No Repeats: {no_repeats_valid}")
            self._log(f"    Protein Rotation: {protein_rotation_valid}")
            self._log(f"    Fasting Window: {fasting_window_valid}")
            self._log(f"    Net Carb Caps: {netcarb_caps_valid}")
            self._log(f"    Sodium Valid: {sodium_valid}")
            self._log(f"    Ingredient Diversity: {ingredient_diversity_valid}")
            self._log(f"    Enhanced Nutrition: {enhanced_nutrition_valid}")
            self._log(f"    Tool Usage Valid: {tool_usage_valid}")
            self._log(f"    Tools Used: {tools_used}")
            self._log(f"    Proposed Dishes: {proposed_dishes}")
            self._log(f"    Primary Proteins: {primary_proteins}")
            self._log(f"    Ingredients: {ingredients}")
            self._log(f"    Total Latency (s): {total_latency:.3f}")
            self._log(f"    Total Input Tokens: {total_input_tokens}")
            self._log(f"    Total Output Tokens: {total_output_tokens}")
            
            return result
            
        except Exception as e:
            print(f"Error in test {test_id}: {e}")
            return {
                'test_id': test_id,
                'user_id': user_id,
                'question': question,
                'response': None,
                'passed': False,
                'error': str(e),
                'nutrition_valid': False,
                'user_req_score': 0,
                'context_score': 0,
                'tool_usage_valid': False
            }
            
    def _parse_macro_criteria(self, criteria_text: str):
        """
        从 ground_truth['final_answer_criteria'] 里解析：
        - 是否是整日 Total
        - kcal 区间
        - protein 最小值
        - fiber 最小值
        """
        import re

        s = (criteria_text or "").lower()
        macros = {
            "is_total": "total" in s,   # 包含 "Total 1710–1890 kcal" 之类
            "kcal_low": None,
            "kcal_high": None,
            "protein_min": None,
            "fiber_min": None,
        }

        # 1) Total 1520–1680 kcal / Total 1710–1890 kcal 这类整日目标
        m_total = re.search(r"total\s+(\d{3,4})\s*[-–]\s*(\d{3,4})\s*kcal", s)
        if m_total:
            lo, hi = float(m_total.group(1)), float(m_total.group(2))
            if lo > hi:
                lo, hi = hi, lo
            macros["kcal_low"], macros["kcal_high"] = lo, hi
        else:
            # 2) 一般的区间，比如 "540–660 kcal"、"360–440 kcal"
            m = re.search(r"(\d{3,4})\s*[-–]\s*(\d{3,4})\s*kcal", s)
            if m:
                lo, hi = float(m.group(1)), float(m.group(2))
                if lo > hi:
                    lo, hi = hi, lo
                macros["kcal_low"], macros["kcal_high"] = lo, hi

        # 3) 只有一个 kcal 数字，例如 "around 300 kcal"
        if macros["kcal_low"] is None and macros["kcal_high"] is None:
            m1 = re.search(r"(\d{3,4})\s*(?:kcal|cal|cals)", s)
            if m1:
                center = float(m1.group(1))
                macros["kcal_low"] = center * 0.8
                macros["kcal_high"] = center * 1.2

        # 4) protein ≥XX g / ≥XX g protein
        m_prot = re.search(r"protein\s*(?:≥|>=)\s*(\d{1,3})\s*g", s)
        if not m_prot:
            m_prot = re.search(r"(?:≥|>=)\s*(\d{1,3})\s*g\s*protein", s)
        if m_prot:
            macros["protein_min"] = float(m_prot.group(1))

        # 5) fiber ≥XX g / ≥XX g fiber
        m_fib = re.search(r"fiber\s*(?:≥|>=)\s*(\d{1,3})\s*g", s)
        if not m_fib:
            m_fib = re.search(r"(?:≥|>=)\s*(\d{1,3})\s*g\s*fiber", s)
        if m_fib:
            macros["fiber_min"] = float(m_fib.group(1))

        return macros
    
    
    def _evaluate_nutrition(
        self,
        test: Dict[str, Any],
        response: str,
        nutrition_numbers: Dict[str, float],
    ) -> bool:
        user_id = test.get("user_id")
        ground_truth = test.get("ground_truth", {}) or {}
        criteria_text = ground_truth.get("final_answer_criteria", "")

        macros = self._parse_macro_criteria(criteria_text)

        if (not criteria_text) or (
            macros["kcal_low"] is None
            and macros["protein_min"] is None
            and macros["fiber_min"] is None
        ):
            return evaluate_day(user_id, response, 0.1)

        if not nutrition_numbers:
            return False

        kcal = nutrition_numbers.get("kcal", 0.0)
        protein = nutrition_numbers.get("protein_g", 0.0)
        fiber = nutrition_numbers.get("fiber_g", 0.0)

        kcal_ok = True
        protein_ok = True
        fiber_ok = True

        if macros["kcal_low"] is not None and macros["kcal_high"] is not None and kcal:
            kcal_ok = macros["kcal_low"] <= kcal <= macros["kcal_high"]
        elif macros["is_total"] and user_id in ANCHORS:
            target = ANCHORS[user_id]["kcal"]
            kcal_ok = abs(kcal - target) <= 0.1 * target

        if macros["protein_min"] is not None:
            protein_ok = protein >= macros["protein_min"]
        elif macros["is_total"] and user_id in ANCHORS and "protein_min_g" in ANCHORS[user_id]:
            protein_ok = protein >= ANCHORS[user_id]["protein_min_g"]

        if macros["fiber_min"] is not None:
            fiber_ok = fiber >= macros["fiber_min"]
        elif macros["is_total"]:
            anchor_f = ANCHORS.get(user_id, {}).get("fiber_min_g", None)
            if anchor_f is not None:
                fiber_ok = fiber >= anchor_f

        return kcal_ok and protein_ok and fiber_ok


    
    def run_evaluation(self) -> Dict[str, Any]:
        """Run comprehensive baseline evaluation."""
        self._log("Starting Comprehensive Baseline Evaluation")
        self._log("=" * 60)
        
        self.results = []
        
        for i, test in enumerate(self.tests):
            self._log(f"\nProgress: {i+1}/{len(self.tests)}")
            result = self._evaluate_single_test(test)
            self.results.append(result)
            
            # Log immediate feedback
            status = "PASS" if result['passed'] else "FAIL"
            self._log(f"Test {result['test_id']}: {status}")
            if 'error' in result:
                self._log(f"  Error: {result['error']}")
            else:
                self._log(f"  Nutrition: {'Valid' if result['nutrition_valid'] else 'Invalid'}")
                self._log(f"  Nutrition Numbers: {result['nutrition_numbers']}")
                self._log(f"  User Requirements: {result['user_req_score']:.2f}")
                self._log(f"  User Requirements Details: {result['user_requirements']}")
                self._log(f"  Context Handling: {result['context_score']:.2f}")
                self._log(f"  Advanced Score: {result.get('advanced_score', 0):.2f}")
                self._log(f"  No Repeats: {'Valid' if result.get('no_repeats_valid', False) else 'Invalid'}")
                self._log(f"  Protein Rotation: {'Valid' if result.get('protein_rotation_valid', False) else 'Invalid'}")
                self._log(f"  Fasting Window: {'Valid' if result.get('fasting_window_valid', False) else 'Invalid'}")
                self._log(f"  Net Carb Caps: {'Valid' if result.get('netcarb_caps_valid', False) else 'Invalid'}")
                self._log(f"  Sodium: {'Valid' if result.get('sodium_valid', False) else 'Invalid'}")
                self._log(f"  Tool Usage: {'Valid' if result['tool_usage_valid'] else 'Invalid'}")
                self._log(f"  Tools Used: {result.get('tools_used', [])}")
        
        # Save overall results
        self._save_overall_results()
        
        return self._analyze_results()
    
    def _analyze_results(self) -> Dict[str, Any]:
        """Analyze evaluation results and calculate metrics."""
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r['passed'])
        
        # Task completion rate
        completion_rate = passed_tests / total_tests if total_tests > 0 else 0
        
        total_latency_sum = sum(r.get('latency_total', 0.0) for r in self.results)
        total_input_tokens_sum = sum(r.get('input_tokens_total', 0) for r in self.results)
        total_output_tokens_sum = sum(r.get('output_tokens_total', 0) for r in self.results)

        avg_latency_per_test = total_latency_sum / total_tests if total_tests > 0 else 0.0
        avg_input_tokens_per_test = total_input_tokens_sum / total_tests if total_tests > 0 else 0.0
        avg_output_tokens_per_test = total_output_tokens_sum / total_tests if total_tests > 0 else 0.0
        
        # Tool call efficiency (simplified)
        tool_efficiency = sum(1 for r in self.results if r.get('tool_usage_valid', False)) / total_tests
        
        # Response quality metrics
        nutrition_valid_rate = sum(1 for r in self.results if r.get('nutrition_valid', False)) / total_tests
        avg_user_req_score = sum(r.get('user_req_score', 0) for r in self.results) / total_tests
        avg_context_score = sum(r.get('context_score', 0) for r in self.results) / total_tests
        avg_advanced_score = sum(r.get('advanced_score', 0) for r in self.results) / total_tests
        
        # Advanced evaluation metrics
        no_repeats_rate = sum(1 for r in self.results if r.get('no_repeats_valid', False)) / total_tests
        protein_rotation_rate = sum(1 for r in self.results if r.get('protein_rotation_valid', False)) / total_tests
        fasting_window_rate = sum(1 for r in self.results if r.get('fasting_window_valid', False)) / total_tests
        netcarb_caps_rate = sum(1 for r in self.results if r.get('netcarb_caps_valid', False)) / total_tests
        sodium_rate = sum(1 for r in self.results if r.get('sodium_valid', False)) / total_tests
        ingredient_diversity_rate = sum(1 for r in self.results if r.get('ingredient_diversity_valid', False)) / total_tests
        enhanced_nutrition_rate = sum(1 for r in self.results if r.get('enhanced_nutrition_valid', False)) / total_tests
        
        # Context management weaknesses
        inter_session_tests = [r for r in self.results if r.get('memory_dependencies', {}).get('inter_session')]
        inter_session_pass_rate = sum(1 for r in inter_session_tests if r['passed']) / len(inter_session_tests) if inter_session_tests else 0
        
        # Length-based analysis
        short_tests = [r for r in self.results if r.get('length') == 'short']
        medium_tests = [r for r in self.results if r.get('length') == 'medium']
        long_tests = [r for r in self.results if r.get('length') == 'long']
        
        short_pass_rate = sum(1 for r in short_tests if r['passed']) / len(short_tests) if short_tests else 0
        medium_pass_rate = sum(1 for r in medium_tests if r['passed']) / len(medium_tests) if medium_tests else 0
        long_pass_rate = sum(1 for r in long_tests if r['passed']) / len(long_tests) if long_tests else 0
        
        # User-specific analysis
        user_analysis = {}
        for user_id in ['u01', 'u02', 'u03', 'u04']:
            user_tests = [r for r in self.results if r['user_id'] == user_id]
            if user_tests:
                user_pass_rate = sum(1 for r in user_tests if r['passed']) / len(user_tests)
                user_avg_req_score = sum(r.get('user_req_score', 0) for r in user_tests) / len(user_tests)
                user_analysis[user_id] = {
                    'pass_rate': user_pass_rate,
                    'avg_requirement_score': user_avg_req_score,
                    'total_tests': len(user_tests)
                }
        
        analysis = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'task_completion_rate': completion_rate,
            'tool_call_efficiency': tool_efficiency,
            'response_quality': {
                'nutrition_valid_rate': nutrition_valid_rate,
                'avg_user_requirement_score': avg_user_req_score,
                'avg_context_score': avg_context_score,
                'avg_advanced_score': avg_advanced_score
            },
            'advanced_evaluation': {
                'no_repeats_rate': no_repeats_rate,
                'protein_rotation_rate': protein_rotation_rate,
                'fasting_window_rate': fasting_window_rate,
                'netcarb_caps_rate': netcarb_caps_rate,
                'sodium_rate': sodium_rate,
                'ingredient_diversity_rate': ingredient_diversity_rate,
                'enhanced_nutrition_rate': enhanced_nutrition_rate
            },
            'context_weaknesses': {
                'inter_session_pass_rate': inter_session_pass_rate,
                'inter_session_tests': len(inter_session_tests)
            },
            'length_analysis': {
                'short_pass_rate': short_pass_rate,
                'medium_pass_rate': medium_pass_rate,
                'long_pass_rate': long_pass_rate,
                'short_tests': len(short_tests),
                'medium_tests': len(medium_tests),
                'long_tests': len(long_tests)
            },
            'latency_and_tokens': {
                'total_latency': total_latency_sum,
                'total_input_tokens': total_input_tokens_sum,
                'total_output_tokens': total_output_tokens_sum,
                'avg_latency_per_test': avg_latency_per_test,
                'avg_input_tokens_per_test': avg_input_tokens_per_test,
                'avg_output_tokens_per_test': avg_output_tokens_per_test,
            },
            'user_analysis': user_analysis,
            'failed_tests': [r for r in self.results if not r['passed']]
        }
        
        return analysis
    
    def _save_overall_results(self):
        """Save overall evaluation results and statistics."""
        # Save summary statistics
        summary_file = f"evaluation_results/summary_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Calculate basic statistics
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r['passed'])
        completion_rate = passed_tests / total_tests if total_tests > 0 else 0
        
        # User-specific statistics
        user_stats = {}
        for user_id in ['u01', 'u02', 'u03', 'u04']:
            user_tests = [r for r in self.results if r['user_id'] == user_id]
            if user_tests:
                user_passed = sum(1 for r in user_tests if r['passed'])
                user_stats[user_id] = {
                    'total_tests': len(user_tests),
                    'passed_tests': user_passed,
                    'pass_rate': user_passed / len(user_tests)
                }
        
        # Length-based statistics
        length_stats = {}
        for length in ['short', 'medium', 'long']:
            length_tests = [r for r in self.results if r.get('length') == length]
            if length_tests:
                length_passed = sum(1 for r in length_tests if r['passed'])
                length_stats[length] = {
                    'total_tests': len(length_tests),
                    'passed_tests': length_passed,
                    'pass_rate': length_passed / len(length_tests)
                }
        
        summary = {
            'evaluation_timestamp': datetime.datetime.now().isoformat(),
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'completion_rate': completion_rate,
            'user_statistics': user_stats,
            'length_statistics': length_stats,
            'advanced_evaluation_summary': {
                'no_repeats_rate': sum(1 for r in self.results if r.get('no_repeats_valid', False)) / total_tests,
                'protein_rotation_rate': sum(1 for r in self.results if r.get('protein_rotation_valid', False)) / total_tests,
                'fasting_window_rate': sum(1 for r in self.results if r.get('fasting_window_valid', False)) / total_tests,
                'netcarb_caps_rate': sum(1 for r in self.results if r.get('netcarb_caps_valid', False)) / total_tests,
                'sodium_rate': sum(1 for r in self.results if r.get('sodium_valid', False)) / total_tests,
                'ingredient_diversity_rate': sum(1 for r in self.results if r.get('ingredient_diversity_valid', False)) / total_tests,
                'enhanced_nutrition_rate': sum(1 for r in self.results if r.get('enhanced_nutrition_valid', False)) / total_tests
            },
            'test_results': [
                {
                    'test_id': r['test_id'],
                    'user_id': r['user_id'],
                    'passed': r['passed'],
                    'nutrition_valid': r.get('nutrition_valid', False),
                    'user_req_score': r.get('user_req_score', 0),
                    'context_score': r.get('context_score', 0),
                    'advanced_score': r.get('advanced_score', 0),
                    'no_repeats_valid': r.get('no_repeats_valid', False),
                    'protein_rotation_valid': r.get('protein_rotation_valid', False),
                    'fasting_window_valid': r.get('fasting_window_valid', False),
                    'netcarb_caps_valid': r.get('netcarb_caps_valid', False),
                    'sodium_valid': r.get('sodium_valid', False),
                    'length': r.get('length', 'unknown'), 
                    'latency_total': r.get('latency_total', 0.0),
                    'input_tokens_total': r.get('input_tokens_total', 0),
                    'output_tokens_total': r.get('output_tokens_total', 0),
                } for r in self.results
            ]
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        self._log(f"Saved overall results to {summary_file}")
    
    def print_detailed_report(self, analysis: Dict[str, Any]):
        """Print detailed evaluation report."""
        print("\n" + "=" * 80)
        print("COMPREHENSIVE BASELINE EVALUATION REPORT")
        print("=" * 80)
        
        # Overall metrics
        print(f"\nOVERALL METRICS:")
        print(f"Total Tests: {analysis['total_tests']}")
        print(f"Passed Tests: {analysis['passed_tests']}")
        print(f"Task Completion Rate: {analysis['task_completion_rate']:.2%}")
        print(f"Tool Call Efficiency: {analysis['tool_call_efficiency']:.2%}")
        lt = analysis.get('latency_and_tokens', {})
        print(f"\nLATENCY & TOKEN USAGE:")
        print(f"Total Latency (s): {lt.get('total_latency', 0.0):.3f}")
        print(f"Total Input Tokens: {lt.get('total_input_tokens', 0)}")
        print(f"Total Output Tokens: {lt.get('total_output_tokens', 0)}")
        print(f"Avg Latency per Test (s): {lt.get('avg_latency_per_test', 0.0):.3f}")
        print(f"Avg Input Tokens per Test: {lt.get('avg_input_tokens_per_test', 0):.1f}")
        print(f"Avg Output Tokens per Test: {lt.get('avg_output_tokens_per_test', 0):.1f}")
        
        # Response quality
        print(f"\nRESPONSE QUALITY:")
        print(f"Nutrition Valid Rate: {analysis['response_quality']['nutrition_valid_rate']:.2%}")
        print(f"Average User Requirement Score: {analysis['response_quality']['avg_user_requirement_score']:.2f}")
        print(f"Average Context Score: {analysis['response_quality']['avg_context_score']:.2f}")
        print(f"Average Advanced Score: {analysis['response_quality']['avg_advanced_score']:.2f}")
        
        # Advanced evaluation
        print(f"\nADVANCED EVALUATION:")
        print(f"No Repeats Rate: {analysis['advanced_evaluation']['no_repeats_rate']:.2%}")
        print(f"Protein Rotation Rate: {analysis['advanced_evaluation']['protein_rotation_rate']:.2%}")
        print(f"Fasting Window Rate: {analysis['advanced_evaluation']['fasting_window_rate']:.2%}")
        print(f"Net Carb Caps Rate: {analysis['advanced_evaluation']['netcarb_caps_rate']:.2%}")
        print(f"Sodium Rate: {analysis['advanced_evaluation']['sodium_rate']:.2%}")
        print(f"Ingredient Diversity Rate: {analysis['advanced_evaluation']['ingredient_diversity_rate']:.2%}")
        print(f"Enhanced Nutrition Rate: {analysis['advanced_evaluation']['enhanced_nutrition_rate']:.2%}")
        
        # Context management weaknesses
        print(f"\nCONTEXT MANAGEMENT WEAKNESSES:")
        print(f"Inter-session Memory Pass Rate: {analysis['context_weaknesses']['inter_session_pass_rate']:.2%}")
        print(f"Inter-session Tests: {analysis['context_weaknesses']['inter_session_tests']}")
        
        # Length-based analysis
        print(f"\nLENGTH-BASED ANALYSIS:")
        print(f"Short Tests: {analysis['length_analysis']['short_tests']} (Pass Rate: {analysis['length_analysis']['short_pass_rate']:.2%})")
        print(f"Medium Tests: {analysis['length_analysis']['medium_tests']} (Pass Rate: {analysis['length_analysis']['medium_pass_rate']:.2%})")
        print(f"Long Tests: {analysis['length_analysis']['long_tests']} (Pass Rate: {analysis['length_analysis']['long_pass_rate']:.2%})")
        
        # User-specific analysis
        print(f"\nUSER-SPECIFIC ANALYSIS:")
        for user_id, user_data in analysis['user_analysis'].items():
            print(f"{user_id}: {user_data['total_tests']} tests, Pass Rate: {user_data['pass_rate']:.2%}, Avg Requirement Score: {user_data['avg_requirement_score']:.2f}")
        
        # Failed tests analysis
        failed_tests = analysis['failed_tests']
        print(f"\nFAILED TESTS ANALYSIS:")
        print(f"Total Failed: {len(failed_tests)}")
        
        for test in failed_tests[:5]:  # Show first 5 failed tests
            print(f"\nFailed Test {test['test_id']} ({test['user_id']}):")
            print(f"  Question: {test['question'][:100]}...")
            if 'error' in test:
                print(f"  Error: {test['error']}")
            else:
                print(f"  Nutrition Valid: {test.get('nutrition_valid', False)}")
                print(f"  User Requirements Score: {test.get('user_req_score', 0):.2f}")
                print(f"  Context Score: {test.get('context_score', 0):.2f}")
        
        if len(failed_tests) > 5:
            print(f"  ... and {len(failed_tests) - 5} more failed tests")
    
    def demonstrate_context_weaknesses(self):
        """Demonstrate specific context management weaknesses."""
        print("\n" + "=" * 80)
        print("CONTEXT MANAGEMENT WEAKNESSES DEMONSTRATION")
        print("=" * 80)
        
        # Find tests with inter-session dependencies
        inter_session_tests = [r for r in self.results if r.get('memory_dependencies', {}).get('inter_session')]
        
        print(f"\nINTER-SESSION MEMORY WEAKNESSES:")
        print(f"Tests requiring inter-session memory: {len(inter_session_tests)}")
        
        for test in inter_session_tests[:3]:  # Show first 3 examples
            print(f"\nTest {test['test_id']} ({test['user_id']}):")
            print(f"  Question: {test['question']}")
            print(f"  Inter-session Dependencies: {test.get('memory_dependencies', {}).get('inter_session', [])}")
            print(f"  Passed: {test['passed']}")
            print(f"  Context Score: {test.get('context_score', 0):.2f}")
            if test.get('response'):
                print(f"  Response Preview: {test['response'][:200]}...")
        
        # Find tests with ambiguous references
        ambiguous_tests = [r for r in self.results if any(
            phrase in r['question'].lower() for phrase in ["my usual", "same targets", "like before", "don't repeat"]
        )]
        
        print(f"\nAMBIGUOUS REFERENCE WEAKNESSES:")
        print(f"Tests with ambiguous references: {len(ambiguous_tests)}")
        
        for test in ambiguous_tests[:3]:  # Show first 3 examples
            print(f"\nTest {test['test_id']} ({test['user_id']}):")
            print(f"  Question: {test['question']}")
            print(f"  Passed: {test['passed']}")
            print(f"  User Requirements Score: {test.get('user_req_score', 0):.2f}")
            if test.get('response'):
                print(f"  Response Preview: {test['response'][:200]}...")

def main():
    """Main function to run memory-enhanced evaluation."""
    print("Context/Memory Management Enhanced Evaluation")
    print("Evaluating agent performance WITH memory management + RAG")
    print("=" * 60)

    # Initialize evaluator
    evaluator = MemoryEnhancedEvaluator()

    if not evaluator.tests:
        print("No test cases loaded. Exiting.")
        return

    # Run evaluation
    analysis = evaluator.run_evaluation()

    # Print detailed report
    evaluator.print_detailed_report(analysis)

    # Demonstrate context weaknesses
    evaluator.demonstrate_context_weaknesses()

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)

    # Log completion
    evaluator._log("EVALUATION COMPLETE")
    evaluator._log(f"Log file: {evaluator.log_file}")
    evaluator._log(f"Results directory: evaluation_results/")
    evaluator._log(f"Individual test results saved to: evaluation_results/*_result.json")
    evaluator._log(f"Summary statistics saved to: evaluation_results/summary_*.json")

if __name__ == "__main__":
    main()
