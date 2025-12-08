# grading_helpers.py
# Minimal helpers to evaluate agent outputs for the healthy-diet benchmark.
# These are library-style functions; plug into your harness or notebooks.
from typing import List, Dict, Tuple, Optional
import re
from collections import Counter

# Anchors: how to interpret "my usual" etc.
ANCHORS: Dict[str, Dict] = {
    "u01": {"kcal": 1800, "protein_min_g": 140, "fiber_min_g": 30, "units": "US",
            "allergies": ["peanut"], "equipment": ["stovetop", "oven"], "no_blender": True,
            "cuisines": ["Mediterranean","Mexican"]},
    "u02": {"kcal": 1600, "protein_min_g": 110, "units": "metric", "diet": "vegetarian",
            "lactose_free": True, "fasting_window": (12, 20), "equipment": ["microwave","rice cooker"]},
    "u03": {"kcal": 2000, "protein_min_g": 150, "units": "US", "diet": "halal",
            "low_glycemic": True, "fiber_min_g": 30, "equipment": ["grill", "air fryer"]},
    "u04": {"kcal": 2200, "protein_min_g": 130, "fiber_min_g": 30, "units": "US",
            "diet": "pescatarian", "exclude": ["tuna"], "cuisines": ["Japanese","Thai"]},
}

def within_pct_window(value: float, target: float, pct: float) -> bool:
    return abs(value - target) <= pct * target

def check_daily_targets(day_totals: Dict[str, float],
                        target_kcal: float,
                        kcal_window_pct: float,
                        protein_min: float,
                        fiber_min: Optional[float] = None) -> bool:
    kcal_ok = within_pct_window(day_totals.get("kcal", 0.0), target_kcal, kcal_window_pct)
    protein_ok = day_totals.get("protein_g", 0.0) >= protein_min
    fiber_ok = True if fiber_min is None else (day_totals.get("fiber_g", 0.0) >= fiber_min)
    return kcal_ok and protein_ok and fiber_ok

def check_no_repeats(menu_history: List[str], proposed_dishes: List[str]) -> bool:
    history_set = set([d.strip().lower() for d in menu_history])
    for d in proposed_dishes:
        if d.strip().lower() in history_set:
            return False
    return True

def check_protein_rotation(primary_proteins: List[str], cap: int) -> bool:
    counts = Counter([p.strip().lower() for p in primary_proteins])
    return all(c <= cap for c in counts.values())

def check_fasting_window(meal_times_hhmm: List[str], window_start: int, window_end: int) -> bool:
    def _hour(s: str) -> int:
        m = re.match(r"^\s*(\d{1,2})\s*:\s*\d{2}\s*$", s)
        return int(m.group(1)) if m else -1
    hours = [_hour(t) for t in meal_times_hhmm]
    return all(window_start <= h <= window_end for h in hours if h >= 0)

def check_netcarb_caps(meal_netcarbs: Dict[str, float], caps: Dict[str, float]) -> bool:
    # caps examples: {"lunch": 40.0, "dinner": 30.0}
    for meal, cap in caps.items():
        if meal in meal_netcarbs and meal_netcarbs[meal] > cap:
            return False
    return True

def jaccard(a: List[str], b: List[str]) -> float:
    A, B = set(a), set(b)
    if not A and not B:
        return 0.0
    return len(A & B) / len(A | B)

def check_ingredient_diversity(day_ingredients: List[List[str]], max_jaccard: float) -> bool:
    # day_ingredients: list of ingredient lists per day
    for i in range(len(day_ingredients)):
        for j in range(i+1, len(day_ingredients)):
            if jaccard(day_ingredients[i], day_ingredients[j]) > max_jaccard:
                return False
    return True

def check_sodium(day_sodium_mg: float, limit_mg: float) -> bool:
    return day_sodium_mg <= limit_mg

def check_tool_usage(tool_logs: List[str], must_use: List[str]) -> bool:
    # tool_logs: e.g., ["search", "calculator", "search"]
    for i in tool_logs:
        flag = False
        for j in must_use:
            if j in i:
                flag = True
                break
        if flag is False:
            return False
    return True

from typing import List, Dict, Tuple, Optional
import re
from collections import Counter

def _find_labeled_number(text: str, labels: List[str]) -> Optional[float]:
    for label in labels:
        pattern = re.compile(re.escape(label) + r"\s*:", re.I)
        for m in pattern.finditer(text):
            end = m.end()
            tail = text[end:]
            num_match = re.search(r"(\d+(\.\d+)?)", tail)
            if num_match:
                return float(num_match.group(1))

    for label in labels:
        pattern = re.compile(re.escape(label), re.I)
        m = pattern.search(text)
        if m:
            start = m.start()
            prefix = text[:start]
            nums = list(re.finditer(r"(\d+(\.\d+)?)", prefix))
            if nums:
                return float(nums[-1].group(1))

    return None



def extract_numbers_from_text(text: str) -> Dict[str, float]:
    """
      - Calories / calorie / cal / kcal / kcals  → kcal
      - Protein                                 → protein_g
      - Fiber / fibre                           → fiber_g
      - Net Carbs / net carb                    → net_carbs_g
    """
    out: Dict[str, float] = {}

    kcal = _find_labeled_number(
        text,
        ["calories", "calorie", "cal", "kcal", "kcals"]
    )

    prot = _find_labeled_number(
        text,
        ["protein", "proteins"]
    )

    fiber = _find_labeled_number(
        text,
        ["fiber", "fibre"]
    )

    netc = _find_labeled_number(
        text,
        ["net carbs", "net carb"]
    )

    if kcal is not None:
        out["kcal"] = kcal
    if prot is not None:
        out["protein_g"] = prot
    if fiber is not None:
        out["fiber_g"] = fiber
    if netc is not None:
        out["net_carbs_g"] = netc

    return out

# Example high-level check for a single day (you can adapt to multi-day/week):
def evaluate_day(user_id: str,
                 agent_text: str,
                 target_kcal_window_pct: float,
                 protein_min_override: Optional[float]=None,
                 fiber_min_override: Optional[float]=None) -> bool:
    anchors = ANCHORS[user_id]
    numbers = extract_numbers_from_text(agent_text)
    kcal_tgt = anchors["kcal"]
    protein_min = protein_min_override if protein_min_override is not None else anchors.get("protein_min_g", 0)
    fiber_min = fiber_min_override if fiber_min_override is not None else anchors.get("fiber_min_g", None)
    return check_daily_targets(numbers, kcal_tgt, target_kcal_window_pct, protein_min, fiber_min)
