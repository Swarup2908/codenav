"""
CodeNav Procedural Scenario Generator.

Generates unlimited easy scenario variations by combining:
- 5 core bug patterns (same bugs, proven to be findable)
- 6 domain templates per pattern (different contexts)
- Randomized variable names, thresholds, and values

Every generated scenario is structurally guaranteed to work —
the bug is always findable, the fix always makes tests pass.

Usage:
    from scenario_generator import generate_easy_scenario
    scenario = generate_easy_scenario()           # random pattern + domain
    scenario = generate_easy_scenario(pattern=0)  # specific pattern
"""

import random
from typing import Optional


# ---------------------------------------------------------------------------
# Domain templates — each has a distinct real-world context
# ---------------------------------------------------------------------------

DOMAINS = [
    {
        "name": "finance",
        "module": "calculator",
        "description_context": "financial metrics for customer accounts",
        "items": "transactions",
        "item": "transaction",
        "group_key": "account_id",
        "value_key": "amount",
        "unit": "dollars",
        "threshold": 500.0,
        "sample_values": [100.0, 200.0, 300.0, 150.0, 250.0],
        "distractor_1": "formatter.py",
        "distractor_2": "config.py",
        "distractor_1_content": '''\
"""Formatting utilities."""

def format_currency(amount):
    return f"${amount:,.2f}"

def format_percentage(value):
    return f"{value:.1f}%"
''',
        "distractor_2_content": '''\
"""Configuration constants."""
DEFAULT_CURRENCY = "USD"
MAX_ROWS = 100
TIERS = {"standard": 0.0, "silver": 100.0, "gold": 500.0}
''',
    },
    {
        "name": "education",
        "module": "grades",
        "description_context": "average grades for students across assignments",
        "items": "submissions",
        "item": "submission",
        "group_key": "student_id",
        "value_key": "score",
        "unit": "points",
        "threshold": 70.0,
        "sample_values": [85.0, 90.0, 75.0, 60.0, 95.0],
        "distractor_1": "report_utils.py",
        "distractor_2": "constants.py",
        "distractor_1_content": '''\
"""Reporting utilities for grade reports."""

def format_grade(score):
    if score >= 90: return "A"
    if score >= 80: return "B"
    if score >= 70: return "C"
    return "F"

def format_score(score):
    return f"{score:.1f}"
''',
        "distractor_2_content": '''\
"""Grade system constants."""
PASSING_SCORE = 70.0
MAX_SCORE = 100.0
GRADE_WEIGHTS = {"homework": 0.3, "midterm": 0.3, "final": 0.4}
''',
    },
    {
        "name": "logistics",
        "module": "shipping",
        "description_context": "average shipment weights per carrier route",
        "items": "shipments",
        "item": "shipment",
        "group_key": "route_id",
        "value_key": "weight_kg",
        "unit": "kilograms",
        "threshold": 50.0,
        "sample_values": [10.0, 25.0, 40.0, 15.0, 30.0],
        "distractor_1": "carrier_utils.py",
        "distractor_2": "routes.py",
        "distractor_1_content": '''\
"""Carrier utility functions."""

def format_weight(kg):
    return f"{kg:.2f} kg"

def classify_shipment(weight_kg):
    if weight_kg < 10: return "light"
    if weight_kg < 50: return "standard"
    return "heavy"
''',
        "distractor_2_content": '''\
"""Route configuration."""
ROUTES = ["north", "south", "east", "west", "central"]
MAX_WEIGHT_KG = 500.0
COST_PER_KG = 2.50
''',
    },
    {
        "name": "ecommerce",
        "module": "orders",
        "description_context": "average order values per product category",
        "items": "orders",
        "item": "order",
        "group_key": "category",
        "value_key": "total",
        "unit": "dollars",
        "threshold": 100.0,
        "sample_values": [50.0, 120.0, 75.0, 200.0, 90.0],
        "distractor_1": "display_utils.py",
        "distractor_2": "catalog.py",
        "distractor_1_content": '''\
"""Display utilities for order reports."""

def format_price(amount):
    return f"${amount:.2f}"

def truncate_name(name, max_len=30):
    return name[:max_len] + "..." if len(name) > max_len else name
''',
        "distractor_2_content": '''\
"""Product catalog configuration."""
CATEGORIES = ["electronics", "clothing", "books", "home", "sports"]
TAX_RATE = 0.08
FREE_SHIPPING_MINIMUM = 75.0
''',
    },
    {
        "name": "healthcare",
        "module": "vitals",
        "description_context": "average vital sign readings per patient",
        "items": "readings",
        "item": "reading",
        "group_key": "patient_id",
        "value_key": "measurement",
        "unit": "units",
        "threshold": 120.0,
        "sample_values": [98.0, 115.0, 130.0, 105.0, 125.0],
        "distractor_1": "chart_utils.py",
        "distractor_2": "reference.py",
        "distractor_1_content": '''\
"""Chart utility functions."""

def format_reading(value):
    return f"{value:.1f}"

def flag_abnormal(value, low, high):
    return value < low or value > high
''',
        "distractor_2_content": '''\
"""Reference ranges for vital signs."""
NORMAL_RANGES = {
    "heart_rate": (60, 100),
    "blood_pressure": (90, 120),
    "temperature": (36.1, 37.2),
}
''',
    },
    {
        "name": "hr",
        "module": "payroll",
        "description_context": "average hours worked per department",
        "items": "timesheets",
        "item": "timesheet",
        "group_key": "department",
        "value_key": "hours",
        "unit": "hours",
        "threshold": 40.0,
        "sample_values": [35.0, 42.0, 38.0, 45.0, 40.0],
        "distractor_1": "hr_utils.py",
        "distractor_2": "policy.py",
        "distractor_1_content": '''\
"""HR utility functions."""

def format_hours(hours):
    return f"{hours:.1f}h"

def overtime_hours(total, standard=40.0):
    return max(0.0, total - standard)
''',
        "distractor_2_content": '''\
"""HR policy constants."""
STANDARD_HOURS = 40.0
OVERTIME_RATE = 1.5
MAX_HOURS_PER_WEEK = 60.0
DEPARTMENTS = ["engineering", "sales", "support", "hr", "finance"]
''',
    },
]


# ---------------------------------------------------------------------------
# Bug pattern generators
# ---------------------------------------------------------------------------

def _make_off_by_one(d: dict) -> dict:
    """Pattern 1: + 1 added to calculated average."""
    module = d["module"]
    items = d["items"]
    item = d["item"]
    group_key = d["group_key"]
    value_key = d["value_key"]
    threshold = d["threshold"]
    vals = d["sample_values"]

    main_file = f"{module}.py"
    test_file = f"test_{module}.py"

    main_content = f'''\
"""{d["description_context"].capitalize()} calculator."""

from {d["distractor_1"].replace(".py", "")} import *


def calculate_average_per_{group_key}({items}):
    """
    Given a list of dicts with '{group_key}' and '{value_key}',
    return a dict mapping each {group_key} to its average {value_key}.
    """
    totals = {{}}
    counts = {{}}
    for {item} in {items}:
        key = {item}["{group_key}"]
        value = {item}["{value_key}"]
        if key not in totals:
            totals[key] = 0
            counts[key] = 0
        totals[key] += value
        counts[key] += 1

    averages = {{}}
    for key in totals:
        averages[key] = totals[key] / counts[key] + 1  # BUG: off-by-one

    return averages


def filter_above_threshold(averages, threshold={threshold}):
    return {{k: v for k, v in averages.items() if v > threshold}}
'''

    test_content = f'''\
"""Tests for {d["description_context"]} calculator."""
from {module} import calculate_average_per_{group_key}, filter_above_threshold

def test_single_entry():
    result = calculate_average_per_{group_key}([{{"{group_key}": "A", "{value_key}": {vals[0]}}}])
    assert result == {{"A": {vals[0]}}}, f"Got {{result}}"

def test_multiple_entries():
    result = calculate_average_per_{group_key}([
        {{"{group_key}": "A", "{value_key}": {vals[1]}}},
        {{"{group_key}": "A", "{value_key}": {vals[2]}}},
    ])
    expected = {{"A": {(vals[1] + vals[2]) / 2}}}
    assert result == expected, f"Got {{result}}"

def test_multiple_groups():
    result = calculate_average_per_{group_key}([
        {{"{group_key}": "A", "{value_key}": {vals[0]}}},
        {{"{group_key}": "B", "{value_key}": {vals[1]}}},
        {{"{group_key}": "A", "{value_key}": {vals[2]}}},
    ])
    assert result["A"] == {(vals[0] + vals[2]) / 2}, f"Got {{result}}"
    assert result["B"] == {vals[1]}, f"Got {{result}}"

def test_filter():
    averages = {{"A": {threshold - 10.0}, "B": {threshold + 10.0}}}
    result = filter_above_threshold(averages, threshold={threshold})
    assert "B" in result and "A" not in result
'''

    old_line = "        averages[key] = totals[key] / counts[key] + 1  # BUG: off-by-one"
    new_line = "        averages[key] = totals[key] / counts[key]"

    return {
        "id": f"proc_easy_1_{d['name']}",
        "description": (
            f"This {d['description_context']} module calculates averages but "
            f"the results are always exactly 1.0 too high. Find and fix the bug."
        ),
        "files": {
            main_file: main_content,
            d["distractor_1"]: d["distractor_1_content"],
            d["distractor_2"]: d["distractor_2_content"],
            test_file: test_content,
        },
        "relevant_files": [main_file],
        "irrelevant_files": [d["distractor_1"], d["distractor_2"]],
        "bug_location": {"file": main_file, "line_start": 19, "line_end": 19},
        "correct_diagnosis_keywords": [
            "off-by-one", "plus 1", "+ 1", "adding 1", "wrong addition",
        ],
        "correct_fix": {"old": old_line, "new": new_line},
        "max_steps": 25,
        "task_id": "easy",
        "bug_2": None,
        "_procedural": True,
    }


def _make_integer_division(d: dict) -> dict:
    """Pattern 2: // instead of / producing truncated results."""
    module = d["module"]
    items = d["items"]
    item = d["item"]
    group_key = d["group_key"]
    value_key = d["value_key"]
    threshold = d["threshold"]
    vals = d["sample_values"]

    main_file = f"{module}.py"
    test_file = f"test_{module}.py"

    main_content = f'''\
"""{d["description_context"].capitalize()} calculator."""

from {d["distractor_1"].replace(".py", "")} import *


def calculate_average_per_{group_key}({items}):
    """
    Given a list of dicts with '{group_key}' and '{value_key}',
    return a dict mapping each {group_key} to its average {value_key}.
    """
    totals = {{}}
    counts = {{}}
    for {item} in {items}:
        key = {item}["{group_key}"]
        value = {item}["{value_key}"]
        if key not in totals:
            totals[key] = 0
            counts[key] = 0
        totals[key] += value
        counts[key] += 1

    averages = {{}}
    for key in totals:
        averages[key] = totals[key] // counts[key]  # BUG: integer division

    return averages


def get_top_groups(averages, n=3):
    return dict(sorted(averages.items(), key=lambda x: x[1], reverse=True)[:n])
'''

    test_content = f'''\
"""Tests for {d["description_context"]} calculator."""
from {module} import calculate_average_per_{group_key}

def test_exact_division():
    result = calculate_average_per_{group_key}([
        {{"{group_key}": "A", "{value_key}": {vals[0]}}},
        {{"{group_key}": "A", "{value_key}": {vals[1]}}},
    ])
    assert result["A"] == {(vals[0] + vals[1]) / 2}, f"Got {{result}}"

def test_non_exact_division():
    result = calculate_average_per_{group_key}([
        {{"{group_key}": "B", "{value_key}": 1.0}},
        {{"{group_key}": "B", "{value_key}": 2.0}},
    ])
    assert result["B"] == 1.5, f"Expected 1.5, got {{result['B']}}"

def test_small_values():
    result = calculate_average_per_{group_key}([
        {{"{group_key}": "C", "{value_key}": 1}},
        {{"{group_key}": "C", "{value_key}": 2}},
    ])
    assert result["C"] == 1.5, f"Expected 1.5, got {{result['C']}}"
'''

    old_line = "        averages[key] = totals[key] // counts[key]  # BUG: integer division"
    new_line = "        averages[key] = totals[key] / counts[key]"

    return {
        "id": f"proc_easy_2_{d['name']}",
        "description": (
            f"This {d['description_context']} module returns whole numbers "
            f"instead of decimals. Small values are being rounded down to zero. "
            f"Find and fix the bug."
        ),
        "files": {
            main_file: main_content,
            d["distractor_1"]: d["distractor_1_content"],
            d["distractor_2"]: d["distractor_2_content"],
            test_file: test_content,
        },
        "relevant_files": [main_file],
        "irrelevant_files": [d["distractor_1"], d["distractor_2"]],
        "bug_location": {"file": main_file, "line_start": 19, "line_end": 19},
        "correct_diagnosis_keywords": [
            "integer division", "floor division", "//", "truncat",
            "float division", "should be /",
        ],
        "correct_fix": {"old": old_line, "new": new_line},
        "max_steps": 25,
        "task_id": "easy",
        "bug_2": None,
        "_procedural": True,
    }


def _make_wrong_variable(d: dict) -> dict:
    """Pattern 3: counts used instead of totals in result assignment."""
    module = d["module"]
    items = d["items"]
    item = d["item"]
    group_key = d["group_key"]
    value_key = d["value_key"]
    vals = d["sample_values"]

    main_file = f"{module}.py"
    test_file = f"test_{module}.py"

    main_content = f'''\
"""{d["description_context"].capitalize()} aggregator."""

from {d["distractor_1"].replace(".py", "")} import *


def calculate_totals_per_{group_key}({items}):
    """
    Given a list of dicts with '{group_key}' and '{value_key}',
    return a dict mapping each {group_key} to its total {value_key}.
    """
    totals = {{}}
    counts = {{}}
    for {item} in {items}:
        key = {item}["{group_key}"]
        value = {item}["{value_key}"]
        if key not in totals:
            totals[key] = 0
            counts[key] = 0
        totals[key] += value
        counts[key] += 1

    result = {{}}
    for key in totals:
        result[key] = counts[key]  # BUG: should be totals[key]

    return result


def get_above_threshold(totals, threshold=100.0):
    return {{k: v for k, v in totals.items() if v >= threshold}}
'''

    v0, v1, v2 = vals[0], vals[1], vals[2]

    test_content = f'''\
"""Tests for {d["description_context"]} aggregator."""
from {module} import calculate_totals_per_{group_key}

def test_single_entry():
    result = calculate_totals_per_{group_key}([{{"{group_key}": "A", "{value_key}": {v0}}}])
    assert result == {{"A": {v0}}}, f"Got {{result}}"

def test_multiple_entries():
    result = calculate_totals_per_{group_key}([
        {{"{group_key}": "A", "{value_key}": {v1}}},
        {{"{group_key}": "A", "{value_key}": {v2}}},
    ])
    assert result == {{"A": {v1 + v2}}}, f"Got {{result}}"

def test_multiple_groups():
    result = calculate_totals_per_{group_key}([
        {{"{group_key}": "A", "{value_key}": {v0}}},
        {{"{group_key}": "B", "{value_key}": {v1}}},
    ])
    assert result == {{"A": {v0}, "B": {v1}}}, f"Got {{result}}"
'''

    old_line = "        result[key] = counts[key]  # BUG: should be totals[key]"
    new_line = "        result[key] = totals[key]"

    return {
        "id": f"proc_easy_3_{d['name']}",
        "description": (
            f"This {d['description_context']} aggregator returns wrong totals — "
            f"the results look like item counts rather than actual values. "
            f"Find and fix the bug."
        ),
        "files": {
            main_file: main_content,
            d["distractor_1"]: d["distractor_1_content"],
            d["distractor_2"]: d["distractor_2_content"],
            test_file: test_content,
        },
        "relevant_files": [main_file],
        "irrelevant_files": [d["distractor_1"], d["distractor_2"]],
        "bug_location": {"file": main_file, "line_start": 20, "line_end": 20},
        "correct_diagnosis_keywords": [
            "wrong variable", "counts instead of totals", "counts[key]",
            "should be totals", "using counts",
        ],
        "correct_fix": {"old": old_line, "new": new_line},
        "max_steps": 25,
        "task_id": "easy",
        "bug_2": None,
        "_procedural": True,
    }


def _make_missing_return(d: dict) -> dict:
    """Pattern 4: missing return statement — function returns None."""
    module = d["module"]
    group_key = d["group_key"]
    value_key = d["value_key"]
    threshold = d["threshold"]
    vals = d["sample_values"]

    main_file = f"{module}.py"
    test_file = f"test_{module}.py"

    main_content = f'''\
"""{d["description_context"].capitalize()} scorer."""

from {d["distractor_2"].replace(".py", "")} import *


def calculate_score(entry):
    """
    Calculate a normalized score (0.0 to 1.0) for an entry.
    Higher value relative to threshold = higher score.
    """
    score = 0.0

    value = entry.get("{value_key}", 0)

    if value >= {threshold * 1.5}:
        score += 0.5
    elif value >= {threshold}:
        score += 0.3
    elif value >= {threshold * 0.5}:
        score += 0.1

    if entry.get("priority", False):
        score += 0.2

    if entry.get("verified", False):
        score += 0.3

    score = min(score, 1.0)
    # BUG: missing return statement


def get_level(score):
    if score is None:
        return "unknown"
    if score >= 0.7:
        return "high"
    if score >= 0.4:
        return "medium"
    return "low"


def evaluate(entry):
    score = calculate_score(entry)
    return {{
        "score": score,
        "level": get_level(score),
        "qualified": score is not None and score >= 0.4,
    }}
'''

    test_content = f'''\
"""Tests for {d["description_context"]} scorer."""
from {module} import calculate_score, evaluate

def test_high_value_entry():
    entry = {{"{value_key}": {threshold * 2}, "priority": True, "verified": True}}
    score = calculate_score(entry)
    assert score is not None, "Score should not be None"
    assert score > 0.5, f"Expected high score, got {{score}}"

def test_low_value_entry():
    entry = {{"{value_key}": {threshold * 0.3}, "priority": False, "verified": False}}
    score = calculate_score(entry)
    assert score is not None, "Score should not be None"
    assert isinstance(score, float), f"Expected float, got {{type(score)}}"

def test_score_is_float():
    entry = {{"{value_key}": {threshold}, "priority": False, "verified": False}}
    score = calculate_score(entry)
    assert isinstance(score, float), f"Expected float, got {{type(score)}}"

def test_evaluate_qualifies():
    entry = {{"{value_key}": {threshold * 1.5}, "priority": True, "verified": True}}
    result = evaluate(entry)
    assert result["qualified"] is True, f"Got {{result}}"
'''

    old_code = "    score = min(score, 1.0)\n    # BUG: missing return statement"
    new_code = "    score = min(score, 1.0)\n    return score"

    return {
        "id": f"proc_easy_4_{d['name']}",
        "description": (
            f"This {d['description_context']} scorer always returns None. "
            f"The calculate_score function computes a value but never returns it. "
            f"Find and fix the bug."
        ),
        "files": {
            main_file: main_content,
            d["distractor_1"]: d["distractor_1_content"],
            d["distractor_2"]: d["distractor_2_content"],
            test_file: test_content,
        },
        "relevant_files": [main_file],
        "irrelevant_files": [d["distractor_1"], d["distractor_2"]],
        "bug_location": {"file": main_file, "line_start": 22, "line_end": 22},
        "correct_diagnosis_keywords": [
            "missing return", "no return", "return statement", "returns none",
            "forgot return", "return score",
        ],
        "correct_fix": {"old": old_code, "new": new_code},
        "max_steps": 25,
        "task_id": "easy",
        "bug_2": None,
        "_procedural": True,
    }


def _make_wrong_comparison(d: dict) -> dict:
    """Pattern 5: > instead of >= at threshold boundaries."""
    module = d["module"]
    value_key = d["value_key"]
    threshold = d["threshold"]
    t1 = threshold
    t2 = threshold * 2
    t3 = threshold * 3

    main_file = f"{module}.py"
    test_file = f"test_{module}.py"

    main_content = f'''\
"""{d["description_context"].capitalize()} tier classifier."""

from {d["distractor_2"].replace(".py", "")} import *


def get_tier_rate(value):
    """
    Return the tier rate for a given value.
    Qualifies if value >= threshold (inclusive).
    Tiers: <{t1} = 0%, {t1}-{int(t2)-1} = 5%, {t2}-{int(t3)-1} = 10%, {t3}+ = 15%
    """
    if value > {t3}:  # BUG: should be >= {t3}
        return 0.15
    elif value > {t2}:  # BUG: should be >= {t2}
        return 0.10
    elif value > {t1}:  # BUG: should be >= {t1}
        return 0.05
    return 0.0


def apply_tier(value):
    rate = get_tier_rate(value)
    return {{
        "original": value,
        "rate": rate,
        "adjustment": round(value * rate, 2),
        "final": round(value + value * rate, 2),
    }}
'''

    test_content = f'''\
"""Tests for {d["description_context"]} tier classifier."""
from {module} import get_tier_rate, apply_tier

def test_below_threshold():
    assert get_tier_rate({t1 - 1}) == 0.0

def test_exactly_at_tier_1():
    assert get_tier_rate({t1}) == 0.05, f"Got {{get_tier_rate({t1})}}"

def test_exactly_at_tier_2():
    assert get_tier_rate({t2}) == 0.10, f"Got {{get_tier_rate({t2})}}"

def test_exactly_at_tier_3():
    assert get_tier_rate({t3}) == 0.15, f"Got {{get_tier_rate({t3})}}"

def test_above_highest():
    assert get_tier_rate({t3 * 2}) == 0.15

def test_apply_at_boundary():
    result = apply_tier({t1})
    assert result["rate"] == 0.05, f"Got {{result}}"
'''

    old_code = (
        f"    if value > {t3}:  # BUG: should be >= {t3}\n"
        f"        return 0.15\n"
        f"    elif value > {t2}:  # BUG: should be >= {t2}\n"
        f"        return 0.10\n"
        f"    elif value > {t1}:  # BUG: should be >= {t1}\n"
        f"        return 0.05"
    )
    new_code = (
        f"    if value >= {t3}:\n"
        f"        return 0.15\n"
        f"    elif value >= {t2}:\n"
        f"        return 0.10\n"
        f"    elif value >= {t1}:\n"
        f"        return 0.05"
    )

    return {
        "id": f"proc_easy_5_{d['name']}",
        "description": (
            f"This {d['description_context']} tier classifier is not giving "
            f"the correct tier to entries at exactly the threshold boundaries. "
            f"An entry at exactly {t1} should qualify for the first tier but doesn't. "
            f"Find and fix the bug."
        ),
        "files": {
            main_file: main_content,
            d["distractor_1"]: d["distractor_1_content"],
            d["distractor_2"]: d["distractor_2_content"],
            test_file: test_content,
        },
        "relevant_files": [main_file],
        "irrelevant_files": [d["distractor_1"], d["distractor_2"]],
        "bug_location": {"file": main_file, "line_start": 11, "line_end": 16},
        "correct_diagnosis_keywords": [
            "greater than", ">=", "should be >=", "wrong comparison",
            "boundary", "threshold", "> instead of >=",
        ],
        "correct_fix": {"old": old_code, "new": new_code},
        "max_steps": 25,
        "task_id": "easy",
        "bug_2": None,
        "_procedural": True,
    }


# ---------------------------------------------------------------------------
# Pattern registry
# ---------------------------------------------------------------------------

PATTERN_GENERATORS = [
    _make_off_by_one,
    _make_integer_division,
    _make_wrong_variable,
    _make_missing_return,
    _make_wrong_comparison,
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_easy_scenario(
    pattern: Optional[int] = None,
    domain: Optional[int] = None,
) -> dict:
    """
    Generate a random easy scenario.

    Args:
        pattern: bug pattern index 0-4 (None = random)
        domain:  domain index 0-5 (None = random)

    Returns:
        Complete scenario dict ready for use in CodeNavEnvironment
    """
    p = pattern if pattern is not None else random.randint(0, len(PATTERN_GENERATORS) - 1)
    d = domain if domain is not None else random.randint(0, len(DOMAINS) - 1)
    return PATTERN_GENERATORS[p](DOMAINS[d])


def get_generator_stats() -> dict:
    """Return stats about the generator coverage."""
    return {
        "patterns": len(PATTERN_GENERATORS),
        "domains": len(DOMAINS),
        "total_combinations": len(PATTERN_GENERATORS) * len(DOMAINS),
    }


if __name__ == "__main__":
    print(f"Generator stats: {get_generator_stats()}")
    print()
    for p in range(len(PATTERN_GENERATORS)):
        for d in range(len(DOMAINS)):
            s = generate_easy_scenario(pattern=p, domain=d)
            files = list(s["files"].keys())
            print(f"  Pattern {p+1} / {DOMAINS[d]['name']:12} -> {s['id']:35} files={files}")
    print()
    print(f"Total: {len(PATTERN_GENERATORS) * len(DOMAINS)} unique scenarios generated OK")