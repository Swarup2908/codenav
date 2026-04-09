"""
CodeNav Scenario Pool — 19 scenarios across 3 difficulty levels.

Easy:   5 scenarios — single file bugs, 2 distractor files, no bug_2
Medium: 6 scenarios — cross-file bugs, 2 distractor files, bug_2
Hard:   8 scenarios — multi-file silent bugs, 2 distractors, bug_2

Usage:
    from scenarios import get_scenario, get_pool_size
    scenario = get_scenario("easy")           # random pick
    scenario = get_scenario("hard", index=2)  # specific scenario
"""

import random
from typing import Optional


# ---------------------------------------------------------------------------
# Easy Scenarios (5)
# ---------------------------------------------------------------------------

EASY_SCENARIOS = [

# ── easy_1 ── off-by-one in average calculation ───────────────────────────
{
    "id": "easy_1",
    "description": (
        "This data processing pipeline calculates the average transaction value "
        "per customer and generates a summary report. "
        "The averages are wrong. Find and fix the bug."
    ),
    "files": {
        "processor.py": '''\
"""Transaction processor — calculates per-customer metrics."""

from formatter import format_currency


def calculate_average_per_customer(transactions):
    """
    Given a list of dicts with 'customer_id' and 'amount',
    return a dict mapping each customer_id to their average transaction amount.
    """
    totals = {}
    counts = {}
    for tx in transactions:
        cid = tx["customer_id"]
        amount = tx["amount"]
        if cid not in totals:
            totals[cid] = 0
            counts[cid] = 0
        totals[cid] += amount
        counts[cid] += 1

    averages = {}
    for cid in totals:
        averages[cid] = totals[cid] / counts[cid] + 1  # BUG: off-by-one error

    return averages


def filter_high_value(averages, threshold=100.0):
    return {cid: avg for cid, avg in averages.items() if avg > threshold}


def generate_report(transactions, threshold=100.0):
    averages = calculate_average_per_customer(transactions)
    high_value = filter_high_value(averages, threshold)
    return {
        "total_customers": len(averages),
        "high_value_customers": len(high_value),
        "averages": {cid: format_currency(avg) for cid, avg in averages.items()},
    }
''',
        "formatter.py": '''\
"""Formatting utilities for the transaction processor."""

def format_currency(amount):
    return f"${amount:,.2f}"

def format_percentage(value, decimals=1):
    return f"{value:.{decimals}f}%"

def truncate_string(s, max_length=50):
    if len(s) <= max_length:
        return s
    return s[:max_length - 3] + "..."
''',
        "config.py": '''\
"""Configuration for the transaction processor."""
HIGH_VALUE_THRESHOLD = 100.0
MAX_REPORT_ROWS = 100
CUSTOMER_TIERS = {
    "standard": 0.0, "silver": 100.0, "gold": 500.0, "platinum": 1000.0,
}

def get_customer_tier(average_amount):
    tier = "standard"
    for name, threshold in sorted(CUSTOMER_TIERS.items(), key=lambda x: x[1]):
        if average_amount >= threshold:
            tier = name
    return tier
''',
        "test_processor.py": '''\
"""Tests for the transaction processor."""
from processor import calculate_average_per_customer, filter_high_value, generate_report

def test_single_customer():
    result = calculate_average_per_customer([{"customer_id": "A", "amount": 50.0}])
    assert result == {"A": 50.0}, f"Got {result}"

def test_multiple_transactions():
    result = calculate_average_per_customer([
        {"customer_id": "A", "amount": 100.0},
        {"customer_id": "A", "amount": 200.0},
    ])
    assert result == {"A": 150.0}, f"Got {result}"

def test_multiple_customers():
    result = calculate_average_per_customer([
        {"customer_id": "A", "amount": 100.0},
        {"customer_id": "B", "amount": 200.0},
        {"customer_id": "A", "amount": 300.0},
    ])
    assert result == {"A": 200.0, "B": 200.0}, f"Got {result}"

def test_filter_high_value():
    result = filter_high_value({"A": 50.0, "B": 150.0}, threshold=100.0)
    assert result == {"B": 150.0}, f"Got {result}"

def test_report_counts():
    report = generate_report([
        {"customer_id": "A", "amount": 200.0},
        {"customer_id": "B", "amount": 50.0},
    ], threshold=100.0)
    assert report["total_customers"] == 2
    assert report["high_value_customers"] == 1
''',
    },
    "relevant_files": ["processor.py"],
    "irrelevant_files": ["formatter.py", "config.py"],
    "bug_location": {"file": "processor.py", "line_start": 22, "line_end": 22},
    "correct_diagnosis_keywords": [
        "off-by-one", "plus 1", "+ 1", "adding 1", "wrong addition",
    ],
    "correct_fix": {
        "old": "        averages[cid] = totals[cid] / counts[cid] + 1  # BUG: off-by-one error",
        "new": "        averages[cid] = totals[cid] / counts[cid]",
    },
    "max_steps": 25,
    "task_id": "easy",
    "bug_2": None,
},

# ── easy_2 ── integer division truncating results ─────────────────────────
{
    "id": "easy_2",
    "description": (
        "This sales report generator calculates average order values per region. "
        "Results are always whole numbers even when they should have decimals. "
        "Small orders are being rounded down to zero. Find and fix the bug."
    ),
    "files": {
        "sales.py": '''\
"""Sales report generator — average order value per region."""

from utils import format_result


def calculate_regional_averages(orders):
    """
    Given a list of dicts with 'region' and 'value',
    return a dict mapping each region to its average order value.
    """
    totals = {}
    counts = {}
    for order in orders:
        region = order["region"]
        value = order["value"]
        if region not in totals:
            totals[region] = 0
            counts[region] = 0
        totals[region] += value
        counts[region] += 1

    averages = {}
    for region in totals:
        averages[region] = totals[region] // counts[region]  # BUG: integer division

    return averages


def get_top_regions(averages, n=3):
    return dict(sorted(averages.items(), key=lambda x: x[1], reverse=True)[:n])


def generate_sales_report(orders):
    averages = calculate_regional_averages(orders)
    return {
        "regional_averages": averages,
        "top_regions": get_top_regions(averages),
        "formatted": {r: format_result(v) for r, v in averages.items()},
    }
''',
        "utils.py": '''\
"""Utility functions for sales reporting."""

def format_result(value):
    return f"{value:.2f}"

def normalize_region(region):
    return region.strip().upper()

def validate_order(order):
    return "region" in order and "value" in order and order["value"] >= 0
''',
        "constants.py": '''\
"""Sales reporting constants."""
MIN_SAMPLE_SIZE = 5
DEFAULT_TOP_N = 3
CURRENCY_SYMBOL = "$"
REGIONS = ["North", "South", "East", "West", "Central"]
''',
        "test_sales.py": '''\
"""Tests for the sales report generator."""
from sales import calculate_regional_averages, get_top_regions

def test_exact_division():
    result = calculate_regional_averages([
        {"region": "North", "value": 100},
        {"region": "North", "value": 200},
    ])
    assert result == {"North": 150.0}, f"Got {result}"

def test_non_exact_division():
    result = calculate_regional_averages([
        {"region": "South", "value": 10},
        {"region": "South", "value": 21},
    ])
    assert result["South"] == 15.5, f"Expected 15.5, got {result['South']}"

def test_small_values():
    result = calculate_regional_averages([
        {"region": "East", "value": 1},
        {"region": "East", "value": 2},
    ])
    assert result["East"] == 1.5, f"Expected 1.5, got {result['East']}"

def test_top_regions():
    averages = {"A": 300.0, "B": 100.0, "C": 200.0}
    result = get_top_regions(averages, n=2)
    assert list(result.keys()) == ["A", "C"]
''',
    },
    "relevant_files": ["sales.py"],
    "irrelevant_files": ["utils.py", "constants.py"],
    "bug_location": {"file": "sales.py", "line_start": 22, "line_end": 22},
    "correct_diagnosis_keywords": [
        "integer division", "floor division", "//", "truncat",
        "float division", "should be /", "integer instead of float",
    ],
    "correct_fix": {
        "old": "        averages[region] = totals[region] // counts[region]  # BUG: integer division",
        "new": "        averages[region] = totals[region] / counts[region]",
    },
    "max_steps": 25,
    "task_id": "easy",
    "bug_2": None,
},

# ── easy_3 ── wrong variable (counts used instead of totals) ──────────────
{
    "id": "easy_3",
    "description": (
        "This inventory tracker calculates total stock value per warehouse. "
        "The totals are wrong — they seem to reflect item counts instead of values. "
        "Find and fix the bug."
    ),
    "files": {
        "inventory.py": '''\
"""Inventory tracker — calculates stock value per warehouse."""

from reporter import format_value


def calculate_warehouse_totals(items):
    """
    Given a list of dicts with 'warehouse', 'quantity', and 'unit_price',
    return a dict mapping each warehouse to its total stock value.
    """
    totals = {}
    counts = {}
    for item in items:
        wh = item["warehouse"]
        value = item["quantity"] * item["unit_price"]
        if wh not in totals:
            totals[wh] = 0
            counts[wh] = 0
        totals[wh] += value
        counts[wh] += 1

    result = {}
    for wh in totals:
        result[wh] = counts[wh]  # BUG: should be totals[wh]

    return result


def get_high_value_warehouses(totals, threshold=1000.0):
    return {wh: v for wh, v in totals.items() if v >= threshold}
''',
        "reporter.py": '''\
"""Reporting utilities."""

def format_value(v):
    return f"${v:,.2f}"

def format_count(n):
    return f"{n:,} items"

def percent_of_total(value, total):
    if total == 0:
        return 0.0
    return round(value / total * 100, 2)
''',
        "locations.py": '''\
"""Warehouse location data."""
WAREHOUSE_LOCATIONS = {
    "A": "New York", "B": "Los Angeles", "C": "Chicago", "D": "Houston",
}

def get_location(warehouse_id):
    return WAREHOUSE_LOCATIONS.get(warehouse_id, "Unknown")
''',
        "test_inventory.py": '''\
"""Tests for the inventory tracker."""
from inventory import calculate_warehouse_totals, get_high_value_warehouses

def test_single_item():
    items = [{"warehouse": "A", "quantity": 10, "unit_price": 5.0}]
    result = calculate_warehouse_totals(items)
    assert result == {"A": 50.0}, f"Got {result}"

def test_multiple_items_same_warehouse():
    items = [
        {"warehouse": "A", "quantity": 10, "unit_price": 5.0},
        {"warehouse": "A", "quantity": 20, "unit_price": 3.0},
    ]
    result = calculate_warehouse_totals(items)
    assert result == {"A": 110.0}, f"Got {result}"

def test_multiple_warehouses():
    items = [
        {"warehouse": "A", "quantity": 5, "unit_price": 100.0},
        {"warehouse": "B", "quantity": 10, "unit_price": 50.0},
    ]
    result = calculate_warehouse_totals(items)
    assert result == {"A": 500.0, "B": 500.0}, f"Got {result}"

def test_high_value_filter():
    totals = {"A": 500.0, "B": 1500.0, "C": 1000.0}
    result = get_high_value_warehouses(totals, threshold=1000.0)
    assert "B" in result and "C" in result and "A" not in result
''',
    },
    "relevant_files": ["inventory.py"],
    "irrelevant_files": ["reporter.py", "locations.py"],
    "bug_location": {"file": "inventory.py", "line_start": 20, "line_end": 20},
    "correct_diagnosis_keywords": [
        "wrong variable", "counts instead of totals", "counts[wh]",
        "should be totals", "using counts", "result[wh] = counts",
    ],
    "correct_fix": {
        "old": "        result[wh] = counts[wh]  # BUG: should be totals[wh]",
        "new": "        result[wh] = totals[wh]",
    },
    "max_steps": 25,
    "task_id": "easy",
    "bug_2": None,
},

# ── easy_4 ── missing return statement ────────────────────────────────────
{
    "id": "easy_4",
    "description": (
        "This risk scoring system calculates scores for loan applications. "
        "The calculate_risk_score function always returns None. "
        "Find and fix the bug."
    ),
    "files": {
        "scoring.py": '''\
"""Loan application risk scoring system."""

from thresholds import RISK_LEVELS


def calculate_risk_score(applicant):
    """
    Calculate a risk score (0.0 to 1.0) for a loan applicant.
    Higher score = higher risk.
    """
    score = 0.0

    if applicant["credit_score"] < 580:
        score += 0.4
    elif applicant["credit_score"] < 670:
        score += 0.2
    elif applicant["credit_score"] < 740:
        score += 0.1

    dti = applicant["debt"] / applicant["income"] if applicant["income"] > 0 else 1.0
    if dti > 0.5:
        score += 0.4
    elif dti > 0.35:
        score += 0.2
    elif dti > 0.2:
        score += 0.1

    if applicant["age"] < 25:
        score += 0.1

    score = min(score, 1.0)
    # BUG: missing return statement


def get_risk_level(score):
    if score is None:
        return "unknown"
    for level, threshold in RISK_LEVELS.items():
        if score <= threshold:
            return level
    return "very_high"


def evaluate_application(applicant):
    score = calculate_risk_score(applicant)
    return {
        "score": score,
        "level": get_risk_level(score),
        "approved": score is not None and score < 0.5,
    }
''',
        "thresholds.py": '''\
"""Risk level thresholds."""
RISK_LEVELS = {
    "low": 0.2, "medium": 0.4, "high": 0.6, "very_high": 0.8,
}
MIN_INCOME = 20000
MAX_DTI = 0.5
''',
        "applicant_utils.py": '''\
"""Applicant data utilities."""

def normalize_applicant(raw):
    return {
        "credit_score": int(raw.get("credit_score", 650)),
        "income": float(raw.get("income", 0)),
        "debt": float(raw.get("debt", 0)),
        "age": int(raw.get("age", 30)),
    }

def format_score(score):
    if score is None:
        return "N/A"
    return f"{score:.2f}"
''',
        "test_scoring.py": '''\
"""Tests for the risk scoring system."""
from scoring import calculate_risk_score, evaluate_application

def test_low_risk_applicant():
    applicant = {"credit_score": 780, "income": 80000, "debt": 10000, "age": 35}
    score = calculate_risk_score(applicant)
    assert score is not None, "Score should not be None"
    assert score < 0.3, f"Expected low risk, got {score}"

def test_high_risk_applicant():
    applicant = {"credit_score": 550, "income": 30000, "debt": 20000, "age": 22}
    score = calculate_risk_score(applicant)
    assert score is not None, "Score should not be None"
    assert score >= 0.5, f"Expected high risk, got {score}"

def test_score_is_float():
    applicant = {"credit_score": 700, "income": 50000, "debt": 15000, "age": 30}
    score = calculate_risk_score(applicant)
    assert isinstance(score, float), f"Expected float, got {type(score)}"

def test_evaluate_approved():
    applicant = {"credit_score": 780, "income": 80000, "debt": 5000, "age": 35}
    result = evaluate_application(applicant)
    assert result["approved"] is True, f"Got {result}"
''',
    },
    "relevant_files": ["scoring.py"],
    "irrelevant_files": ["thresholds.py", "applicant_utils.py"],
    "bug_location": {"file": "scoring.py", "line_start": 28, "line_end": 28},
    "correct_diagnosis_keywords": [
        "missing return", "no return", "return statement", "returns none",
        "forgot return", "return score",
    ],
    "correct_fix": {
        "old": "    score = min(score, 1.0)\n    # BUG: missing return statement",
        "new": "    score = min(score, 1.0)\n    return score",
    },
    "max_steps": 25,
    "task_id": "easy",
    "bug_2": None,
},

# ── easy_5 ── wrong comparison operator (> instead of >=) ─────────────────
{
    "id": "easy_5",
    "description": (
        "This discount engine applies tiered discounts to orders. "
        "Orders at exactly the threshold amount are not getting the discount "
        "they should qualify for. Find and fix the bug."
    ),
    "files": {
        "discounts.py": '''\
"""Tiered discount engine for order processing."""

from catalog import DISCOUNT_TIERS


def get_discount_rate(order_total):
    """
    Return the discount rate for a given order total.
    Customers qualify if their total is >= the threshold.
    Tiers: <100 = 0%, 100-499 = 5%, 500-999 = 10%, 1000+ = 15%
    """
    if order_total > 1000:  # BUG: should be >= 1000
        return 0.15
    elif order_total > 500:  # BUG: should be >= 500
        return 0.10
    elif order_total > 100:  # BUG: should be >= 100
        return 0.05
    return 0.0


def apply_discount(order_total):
    rate = get_discount_rate(order_total)
    discount = order_total * rate
    return {
        "original": order_total,
        "discount_rate": rate,
        "discount_amount": round(discount, 2),
        "final_total": round(order_total - discount, 2),
    }
''',
        "catalog.py": '''\
"""Product catalog and discount configuration."""
DISCOUNT_TIERS = {
    "bronze": {"min_order": 100, "rate": 0.05},
    "silver": {"min_order": 500, "rate": 0.10},
    "gold":   {"min_order": 1000, "rate": 0.15},
}
TAX_RATE = 0.08
FREE_SHIPPING_THRESHOLD = 75.0
''',
        "order_utils.py": '''\
"""Order utility functions."""

def validate_order_total(total):
    return isinstance(total, (int, float)) and total >= 0

def format_discount(rate):
    return f"{rate * 100:.0f}%"

def calculate_tax(total, rate=0.08):
    return round(total * rate, 2)
''',
        "test_discounts.py": '''\
"""Tests for the discount engine."""
from discounts import get_discount_rate, apply_discount

def test_no_discount_below_threshold():
    assert get_discount_rate(99.99) == 0.0

def test_exactly_at_100():
    assert get_discount_rate(100.0) == 0.05, f"Got {get_discount_rate(100.0)}"

def test_exactly_at_500():
    assert get_discount_rate(500.0) == 0.10, f"Got {get_discount_rate(500.0)}"

def test_exactly_at_1000():
    assert get_discount_rate(1000.0) == 0.15, f"Got {get_discount_rate(1000.0)}"

def test_above_threshold():
    assert get_discount_rate(1500.0) == 0.15

def test_apply_discount_boundary():
    result = apply_discount(100.0)
    assert result["discount_rate"] == 0.05, f"Got {result}"
    assert result["final_total"] == 95.0, f"Got {result}"
''',
    },
    "relevant_files": ["discounts.py"],
    "irrelevant_files": ["catalog.py", "order_utils.py"],
    "bug_location": {"file": "discounts.py", "line_start": 11, "line_end": 16},
    "correct_diagnosis_keywords": [
        "greater than", ">=", "should be >=", "wrong comparison",
        "boundary", "threshold", "exactly at", "> instead of >=",
    ],
    "correct_fix": {
        "old": (
            "    if order_total > 1000:  # BUG: should be >= 1000\n"
            "        return 0.15\n"
            "    elif order_total > 500:  # BUG: should be >= 500\n"
            "        return 0.10\n"
            "    elif order_total > 100:  # BUG: should be >= 100\n"
            "        return 0.05"
        ),
        "new": (
            "    if order_total >= 1000:\n"
            "        return 0.15\n"
            "    elif order_total >= 500:\n"
            "        return 0.10\n"
            "    elif order_total >= 100:\n"
            "        return 0.05"
        ),
    },
    "max_steps": 25,
    "task_id": "easy",
    "bug_2": None,
},

]  # end EASY_SCENARIOS


# ---------------------------------------------------------------------------
# Medium Scenarios (6)
# ---------------------------------------------------------------------------

MEDIUM_SCENARIOS = [

# ── medium_1 ── None check missing after sanitization ─────────────────────
{
    "id": "medium_1",
    "description": (
        "Users are reporting that the login endpoint sometimes returns a 500 error. "
        "It does not happen every time — only for certain usernames. "
        "Find the root cause and fix it."
    ),
    "files": {
        "api_handler.py": '''\
"""Login API handler."""
from utils import sanitize_username, hash_password
from session import create_session, validate_session

USER_DB = {
    "alice": "5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8",
    "bob":   "6b3a55e0261b0304143f805a24924d0c1c44524821305f31d9277843b8a10f4e",
}

def handle_login(username: str, password: str) -> dict:
    """Handle a login request."""
    clean_username = sanitize_username(username)
    # BUG: no None check — if sanitize_username returns None,
    # the next line raises AttributeError (500 error)
    if clean_username.lower() not in USER_DB:
        return {"success": False, "message": "User not found"}
    stored_hash = USER_DB[clean_username.lower()]
    provided_hash = hash_password(password)
    if stored_hash == provided_hash:
        token = create_session(clean_username)
        return {"success": True, "message": "Login successful", "token": token}
    return {"success": False, "message": "Invalid password"}

def handle_logout(token: str) -> dict:
    if validate_session(token):
        return {"success": True, "message": "Logged out"}
    return {"success": False, "message": "Invalid session"}
''',
        "utils.py": '''\
"""Utility functions for the API."""
import hashlib, re

def sanitize_username(username: str):
    username = username.strip()
    if not re.match(r"^[a-zA-Z0-9_]+$", username):
        return None  # silent failure — caller must check
    return username

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def sanitize_email(email: str):
    email = email.strip().lower()
    if "@" not in email or "." not in email.split("@")[-1]:
        return None
    return email
''',
        "session.py": '''\
"""Session management utilities."""
import hashlib, time
_SESSIONS = {}

def create_session(username: str) -> str:
    token = hashlib.sha256(f"{username}{time.time()}".encode()).hexdigest()[:32]
    _SESSIONS[token] = {"username": username, "expires_at": time.time() + 3600}
    return token

def validate_session(token: str) -> bool:
    if token not in _SESSIONS:
        return False
    if time.time() > _SESSIONS[token]["expires_at"]:
        del _SESSIONS[token]
        return False
    return True
''',
        "middleware.py": '''\
"""Request middleware for the API."""
import time

def rate_limit_check(ip: str, log: dict, max_req: int = 100) -> bool:
    now = time.time()
    log.setdefault(ip, [])
    log[ip] = [t for t in log[ip] if t > now - 60]
    if len(log[ip]) >= max_req:
        return False
    log[ip].append(now)
    return True

def extract_bearer_token(auth_header: str):
    if not auth_header or not auth_header.startswith("Bearer "):
        return None
    return auth_header[7:]
''',
        "test_api.py": '''\
"""Tests for the login API handler."""
from api_handler import handle_login

def test_valid_login():
    result = handle_login("alice", "password")
    assert result["success"] is True

def test_wrong_password():
    result = handle_login("alice", "wrongpassword")
    assert result["success"] is False

def test_unknown_user():
    result = handle_login("charlie", "password")
    assert result["success"] is False

def test_special_character_username():
    result = handle_login("alice@domain.com", "password")
    assert result["success"] is False
    assert "message" in result

def test_whitespace_username():
    result = handle_login("  alice  ", "password")
    assert result["success"] is True
''',
    },
    "relevant_files": ["api_handler.py", "utils.py"],
    "irrelevant_files": ["session.py", "middleware.py"],
    "bug_location": {"file": "api_handler.py", "line_start": 14, "line_end": 14},
    "correct_diagnosis_keywords": [
        "none check", "none", "sanitize_username returns none",
        "attributeerror", "null check", "not checking for none",
        "special character", "clean_username is none",
    ],
    "correct_fix": {
        "old": (
            "    # BUG: no None check — if sanitize_username returns None,\n"
            "    # the next line raises AttributeError (500 error)\n"
            "    if clean_username.lower() not in USER_DB:"
        ),
        "new": (
            "    if clean_username is None:\n"
            "        return {\"success\": False, \"message\": \"Invalid username\"}\n"
            "    if clean_username.lower() not in USER_DB:"
        ),
    },
    "max_steps": 35,
    "task_id": "medium",
    "bug_2": {
        "description": (
            "New report: usernames longer than 20 characters are being accepted "
            "by the login endpoint but failing silently downstream. "
            "The sanitizer should reject usernames that are too long."
        ),
        "relevant_files": ["utils.py"],
        "correct_diagnosis_keywords": [
            "length", "too long", "max length", "20 characters",
            "len(username)", "username length", "longer than",
        ],
        "correct_fix": {
            "old": '    if not re.match(r"^[a-zA-Z0-9_]+$", username):',
            "new": (
                "    if len(username) > 20:\n"
                "        return None\n"
                '    if not re.match(r"^[a-zA-Z0-9_]+$", username):'
            ),
        },
        "test_additions": {
            "test_api.py": '''
def test_long_username():
    result = handle_login("a" * 25, "password")
    assert result["success"] is False
    assert "message" in result
'''
        },
    },
},

# ── medium_2 ── empty string accepted as valid username ───────────────────
{
    "id": "medium_2",
    "description": (
        "The user registration endpoint is accepting empty usernames and creating "
        "accounts with blank usernames, causing database key conflicts. "
        "The validator should reject empty usernames. Find and fix the bug."
    ),
    "files": {
        "registration.py": '''\
"""User registration handler."""
from validator import validate_username, validate_password, validate_email
from storage import save_user, user_exists

def register_user(username: str, password: str, email: str) -> dict:
    username_result = validate_username(username)
    if not username_result["valid"]:
        return {"success": False, "message": username_result["error"]}
    password_result = validate_password(password)
    if not password_result["valid"]:
        return {"success": False, "message": password_result["error"]}
    if user_exists(username):
        return {"success": False, "message": "Username already taken"}
    save_user(username, password, email)
    return {"success": True, "message": "Registration successful"}
''',
        "validator.py": '''\
"""Input validation utilities."""
import re

def validate_username(username: str) -> dict:
    if username is None:
        return {"valid": False, "error": "Username is required"}
    username = username.strip()
    # BUG: missing empty string check — empty string passes all remaining checks
    if len(username) < 3:
        return {"valid": False, "error": "Username must be at least 3 characters"}
    if len(username) > 20:
        return {"valid": False, "error": "Username must be at most 20 characters"}
    if not re.match(r"^[a-zA-Z0-9_]+$", username):
        return {"valid": False, "error": "Username must be alphanumeric"}
    return {"valid": True, "error": None}

def validate_password(password: str) -> dict:
    if not password or len(password) < 8:
        return {"valid": False, "error": "Password must be at least 8 characters"}
    return {"valid": True, "error": None}

def validate_email(email: str) -> dict:
    if not email or "@" not in email:
        return {"valid": False, "error": "Invalid email address"}
    return {"valid": True, "error": None}
''',
        "storage.py": '''\
"""User storage simulation."""
_USERS = {}

def save_user(username: str, password: str, email: str) -> None:
    _USERS[username] = {"password": password, "email": email}

def user_exists(username: str) -> bool:
    return username in _USERS

def get_user(username: str) -> dict:
    return _USERS.get(username)
''',
        "audit_log.py": '''\
"""Audit logging for registration events."""
import time
_LOG = []

def log_registration(username: str, success: bool) -> None:
    _LOG.append({"username": username, "success": success, "ts": time.time()})

def get_recent_logs(n: int = 10) -> list:
    return _LOG[-n:]
''',
        "test_registration.py": '''\
"""Tests for user registration."""
from registration import register_user

def test_valid_registration():
    result = register_user("alice123", "securepass", "alice@example.com")
    assert result["success"] is True

def test_empty_username():
    result = register_user("", "securepass", "test@example.com")
    assert result["success"] is False
    assert "message" in result

def test_whitespace_only_username():
    result = register_user("   ", "securepass", "test@example.com")
    assert result["success"] is False

def test_short_username():
    result = register_user("ab", "securepass", "test@example.com")
    assert result["success"] is False

def test_invalid_chars():
    result = register_user("alice!", "securepass", "test@example.com")
    assert result["success"] is False
''',
    },
    "relevant_files": ["validator.py", "registration.py"],
    "irrelevant_files": ["storage.py", "audit_log.py"],
    "bug_location": {"file": "validator.py", "line_start": 8, "line_end": 8},
    "correct_diagnosis_keywords": [
        "empty string", "empty username", "missing empty check",
        "empty", "len == 0", "not username", "blank",
    ],
    "correct_fix": {
        "old": (
            "    # BUG: missing empty string check — empty string passes all remaining checks\n"
            "    if len(username) < 3:"
        ),
        "new": (
            "    if not username:\n"
            "        return {\"valid\": False, \"error\": \"Username cannot be empty\"}\n"
            "    if len(username) < 3:"
        ),
    },
    "max_steps": 35,
    "task_id": "medium",
    "bug_2": {
        "description": (
            "New issue: users are registering with email addresses that have no TLD "
            "(e.g. 'user@domain' without '.com'). The email validator is too permissive — "
            "it should require a dot in the domain part after the @ sign."
        ),
        "relevant_files": ["validator.py"],
        "correct_diagnosis_keywords": [
            "tld", "dot", "period", "email validation", "after @",
            "domain", "no dot", "missing dot",
        ],
        "correct_fix": {
            "old": (
                '    if not email or "@" not in email:\n'
                '        return {"valid": False, "error": "Invalid email address"}'
            ),
            "new": (
                '    if not email or "@" not in email:\n'
                '        return {"valid": False, "error": "Invalid email address"}\n'
                '    domain = email.split("@")[-1]\n'
                '    if "." not in domain:\n'
                '        return {"valid": False, "error": "Invalid email address"}'
            ),
        },
        "test_additions": {
            "test_registration.py": '''
def test_email_without_tld():
    result = register_user("validuser", "securepass", "user@nodot")
    assert result["success"] is False, f"Should reject email without TLD, got {result}"
'''
        },
    },
},

# ── medium_3 ── case sensitivity bug in password verification ─────────────
{
    "id": "medium_3",
    "description": (
        "Users report that login fails when they type their username with capital "
        "letters even though they registered with lowercase. The system should be "
        "case-insensitive for usernames. Find and fix the bug."
    ),
    "files": {
        "auth.py": '''\
"""Authentication handler."""
from hasher import hash_password, verify_password
from db import get_user_record

def authenticate(username: str, password: str) -> dict:
    normalized = username.lower().strip()
    user = get_user_record(normalized)
    if user is None:
        return {"success": False, "message": "User not found"}
    # BUG: using original username case instead of normalized for verification
    if not verify_password(password, user["password_hash"], username):
        return {"success": False, "message": "Invalid password"}
    return {"success": True, "message": "Authenticated", "user_id": user["id"]}

def change_password(username: str, old_password: str, new_password: str) -> dict:
    normalized = username.lower().strip()
    user = get_user_record(normalized)
    if user is None:
        return {"success": False, "message": "User not found"}
    if not verify_password(old_password, user["password_hash"], normalized):
        return {"success": False, "message": "Invalid current password"}
    return {"success": True, "message": "Password changed"}
''',
        "hasher.py": '''\
"""Password hashing utilities."""
import hashlib

def hash_password(password: str, username: str) -> str:
    salt = username.lower()
    return hashlib.sha256(f"{salt}:{password}".encode()).hexdigest()

def verify_password(password: str, stored_hash: str, username: str) -> bool:
    return hash_password(password, username) == stored_hash
''',
        "db.py": '''\
"""Simulated user database."""
import hashlib

def _make_hash(username, password):
    salt = username.lower()
    return hashlib.sha256(f"{salt}:{password}".encode()).hexdigest()

_USERS = {
    "alice": {"id": 1, "password_hash": _make_hash("alice", "secret123")},
    "bob":   {"id": 2, "password_hash": _make_hash("bob", "mypassword")},
}

def get_user_record(username: str) -> dict:
    return _USERS.get(username.lower())
''',
        "rate_limiter.py": '''\
"""Rate limiting for auth endpoints."""
import time
_ATTEMPTS = {}

def check_rate_limit(username: str, max_attempts: int = 5, window: int = 300) -> bool:
    now = time.time()
    key = username.lower()
    _ATTEMPTS.setdefault(key, [])
    _ATTEMPTS[key] = [t for t in _ATTEMPTS[key] if t > now - window]
    if len(_ATTEMPTS[key]) >= max_attempts:
        return False
    _ATTEMPTS[key].append(now)
    return True

def reset_attempts(username: str) -> None:
    _ATTEMPTS.pop(username.lower(), None)
''',
        "test_auth.py": '''\
"""Tests for the authentication handler."""
from auth import authenticate

def test_lowercase_login():
    result = authenticate("alice", "secret123")
    assert result["success"] is True, f"Got {result}"

def test_uppercase_username():
    result = authenticate("ALICE", "secret123")
    assert result["success"] is True, f"Uppercase should work, got {result}"

def test_mixed_case_username():
    result = authenticate("Alice", "secret123")
    assert result["success"] is True, f"Mixed case should work, got {result}"

def test_wrong_password():
    result = authenticate("alice", "wrongpassword")
    assert result["success"] is False

def test_unknown_user():
    result = authenticate("charlie", "password")
    assert result["success"] is False
''',
    },
    "relevant_files": ["auth.py", "hasher.py"],
    "irrelevant_files": ["rate_limiter.py", "db.py"],
    "bug_location": {"file": "auth.py", "line_start": 10, "line_end": 10},
    "correct_diagnosis_keywords": [
        "case sensitivity", "username case", "normalized", "lowercase",
        "original username", "should use normalized", "verify_password username",
    ],
    "correct_fix": {
        "old": '    if not verify_password(password, user["password_hash"], username):',
        "new": '    if not verify_password(password, user["password_hash"], normalized):',
    },
    "max_steps": 35,
    "task_id": "medium",
    "bug_2": {
        "description": (
            "New report: change_password is not rate-limited. "
            "Attackers can brute-force password changes without throttling. "
            "The rate_limiter module exists but is not being called in change_password."
        ),
        "relevant_files": ["auth.py", "rate_limiter.py"],
        "correct_diagnosis_keywords": [
            "rate limit", "rate_limit", "check_rate_limit", "not called",
            "missing rate", "brute force", "throttle",
        ],
        "correct_fix": {
            "old": (
                "def change_password(username: str, old_password: str, new_password: str) -> dict:\n"
                "    normalized = username.lower().strip()\n"
                "    user = get_user_record(normalized)"
            ),
            "new": (
                "def change_password(username: str, old_password: str, new_password: str) -> dict:\n"
                "    normalized = username.lower().strip()\n"
                "    from rate_limiter import check_rate_limit\n"
                "    if not check_rate_limit(normalized):\n"
                "        return {\"success\": False, \"message\": \"Too many attempts\"}\n"
                "    user = get_user_record(normalized)"
            ),
        },
        "test_additions": {
            "test_auth.py": '''
def test_change_password_rate_limited():
    from auth import change_password
    from rate_limiter import reset_attempts
    reset_attempts("alice")
    for _ in range(5):
        change_password("alice", "wrongpass", "newpass")
    result = change_password("alice", "wrongpass", "newpass")
    assert result["success"] is False
    assert "attempts" in result["message"].lower() or "too many" in result["message"].lower()
'''
        },
    },
},

# ── medium_4 ── wrong dictionary key in API response ──────────────────────
{
    "id": "medium_4",
    "description": (
        "The payment API returns responses but the frontend cannot read the "
        "transaction ID. The field is present in the processor result but "
        "missing from the API response. Find and fix the bug."
    ),
    "files": {
        "payment_handler.py": '''\
"""Payment processing handler."""
from payment_processor import process_payment, get_transaction
from validator import validate_payment_request

def handle_payment(request: dict) -> dict:
    validation = validate_payment_request(request)
    if not validation["valid"]:
        return {"success": False, "error": validation["message"]}

    result = process_payment(
        amount=request["amount"],
        currency=request.get("currency", "USD"),
        card_token=request["card_token"],
    )

    if result["status"] == "success":
        return {
            "success": True,
            # BUG: result uses 'transaction_id' but we return 'txn_id'
            "txn_id": result["transaction_id"],
            "amount": result["amount"],
            "currency": result["currency"],
        }
    return {"success": False, "error": result.get("error", "Payment failed")}

def handle_refund(transaction_id: str, amount: float) -> dict:
    txn = get_transaction(transaction_id)
    if not txn:
        return {"success": False, "error": "Transaction not found"}
    return {"success": True, "refunded": amount}
''',
        "payment_processor.py": '''\
"""Simulated payment processor."""
import uuid

def process_payment(amount: float, currency: str, card_token: str) -> dict:
    if not card_token or card_token == "invalid":
        return {"status": "failed", "error": "Invalid card token"}
    if amount <= 0:
        return {"status": "failed", "error": "Invalid amount"}
    return {
        "status": "success",
        "transaction_id": str(uuid.uuid4()),
        "amount": amount,
        "currency": currency,
    }

def get_transaction(transaction_id: str) -> dict:
    return {"id": transaction_id, "status": "completed"}
''',
        "validator.py": '''\
"""Payment request validator."""
SUPPORTED_CURRENCIES = {"USD", "EUR", "GBP", "JPY"}

def validate_payment_request(request: dict) -> dict:
    if "amount" not in request:
        return {"valid": False, "message": "amount is required"}
    if request["amount"] <= 0:
        return {"valid": False, "message": "amount must be positive"}
    if "card_token" not in request:
        return {"valid": False, "message": "card_token is required"}
    currency = request.get("currency", "USD")
    if currency not in SUPPORTED_CURRENCIES:
        return {"valid": False, "message": f"Unsupported currency: {currency}"}
    return {"valid": True, "message": None}
''',
        "webhook.py": '''\
"""Webhook handler for payment events."""
import hmac, hashlib

def verify_webhook_signature(payload: bytes, signature: str, secret: str) -> bool:
    expected = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)

def parse_webhook_event(payload: dict) -> dict:
    return {"type": payload.get("type", "unknown"), "data": payload.get("data", {})}
''',
        "test_payment.py": '''\
"""Tests for the payment handler."""
from payment_handler import handle_payment

def test_successful_payment_has_transaction_id():
    result = handle_payment({
        "amount": 100.0, "card_token": "tok_valid", "currency": "USD",
    })
    assert result["success"] is True
    assert "transaction_id" in result, f"Missing transaction_id in {result}"

def test_failed_payment_has_error():
    result = handle_payment({"amount": 100.0, "card_token": "invalid", "currency": "USD"})
    assert result["success"] is False

def test_missing_amount():
    result = handle_payment({"card_token": "tok_valid"})
    assert result["success"] is False

def test_negative_amount():
    result = handle_payment({"amount": -10.0, "card_token": "tok_valid"})
    assert result["success"] is False
''',
    },
    "relevant_files": ["payment_handler.py", "payment_processor.py"],
    "irrelevant_files": ["validator.py", "webhook.py"],
    "bug_location": {"file": "payment_handler.py", "line_start": 18, "line_end": 18},
    "correct_diagnosis_keywords": [
        "txn_id", "transaction_id", "wrong key", "key name",
        "field name", "missing field", "response key",
    ],
    "correct_fix": {
        "old": (
            "            # BUG: result uses 'transaction_id' but we return 'txn_id'\n"
            '            "txn_id": result["transaction_id"],'
        ),
        "new": '            "transaction_id": result["transaction_id"],',
    },
    "max_steps": 35,
    "task_id": "medium",
    "bug_2": {
        "description": (
            "New report: payments with JPY currency are being rejected even though "
            "JPY is in the supported currencies list. JPY uses whole numbers — "
            "an amount of 100 JPY is valid but being treated as invalid."
        ),
        "relevant_files": ["validator.py"],
        "correct_diagnosis_keywords": [
            "jpy", "whole number", "integer", "amount validation",
            "positive", "100 jpy", "currency specific",
        ],
        "correct_fix": {
            "old": (
                '    if request["amount"] <= 0:\n'
                '        return {"valid": False, "message": "amount must be positive"}'
            ),
            "new": (
                '    if request["amount"] <= 0:\n'
                '        return {"valid": False, "message": "amount must be positive"}\n'
                '    currency = request.get("currency", "USD")\n'
                '    if currency != "JPY" and request["amount"] < 0.01:\n'
                '        return {"valid": False, "message": "amount too small for currency"}'
            ),
        },
        "test_additions": {
            "test_payment.py": '''
def test_jpy_whole_number_amount():
    result = handle_payment({"amount": 100, "card_token": "tok_valid", "currency": "JPY"})
    assert result["success"] is True, f"JPY amount should be valid, got {result}"
    assert "transaction_id" in result
'''
        },
    },
},

# ── medium_5 ── missing strip on password ─────────────────────────────────
{
    "id": "medium_5",
    "description": (
        "Users who copy-paste their password from a document cannot log in "
        "even though the password is correct. It only affects passwords with "
        "leading or trailing spaces. Find and fix the bug."
    ),
    "files": {
        "login.py": '''\
"""Login handler for the user portal."""
from crypto import hash_value, safe_compare
from user_store import lookup_user

def process_login(username: str, password: str) -> dict:
    clean_username = username.strip().lower()
    if not clean_username:
        return {"authenticated": False, "reason": "Invalid username"}
    user = lookup_user(clean_username)
    if user is None:
        return {"authenticated": False, "reason": "User not found"}
    # BUG: password is not stripped before hashing
    # passwords with leading/trailing whitespace won\'t match
    password_hash = hash_value(password)
    if not safe_compare(password_hash, user["password_hash"]):
        return {"authenticated": False, "reason": "Incorrect password"}
    return {"authenticated": True, "user_id": user["id"], "username": clean_username}

def update_password(username: str, current: str, new_password: str) -> dict:
    clean = username.strip().lower()
    user = lookup_user(clean)
    if user is None:
        return {"success": False, "reason": "User not found"}
    if not safe_compare(hash_value(current.strip()), user["password_hash"]):
        return {"success": False, "reason": "Incorrect current password"}
    return {"success": True}
''',
        "crypto.py": '''\
"""Cryptographic utilities."""
import hashlib, hmac

def hash_value(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()

def safe_compare(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode(), b.encode())

def generate_token(length: int = 32) -> str:
    import secrets
    return secrets.token_hex(length)
''',
        "user_store.py": '''\
"""User storage layer."""
import hashlib

def _h(v):
    return hashlib.sha256(v.strip().encode()).hexdigest()

_USERS = {
    "alice": {"id": 1, "password_hash": _h("correcthorse")},
    "bob":   {"id": 2, "password_hash": _h("battery staple")},
}

def lookup_user(username: str) -> dict:
    return _USERS.get(username.lower())
''',
        "logger.py": '''\
"""Login event logger."""
import time
_EVENTS = []

def log_login_attempt(username: str, success: bool) -> None:
    _EVENTS.append({"username": username, "success": success, "ts": time.time()})

def get_failed_attempts(username: str) -> list:
    return [e for e in _EVENTS if e["username"] == username and not e["success"]]
''',
        "test_login.py": '''\
"""Tests for the login handler."""
from login import process_login

def test_correct_password():
    result = process_login("alice", "correcthorse")
    assert result["authenticated"] is True, f"Got {result}"

def test_wrong_password():
    result = process_login("alice", "wrongpassword")
    assert result["authenticated"] is False

def test_unknown_user():
    result = process_login("charlie", "password")
    assert result["authenticated"] is False

def test_password_with_trailing_space():
    result = process_login("alice", "correcthorse ")
    assert result["authenticated"] is True, f"Trailing space should be stripped, got {result}"

def test_password_with_leading_space():
    result = process_login("alice", " correcthorse")
    assert result["authenticated"] is True, f"Leading space should be stripped, got {result}"
''',
    },
    "relevant_files": ["login.py", "crypto.py"],
    "irrelevant_files": ["logger.py", "user_store.py"],
    "bug_location": {"file": "login.py", "line_start": 12, "line_end": 12},
    "correct_diagnosis_keywords": [
        "strip", "whitespace", "trim", "trailing space", "leading space",
        "password not stripped", "missing strip", "hash_value(password)",
    ],
    "correct_fix": {
        "old": (
            "    # BUG: password is not stripped before hashing\n"
            "    # passwords with leading/trailing whitespace won't match\n"
            "    password_hash = hash_value(password)"
        ),
        "new": "    password_hash = hash_value(password.strip())",
    },
    "max_steps": 35,
    "task_id": "medium",
    "bug_2": {
        "description": (
            "New report: update_password is not enforcing a minimum password length. "
            "Users can set their password to a single character. "
            "New passwords should be at least 8 characters."
        ),
        "relevant_files": ["login.py"],
        "correct_diagnosis_keywords": [
            "minimum length", "password length", "8 characters",
            "too short", "len(new_password)", "length check",
        ],
        "correct_fix": {
            "old": (
                '    if not safe_compare(hash_value(current.strip()), user["password_hash"]):\n'
                '        return {"success": False, "reason": "Incorrect current password"}\n'
                '    return {"success": True}'
            ),
            "new": (
                '    if not safe_compare(hash_value(current.strip()), user["password_hash"]):\n'
                '        return {"success": False, "reason": "Incorrect current password"}\n'
                '    if len(new_password.strip()) < 8:\n'
                '        return {"success": False, "reason": "Password must be at least 8 characters"}\n'
                '    return {"success": True}'
            ),
        },
        "test_additions": {
            "test_login.py": '''
def test_update_password_too_short():
    from login import update_password
    result = update_password("alice", "correcthorse", "abc")
    assert result["success"] is False
    assert "8" in result["reason"] or "characters" in result["reason"]
'''
        },
    },
},

# ── medium_6 ── sessions not invalidated on account lockout ───────────────
{
    "id": "medium_6",
    "description": (
        "A security audit found that when a user's account is locked after too many "
        "failed login attempts, their existing active sessions are not invalidated. "
        "An attacker with a valid session token before lockout can still use it. "
        "Find and fix the bug."
    ),
    "files": {
        "account.py": '''\
"""Account management — login attempts and lockout."""
from session_store import get_sessions, invalidate_all_sessions, create_session
from attempt_tracker import record_attempt, get_attempt_count, reset_attempts

MAX_ATTEMPTS = 5

def attempt_login(user_id: str, success: bool) -> dict:
    if success:
        reset_attempts(user_id)
        return {"locked": False, "attempts": 0}
    record_attempt(user_id)
    count = get_attempt_count(user_id)
    if count >= MAX_ATTEMPTS:
        # BUG: account locked but existing sessions NOT invalidated
        return {"locked": True, "attempts": count, "message": "Account locked"}
    return {"locked": False, "attempts": count}

def unlock_account(user_id: str) -> dict:
    reset_attempts(user_id)
    invalidate_all_sessions(user_id)
    return {"success": True, "message": "Account unlocked"}
''',
        "session_store.py": '''\
"""Session storage and management."""
import time
_SESSIONS = {}

def create_session(user_id: str) -> str:
    import secrets
    token = secrets.token_hex(32)
    _SESSIONS[token] = {"user_id": user_id, "created_at": time.time()}
    return token

def get_sessions(user_id: str) -> list:
    return [t for t, s in _SESSIONS.items() if s["user_id"] == user_id]

def validate_token(token: str) -> dict:
    return _SESSIONS.get(token)

def invalidate_token(token: str) -> None:
    _SESSIONS.pop(token, None)

def invalidate_all_sessions(user_id: str) -> None:
    for token in get_sessions(user_id):
        _SESSIONS.pop(token, None)
''',
        "attempt_tracker.py": '''\
"""Login attempt tracking."""
import time
_ATTEMPTS = {}

def record_attempt(user_id: str) -> None:
    _ATTEMPTS.setdefault(user_id, [])
    _ATTEMPTS[user_id].append(time.time())

def get_attempt_count(user_id: str, window: int = 900) -> int:
    now = time.time()
    return sum(1 for t in _ATTEMPTS.get(user_id, []) if t > now - window)

def reset_attempts(user_id: str) -> None:
    _ATTEMPTS.pop(user_id, None)
''',
        "notification.py": '''\
"""Security notification service."""
_NOTIFICATIONS = []

def send_lockout_notice(user_id: str, email: str) -> None:
    _NOTIFICATIONS.append({"type": "lockout", "user_id": user_id, "email": email})

def send_unlock_notice(user_id: str, email: str) -> None:
    _NOTIFICATIONS.append({"type": "unlock", "user_id": user_id, "email": email})

def get_notifications() -> list:
    return list(_NOTIFICATIONS)
''',
        "test_account.py": '''\
"""Tests for account lockout behavior."""
from account import attempt_login, unlock_account
from session_store import create_session, validate_token

def test_successful_login_resets_attempts():
    result = attempt_login("user1", success=True)
    assert result["locked"] is False

def test_lockout_after_max_attempts():
    for _ in range(5):
        attempt_login("user_lock_test", success=False)
    result = attempt_login("user_lock_test", success=False)
    assert result["locked"] is True

def test_sessions_invalidated_on_lockout():
    token = create_session("user3")
    assert validate_token(token) is not None
    for _ in range(5):
        attempt_login("user3", success=False)
    attempt_login("user3", success=False)
    assert validate_token(token) is None, "Session should be invalidated after lockout"

def test_unlock_invalidates_sessions():
    token = create_session("user4")
    unlock_account("user4")
    assert validate_token(token) is None
''',
    },
    "relevant_files": ["account.py", "session_store.py"],
    "irrelevant_files": ["notification.py", "attempt_tracker.py"],
    "bug_location": {"file": "account.py", "line_start": 11, "line_end": 13},
    "correct_diagnosis_keywords": [
        "invalidate", "sessions not invalidated", "session still valid",
        "invalidate_all_sessions", "missing invalidation", "lockout sessions",
    ],
    "correct_fix": {
        "old": (
            "    if count >= MAX_ATTEMPTS:\n"
            "        # BUG: account locked but existing sessions NOT invalidated\n"
            '        return {"locked": True, "attempts": count, "message": "Account locked"}'
        ),
        "new": (
            "    if count >= MAX_ATTEMPTS:\n"
            "        invalidate_all_sessions(user_id)\n"
            '        return {"locked": True, "attempts": count, "message": "Account locked"}'
        ),
    },
    "max_steps": 35,
    "task_id": "medium",
    "bug_2": {
        "description": (
            "New finding: successful logins reset the attempt counter but do not "
            "create a session token. Users authenticate but receive no token back. "
            "The login flow should return a session token on success."
        ),
        "relevant_files": ["account.py", "session_store.py"],
        "correct_diagnosis_keywords": [
            "create_session", "no session", "session token", "missing session",
            "not creating", "session not returned",
        ],
        "correct_fix": {
            "old": (
                "    if success:\n"
                "        reset_attempts(user_id)\n"
                '        return {"locked": False, "attempts": 0}'
            ),
            "new": (
                "    if success:\n"
                "        reset_attempts(user_id)\n"
                "        token = create_session(user_id)\n"
                '        return {"locked": False, "attempts": 0, "session_token": token}'
            ),
        },
        "test_additions": {
            "test_account.py": '''
def test_successful_login_returns_session_token():
    result = attempt_login("user5", success=True)
    assert "session_token" in result, f"Should return session token, got {result}"
    assert result["session_token"] is not None
'''
        },
    },
},

]  # end MEDIUM_SCENARIOS


# ---------------------------------------------------------------------------
# Hard Scenarios (8)
# ---------------------------------------------------------------------------

HARD_SCENARIOS = [

# ── hard_1 ── shallow merge corrupting nested config ──────────────────────
{
    "id": "hard_1",
    "description": (
        "Our configuration system works fine in development but silently corrupts "
        "user settings in production. When users have both custom theme preferences "
        "AND notification preferences set, their theme gets reset to defaults. "
        "No error is raised. Find the root cause and fix it."
    ),
    "files": {
        "config_manager.py": '''\
"""Configuration manager — entry point for loading and applying user config."""
from loader import load_config
from validator import validate_config
from audit import log_config_change

def get_user_config(user_id: str, user_prefs: dict) -> dict:
    validated = validate_config(user_prefs)
    final = load_config(user_id, validated)
    log_config_change(user_id, user_prefs, final)
    return final

def reset_user_config(user_id: str) -> dict:
    from defaults import DEFAULT_CONFIG
    log_config_change(user_id, {}, DEFAULT_CONFIG)
    return dict(DEFAULT_CONFIG)
''',
        "loader.py": '''\
"""Config loader — merges user preferences with system defaults."""
from defaults import DEFAULT_CONFIG

def load_config(user_id: str, user_prefs: dict) -> dict:
    merged = {}
    for key, value in DEFAULT_CONFIG.items():
        merged[key] = value
    for key, value in user_prefs.items():
        if key in merged and isinstance(merged[key], dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    # BUG: post-process block re-merges theme using merged["theme"] which has
    # already lost user values when notification_preferences is also present
    if "notification_preferences" in merged:
        merged["notification_preferences"] = {
            **DEFAULT_CONFIG.get("notification_preferences", {}),
            **merged.get("notification_preferences", {}),
        }
        if "theme" in DEFAULT_CONFIG:
            merged["theme"] = {
                **DEFAULT_CONFIG["theme"],
                **merged.get("theme", {}),
            }
    return merged
''',
        "validator.py": '''\
"""Config validator — checks that user preferences are well-formed."""
ALLOWED_KEYS = {"theme","notification_preferences","language","timezone","display_density"}
ALLOWED_THEMES = {"dark", "light", "system"}
ALLOWED_DENSITIES = {"compact", "comfortable", "spacious"}

def validate_config(user_prefs: dict) -> dict:
    cleaned = {}
    for key, value in user_prefs.items():
        if key not in ALLOWED_KEYS:
            continue
        if key == "theme" and isinstance(value, dict):
            if value.get("mode", "system") not in ALLOWED_THEMES:
                value["mode"] = "system"
            cleaned[key] = value
        elif key == "notification_preferences" and isinstance(value, dict):
            cleaned[key] = value
        elif key == "display_density":
            # BUG 2: missing validation — invalid densities pass through silently
            cleaned[key] = value
        else:
            cleaned[key] = value
    return cleaned
''',
        "defaults.py": '''\
"""System default configuration."""
DEFAULT_CONFIG = {
    "theme": {"mode": "system", "font_size": 14, "accent_color": "#0066cc"},
    "notification_preferences": {"email": True, "push": True, "sms": False, "frequency": "daily"},
    "language": "en",
    "timezone": "UTC",
    "display_density": "comfortable",
}
''',
        "audit.py": '''\
"""Audit logging for configuration changes."""
import time
_LOG = []

def log_config_change(user_id, old, new):
    _LOG.append({"user_id": user_id, "timestamp": time.time(), "old": old, "new": new})

def get_audit_log(user_id=None):
    if user_id is None:
        return list(_LOG)
    return [e for e in _LOG if e["user_id"] == user_id]
''',
        "cache.py": '''\
"""Config caching layer."""
import time
_CACHE = {}
_TTL = 300

def get_cached_config(user_id):
    entry = _CACHE.get(user_id)
    if entry is None:
        return None
    if time.time() - entry["cached_at"] > _TTL:
        del _CACHE[user_id]
        return None
    return entry["config"]

def set_cached_config(user_id, config):
    _CACHE[user_id] = {"config": config, "cached_at": time.time()}

def invalidate_cache(user_id):
    _CACHE.pop(user_id, None)
''',
        "test_config.py": '''\
"""Tests for the config system."""
from config_manager import get_user_config

def test_theme_only():
    result = get_user_config("u1", {"theme": {"mode": "dark", "font_size": 16}})
    assert result["theme"]["mode"] == "dark"
    assert result["theme"]["font_size"] == 16

def test_notifications_only():
    result = get_user_config("u1", {"notification_preferences": {"email": False}})
    assert result["notification_preferences"]["email"] is False

def test_theme_and_notifications_together():
    prefs = {
        "theme": {"mode": "dark", "font_size": 18, "accent_color": "#ff0000"},
        "notification_preferences": {"email": False, "push": False},
    }
    result = get_user_config("u1", prefs)
    assert result["theme"]["mode"] == "dark", f"Theme reset. Got: {result['theme']}"
    assert result["theme"]["font_size"] == 18
    assert result["theme"]["accent_color"] == "#ff0000"

def test_defaults_applied():
    result = get_user_config("u1", {"language": "fr"})
    assert result["language"] == "fr"
    assert "theme" in result
''',
    },
    "relevant_files": ["loader.py", "defaults.py", "config_manager.py"],
    "irrelevant_files": ["audit.py", "cache.py"],
    "bug_location": {"file": "loader.py", "line_start": 14, "line_end": 22},
    "correct_diagnosis_keywords": [
        "shallow merge", "shallow copy", "nested dict", "theme overwritten",
        "notification", "post-process", "merged[key]", "lost user values", "resets theme",
    ],
    "correct_fix": {
        "old": (
            '    # BUG: post-process block re-merges theme using merged["theme"] which has\n'
            '    # already lost user values when notification_preferences is also present\n'
            '    if "notification_preferences" in merged:\n'
            '        merged["notification_preferences"] = {\n'
            '            **DEFAULT_CONFIG.get("notification_preferences", {}),\n'
            '            **merged.get("notification_preferences", {}),\n'
            '        }\n'
            '        if "theme" in DEFAULT_CONFIG:\n'
            '            merged["theme"] = {\n'
            '                **DEFAULT_CONFIG["theme"],\n'
            '                **merged.get("theme", {}),\n'
            '            }'
        ),
        "new": (
            '    if "notification_preferences" in user_prefs:\n'
            '        merged["notification_preferences"] = {\n'
            '            **DEFAULT_CONFIG.get("notification_preferences", {}),\n'
            '            **user_prefs.get("notification_preferences", {}),\n'
            '        }'
        ),
    },
    "max_steps": 45,
    "task_id": "hard",
    "bug_2": {
        "description": (
            "QA found: users setting display_density to invalid values like "
            "'ultra-compact' have it silently accepted. The validator should "
            "reject invalid densities and fall back to 'comfortable'."
        ),
        "relevant_files": ["validator.py"],
        "correct_diagnosis_keywords": [
            "display_density", "allowed_densities", "validation",
            "invalid density", "missing check", "density check",
        ],
        "correct_fix": {
            "old": (
                '        elif key == "display_density":\n'
                '            # BUG 2: missing validation — invalid densities pass through silently\n'
                '            cleaned[key] = value'
            ),
            "new": (
                '        elif key == "display_density":\n'
                '            if value not in ALLOWED_DENSITIES:\n'
                '                value = "comfortable"\n'
                '            cleaned[key] = value'
            ),
        },
        "test_additions": {
            "test_config.py": '''
def test_invalid_display_density():
    result = get_user_config("u1", {"display_density": "ultra-compact"})
    assert result["display_density"] == "comfortable", f"Got {result['display_density']}"
'''
        },
    },
},

# ── hard_2 ── mutable default argument ────────────────────────────────────
{
    "id": "hard_2",
    "description": (
        "Our feature flag system is behaving strangely. After the first user's "
        "flags are evaluated, subsequent users are getting the first user's flag "
        "overrides mixed into their own. The flags seem to be leaking between users. "
        "No error is raised. Find the root cause and fix it."
    ),
    "files": {
        "feature_flags.py": '''\
"""Feature flag evaluation system."""
from flag_store import get_user_overrides, get_default_flags
from rules import evaluate_rollout_rules
from logger import log_flag_evaluation


def get_flags_for_user(user_id: str, context: dict = {}) -> dict:
    """
    Get the effective feature flags for a user.
    BUG: mutable default argument — the context dict is shared across
    all calls that don\'t provide a context argument.
    """
    context["user_id"] = user_id
    context["timestamp"] = __import__("time").time()
    overrides = get_user_overrides(user_id)
    context["has_overrides"] = len(overrides) > 0
    flags = evaluate_rollout_rules(user_id, context)
    flags.update(overrides)
    log_flag_evaluation(user_id, flags)
    return flags


def clear_user_context(user_id: str) -> None:
    """Clear any cached context for a user."""
    pass  # currently a no-op
''',
        "flag_store.py": '''\
"""Feature flag storage."""
_OVERRIDES = {
    "user_premium_1": {"new_dashboard": True, "beta_export": True},
    "user_basic_2": {"new_dashboard": False},
}
_DEFAULT_FLAGS = {
    "new_dashboard": False, "beta_export": False,
    "dark_mode": True, "analytics": True,
}

def get_user_overrides(user_id: str) -> dict:
    return dict(_OVERRIDES.get(user_id, {}))

def get_default_flags() -> dict:
    return dict(_DEFAULT_FLAGS)
''',
        "rules.py": '''\
"""Rollout rule evaluation."""
from flag_store import get_default_flags

def evaluate_rollout_rules(user_id: str, context: dict) -> dict:
    flags = get_default_flags()
    if hash(user_id) % 100 < 10:
        flags["beta_export"] = True
    return flags
''',
        "logger.py": '''\
"""Flag evaluation logger."""
import time
_LOG = []

def log_flag_evaluation(user_id: str, flags: dict) -> None:
    _LOG.append({"user_id": user_id, "flags": dict(flags), "ts": time.time()})

def get_log() -> list:
    return list(_LOG)

def clear_log() -> None:
    _LOG.clear()
''',
        "analytics.py": '''\
"""Flag analytics."""
_EVENTS = []

def track_feature_use(user_id: str, feature: str, action: str) -> None:
    _EVENTS.append({"user_id": user_id, "feature": feature, "action": action})

def get_feature_stats(feature: str) -> dict:
    return {"total": len([e for e in _EVENTS if e["feature"] == feature])}
''',
        "test_flags.py": '''\
"""Tests for the feature flag system."""
from feature_flags import get_flags_for_user

def test_basic_user_gets_defaults():
    flags = get_flags_for_user("user_basic_2")
    assert flags["dark_mode"] is True

def test_context_not_shared_between_calls():
    """Bug trigger: if context is shared, second call gets contaminated."""
    get_flags_for_user("user_premium_1")
    flags_basic = get_flags_for_user("regular_user_99")
    assert flags_basic.get("new_dashboard") is False, (
        f"Context leaked from previous call. Got: {flags_basic}"
    )

def test_premium_user_gets_overrides():
    flags = get_flags_for_user("user_premium_1")
    assert flags["new_dashboard"] is True
''',
    },
    "relevant_files": ["feature_flags.py", "rules.py"],
    "irrelevant_files": ["analytics.py", "logger.py"],
    "bug_location": {"file": "feature_flags.py", "line_start": 6, "line_end": 6},
    "correct_diagnosis_keywords": [
        "mutable default", "default argument", "shared dict", "context = {}",
        "mutable default argument", "persists between calls", "leaked context",
    ],
    "correct_fix": {
        "old": "def get_flags_for_user(user_id: str, context: dict = {}) -> dict:",
        "new": (
            "def get_flags_for_user(user_id: str, context: dict = None) -> dict:\n"
            "    if context is None:\n"
            "        context = {}"
        ),
    },
    "max_steps": 45,
    "task_id": "hard",
    "bug_2": {
        "description": (
            "New issue: clear_user_context is a no-op but is being called by "
            "the cleanup pipeline expecting it to clear log entries for that user. "
            "The log is growing unboundedly. clear_user_context should remove "
            "all log entries for the given user_id."
        ),
        "relevant_files": ["feature_flags.py", "logger.py"],
        "correct_diagnosis_keywords": [
            "no-op", "clear log", "log growing", "clear_log",
            "unbounded", "log entries", "clear_user_context",
        ],
        "correct_fix": {
            "old": (
                "def clear_user_context(user_id: str) -> None:\n"
                "    \"\"\"Clear any cached context for a user.\"\"\"\n"
                "    pass  # currently a no-op"
            ),
            "new": (
                "def clear_user_context(user_id: str) -> None:\n"
                "    \"\"\"Clear any cached context for a user.\"\"\"\n"
                "    from logger import _LOG\n"
                '    _LOG[:] = [e for e in _LOG if e["user_id"] != user_id]'
            ),
        },
        "test_additions": {
            "test_flags.py": '''
def test_clear_user_context_removes_logs():
    from feature_flags import clear_user_context
    from logger import get_log, clear_log
    clear_log()
    get_flags_for_user("user_to_clear")
    assert any(e["user_id"] == "user_to_clear" for e in get_log())
    clear_user_context("user_to_clear")
    assert not any(e["user_id"] == "user_to_clear" for e in get_log())
'''
        },
    },
},

# ── hard_3 ── cache invalidation in wrong order ────────────────────────────
{
    "id": "hard_3",
    "description": (
        "Our user profile system has a subtle caching bug. When a user updates "
        "their profile, other services read the updated data correctly. But if "
        "the same user reads their own profile within seconds of updating, they "
        "see the old data. The cache invalidation is happening in the wrong order. "
        "Find and fix it."
    ),
    "files": {
        "profile_service.py": '''\
"""User profile service."""
from profile_cache import get_cached_profile, set_cached_profile, invalidate_profile
from profile_store import read_profile, write_profile
from event_bus import publish_profile_updated


def get_profile(user_id: str) -> dict:
    cached = get_cached_profile(user_id)
    if cached is not None:
        return cached
    profile = read_profile(user_id)
    if profile:
        set_cached_profile(user_id, profile)
    return profile


def update_profile(user_id: str, updates: dict) -> dict:
    """
    Update a user profile.
    BUG: cache is invalidated AFTER the write instead of BEFORE.
    """
    current = read_profile(user_id)
    if current is None:
        return {"success": False, "error": "User not found"}
    updated = {**current, **updates}
    write_profile(user_id, updated)          # 1. write new data
    publish_profile_updated(user_id, updated)
    invalidate_profile(user_id)              # 2. BUG: invalidate AFTER write
    return {"success": True, "profile": updated}


def delete_profile(user_id: str) -> dict:
    invalidate_profile(user_id)
    write_profile(user_id, None)
    return {"success": True}
''',
        "profile_cache.py": '''\
"""Profile caching layer."""
import time
_CACHE = {}
_TTL = 300

def get_cached_profile(user_id: str) -> dict:
    entry = _CACHE.get(user_id)
    if entry is None:
        return None
    if time.time() - entry["cached_at"] > _TTL:
        del _CACHE[user_id]
        return None
    return entry["profile"]

def set_cached_profile(user_id: str, profile: dict) -> None:
    _CACHE[user_id] = {"profile": dict(profile), "cached_at": time.time()}

def invalidate_profile(user_id: str) -> None:
    _CACHE.pop(user_id, None)

def get_cache_entry(user_id: str) -> dict:
    return _CACHE.get(user_id)
''',
        "profile_store.py": '''\
"""Profile persistence layer."""
_STORE = {
    "user1": {"name": "Alice", "email": "alice@example.com", "bio": "Engineer"},
    "user2": {"name": "Bob",   "email": "bob@example.com",   "bio": "Designer"},
}

def read_profile(user_id: str) -> dict:
    p = _STORE.get(user_id)
    return dict(p) if p else None

def write_profile(user_id: str, profile: dict) -> None:
    if profile is None:
        _STORE.pop(user_id, None)
    else:
        _STORE[user_id] = dict(profile)
''',
        "event_bus.py": '''\
"""Simple event bus for profile events."""
_HANDLERS = []
_EVENTS = []

def subscribe(handler) -> None:
    _HANDLERS.append(handler)

def publish_profile_updated(user_id: str, profile: dict) -> None:
    event = {"type": "profile_updated", "user_id": user_id, "profile": profile}
    _EVENTS.append(event)
    for handler in _HANDLERS:
        try:
            handler(event)
        except Exception:
            pass

def get_events() -> list:
    return list(_EVENTS)
''',
        "notifications.py": '''\
"""Profile update notifications."""

def notify_profile_change(user_id: str, changed_fields: list) -> None:
    pass

def get_changed_fields(old: dict, new: dict) -> list:
    return [k for k in new if old.get(k) != new.get(k)]
''',
        "test_profile.py": '''\
"""Tests for the profile service."""
from profile_service import get_profile, update_profile
from profile_cache import set_cached_profile, get_cache_entry

def test_get_profile():
    profile = get_profile("user1")
    assert profile["name"] == "Alice"

def test_update_profile():
    result = update_profile("user1", {"bio": "Senior Engineer"})
    assert result["success"] is True
    assert result["profile"]["bio"] == "Senior Engineer"

def test_cache_invalidated_before_write():
    """Bug trigger: cache should be gone after update."""
    set_cached_profile("user2", {"name": "Bob", "email": "bob@example.com", "bio": "Designer"})
    update_profile("user2", {"bio": "Lead Designer"})
    cache_entry = get_cache_entry("user2")
    assert cache_entry is None, f"Cache should be invalidated, found: {cache_entry}"
    fresh = get_profile("user2")
    assert fresh["bio"] == "Lead Designer", f"Expected updated bio, got {fresh}"

def test_update_nonexistent_user():
    result = update_profile("ghost_user", {"bio": "Nobody"})
    assert result["success"] is False
''',
    },
    "relevant_files": ["profile_service.py", "profile_cache.py"],
    "irrelevant_files": ["event_bus.py", "notifications.py"],
    "bug_location": {"file": "profile_service.py", "line_start": 22, "line_end": 25},
    "correct_diagnosis_keywords": [
        "cache invalidation order", "invalidate before write", "order",
        "after write", "before write", "stale cache", "wrong order", "invalidate first",
    ],
    "correct_fix": {
        "old": (
            "    write_profile(user_id, updated)          # 1. write new data\n"
            "    publish_profile_updated(user_id, updated)\n"
            "    invalidate_profile(user_id)              # 2. BUG: invalidate AFTER write"
        ),
        "new": (
            "    invalidate_profile(user_id)              # 1. invalidate BEFORE write\n"
            "    write_profile(user_id, updated)          # 2. write new data\n"
            "    publish_profile_updated(user_id, updated)"
        ),
    },
    "max_steps": 45,
    "task_id": "hard",
    "bug_2": {
        "description": (
            "New bug: get_profile is caching None when a deleted user's profile "
            "is requested. Subsequent calls return None from cache instead of "
            "hitting the store. get_profile should not cache None profiles."
        ),
        "relevant_files": ["profile_service.py"],
        "correct_diagnosis_keywords": [
            "cache none", "caching none", "none profile", "should not cache",
            "null profile", "if profile", "profile is not none",
        ],
        "correct_fix": {
            "old": (
                "    profile = read_profile(user_id)\n"
                "    if profile:\n"
                "        set_cached_profile(user_id, profile)\n"
                "    return profile"
            ),
            "new": (
                "    profile = read_profile(user_id)\n"
                "    if profile is not None:\n"
                "        set_cached_profile(user_id, profile)\n"
                "    return profile"
            ),
        },
        "test_additions": {
            "test_profile.py": '''
def test_deleted_profile_not_cached():
    from profile_service import delete_profile
    from profile_cache import get_cache_entry
    update_profile("user2", {"bio": "temp"})
    delete_profile("user2")
    get_profile("user2")
    assert get_cache_entry("user2") is None, "Deleted profile should not be cached"
'''
        },
    },
},

# ── hard_4 ── wrong merge direction ───────────────────────────────────────
{
    "id": "hard_4",
    "description": (
        "Our notification settings merger is silently ignoring user preferences. "
        "Users set their notification channels but after saving, the settings "
        "revert to system defaults. No error is raised. Find the root cause and fix it."
    ),
    "files": {
        "notification_manager.py": '''\
"""Notification settings manager."""
from settings_store import load_settings, save_settings
from defaults import DEFAULT_NOTIFICATION_SETTINGS
from validator import validate_notification_settings
from history import record_settings_change


def update_notification_settings(user_id: str, user_settings: dict) -> dict:
    validated = validate_notification_settings(user_settings)
    if not validated["valid"]:
        return {"success": False, "error": validated["error"]}
    current = load_settings(user_id) or {}
    # BUG: merge direction is wrong — defaults overwrite user settings
    merged = {**user_settings, **DEFAULT_NOTIFICATION_SETTINGS}
    save_settings(user_id, merged)
    record_settings_change(user_id, current, merged)
    return {"success": True, "settings": merged}


def get_notification_settings(user_id: str) -> dict:
    stored = load_settings(user_id)
    if stored is None:
        return dict(DEFAULT_NOTIFICATION_SETTINGS)
    return {**DEFAULT_NOTIFICATION_SETTINGS, **stored}


def reset_notification_settings(user_id: str) -> dict:
    save_settings(user_id, None)
    return {"success": True, "settings": dict(DEFAULT_NOTIFICATION_SETTINGS)}
''',
        "defaults.py": '''\
"""Default notification settings."""
DEFAULT_NOTIFICATION_SETTINGS = {
    "email_enabled": True,
    "push_enabled": True,
    "sms_enabled": False,
    "digest_frequency": "daily",
    "quiet_hours_start": 22,
    "quiet_hours_end": 8,
}
''',
        "settings_store.py": '''\
"""Settings persistence."""
_SETTINGS = {}

def load_settings(user_id: str) -> dict:
    return dict(_SETTINGS[user_id]) if user_id in _SETTINGS else None

def save_settings(user_id: str, settings: dict) -> None:
    if settings is None:
        _SETTINGS.pop(user_id, None)
    else:
        _SETTINGS[user_id] = dict(settings)
''',
        "validator.py": '''\
"""Notification settings validator."""
VALID_FREQUENCIES = {"realtime", "hourly", "daily", "weekly"}

def validate_notification_settings(settings: dict) -> dict:
    if not isinstance(settings, dict):
        return {"valid": False, "error": "Settings must be a dict"}
    freq = settings.get("digest_frequency")
    if freq is not None and freq not in VALID_FREQUENCIES:
        return {"valid": False, "error": f"Invalid frequency: {freq}"}
    return {"valid": True, "error": None}
''',
        "history.py": '''\
"""Settings change history."""
import time
_HISTORY = []

def record_settings_change(user_id, old, new):
    _HISTORY.append({"user_id": user_id, "old": old, "new": new, "ts": time.time()})

def get_history(user_id):
    return [h for h in _HISTORY if h["user_id"] == user_id]
''',
        "templates.py": '''\
"""Notification templates."""
_TEMPLATES = {
    "welcome": "Welcome to our service!",
    "digest": "Here is your {frequency} digest.",
}

def get_template(name, **kwargs):
    template = _TEMPLATES.get(name, "")
    return template.format(**kwargs) if kwargs else template
''',
        "test_notifications.py": '''\
"""Tests for notification settings manager."""
from notification_manager import update_notification_settings

def test_disable_email():
    result = update_notification_settings("user1", {"email_enabled": False})
    assert result["success"] is True
    assert result["settings"]["email_enabled"] is False, f"Got {result['settings']}"

def test_disable_push():
    result = update_notification_settings("user2", {"push_enabled": False})
    assert result["success"] is True
    assert result["settings"]["push_enabled"] is False, f"Got {result['settings']}"

def test_enable_sms():
    result = update_notification_settings("user3", {"sms_enabled": True})
    assert result["success"] is True
    assert result["settings"]["sms_enabled"] is True, f"Got {result['settings']}"

def test_set_digest_frequency():
    result = update_notification_settings("user4", {"digest_frequency": "weekly"})
    assert result["success"] is True
    assert result["settings"]["digest_frequency"] == "weekly"
''',
    },
    "relevant_files": ["notification_manager.py", "defaults.py"],
    "irrelevant_files": ["history.py", "templates.py"],
    "bug_location": {"file": "notification_manager.py", "line_start": 13, "line_end": 13},
    "correct_diagnosis_keywords": [
        "merge direction", "wrong merge", "defaults overwrite", "overwriting user",
        "user settings overwritten", "wrong order", "defaults applied over",
    ],
    "correct_fix": {
        "old": "    # BUG: merge direction is wrong — defaults overwrite user settings\n    merged = {**user_settings, **DEFAULT_NOTIFICATION_SETTINGS}",
        "new": "    merged = {**DEFAULT_NOTIFICATION_SETTINGS, **current, **user_settings}",
    },
    "max_steps": 45,
    "task_id": "hard",
    "bug_2": {
        "description": (
            "New bug: reset_notification_settings returns DEFAULT_NOTIFICATION_SETTINGS "
            "directly — not a copy. Callers who modify the returned dict are mutating "
            "the shared defaults object. It should return a copy."
        ),
        "relevant_files": ["notification_manager.py", "defaults.py"],
        "correct_diagnosis_keywords": [
            "dict copy", "shared reference", "mutates defaults", "not a copy",
            "direct reference", "DEFAULT_NOTIFICATION_SETTINGS", "dict()",
        ],
        "correct_fix": {
            "old": '    return {"success": True, "settings": dict(DEFAULT_NOTIFICATION_SETTINGS)}',
            "new": '    return {"success": True, "settings": {**DEFAULT_NOTIFICATION_SETTINGS}}',
        },
        "test_additions": {
            "test_notifications.py": '''
def test_reset_returns_copy_not_reference():
    from notification_manager import reset_notification_settings
    from defaults import DEFAULT_NOTIFICATION_SETTINGS
    result = reset_notification_settings("copy_test_user")
    result["settings"]["email_enabled"] = False
    assert DEFAULT_NOTIFICATION_SETTINGS["email_enabled"] is True, (
        "Mutating returned settings should not affect DEFAULT_NOTIFICATION_SETTINGS"
    )
'''
        },
    },
},

# ── hard_5 ── default config mutation ─────────────────────────────────────
{
    "id": "hard_5",
    "description": (
        "Our application config is behaving strangely after the first user modifies "
        "their settings. Subsequent users who have never changed their settings are "
        "getting the first user's customizations. The defaults appear to be mutated. "
        "No error is raised. Find and fix it."
    ),
    "files": {
        "app_config.py": '''\
"""Application configuration management."""
from config_defaults import APP_DEFAULTS
from user_prefs import get_user_prefs, save_user_prefs
from config_validator import validate_prefs


def get_effective_config(user_id: str) -> dict:
    """
    Get the effective configuration for a user.
    BUG: returns direct reference to APP_DEFAULTS, and mutates it with update().
    """
    user_prefs = get_user_prefs(user_id)
    if not user_prefs:
        return APP_DEFAULTS  # BUG: direct reference — caller can mutate defaults
    # BUG: mutates APP_DEFAULTS in place instead of creating a new dict
    APP_DEFAULTS.update(user_prefs)
    return APP_DEFAULTS


def set_user_config(user_id: str, prefs: dict) -> dict:
    result = validate_prefs(prefs)
    if not result["valid"]:
        return {"success": False, "error": result["error"]}
    save_user_prefs(user_id, prefs)
    return {"success": True}


def get_config_value(user_id: str, key: str, default=None):
    config = get_effective_config(user_id)
    return config.get(key, default)
''',
        "config_defaults.py": '''\
"""Application default configuration."""
APP_DEFAULTS = {
    "theme": "light",
    "language": "en",
    "timezone": "UTC",
    "items_per_page": 20,
    "show_tooltips": True,
    "compact_view": False,
    "notifications": True,
}
''',
        "user_prefs.py": '''\
"""User preference storage."""
_PREFS = {}

def get_user_prefs(user_id: str) -> dict:
    return dict(_PREFS.get(user_id, {}))

def save_user_prefs(user_id: str, prefs: dict) -> None:
    _PREFS[user_id] = dict(prefs)

def delete_user_prefs(user_id: str) -> None:
    _PREFS.pop(user_id, None)
''',
        "config_validator.py": '''\
"""Configuration preference validator."""
VALID_THEMES = {"light", "dark", "system"}
VALID_LANGUAGES = {"en", "fr", "de", "es", "ja", "zh"}

def validate_prefs(prefs: dict) -> dict:
    if "theme" in prefs and prefs["theme"] not in VALID_THEMES:
        return {"valid": False, "error": f"Invalid theme: {prefs['theme']}"}
    if "language" in prefs and prefs["language"] not in VALID_LANGUAGES:
        return {"valid": False, "error": f"Invalid language: {prefs['language']}"}
    if "items_per_page" in prefs:
        v = prefs["items_per_page"]
        if not isinstance(v, int) or v < 5 or v > 100:
            return {"valid": False, "error": "items_per_page must be 5-100"}
    return {"valid": True, "error": None}
''',
        "audit.py": '''\
"""Config audit trail."""
import time
_AUDIT = []

def log_config_access(user_id: str, config: dict) -> None:
    _AUDIT.append({"user_id": user_id, "config": dict(config), "ts": time.time()})

def get_audit_trail(user_id: str = None) -> list:
    if user_id:
        return [a for a in _AUDIT if a["user_id"] == user_id]
    return list(_AUDIT)
''',
        "test_app_config.py": '''\
"""Tests for application config management."""
from app_config import get_effective_config, set_user_config
from config_defaults import APP_DEFAULTS

def test_default_user_gets_defaults():
    config = get_effective_config("new_user_abc")
    assert config["theme"] == "light"
    assert config["language"] == "en"

def test_user_with_prefs_gets_merged():
    set_user_config("user_dark", {"theme": "dark"})
    config = get_effective_config("user_dark")
    assert config["theme"] == "dark"
    assert config["language"] == "en"

def test_defaults_not_mutated():
    """Bug trigger: after user sets dark theme, defaults should still be light."""
    set_user_config("user_custom", {"theme": "dark", "language": "fr"})
    get_effective_config("user_custom")
    config_new = get_effective_config("brand_new_user_xyz")
    assert config_new["theme"] == "light", f"Defaults mutated. Got {config_new['theme']}"
    assert APP_DEFAULTS["theme"] == "light", f"APP_DEFAULTS mutated. Got {APP_DEFAULTS['theme']}"

def test_get_config_value():
    val = get_config_value("some_user", "items_per_page", default=10)
    assert isinstance(val, int)

from app_config import get_config_value
''',
    },
    "relevant_files": ["app_config.py", "config_defaults.py"],
    "irrelevant_files": ["audit.py", "config_validator.py"],
    "bug_location": {"file": "app_config.py", "line_start": 12, "line_end": 17},
    "correct_diagnosis_keywords": [
        "mutates defaults", "default mutation", "APP_DEFAULTS",
        "direct reference", "update in place", "not a copy",
        "dict.update", "shared reference", "mutable",
    ],
    "correct_fix": {
        "old": (
            "    if not user_prefs:\n"
            "        return APP_DEFAULTS  # BUG: direct reference — caller can mutate defaults\n"
            "    # BUG: mutates APP_DEFAULTS in place instead of creating a new dict\n"
            "    APP_DEFAULTS.update(user_prefs)\n"
            "    return APP_DEFAULTS"
        ),
        "new": (
            "    if not user_prefs:\n"
            "        return dict(APP_DEFAULTS)\n"
            "    return {**APP_DEFAULTS, **user_prefs}"
        ),
    },
    "max_steps": 45,
    "task_id": "hard",
    "bug_2": {
        "description": (
            "New issue: validate_prefs is not checking for unknown keys. "
            "Users can submit arbitrary keys like 'admin_mode': True which get "
            "saved and served as part of their config. The validator should "
            "reject unknown configuration keys."
        ),
        "relevant_files": ["config_validator.py", "config_defaults.py"],
        "correct_diagnosis_keywords": [
            "unknown keys", "arbitrary keys", "allowed keys",
            "whitelist", "key validation", "unexpected keys",
        ],
        "correct_fix": {
            "old": "def validate_prefs(prefs: dict) -> dict:\n    if \"theme\" in prefs and prefs[\"theme\"] not in VALID_THEMES:",
            "new": (
                'ALLOWED_KEYS = {"theme", "language", "timezone", "items_per_page", "show_tooltips", "compact_view", "notifications"}\n\n'
                "def validate_prefs(prefs: dict) -> dict:\n"
                "    unknown = set(prefs.keys()) - ALLOWED_KEYS\n"
                "    if unknown:\n"
                "        return {\"valid\": False, \"error\": f\"Unknown config keys: {unknown}\"}\n"
                "    if \"theme\" in prefs and prefs[\"theme\"] not in VALID_THEMES:"
            ),
        },
        "test_additions": {
            "test_app_config.py": '''
def test_unknown_config_key_rejected():
    result = set_user_config("user_hack", {"admin_mode": True, "theme": "dark"})
    assert result["success"] is False, f"Should reject unknown keys, got {result}"
    assert "admin_mode" in result.get("error", "") or "Unknown" in result.get("error", "")
'''
        },
    },
},

# ── hard_6 ── timezone conversion applied twice ────────────────────────────
{
    "id": "hard_6",
    "description": (
        "Our event scheduling system is saving events at the wrong time. "
        "Events end up scheduled 2x their timezone offset away from UTC. "
        "A user in UTC+5:30 scheduling a meeting at 9:00 AM gets it saved "
        "at 20:00 instead of 03:30 UTC. The timezone conversion is being "
        "applied twice. Find and fix it."
    ),
    "files": {
        "scheduler.py": '''\
"""Event scheduling service."""
from time_utils import local_to_utc, utc_to_local, format_time
from event_store import save_event, get_event
from validator import validate_event


def create_event(user_id: str, title: str, local_time: str,
                 timezone_offset: int, duration_minutes: int = 60) -> dict:
    """
    Create a scheduled event.
    Args:
        local_time: time in HH:MM format (local to user)
        timezone_offset: user\'s UTC offset in minutes (e.g. 330 for UTC+5:30)
    """
    validation = validate_event(title, local_time, duration_minutes)
    if not validation["valid"]:
        return {"success": False, "error": validation["error"]}

    # Convert local time to UTC for storage
    # BUG: local_to_utc already converts, but we pass the result through
    # utc_to_local again which applies the offset a second time in reverse
    utc_minutes = local_to_utc(local_time, timezone_offset)
    # This line incorrectly re-applies the timezone conversion
    utc_minutes = utc_to_local(utc_minutes, -timezone_offset)

    event = {
        "user_id": user_id, "title": title,
        "utc_minutes_from_midnight": utc_minutes,
        "local_time": local_time,
        "timezone_offset": timezone_offset,
        "duration_minutes": duration_minutes,
    }
    event_id = save_event(event)
    return {"success": True, "event_id": event_id, "event": event}


def get_event_local_time(event_id: str) -> str:
    """Get an event\'s local time string."""
    event = get_event(event_id)
    if not event:
        return None
    return utc_to_local(event["utc_minutes_from_midnight"], event["timezone_offset"])
''',
        "time_utils.py": '''\
"""Time conversion utilities."""

def local_to_utc(local_time: str, offset_minutes: int) -> int:
    """Convert local time string to UTC minutes from midnight."""
    h, m = map(int, local_time.split(":"))
    local_minutes = h * 60 + m
    utc_minutes = (local_minutes - offset_minutes) % (24 * 60)
    return utc_minutes


def utc_to_local(utc_minutes: int, offset_minutes: int) -> str:
    """Convert UTC minutes from midnight to local time string."""
    local_minutes = (utc_minutes + offset_minutes) % (24 * 60)
    h = local_minutes // 60
    m = local_minutes % 60
    return f"{h:02d}:{m:02d}"


def format_time(minutes_from_midnight: int) -> str:
    h = minutes_from_midnight // 60
    m = minutes_from_midnight % 60
    return f"{h:02d}:{m:02d}"
''',
        "event_store.py": '''\
"""Event persistence."""
import uuid
_EVENTS = {}

def save_event(event: dict) -> str:
    event_id = str(uuid.uuid4())[:8]
    _EVENTS[event_id] = dict(event)
    return event_id

def get_event(event_id: str) -> dict:
    return dict(_EVENTS[event_id]) if event_id in _EVENTS else None

def list_events(user_id: str) -> list:
    return [e for e in _EVENTS.values() if e.get("user_id") == user_id]
''',
        "validator.py": '''\
"""Event validation."""

def validate_event(title: str, local_time: str, duration_minutes: int) -> dict:
    if not title or not title.strip():
        return {"valid": False, "error": "Title is required"}
    try:
        h, m = map(int, local_time.split(":"))
        if not (0 <= h <= 23 and 0 <= m <= 59):
            raise ValueError()
    except (ValueError, AttributeError):
        return {"valid": False, "error": "Invalid time format (use HH:MM)"}
    if duration_minutes < 1 or duration_minutes > 480:
        return {"valid": False, "error": "Duration must be 1-480 minutes"}
    return {"valid": True, "error": None}
''',
        "reminders.py": '''\
"""Event reminder system."""
_REMINDERS = {}

def schedule_reminder(event_id: str, minutes_before: int = 15) -> None:
    _REMINDERS[event_id] = {"minutes_before": minutes_before, "sent": False}

def get_pending_reminders() -> list:
    return [{"event_id": eid, **r} for eid, r in _REMINDERS.items() if not r["sent"]]

def mark_reminder_sent(event_id: str) -> None:
    if event_id in _REMINDERS:
        _REMINDERS[event_id]["sent"] = True
''',
        "test_scheduler.py": '''\
"""Tests for the event scheduler."""
from scheduler import create_event, get_event_local_time

def test_utc_user_scheduling():
    """UTC user: local == UTC, no conversion needed."""
    result = create_event("u1", "Meeting", "09:00", timezone_offset=0)
    assert result["success"] is True
    assert result["event"]["utc_minutes_from_midnight"] == 9 * 60

def test_positive_offset_user():
    """UTC+5:30 user at 9:00 AM local -> 03:30 UTC (210 min)."""
    result = create_event("u2", "Standup", "09:00", timezone_offset=330)
    assert result["success"] is True
    utc_mins = result["event"]["utc_minutes_from_midnight"]
    assert utc_mins == 210, f"Expected 210 (03:30 UTC), got {utc_mins}"

def test_negative_offset_user():
    """UTC-5 user at 9:00 AM local -> 14:00 UTC (840 min)."""
    result = create_event("u3", "Lunch", "09:00", timezone_offset=-300)
    assert result["success"] is True
    utc_mins = result["event"]["utc_minutes_from_midnight"]
    assert utc_mins == 840, f"Expected 840 (14:00 UTC), got {utc_mins}"

def test_roundtrip_conversion():
    """Local -> UTC -> local should return original time."""
    result = create_event("u4", "Call", "14:30", timezone_offset=60)
    assert result["success"] is True
    local_time = get_event_local_time(result["event_id"])
    assert local_time == "14:30", f"Round-trip failed. Got {local_time}"
''',
    },
    "relevant_files": ["scheduler.py", "time_utils.py"],
    "irrelevant_files": ["reminders.py", "validator.py"],
    "bug_location": {"file": "scheduler.py", "line_start": 22, "line_end": 25},
    "correct_diagnosis_keywords": [
        "applied twice", "double conversion", "twice", "re-applies",
        "utc_to_local again", "second conversion", "offset applied twice",
    ],
    "correct_fix": {
        "old": (
            "    # Convert local time to UTC for storage\n"
            "    # BUG: local_to_utc already converts, but we pass the result through\n"
            "    # utc_to_local again which applies the offset a second time in reverse\n"
            "    utc_minutes = local_to_utc(local_time, timezone_offset)\n"
            "    # This line incorrectly re-applies the timezone conversion\n"
            "    utc_minutes = utc_to_local(utc_minutes, -timezone_offset)"
        ),
        "new": "    utc_minutes = local_to_utc(local_time, timezone_offset)",
    },
    "max_steps": 45,
    "task_id": "hard",
    "bug_2": {
        "description": (
            "New bug: get_event_local_time returns a time string but some callers "
            "expect an integer (minutes from midnight) for downstream calculations. "
            "The function signature says it returns a str but the callers need int. "
            "Change it to return numeric minutes from midnight instead."
        ),
        "relevant_files": ["scheduler.py", "time_utils.py"],
        "correct_diagnosis_keywords": [
            "return type", "string instead of int", "minutes not string",
            "utc_to_local returns string", "numeric", "integer minutes",
        ],
        "correct_fix": {
            "old": '    return utc_to_local(event["utc_minutes_from_midnight"], event["timezone_offset"])',
            "new": (
                '    offset = event["timezone_offset"]\n'
                '    local_minutes = (event["utc_minutes_from_midnight"] + offset) % (24 * 60)\n'
                '    return local_minutes'
            ),
        },
        "test_additions": {
            "test_scheduler.py": '''
def test_get_event_local_time_returns_minutes():
    result = create_event("u5", "Morning", "08:00", timezone_offset=60)
    assert result["success"] is True
    local = get_event_local_time(result["event_id"])
    assert isinstance(local, int), f"Expected int (minutes), got {type(local)}: {local}"
    assert local == 9 * 60, f"Expected 540 (09:00), got {local}"
'''
        },
    },
},

# ── hard_7 ── config key collision between two modules ─────────────────────
{
    "id": "hard_7",
    "description": (
        "Two recently-launched features are silently interfering with each other. "
        "When both the A/B testing module and the personalization module are active "
        "for the same user, enabling one feature disables the other. "
        "No error is raised. The settings appear to save successfully. "
        "Find the root cause — it involves how the two modules share configuration."
    ),
    "files": {
        "feature_config.py": '''\
"""Feature configuration manager — merges configs from multiple modules."""
from ab_testing import get_ab_config
from personalization import get_personalization_config
from config_store import load_user_config, save_user_config
from merger import merge_configs


def get_combined_config(user_id: str) -> dict:
    """Get the combined feature configuration for a user."""
    user_config = load_user_config(user_id) or {}
    ab_config = get_ab_config(user_id, user_config)
    personalization_config = get_personalization_config(user_id, user_config)
    # BUG: both modules use the key "enabled" at the top level
    # The second merge silently overwrites the first module\'s "enabled" flag
    combined = merge_configs(ab_config, personalization_config)
    return combined


def update_feature_config(user_id: str, module: str, settings: dict) -> dict:
    current = load_user_config(user_id) or {}
    current[module] = settings
    save_user_config(user_id, current)
    return {"success": True}
''',
        "ab_testing.py": '''\
"""A/B testing configuration module."""
DEFAULT_AB_CONFIG = {
    "enabled": False,       # BUG: same key name as personalization
    "variant": "control",
    "experiment_id": None,
    "sample_rate": 0.1,
}

def get_ab_config(user_id: str, user_config: dict) -> dict:
    ab_overrides = user_config.get("ab_testing", {})
    return {**DEFAULT_AB_CONFIG, **ab_overrides}

def assign_variant(user_id: str) -> str:
    return "control" if hash(user_id) % 2 == 0 else "treatment"
''',
        "personalization.py": '''\
"""Personalization configuration module."""
DEFAULT_PERSONALIZATION_CONFIG = {
    "enabled": False,       # BUG: same key name as ab_testing
    "algorithm": "collaborative",
    "max_recommendations": 10,
    "recency_weight": 0.7,
}

def get_personalization_config(user_id: str, user_config: dict) -> dict:
    p_overrides = user_config.get("personalization", {})
    return {**DEFAULT_PERSONALIZATION_CONFIG, **p_overrides}

def get_recommendations(user_id: str, config: dict) -> list:
    if not config.get("enabled", False):
        return []
    return [f"item_{i}" for i in range(config.get("max_recommendations", 5))]
''',
        "merger.py": '''\
"""Config merging utilities."""

def merge_configs(*configs: dict) -> dict:
    """Merge multiple config dicts. Later configs override earlier ones."""
    result = {}
    for config in configs:
        result.update(config)
    return result

def deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result
''',
        "config_store.py": '''\
"""User config storage."""
_CONFIGS = {}

def load_user_config(user_id: str) -> dict:
    return dict(_CONFIGS.get(user_id, {}))

def save_user_config(user_id: str, config: dict) -> None:
    _CONFIGS[user_id] = dict(config)
''',
        "analytics.py": '''\
"""Feature analytics."""
_EVENTS = []

def track_feature_use(user_id: str, feature: str, action: str) -> None:
    _EVENTS.append({"user_id": user_id, "feature": feature, "action": action})

def get_feature_stats(feature: str) -> dict:
    return {"total": len([e for e in _EVENTS if e["feature"] == feature])}
''',
        "test_feature_config.py": '''\
"""Tests for feature configuration."""
from feature_config import get_combined_config, update_feature_config

def test_enable_ab_testing_only():
    update_feature_config("user1", "ab_testing", {"enabled": True, "variant": "treatment"})
    config = get_combined_config("user1")
    ab_enabled = config.get("ab_testing", {}).get("enabled", False)
    assert ab_enabled is True, f"AB testing should be enabled. Got config: {config}"

def test_enable_personalization_only():
    update_feature_config("user2", "personalization", {"enabled": True})
    config = get_combined_config("user2")
    p_enabled = config.get("personalization", {}).get("enabled", False)
    assert p_enabled is True, f"Personalization should be enabled. Got config: {config}"

def test_both_features_enabled_independently():
    update_feature_config("user3", "ab_testing", {"enabled": True})
    update_feature_config("user3", "personalization", {"enabled": True})
    config = get_combined_config("user3")
    ab_enabled = config.get("ab_testing", {}).get("enabled", False)
    p_enabled = config.get("personalization", {}).get("enabled", False)
    assert ab_enabled is True, f"AB testing should be enabled. Got: {config}"
    assert p_enabled is True, f"Personalization should be enabled. Got: {config}"
''',
    },
    "relevant_files": ["feature_config.py", "ab_testing.py", "personalization.py"],
    "irrelevant_files": ["analytics.py", "config_store.py"],
    "bug_location": {"file": "feature_config.py", "line_start": 11, "line_end": 14},
    "correct_diagnosis_keywords": [
        "key collision", "same key", "enabled key", "namespace",
        "overwrite", "conflict", "both use enabled", "flat merge",
    ],
    "correct_fix": {
        "old": (
            '    # BUG: both modules use the key "enabled" at the top level\n'
            '    # The second merge silently overwrites the first module\'s "enabled" flag\n'
            "    combined = merge_configs(ab_config, personalization_config)\n"
            "    return combined"
        ),
        "new": (
            "    # Namespace each module's config to avoid key collisions\n"
            "    combined = {\n"
            '        "ab_testing": ab_config,\n'
            '        "personalization": personalization_config,\n'
            "    }\n"
            "    return combined"
        ),
    },
    "max_steps": 45,
    "task_id": "hard",
    "bug_2": {
        "description": (
            "New issue: get_recommendations in personalization.py returns an empty list "
            "when called with the combined config because it looks for a top-level 'enabled' "
            "key but the config is now namespaced under 'personalization'. The function "
            "needs to be updated to read from the correct key path."
        ),
        "relevant_files": ["personalization.py", "feature_config.py"],
        "correct_diagnosis_keywords": [
            "nested key", "config path", "personalization key",
            "enabled path", "wrong key", "config structure changed",
        ],
        "correct_fix": {
            "old": (
                "def get_recommendations(user_id: str, config: dict) -> list:\n"
                "    if not config.get(\"enabled\", False):\n"
                "        return []\n"
                "    return [f\"item_{i}\" for i in range(config.get(\"max_recommendations\", 5))]"
            ),
            "new": (
                "def get_recommendations(user_id: str, config: dict) -> list:\n"
                "    p_config = config.get(\"personalization\", config)\n"
                "    if not p_config.get(\"enabled\", False):\n"
                "        return []\n"
                "    return [f\"item_{i}\" for i in range(p_config.get(\"max_recommendations\", 5))]"
            ),
        },
        "test_additions": {
            "test_feature_config.py": '''
def test_recommendations_work_with_combined_config():
    from personalization import get_recommendations
    update_feature_config("user_rec", "personalization", {"enabled": True, "max_recommendations": 3})
    config = get_combined_config("user_rec")
    recs = get_recommendations("user_rec", config)
    assert len(recs) == 3, f"Expected 3 recommendations, got {recs}"
'''
        },
    },
},

# ── hard_8 ── validator silently dropping valid nested keys ────────────────
{
    "id": "hard_8",
    "description": (
        "Users who set both their notification frequency AND their quiet hours "
        "are losing their quiet hours settings after saving. The quiet hours "
        "appear to save successfully but are gone the next time the config is loaded. "
        "No error is raised. Find the root cause and fix it."
    ),
    "files": {
        "preferences_manager.py": '''\
"""User preferences manager."""
from preferences_validator import validate_preferences
from preferences_store import save_preferences, load_preferences
from preferences_defaults import DEFAULT_PREFERENCES
from change_tracker import track_change


def save_user_preferences(user_id: str, prefs: dict) -> dict:
    validated = validate_preferences(prefs)
    if not validated["valid"]:
        return {"success": False, "error": validated["error"]}
    current = load_preferences(user_id) or {}
    merged = {**DEFAULT_PREFERENCES, **current, **validated["data"]}
    save_preferences(user_id, merged)
    track_change(user_id, current, merged)
    return {"success": True, "preferences": merged}


def get_user_preferences(user_id: str) -> dict:
    stored = load_preferences(user_id)
    if stored is None:
        return dict(DEFAULT_PREFERENCES)
    return {**DEFAULT_PREFERENCES, **stored}
''',
        "preferences_validator.py": '''\
"""Preferences validation."""
VALID_FREQUENCIES = {"realtime", "hourly", "daily", "weekly", "never"}
VALID_CHANNELS = {"email", "push", "sms", "in_app"}


def validate_preferences(prefs: dict) -> dict:
    """
    Validate user preferences.
    BUG: when processing \'notifications\' dict, the validator rebuilds
    the dict but only includes known top-level notification keys.
    \'quiet_hours\' is a nested dict inside \'notifications\' but is not
    included in the rebuilt dict — it gets silently dropped.
    """
    if not isinstance(prefs, dict):
        return {"valid": False, "error": "Preferences must be a dict", "data": {}}

    cleaned = {}
    for key, value in prefs.items():
        if key == "notifications" and isinstance(value, dict):
            notifications = {}
            freq = value.get("frequency")
            if freq is not None:
                if freq not in VALID_FREQUENCIES:
                    return {"valid": False, "error": f"Invalid frequency: {freq}", "data": {}}
                notifications["frequency"] = freq
            channels = value.get("channels")
            if channels is not None:
                invalid = set(channels) - VALID_CHANNELS
                if invalid:
                    return {"valid": False, "error": f"Invalid channels: {invalid}", "data": {}}
                notifications["channels"] = channels
            # BUG: quiet_hours is never added to notifications dict
            # it exists in the input but is silently dropped here
            cleaned["notifications"] = notifications
        elif key == "language":
            if not isinstance(value, str) or len(value) != 2:
                return {"valid": False, "error": "Language must be a 2-letter code", "data": {}}
            cleaned["language"] = value
        elif key == "timezone":
            cleaned["timezone"] = value
        elif key == "theme":
            if value not in {"light", "dark", "system"}:
                return {"valid": False, "error": f"Invalid theme: {value}", "data": {}}
            cleaned["theme"] = value
        else:
            cleaned[key] = value

    return {"valid": True, "error": None, "data": cleaned}
''',
        "preferences_defaults.py": '''\
"""Default user preferences."""
DEFAULT_PREFERENCES = {
    "language": "en",
    "timezone": "UTC",
    "theme": "light",
    "notifications": {
        "frequency": "daily",
        "channels": ["email", "push"],
        "quiet_hours": {"start": 22, "end": 8},
    },
}
''',
        "preferences_store.py": '''\
"""Preferences persistence."""
import copy
_STORE = {}

def save_preferences(user_id: str, prefs: dict) -> None:
    _STORE[user_id] = copy.deepcopy(prefs)

def load_preferences(user_id: str) -> dict:
    p = _STORE.get(user_id)
    return copy.deepcopy(p) if p else None

def delete_preferences(user_id: str) -> None:
    _STORE.pop(user_id, None)
''',
        "change_tracker.py": '''\
"""Preference change tracking."""
import time
_CHANGES = []

def track_change(user_id: str, before: dict, after: dict) -> None:
    _CHANGES.append({"user_id": user_id, "before": before, "after": after, "ts": time.time()})

def get_changes(user_id: str) -> list:
    return [c for c in _CHANGES if c["user_id"] == user_id]
''',
        "sync_service.py": '''\
"""Preferences sync service."""
import time
_SYNC_LOG = []

def sync_preferences(user_id: str, preferences: dict) -> dict:
    _SYNC_LOG.append({"user_id": user_id, "preferences": preferences, "ts": time.time()})
    return {"synced": True, "timestamp": time.time()}

def get_sync_status(user_id: str) -> dict:
    entries = [e for e in _SYNC_LOG if e["user_id"] == user_id]
    if not entries:
        return {"synced": False}
    return {"synced": True, "last_sync": entries[-1]["ts"]}
''',
        "test_preferences.py": '''\
"""Tests for the preferences manager."""
from preferences_manager import save_user_preferences, get_user_preferences

def test_save_frequency():
    result = save_user_preferences("u1", {"notifications": {"frequency": "weekly"}})
    assert result["success"] is True
    assert result["preferences"]["notifications"]["frequency"] == "weekly"

def test_save_channels():
    result = save_user_preferences("u2", {"notifications": {"channels": ["email"]}})
    assert result["success"] is True
    assert result["preferences"]["notifications"]["channels"] == ["email"]

def test_quiet_hours_preserved_with_frequency():
    """Bug trigger: quiet_hours should be preserved when frequency is also set."""
    result = save_user_preferences("u3", {
        "notifications": {
            "frequency": "daily",
            "quiet_hours": {"start": 23, "end": 7},
        }
    })
    assert result["success"] is True
    notif = result["preferences"]["notifications"]
    assert "quiet_hours" in notif, f"quiet_hours was dropped. Got: {notif}"
    assert notif["quiet_hours"]["start"] == 23, f"Got: {notif}"

def test_invalid_frequency_rejected():
    result = save_user_preferences("u4", {"notifications": {"frequency": "minutely"}})
    assert result["success"] is False

def test_defaults_applied():
    prefs = get_user_preferences("brand_new_user_pref")
    assert "notifications" in prefs
    assert prefs["language"] == "en"
''',
    },
    "relevant_files": ["preferences_validator.py", "preferences_manager.py"],
    "irrelevant_files": ["sync_service.py", "change_tracker.py"],
    "bug_location": {"file": "preferences_validator.py", "line_start": 30, "line_end": 35},
    "correct_diagnosis_keywords": [
        "quiet_hours", "silently dropped", "not included", "missing key",
        "validator drops", "never added", "notifications dict",
    ],
    "correct_fix": {
        "old": (
            "            # BUG: quiet_hours is never added to notifications dict\n"
            "            # it exists in the input but is silently dropped here\n"
            '            cleaned["notifications"] = notifications'
        ),
        "new": (
            "            # Preserve any extra nested keys not explicitly validated\n"
            "            for notif_key, notif_val in value.items():\n"
            '                if notif_key not in ("frequency", "channels"):\n'
            "                    notifications[notif_key] = notif_val\n"
            '            cleaned["notifications"] = notifications'
        ),
    },
    "max_steps": 45,
    "task_id": "hard",
    "bug_2": {
        "description": (
            "New bug: get_user_preferences merges stored preferences on top of defaults "
            "with a shallow merge. This means if a user has only set notifications.frequency, "
            "the stored notifications dict replaces the entire default notifications dict — "
            "losing the default quiet_hours and channels. The merge should be deep, not shallow."
        ),
        "relevant_files": ["preferences_manager.py", "preferences_defaults.py"],
        "correct_diagnosis_keywords": [
            "shallow merge", "deep merge", "nested merge",
            "notifications dict replaced", "lose defaults",
            "default channels", "default quiet_hours",
        ],
        "correct_fix": {
            "old": (
                "def get_user_preferences(user_id: str) -> dict:\n"
                "    stored = load_preferences(user_id)\n"
                "    if stored is None:\n"
                "        return dict(DEFAULT_PREFERENCES)\n"
                "    return {**DEFAULT_PREFERENCES, **stored}"
            ),
            "new": (
                "def get_user_preferences(user_id: str) -> dict:\n"
                "    stored = load_preferences(user_id)\n"
                "    if stored is None:\n"
                "        return dict(DEFAULT_PREFERENCES)\n"
                "    result = {}\n"
                "    for key, default_val in DEFAULT_PREFERENCES.items():\n"
                "        if key in stored and isinstance(default_val, dict) and isinstance(stored[key], dict):\n"
                "            result[key] = {**default_val, **stored[key]}\n"
                "        else:\n"
                "            result[key] = stored.get(key, default_val)\n"
                "    return result"
            ),
        },
        "test_additions": {
            "test_preferences.py": '''
def test_partial_notifications_preserves_defaults():
    """Setting only frequency should not lose default channels and quiet_hours."""
    save_user_preferences("u5", {"notifications": {"frequency": "hourly"}})
    prefs = get_user_preferences("u5")
    notif = prefs.get("notifications", {})
    assert "channels" in notif, f"Default channels should be preserved. Got: {notif}"
    assert "quiet_hours" in notif, f"Default quiet_hours should be preserved. Got: {notif}"
    assert notif["frequency"] == "hourly"
'''
        },
    },
},

]  # end HARD_SCENARIOS


# ---------------------------------------------------------------------------
# Pool registry and selector
# ---------------------------------------------------------------------------

SCENARIO_POOLS = {
    "easy":   EASY_SCENARIOS,
    "medium": MEDIUM_SCENARIOS,
    "hard":   HARD_SCENARIOS,
}


def get_scenario(task_id: str, index: Optional[int] = None) -> dict:
    """
    Get a scenario for a given difficulty level.

    For easy tasks with no pinned index, uses the procedural generator
    50% of the time — giving effectively unlimited scenario variety.
    For medium and hard, always draws from the fixed pool.
    """
    if task_id not in SCENARIO_POOLS:
        raise ValueError(f"task_id must be one of {list(SCENARIO_POOLS.keys())}")

    pool = SCENARIO_POOLS[task_id]

    if index is not None:
        return pool[index % len(pool)]

    # For easy tasks, use procedural generator 50% of the time
    if task_id == "easy" and random.random() < 0.5:
        try:
            from scenario_generator import generate_easy_scenario
            return generate_easy_scenario()
        except ImportError:
            pass

    return random.choice(pool)


def get_pool_size(task_id: str) -> int:
    """Return the number of scenarios available for a difficulty level."""
    return len(SCENARIO_POOLS[task_id])


if __name__ == "__main__":
    for task_id in ["easy", "medium", "hard"]:
        print(f"{task_id}: {get_pool_size(task_id)} scenarios")
        s = get_scenario(task_id)
        print(f"  sample -> {s['id']}: {s['description'][:60]}...")