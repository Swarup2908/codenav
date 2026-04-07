"""
Debug script - paste and run from inside codenav/ folder
with: python debug_codenav.py
"""
import sys, os, types

# Simulate what _build_module_registry and _run_test_file do
# for the medium task, WITHOUT needing the full environment

FILES = {
    "api_handler.py": '''"""Login API handler."""
from utils import sanitize_username, hash_password

USER_DB = {
    "alice": "5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8",
    "bob":   "6b3a55e0261b0304143f805a24924d0c1c44524821305f31d9277843b8a10f4e",
}

def handle_login(username: str, password: str) -> dict:
    clean_username = sanitize_username(username)
    if clean_username is None:
        return {"success": False, "message": "Invalid username"}
    if clean_username.lower() not in USER_DB:
        return {"success": False, "message": "User not found"}
    stored_hash = USER_DB[clean_username.lower()]
    provided_hash = hash_password(password)
    if stored_hash == provided_hash:
        return {"success": True, "message": "Login successful"}
    return {"success": False, "message": "Invalid password"}
''',
    "utils.py": '''"""Utility functions."""
import hashlib, re

def sanitize_username(username: str) -> str:
    username = username.strip()
    if not re.match(r"^[a-zA-Z0-9_]+$", username):
        return None
    return username

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()
''',
    "test_api.py": '''"""Tests for the login API handler."""
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
'''
}

def build_module_registry(files):
    module_objects = {}
    original_modules = {}

    # Step 1 - register empty shells first
    for f in files:
        if not f.startswith("test_"):
            name = f.replace(".py", "")
            mod = types.ModuleType(name)
            mod.__file__ = f
            module_objects[name] = mod
            original_modules[name] = sys.modules.get(name)
            sys.modules[name] = mod

    # Step 2 - exec into each shell (cross-imports now work)
    for f, src in files.items():
        if not f.startswith("test_"):
            name = f.replace(".py", "")
            try:
                exec(compile(src, f, "exec"), module_objects[name].__dict__)
                print(f"  exec OK: {f}")
            except Exception as e:
                print(f"  exec FAIL: {f}: {e}")

    return module_objects, original_modules

def restore_modules(original_modules):
    for name, original in original_modules.items():
        if original is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original

def run_test_file(fname, content, files):
    module_objects, original_modules = build_module_registry(files)
    namespace = {}
    try:
        exec(compile(content, fname, "exec"), namespace)
        print(f"\n  test file exec OK, namespace keys: {[k for k in namespace if not k.startswith('_')]}")
    except Exception as e:
        restore_modules(original_modules)
        print(f"  test file exec FAIL: {e}")
        return
    finally:
        restore_modules(original_modules)

    test_funcs = {k: v for k, v in namespace.items()
                  if k.startswith("test_") and callable(v)}
    print(f"  found test functions: {list(test_funcs.keys())}")

    for name, func in test_funcs.items():
        try:
            func()
            print(f"    PASS {name}")
        except Exception as e:
            print(f"    FAIL {name}: {e}")

print("=== Building module registry ===")
run_test_file("test_api.py", FILES["test_api.py"], FILES)