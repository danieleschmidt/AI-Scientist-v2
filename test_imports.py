#!/usr/bin/env python3
"""Test import availability"""

def test_import(module_name):
    try:
        __import__(module_name)
        print(f"✅ {module_name} available")
        return True
    except ImportError as e:
        print(f"❌ {module_name} not available: {e}")
        return False

# Test required modules
modules = [
    'rich',
    'rich.console',
    'rich.logging',
    'yaml',
    'argparse',
    'asyncio',
    'logging',
    'os',
    'sys',
    'pathlib'
]

print("Testing import availability:")
available = {}
for module in modules:
    available[module] = test_import(module)

print(f"\nSummary: {sum(available.values())}/{len(available)} modules available")