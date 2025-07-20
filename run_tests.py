#!/usr/bin/env python3
"""
Test runner for AI Scientist v2
"""
import sys
import unittest
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def run_tests():
    """Run all tests and return success status"""
    # Discover and run tests
    loader = unittest.TestLoader()
    start_dir = project_root / 'tests'
    
    try:
        suite = loader.discover(start_dir, pattern='test_*.py')
        
        runner = unittest.TextTestRunner(verbosity=2, buffer=True)
        result = runner.run(suite)
        
        # Print summary
        print(f"\n{'='*50}")
        print(f"Test Summary:")
        print(f"Tests run: {result.testsRun}")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
        print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
        
        if result.failures:
            print(f"\nFailures:")
            for test, traceback in result.failures:
                print(f"  - {test}: {traceback.split(chr(10))[-2] if chr(10) in traceback else traceback}")
        
        if result.errors:
            print(f"\nErrors:")
            for test, traceback in result.errors:
                print(f"  - {test}: {traceback.split(chr(10))[-2] if chr(10) in traceback else traceback}")
        
        success = len(result.failures) == 0 and len(result.errors) == 0
        print(f"\nOverall result: {'PASSED' if success else 'FAILED'}")
        
        return success
        
    except Exception as e:
        print(f"Error running tests: {e}")
        return False


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)