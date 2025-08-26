#!/usr/bin/env python3
"""
SurfCat Test Runner - Simplified test execution script

This script provides convenient ways to run various SurfCat tests.
"""

import sys
import os
import subprocess
from pathlib import Path

def run_command(cmd, description):
    """Run command and display results"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Success")
            if result.stdout:
                print("Output:")
                print(result.stdout)
        else:
            print("❌ Failed")
            if result.stderr:
                print("Error:")
                print(result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Execution failed: {e}")
        return False

def main():
    """Main function"""
    print("SurfCat Test Runner")
    print("="*60)
    
    # Ensure we're in the correct directory
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    if len(sys.argv) == 1:
        print("\nAvailable test options:")
        print("  python run_tests.py basic      - Run basic functionality tests")
        print("  python run_tests.py example    - Run quickstart example")
        print("  python run_tests.py all        - Run all available tests")
        return
    
    test_type = sys.argv[1].lower()
    success = True
    
    if test_type == "basic":
        success = run_command("python tests/test_surfcat.py", "Basic functionality tests")
    
    elif test_type == "example":
        success = run_command("python examples/quickstart_example.py", "Quickstart example")
    
    elif test_type == "all":
        tests = [
            ("python tests/test_surfcat.py", "Basic functionality tests"),
            ("python examples/quickstart_example.py", "Quickstart example"), 
        ]
        
        for cmd, desc in tests:
            if not run_command(cmd, desc):
                success = False
                break
    
    else:
        print(f"❌ Unknown test type: {test_type}")
        success = False
    
    print(f"\n{'='*60}")
    if success:
        print("🎉 All tests completed!")
    else:
        print("⚠️  Some tests failed, please check output above")
    print('='*60)

if __name__ == "__main__":
    main()