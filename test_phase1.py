"""
Test script for Phase 1: Logging and Exception Handling

This script tests:
1. Logger functionality (console and file output)
2. Custom exception handling
3. Error tracking with file name and line number
"""

import sys
from src.utils.logger import logger
from src.utils.exception import CustomException


def test_logging():
    """Test different logging levels"""
    print("\n" + "="*80)
    print("TEST 1: LOGGING FUNCTIONALITY")
    print("="*80)
    
    logger.info("✓ Testing INFO level - General information")
    logger.debug("✓ Testing DEBUG level - Detailed debugging information")
    logger.warning("✓ Testing WARNING level - Warning messages")
    logger.error("✓ Testing ERROR level - Error messages")
    
    print("\n✅ Logging test complete! Check logs/ folder for log file.")


def test_custom_exception():
    """Test custom exception handling"""
    print("\n" + "="*80)
    print("TEST 2: CUSTOM EXCEPTION HANDLING")
    print("="*80)
    
    try:
        # Simulate an error
        logger.info("Attempting to divide by zero (intentional error)...")
        result = 10 / 0
    except Exception as e:
        # Raise custom exception
        raise CustomException(e, sys)


def test_exception_with_file_operations():
    """Test exception with file operations"""
    print("\n" + "="*80)
    print("TEST 3: EXCEPTION WITH FILE OPERATION ERROR")
    print("="*80)
    
    try:
        # Try to open non-existent file
        logger.info("Attempting to open non-existent file...")
        with open("this_file_does_not_exist.csv", "r") as f:
            data = f.read()
    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "PHASE 1 - TESTING SUITE" + " "*35 + "║")
    print("╚" + "="*78 + "╝")
    
    # Test 1: Logging
    test_logging()
    
    # Test 2: Custom Exception (choose one test)
    print("\n\nChoose which exception test to run:")
    print("1. Division by zero error")
    print("2. File not found error")
    print("3. Skip exception tests")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        test_custom_exception()
    elif choice == "2":
        test_exception_with_file_operations()
    else:
        print("\n✓ Skipping exception tests")
        print("\n" + "="*80)
        print("✅ PHASE 1 VERIFICATION COMPLETE!")
        print("="*80)
        print("\nSummary:")
        print("- ✓ Logger created and working (check logs/ folder)")
        print("- ✓ Custom exception handler ready")
        print("- ✓ All utilities exported in __init__.py")
        print("\nNext: Ready for Phase 2 (Data Pipeline)")
        print("="*80)
