"""
Django Pill System Logging - Verification Script
This script verifies that the logging system is correctly configured and working.
"""

import os
import sys
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Detection_and_Analysis_of_Pill.settings')
django.setup()

# Now import logging after Django setup
import logging
from Users.logging_config import (
    log_image_upload,
    log_valid_file_type,
    log_invalid_file_type,
    log_corrupted_image,
    log_preprocessing_started,
    log_preprocessing_completed,
    log_prediction_started,
    log_prediction_completed,
    log_low_confidence_prediction,
    log_system_error,
    log_csv_write,
    log_safety_validation,
    get_pill_logger
)

def test_logging_system():
    """Test all logging functions to verify they work correctly."""
    
    print("=" * 80)
    print("PILL SYSTEM LOGGING VERIFICATION TEST")
    print("=" * 80)
    print()
    
    # Get the pill system logger
    pill_logger = get_pill_logger()
    print(f"✓ Pill system logger initialized: {pill_logger.name}")
    print(f"✓ Log file location: {os.path.join(os.getcwd(), 'pill_system.log')}")
    print()
    
    print("Testing logging functions...")
    print("-" * 80)
    
    try:
        # Test 1: Image upload
        print("1. Testing image upload logging...")
        log_image_upload('test_pill.jpg', 125000, 'test_user')
        print("   ✓ log_image_upload() executed")
        
        # Test 2: Valid file type
        print("2. Testing valid file type logging...")
        log_valid_file_type('test_pill.jpg', 'jpg')
        print("   ✓ log_valid_file_type() executed")
        
        # Test 3: Invalid file type
        print("3. Testing invalid file type logging...")
        log_invalid_file_type('document.pdf', 'pdf', ['jpg', 'png', 'jpeg', 'gif'])
        print("   ✓ log_invalid_file_type() executed")
        
        # Test 4: Corrupted image
        print("4. Testing corrupted image logging...")
        log_corrupted_image('corrupted.jpg', 'cannot identify image file')
        print("   ✓ log_corrupted_image() executed")
        
        # Test 5: Preprocessing started
        print("5. Testing preprocessing started logging...")
        log_preprocessing_started('test_pill.jpg')
        print("   ✓ log_preprocessing_started() executed")
        
        # Test 6: Preprocessing completed
        print("6. Testing preprocessing completed logging...")
        log_preprocessing_completed('test_pill.jpg', ['resize', 'normalization', 'augmentation'])
        print("   ✓ log_preprocessing_completed() executed")
        
        # Test 7: Prediction started
        print("7. Testing prediction started logging...")
        log_prediction_started('test_pill.jpg', 'MobileNetV3Large')
        print("   ✓ log_prediction_started() executed")
        
        # Test 8: Prediction completed
        print("8. Testing prediction completed logging...")
        log_prediction_completed('Paracetamol', '92.45%', {
            'top_3': [('Paracetamol', 0.9245), ('Ibuprofen', 0.0512), ('Aspirin', 0.0243)],
            'processing_time': '245ms'
        })
        print("   ✓ log_prediction_completed() executed")
        
        # Test 9: Low confidence prediction
        print("9. Testing low confidence prediction logging...")
        log_low_confidence_prediction('Unknown Pill', '35.67%', '70%')
        print("   ✓ log_low_confidence_prediction() executed")
        
        # Test 10: Safety validation
        print("10. Testing safety validation logging...")
        log_safety_validation(True, 'Paracetamol', {'check_1': 'passed', 'check_2': 'passed'})
        print("    ✓ log_safety_validation() executed")
        
        # Test 11: CSV write
        print("11. Testing CSV write logging...")
        log_csv_write('pill_predictions.csv', {'pill_name': 'Paracetamol'})
        print("    ✓ log_csv_write() executed")
        
        # Test 12: System error
        print("12. Testing system error logging...")
        try:
            raise ValueError("Test error for logging verification")
        except Exception as e:
            log_system_error(str(e), 'ValueError', 'Test stack trace')
        print("    ✓ log_system_error() executed")
        
        print()
        print("=" * 80)
        print("✓ ALL LOGGING TESTS PASSED")
        print("=" * 80)
        print()
        print("Checking log file...")
        print("-" * 80)
        
        # Check if log file was created
        log_file = os.path.join(os.getcwd(), 'pill_system.log')
        if os.path.exists(log_file):
            print(f"✓ Log file created: {log_file}")
            file_size = os.path.getsize(log_file)
            print(f"✓ Log file size: {file_size} bytes")
            print()
            print("Last 20 lines of log file:")
            print("-" * 80)
            
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines[-20:]:
                    print(line.rstrip())
            
            print()
            print("=" * 80)
            print("✓ LOGGING SYSTEM IS FULLY OPERATIONAL")
            print("=" * 80)
            print()
            print("Next steps:")
            print("1. Run Django: python manage.py runserver")
            print("2. Upload a pill image through the web interface")
            print("3. Check pill_system.log for detailed logging events")
            
        else:
            print(f"✗ Log file not found at: {log_file}")
            print("Checking Django logger configuration...")
            
            # Try to access the logger directly
            logger = logging.getLogger('pill_system')
            print(f"  Logger name: {logger.name}")
            print(f"  Logger level: {logger.level}")
            print(f"  Logger handlers: {logger.handlers}")
            
    except Exception as e:
        print(f"✗ ERROR during logging test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == '__main__':
    success = test_logging_system()
    sys.exit(0 if success else 1)
