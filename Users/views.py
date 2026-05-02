import os
import logging
import csv
from datetime import datetime
from django.conf import settings
from django.shortcuts import render
from django.contrib import messages
from django.core.files.storage import FileSystemStorage
from django.http import JsonResponse

from Users.models import UserRegisteredTable, PillInfo
from Users.utility.requirement import main, predictions
from Users.utility.medical_safety_module import MedicalSafetyValidator
from Users.speech_recognition_handler import SpeechRecognitionHandler, validate_speech_recognition_setup
from Users.logging_config import (
    log_image_upload, log_valid_file_type, log_invalid_file_type,
    log_corrupted_image, log_preprocessing_started, log_preprocessing_completed,
    log_prediction_started, log_prediction_completed, log_low_confidence_prediction,
    log_system_error, log_file_handling_error, log_model_error,
    log_csv_write, log_safety_validation, get_pill_logger
)
try:
    from ..medical_advisor import MedicalAdvisor
except ImportError:
    from medical_advisor import MedicalAdvisor

logger = logging.getLogger(__name__)

# ============================================================================
# CSV OUTPUT HANDLER
# ============================================================================

def write_prediction_to_csv(pill_data):
    """
    Write pill prediction to CSV file.
    
    Args:
        pill_data: Dictionary with pill information
            - pill_name: str
            - confidence: str
            - usage: str
            - dosage: str
            - side_effects: list or str
            - precautions: str
    """
    try:
        pill_system_logger = get_pill_logger()
        csv_file = os.path.join(settings.MEDIA_ROOT, 'pill_predictions.csv')
        
        # Create media directory if it doesn't exist
        os.makedirs(os.path.dirname(csv_file), exist_ok=True)
        
        # Initialize CSV with headers if it doesn't exist
        file_exists = os.path.isfile(csv_file)
        
        # Convert side_effects list to string if needed
        side_effects = pill_data.get('side_effects', '')
        if isinstance(side_effects, list):
            side_effects = '; '.join(side_effects)
        
        with open(csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'timestamp',
                'pill_name',
                'confidence',
                'usage',
                'dosage',
                'side_effects',
                'precautions'
            ])
            
            # Write header if file is new
            if not file_exists:
                writer.writeheader()
            
            # Write prediction record
            writer.writerow({
                'timestamp': datetime.now().isoformat(),
                'pill_name': pill_data.get('pill_name', 'UNKNOWN'),
                'confidence': pill_data.get('confidence', '0%'),
                'usage': pill_data.get('usage', 'N/A'),
                'dosage': pill_data.get('dosage', 'N/A'),
                'side_effects': side_effects,
                'precautions': pill_data.get('precautions', 'Consult healthcare professional')
            })
        
        pill_system_logger.info(f"Prediction recorded to CSV: {pill_data.get('pill_name', 'UNKNOWN')}")
        logger.info(f"Prediction recorded to CSV: {pill_data.get('pill_name', 'UNKNOWN')}")
        return True
    
    except Exception as e:
        pill_system_logger = get_pill_logger()
        message = f"Error writing prediction to CSV: {str(e)}"
        log_file_handling_error('pill_predictions.csv', 'write', str(e))
        logger.error(message)
        pill_system_logger.error(message)
        return False

# ============================================================================
# SAFE PILL INFORMATION MAPPING
# ============================================================================

logger = logging.getLogger(__name__)

# Safe pill information mapping - maps pill names to detailed information
# For the 254-class Kaggle dataset, we have detailed info for original 10 pills
# and generic info for the rest
DETAILED_PILL_INFO = {
    'Alaxan': {
        'dosage': '500mg tablet - Take 1 tablet every 4-6 hours as needed, do not exceed 8 tablets per day',
        'usage': 'Pain relief and fever reduction',
        'consumption_time': 'Anytime',
        'side_effects': 'Nausea, vomiting, stomach upset, dizziness, drowsiness',
        'precautions': 'Do not exceed recommended dose. Consult doctor if pregnant or have liver/kidney problems.'
    },
    'Bactidol': {
        'dosage': '1 lozenge - Suck 1 lozenge every 2 hours as needed, maximum 8 lozenges per day',
        'usage': 'Throat lozenges for sore throat and mouth infections',
        'consumption_time': 'Anytime during the day',
        'side_effects': 'Mild irritation, allergic reactions in sensitive individuals',
        'precautions': 'Do not swallow whole. Keep out of reach of children under 3 years.'
    },
    'Bioflu': {
        'dosage': '500mg capsule - Take 1 capsule every 4-6 hours as needed, maximum 6 capsules per day',
        'usage': 'Relief of flu symptoms including fever, headache, and body aches',
        'consumption_time': 'Anytime',
        'side_effects': 'Drowsiness, dizziness, dry mouth, nausea',
        'precautions': 'May cause drowsiness. Avoid alcohol. Consult doctor if pregnant.'
    },
    'Biogesic': {
        'dosage': '500mg tablet - Take 1 tablet every 4-6 hours, not to exceed 4000mg per day',
        'usage': 'Pain relief and fever reduction',
        'consumption_time': 'Anytime',
        'side_effects': 'Stomach upset, nausea, dizziness, rash',
        'precautions': 'Do not exceed recommended dose. Consult doctor for long-term use.'
    },
    'DayZinc': {
        'dosage': '15mg tablet - Take 1 tablet daily with water, preferably with meals',
        'usage': 'Zinc supplement for immune system support',
        'consumption_time': 'Morning or with meals',
        'side_effects': 'Nausea, vomiting, metallic taste, stomach upset',
        'precautions': 'Take with food to reduce stomach upset. Consult doctor for proper dosage.'
    },
    'Decolgen': {
        'dosage': '2 tablets - Take 2 tablets every 4-6 hours as needed, maximum 8 tablets per day',
        'usage': 'Relief of cold and flu symptoms',
        'consumption_time': 'Anytime',
        'side_effects': 'Drowsiness, dizziness, dry mouth, constipation',
        'precautions': 'May cause drowsiness. Avoid driving. Do not exceed recommended dose.'
    },
    'Fish Oil': {
        'dosage': '1000mg softgel - Take 1-2 softgels daily with meals',
        'usage': 'Omega-3 fatty acid supplement for heart and joint health',
        'consumption_time': 'With meals (breakfast or dinner)',
        'side_effects': 'Fishy aftertaste, belching, stomach upset, loose stools',
        'precautions': 'May interact with blood thinners. Consult doctor if on medication.'
    },
    'Kremil S': {
        'dosage': '2-4 tablets - Take 2-4 tablets after meals or as needed, maximum 12 tablets per day',
        'usage': 'Antacid for relief of acid indigestion, heartburn, and sour stomach',
        'consumption_time': 'After meals or as needed',
        'side_effects': 'Constipation, diarrhea, nausea, vomiting',
        'precautions': 'Do not use for more than 2 weeks without consulting doctor.'
    },
    'Medicol': {
        'dosage': '200mg tablet - Take 1-2 tablets every 4-6 hours as needed, not to exceed 6 tablets per day',
        'usage': 'Pain relief for mild to moderate pain',
        'consumption_time': 'Anytime',
        'side_effects': 'Stomach upset, nausea, dizziness, drowsiness',
        'precautions': 'Do not exceed recommended dose. Consult doctor if pregnant.'
    },
    'Neozep': {
        'dosage': '2 capsules - Take 2 capsules every 4-6 hours as needed, maximum 6 capsules per day',
        'usage': 'Relief of cold and cough symptoms',
        'consumption_time': 'Anytime',
        'side_effects': 'Drowsiness, dizziness, dry mouth, nausea',
        'precautions': 'May cause drowsiness. Avoid alcohol and driving. Do not exceed dose.'
    }
}

def get_pill_information(pill_class):
    """
    Get pill information for a given class number (0-253).
    Returns detailed info for known pills, generic info for others.
    """
    # For the Kaggle dataset, pill classes are just numbers 0-253
    # We don't have specific names for all 254 pills, so we provide generic info
    if str(pill_class) in DETAILED_PILL_INFO:
        return DETAILED_PILL_INFO[str(pill_class)]
    else:
        # Generic information for unknown pills
        return {
            'dosage': 'Consult pharmacist or physician for specific dosage instructions',
            'usage': 'Medication identified - Please consult healthcare professional for usage details',
            'consumption_time': 'Follow prescription instructions or consult healthcare provider',
            'side_effects': 'Information not available - Consult healthcare professional',
            'precautions': 'Always consult a healthcare professional before taking any medication'
        }

# Create your views here.

def userRegister(request):
    if request.method == 'POST':
        # Extract data from the request
        name = request.POST.get('name')
        email = request.POST.get('email')
        password = request.POST.get('password')
        loginid = request.POST.get('loginid')
        mobile = request.POST.get('mobile')
        locality = request.POST.get('locality')  # Locality
        state = request.POST.get('state')  # State

        user = UserRegisteredTable(
            name=name,
            email=email,
            password=password,  # Password will be hashed in the model's save method
            loginid=loginid,
            mobile=mobile,
            locality=locality,
            state=state,
        )
        try:
            if user.full_clean:
                user.save()
                messages.success(request, 'Registration successful!.')
                return render(request,'register.html')
            else:
                messages.error(request,'Entered data is invalid')
                return render(request,'register.html')
        except Exception as e:
            logger.error(f"Registration error: {str(e)}")
            messages.error(request,'Entered data is invalid')
            return render(request,'register.html')

    return render(request, 'register.html')

def userLoginCheck(request):
    if request.method=="POST":
        loginid=request.POST['loginid']
        password=request.POST['password']
        try:
            user=UserRegisteredTable.objects.get(loginid=loginid,password=password)
            status=user.status
            if status=='activated':
                return render(request,'users/userHome.html')
            else:
                messages.error(request,'Status Not Activated')
                return render(request,'userLogin.html')
        except Exception as e:
            logger.error(f"Login error: {str(e)}")
            messages.error(request,'Invalid details')
            return render(request,'userLogin.html')
    else:
        return render(request,'userLogin.html')

def prediction(request):
    """
    MEDICAL-GRADE PILL IDENTIFICATION
    
    Safety Features:
    - 70% confidence threshold enforced
    - "UNKNOWN TABLET" for low-confidence predictions
    - Top-3 candidates logged for audit
    - Full validation before returning response
    """
    if request.method == 'POST' and request.FILES.get('image'):
        try:
            image_file = request.FILES['image']
            
            # ========== LOG IMAGE UPLOAD EVENT ==========
            log_image_upload(image_file.name, image_file.size)
            
            # Validate file type
            allowed_types = ['jpg', 'jpeg', 'png', 'gif', 'bmp']
            file_ext = image_file.name.split('.')[-1].lower() if '.' in image_file.name else ''
            
            if file_ext not in allowed_types:
                # ========== LOG INVALID FILE TYPE ==========
                log_invalid_file_type(image_file.name, file_ext, allowed_types)
                messages.error(request, f"Invalid file type: {file_ext}. Allowed: {', '.join(allowed_types)}")
                return render(request, 'users/predictionForm.html', {
                    'error': 'Invalid file type'
                })
            
            # ========== LOG VALID FILE TYPE ==========
            log_valid_file_type(image_file.name, file_ext)
            
            # Save uploaded file
            fs = FileSystemStorage()
            filename = fs.save(image_file.name, image_file)
            file_path = os.path.join(settings.MEDIA_ROOT, filename)
            logger.info(f"[MEDICAL PREDICTION] Processing image: {file_path}")

            # ========== CHECK FOR CORRUPTED IMAGE ==========
            try:
                from PIL import Image as PILImage
                test_img = PILImage.open(file_path)
                test_img.verify()
            except Exception as img_error:
                # ========== LOG CORRUPTED IMAGE ==========
                log_corrupted_image(image_file.name, str(img_error))
                logger.error(f"[PREDICTION ERROR] Corrupted image: {img_error}")
                messages.error(request, "The image file is corrupted or unreadable. Please upload a valid image.")
                try:
                    os.remove(file_path)
                except:
                    pass
                return render(request, 'users/predictionForm.html', {
                    'error': 'Corrupted image detected'
                })
            
            # ========== START PREPROCESSING LOG ==========
            log_preprocessing_started(image_file.name)

            # Call prediction with medical safety (confidence threshold built-in)
            # Lowered from 0.70 to 0.50 to fix UNKNOWN TABLET issue in v2 model
            
            # ========== START PREDICTION LOG ==========
            log_prediction_started(image_file.name, 'MobileNetV3Large')
            
            result = predictions(file_path, confidence_threshold=0.50)
            
            # ========== LOG PREPROCESSING COMPLETED ==========
            log_preprocessing_completed(image_file.name, ['resize', 'normalization', 'augmentation'])
            
            # Validate prediction for medical safety
            validation = MedicalSafetyValidator.validate_prediction(result)
            
            # Extract results
            tablet_name = result.get('tablet_name', 'UNKNOWN TABLET')
            confidence_str = result.get('confidence', '0%')
            generic_name = result.get('generic_name', 'N/A')
            usage = result.get('usage', 'Information not available')
            dosage = result.get('dosage', 'Information not available')
            consumption_time = result.get('consumption_time', 'Information not available')
            side_effects = result.get('side_effects', [])
            precautions = result.get('precautions', 'Consult healthcare professional')
            disclaimer = result.get('disclaimer', '⚠️ Always consult a pharmacist')
            debug_info = result.get('debug_info', {})
            
            # Log decision
            is_safe = validation['is_safe']
            passed_check = debug_info.get('passed_safety_check', False)
            
            # ========== LOG PREDICTION COMPLETED ==========
            confidence_value = float(confidence_str.replace('%', '')) if '%' in confidence_str else float(confidence_str)
            log_prediction_completed(tablet_name, confidence_str, {
                'top_3': debug_info.get('top_3_candidates', []),
                'processing_time': debug_info.get('processing_time', 'N/A')
            })
            
            # Log low confidence predictions
            if confidence_value < 70:
                log_low_confidence_prediction(tablet_name, confidence_str, '70%')
            
            # Log safety validation
            log_safety_validation(passed_check, tablet_name, validation)
            
            logger.info(f"[MEDICAL PREDICTION] Result: {tablet_name}")
            logger.info(f"[MEDICAL PREDICTION] Confidence: {confidence_str}")
            logger.info(f"[MEDICAL PREDICTION] Safety Check: {passed_check}")
            logger.info(f"[MEDICAL PREDICTION] Validation: {is_safe}")
            
            # Log any warnings
            for warning in validation['warnings']:
                logger.warning(f"[MEDICAL WARNING] {warning}")
            
            for error in validation['errors']:
                logger.error(f"[MEDICAL ERROR] {error}")
            
            # Write prediction to CSV
            csv_data = {
                'pill_name': tablet_name,
                'confidence': confidence_str,
                'usage': usage,
                'dosage': dosage,
                'side_effects': side_effects,
                'precautions': precautions
            }
            write_prediction_to_csv(csv_data)
            
            # ========== LOG CSV WRITE ==========
            log_csv_write('pill_predictions.csv', {'pill_name': tablet_name})
            
            # Display results
            context = {
                'tablet_name': tablet_name,
                'pill_name': tablet_name,  # For backward compatibility
                'confidence': confidence_str,
                'generic_name': generic_name,
                'usage': usage,
                'dosage': dosage,
                'consumption_time': consumption_time,
                'side_effects': side_effects,
                'precautions': precautions,
                'disclaimer': disclaimer,
                'passed_safety_check': passed_check,
                'is_safe': is_safe,
                'top_3_candidates': debug_info.get('top_3_candidates', []),
                'confidence_threshold': debug_info.get('confidence_threshold', '70%'),
            }
            
            # Set success or warning message based on safety
            if tablet_name == 'UNKNOWN TABLET':
                messages.warning(
                    request,
                    f"⚠️ CAUTION: Could not identify tablet with confidence. "
                    f"Confidence: {confidence_str}. Contact your pharmacist."
                )
            elif passed_check:
                messages.success(
                    request,
                    f"✓ Tablet Identified: {tablet_name} (Confidence: {confidence_str})"
                )
            else:
                messages.warning(
                    request,
                    f"⚠️ Identification uncertain. Please verify with pharmacist."
                )
            
            # Clean up temp file
            try:
                os.remove(file_path)
            except:
                pass
            
            return render(request, 'users/predictionForm.html', context)
            
        except Exception as e:
            # ========== LOG SYSTEM ERROR ==========
            log_system_error(str(e), 'PredictionError', logger.exc_info())
            logger.error(f"[PREDICTION ERROR] {str(e)}", exc_info=True)
            messages.error(request, f"System error: {str(e)}")
            return render(request, 'users/predictionForm.html', {
                'error': 'System error occurred',
                'tablet_name': 'ERROR',
                'disclaimer': '⚠️ SAFETY: System error. Please try again or consult a healthcare professional.'
            })

    return render(request, 'users/predictionForm.html', {
        'disclaimer': '⚠️ Upload a clear image of the pill for identification. '
                      'AI prediction must be verified by a pharmacist.'
    })

def classificationView(request):
    """
    Train the pill identification model with enhanced parameters.
    """
    try:
        pill_system_logger = get_pill_logger()
        pill_system_logger.info("Starting model training from web interface")
        logger.info("Starting model training from web interface")
        
        accuracy, model, label_map, reverse_label_map = main()
        
        context = {
            'accuracy': accuracy,
            'accuracy_percentage': f"{accuracy*100:.2f}%",
            'num_classes': len(label_map),
            'classes': list(reverse_label_map.values())
        }
        
        pill_system_logger.info(f'Model training completed! Accuracy: {accuracy*100:.2f}%')
        messages.success(request, f'Model training completed! Accuracy: {accuracy*100:.2f}%')
        return render(request, 'users/classificationView.html', context)
        
    except Exception as e:
        pill_system_logger = get_pill_logger()
        error_msg = f"Classification training error: {str(e)}"
        log_model_error(str(e), 'MobileNetV3Large')
        pill_system_logger.error(error_msg)
        logger.error(error_msg)
        messages.error(request, f"Training failed: {str(e)}")
        return render(request, 'users/classificationView.html', {'Error': str(e)})


def ask_medical_advisor(request):
    """AJAX endpoint: answer user questions about the last identified pill.

    Expected POST JSON or form data:
      - question: str
      - pill_name (optional): if provided, override lookup
      - dosage, usage, side_effects, precautions, consumption_time, confidence (optional)

    Returns JSON: { 'answer': str }
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=400)

    try:
        # Gather pill info from POST fields (fall back to minimal)
        pill_info = {
            'name': request.POST.get('pill_name') or request.POST.get('tablet_name') or request.POST.get('pill_name_input'),
            'dosage': request.POST.get('dosage'),
            'usage': request.POST.get('usage'),
            'side_effects': request.POST.get('side_effects'),
            'precautions': request.POST.get('precautions'),
            'consumption_time': request.POST.get('consumption_time'),
            'confidence': request.POST.get('confidence')
        }

        question = request.POST.get('question') or request.GET.get('question')
        if not question:
            return JsonResponse({'error': 'question is required'}, status=400)

        advisor = MedicalAdvisor()
        answer = advisor.answer_question(pill_info, question)

        return JsonResponse({'answer': answer})

    except Exception as e:
        logger.error(f"[MEDICAL_ADVISOR_ERROR] {str(e)}", exc_info=True)
        return JsonResponse({'error': 'internal error'}, status=500)


# ============================================================================
# VOICE-BASED MEDICATION QUERIES
# ============================================================================

def check_voice_input_availability(request):
    """
    API endpoint: Check if voice input (speech recognition) is available.
    
    Returns JSON:
      {
        'available': bool,
        'setup_status': dict with validation details
      }
    """
    try:
        setup_status = validate_speech_recognition_setup()
        
        return JsonResponse({
            'available': setup_status['ready'],
            'setup_status': setup_status
        })
    
    except Exception as e:
        logger.error(f"Error checking voice input availability: {str(e)}")
        return JsonResponse({
            'available': False,
            'error': str(e)
        }, status=500)


def record_voice_query(request):
    """
    AJAX endpoint: Record voice input and convert to text.
    
    Records audio from microphone and converts speech to text using
    Google Speech Recognition API.
    
    Returns JSON:
      {
        'success': bool,
        'text': recognized text (if successful),
        'error': error message (if failed)
      }
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=400)
    
    try:
        logger.info("[VOICE_QUERY] Starting voice recording...")
        
        # Get language preference from request (default: English)
        language = request.POST.get('language', 'en-US')
        timeout = int(request.POST.get('timeout', '10'))
        
        logger.info(f"[VOICE_QUERY] Language: {language}, Timeout: {timeout}s")
        
        # Initialize speech recognition handler
        handler = SpeechRecognitionHandler(language=language, timeout=timeout)
        
        if not handler.is_available():
            return JsonResponse({
                'success': False,
                'error': 'Speech recognition not available. Please ensure SpeechRecognition '
                        'and PyAudio are installed.',
                'text': ''
            }, status=503)
        
        # Record from microphone and recognize
        logger.info("[VOICE_QUERY] Recording from microphone...")
        success, text, error = handler.recognize_from_microphone()
        
        logger.info(f"[VOICE_QUERY] Recognition result - Success: {success}, Text: {text}")
        
        if success:
            logger.info(f"[VOICE_QUERY] Successfully recognized: {text}")
            return JsonResponse({
                'success': True,
                'text': text,
                'error': ''
            })
        else:
            logger.warning(f"[VOICE_QUERY] Recognition failed: {error}")
            return JsonResponse({
                'success': False,
                'text': '',
                'error': error or 'Could not recognize speech'
            })
    
    except Exception as e:
        error_msg = f"Error processing voice query: {str(e)}"
        logger.error(f"[VOICE_QUERY_ERROR] {error_msg}", exc_info=True)
        return JsonResponse({
            'success': False,
            'text': '',
            'error': error_msg
        }, status=500)


def voice_based_medication_query(request):
    """
    Process a medication query from voice input.
    
    Expected POST data:
      - question: str (the medication query from voice)
      - pill_name (optional): if providing follow-up to identified pill
    
    Returns JSON with medication information or advisor response.
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=400)
    
    try:
        question = request.POST.get('question', '').strip()
        
        if not question:
            return JsonResponse({
                'error': 'Question is required',
                'success': False
            }, status=400)
        
        logger.info(f"[VOICE_MEDICATION_QUERY] Processing: {question}")
        
        # Use medical advisor to answer the question
        pill_name = request.POST.get('pill_name')
        pill_info = {
            'name': pill_name or 'Not specified',
            'dosage': request.POST.get('dosage'),
            'usage': request.POST.get('usage'),
            'side_effects': request.POST.get('side_effects'),
            'precautions': request.POST.get('precautions'),
            'consumption_time': request.POST.get('consumption_time'),
            'confidence': request.POST.get('confidence')
        }
        
        advisor = MedicalAdvisor()
        answer = advisor.answer_question(pill_info, question)
        
        logger.info(f"[VOICE_MEDICATION_QUERY] Answer provided")
        
        return JsonResponse({
            'success': True,
            'answer': answer,
            'original_question': question
        })
    
    except Exception as e:
        error_msg = f"Error processing voice medication query: {str(e)}"
        logger.error(f"[VOICE_QUERY_ERROR] {error_msg}", exc_info=True)
        return JsonResponse({
            'error': error_msg,
            'success': False
        }, status=500)

