"""
Speech Recognition Handler for Medication Queries
===================================================

Provides speech-to-text conversion for medication queries using
the SpeechRecognition library with Google's Speech API.

Features:
- Microphone input capture
- Speech-to-text conversion
- Error handling for recognition failures
- Support for multiple languages
- Timeouts and audio validation
"""

import logging
import io
import wave
from typing import Dict, Tuple, Optional, TYPE_CHECKING

try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False
    sr = None

if TYPE_CHECKING:
    import speech_recognition


logger = logging.getLogger(__name__)


class SpeechRecognitionHandler:
    """
    Handles speech recognition for medication queries.
    
    Provides methods to capture audio from microphone and convert it to text.
    """
    
    def __init__(self, language: str = 'en-US', timeout: int = 10):
        """
        Initialize speech recognition handler.
        
        Args:
            language: Language code (default: 'en-US')
            timeout: Timeout in seconds for recording (default: 10)
        """
        global SPEECH_RECOGNITION_AVAILABLE
        
        self.language = language
        self.timeout = timeout
        self.recognizer = None
        self.microphone = None
        
        if SPEECH_RECOGNITION_AVAILABLE:
            try:
                self.recognizer = sr.Recognizer()
                self.microphone = sr.Microphone()
                
                # Configure recognizer settings for medical terminology
                self.recognizer.energy_threshold = 4000  # More sensitive
                self.recognizer.dynamic_energy_threshold = True
                
                logger.info("Speech recognition initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize speech recognition: {str(e)}")
                SPEECH_RECOGNITION_AVAILABLE = False
        else:
            logger.warning("SpeechRecognition library not available")
    
    def record_from_microphone(self, duration: Optional[int] = None) -> Optional["speech_recognition.AudioData"]:
        """
        Record audio from microphone.
        
        Args:
            duration: Duration to record in seconds (None = until phrase detected)
            
        Returns:
            AudioData object or None if recording fails
        """
        if not SPEECH_RECOGNITION_AVAILABLE or not self.recognizer or not self.microphone:
            logger.error("Speech recognition not available")
            return None
        
        try:
            with self.microphone as source:
                logger.info("Recording audio from microphone...")
                
                # Adapt recognizer to ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                
                if duration:
                    # Record for fixed duration
                    audio = self.recognizer.record(source, duration=duration, timeout=self.timeout)
                else:
                    # Record until silence is detected (phrase ends)
                    audio = self.recognizer.listen(
                        source,
                        timeout=self.timeout,
                        phrase_time_limit=self.timeout
                    )
                
                logger.info(f"Successfully recorded audio ({len(audio.frame_data)} bytes)")
                return audio
                
        except Exception as e:
            # Handle both sr.RequestError and sr.UnknownValueError without direct reference
            error_type = type(e).__name__
            if error_type == 'RequestError':
                logger.error(f"Could not request results from speech recognition API: {str(e)}")
            elif error_type == 'UnknownValueError':
                logger.warning("Could not understand audio")
            else:
                logger.error(f"Error recording from microphone: {str(e)}")
            return None
    
    def recognize_speech_from_audio(self, audio: "speech_recognition.AudioData") -> Tuple[bool, str, str]:
        """
        Convert audio to text using Google Speech Recognition API.
        
        Args:
            audio: AudioData object from recording
            
        Returns:
            Tuple of (success: bool, text: str, error_message: str)
        """
        if not SPEECH_RECOGNITION_AVAILABLE or not self.recognizer:
            return False, "", "Speech recognition not available"
        
        if not audio:
            return False, "", "No audio data provided"
        
        try:
            logger.info(f"Recognizing speech in language: {self.language}")
            
            # Use Google Speech Recognition API (free, but has rate limits)
            text = self.recognizer.recognize_google(audio, language=self.language)
            
            logger.info(f"Successfully recognized speech: {text}")
            return True, text, ""
            
        except Exception as e:
            # Handle both sr.UnknownValueError and sr.RequestError without direct reference
            error_type = type(e).__name__
            if error_type == 'UnknownValueError':
                error_msg = "Could not understand the audio. Please speak clearly."
                logger.warning(error_msg)
            elif error_type == 'RequestError':
                error_msg = f"Could not request results from API: {str(e)}"
                logger.error(error_msg)
            else:
                error_msg = f"Error during speech recognition: {str(e)}"
                logger.error(error_msg)
            return False, "", error_msg
    
    def recognize_from_microphone(self) -> Tuple[bool, str, str]:
        """
        Full pipeline: record from microphone and recognize speech.
        
        Returns:
            Tuple of (success: bool, text: str, error_message: str)
        """
        try:
            # Record audio
            audio = self.record_from_microphone()
            if not audio:
                return False, "", "Failed to record audio from microphone"
            
            # Recognize speech from audio
            success, text, error = self.recognize_speech_from_audio(audio)
            return success, text, error
            
        except Exception as e:
            error_msg = f"Error in recognition pipeline: {str(e)}"
            logger.error(error_msg)
            return False, "", error_msg
    
    def recognize_from_file(self, file_path: str) -> Tuple[bool, str, str]:
        """
        Recognize speech from an audio file.
        
        Args:
            file_path: Path to audio file (wav, mp3, etc.)
            
        Returns:
            Tuple of (success: bool, text: str, error_message: str)
        """
        if not SPEECH_RECOGNITION_AVAILABLE or not self.recognizer:
            return False, "", "Speech recognition not available"
        
        try:
            logger.info(f"Loading audio from file: {file_path}")
            
            with sr.AudioFile(file_path) as source:
                audio = self.recognizer.record(source)
            
            # Recognize speech
            success, text, error = self.recognize_speech_from_audio(audio)
            return success, text, error
            
        except Exception as e:
            error_msg = f"Error reading audio file: {str(e)}"
            logger.error(error_msg)
            return False, "", error_msg
    
    @staticmethod
    def is_available() -> bool:
        """Check if speech recognition is available."""
        return SPEECH_RECOGNITION_AVAILABLE
    
    @staticmethod
    def get_required_packages() -> Dict[str, str]:
        """Get required packages information."""
        return {
            'SpeechRecognition': '3.10.0',
            'PyAudio': '0.2.13'  # May vary by system
        }


def process_voice_query(language: str = 'en-US', timeout: int = 10) -> Dict:
    """
    Process a voice query for medication information.
    
    Args:
        language: Language code for speech recognition
        timeout: Timeout in seconds
        
    Returns:
        Dictionary with:
        - success: bool
        - text: recognized text
        - error: error message if any
    """
    try:
        handler = SpeechRecognitionHandler(language=language, timeout=timeout)
        
        success, text, error = handler.recognize_from_microphone()
        
        return {
            'success': success,
            'text': text,
            'error': error,
            'query_type': 'voice'
        }
        
    except Exception as e:
        logger.error(f"Error processing voice query: {str(e)}")
        return {
            'success': False,
            'text': '',
            'error': str(e),
            'query_type': 'voice'
        }


def validate_speech_recognition_setup() -> Dict:
    """
    Validate speech recognition setup.
    
    Returns:
        Dictionary with validation status and information
    """
    status = {
        'speech_recognition_available': SPEECH_RECOGNITION_AVAILABLE,
        'issues': [],
        'recommendations': []
    }
    
    if not SPEECH_RECOGNITION_AVAILABLE:
        status['issues'].append("SpeechRecognition library not installed")
        status['recommendations'].append("Install SpeechRecognition: pip install SpeechRecognition")
    
    try:
        import pyaudio
    except ImportError:
        status['issues'].append("PyAudio not installed")
        status['recommendations'].append(
            "Install PyAudio: pip install PyAudio "
            "(On Linux: sudo apt-get install portaudio19-dev)"
        )
    
    if not SpeechRecognitionHandler.is_available():
        status['issues'].append("Microphone not detected")
        status['recommendations'].append("Check microphone connection and permissions")
    
    status['ready'] = len(status['issues']) == 0
    
    return status
