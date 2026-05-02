#!/usr/bin/env python
"""Test if speech recognition modules are available"""

import sys
print("Python path:", sys.path)

# Test imports
try:
    import speech_recognition
    print("✓ speech_recognition imported successfully")
    print("  Version:", speech_recognition.__version__)
except ImportError as e:
    print("✗ Failed to import speech_recognition:", e)

try:
    import pyaudio
    print("✓ pyaudio imported successfully")
except ImportError as e:
    print("✗ Failed to import pyaudio:", e)

# Test handler import
try:
    from Users.speech_recognition_handler import SPEECH_RECOGNITION_AVAILABLE, SpeechRecognitionHandler
    print("✓ SpeechRecognitionHandler imported successfully")
    print("  SPEECH_RECOGNITION_AVAILABLE:", SPEECH_RECOGNITION_AVAILABLE)
    
    if SPEECH_RECOGNITION_AVAILABLE:
        handler = SpeechRecognitionHandler()
        print("  Handler initialized, is_available():", handler.is_available())
    else:
        print("  ✗ Speech recognition not available!")
except ImportError as e:
    print("✗ Failed to import SpeechRecognitionHandler:", e)
except Exception as e:
    print("✗ Error:", e)
