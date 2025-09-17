"""
Session State Validation Utility

Provides comprehensive validation for Streamlit session state to prevent UI dead-ends
and ensure graceful handling of missing or corrupted state data.
"""

import streamlit as st
import time
from typing import Dict, Any, Optional, List, Union
from pathlib import Path


class SessionStateValidator:
    """Validates and manages Streamlit session state integrity"""
    
    # Required session state keys and their expected types
    REQUIRED_KEYS = {
        'current_screen': str,
        'processing_results': (dict, type(None)),
        'uploaded_file': (type(None), object),  # Can be None or file object
        'file_url': str,
        'processing_stage': int,
        'processing_stages': list,
        'difficulty_mode': str,
        'output_formats': list,
        'selected_formats': list,
        'processing_error': (str, type(None)),
        'job_history': list,
        'estimated_time': (int, float, type(None)),
        'start_time': (int, float, type(None)),
        'app_initialized': bool
    }
    
    # Optional keys that might be added during processing
    OPTIONAL_KEYS = {
        'ensemble_manager': object,
        'uploaded_file_path': str,
        'raw_ensemble_results': dict,
        'hotspot_manager': object,
        'hotspot_session': dict,
        'hotspot_edits': dict
    }
    
    # Valid screen names
    VALID_SCREENS = ['landing', 'processing', 'results', 'hotspot_review', 'error']
    
    @classmethod
    def initialize_defaults(cls) -> None:
        """Initialize session state with safe defaults"""
        defaults = {
            'current_screen': 'landing',
            'uploaded_file': None,
            'file_url': '',
            'processing_stage': 0,
            'processing_stages': [
                'Upload',
                'Chunking', 
                'Transcription',
                'Speaker Diarization',
                'Consensus',
                'Finalizing'
            ],
            'difficulty_mode': 'Standard',
            'output_formats': ['Transcript (.txt)', 'Subtitles (.srt)', 'Report (.json)'],
            'selected_formats': ['Transcript (.txt)'],
            'processing_results': None,
            'processing_error': None,
            'job_history': [],
            'estimated_time': None,
            'start_time': None,
            'app_initialized': False
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    @classmethod
    def validate_key(cls, key: str, expected_type: Optional[Union[type, tuple]] = None) -> bool:
        """Validate a specific session state key"""
        if key not in st.session_state:
            return False
        
        if expected_type is None:
            return True
        
        value = st.session_state[key]
        
        # Handle multiple allowed types
        if isinstance(expected_type, tuple):
            return any(isinstance(value, t) for t in expected_type)
        else:
            return isinstance(value, expected_type)
    
    @classmethod
    def validate_all_required(cls) -> Dict[str, bool]:
        """Validate all required session state keys"""
        results = {}
        for key, expected_type in cls.REQUIRED_KEYS.items():
            results[key] = cls.validate_key(key, expected_type)
        return results
    
    @classmethod
    def validate_current_screen(cls) -> bool:
        """Validate current_screen is valid"""
        if not cls.validate_key('current_screen', str):
            return False
        return st.session_state.current_screen in cls.VALID_SCREENS
    
    @classmethod
    def validate_processing_results(cls) -> bool:
        """Validate processing_results has required structure"""
        if not cls.validate_key('processing_results', dict):
            return False
        
        results = st.session_state.processing_results
        required_fields = ['file_name', 'transcript', 'duration', 'speakers']
        
        return all(field in results for field in required_fields)
    
    @classmethod
    def validate_for_hotspot_review(cls) -> tuple[bool, str]:
        """Validate session state for hotspot review access"""
        # Check if we have processing results
        if not cls.validate_processing_results():
            return False, "No transcript results available. Please process a file first."
        
        # Check if results have the required structure for hotspot review
        results = st.session_state.processing_results
        
        if 'full_results' not in results:
            return False, "Transcript results are incomplete. Please re-process the file."
        
        # Validate uploaded file path for audio playback
        if not cls.validate_key('uploaded_file_path', str):
            # This is not critical - hotspot review can work without audio
            pass
        
        return True, ""
    
    @classmethod
    def validate_hotspot_session(cls) -> bool:
        """Validate hotspot session structure"""
        if not cls.validate_key('hotspot_session', dict):
            return False
        
        session = st.session_state.hotspot_session
        required_fields = ['clips', 'current_clip_index', 'status', 'start_time']
        
        return all(field in session for field in required_fields)
    
    @classmethod
    def validate_for_results_screen(cls) -> tuple[bool, str]:
        """Validate session state for results screen access"""
        if not cls.validate_processing_results():
            return False, "No processing results available. Please transcribe a file first."
        
        return True, ""
    
    @classmethod
    def reset_session_state(cls, preserve_keys: Optional[List[str]] = None) -> None:
        """Reset session state while preserving specified keys"""
        preserve_keys = preserve_keys or ['app_initialized']
        
        preserved_values = {}
        for key in preserve_keys:
            if key in st.session_state:
                preserved_values[key] = st.session_state[key]
        
        # Clear all session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        
        # Restore preserved values
        for key, value in preserved_values.items():
            st.session_state[key] = value
        
        # Reinitialize defaults
        cls.initialize_defaults()
    
    @classmethod
    def safe_navigate_to_screen(cls, screen: str, validate_access: bool = True) -> bool:
        """Safely navigate to a screen with validation"""
        if screen not in cls.VALID_SCREENS:
            st.error(f"Invalid screen: {screen}")
            return False
        
        # Validate access permissions for certain screens
        if validate_access:
            if screen == 'results':
                is_valid, error_msg = cls.validate_for_results_screen()
                if not is_valid:
                    st.warning(error_msg)
                    cls.safe_navigate_to_screen('landing', validate_access=False)
                    return False
            
            elif screen == 'hotspot_review':
                is_valid, error_msg = cls.validate_for_hotspot_review()
                if not is_valid:
                    st.warning(error_msg)
                    # Try to go back to results if available, otherwise landing
                    if cls.validate_for_results_screen()[0]:
                        cls.safe_navigate_to_screen('results', validate_access=False)
                    else:
                        cls.safe_navigate_to_screen('landing', validate_access=False)
                    return False
        
        # Navigation is valid
        st.session_state.current_screen = screen
        return True
    
    @classmethod
    def get_validation_report(cls) -> Dict[str, Any]:
        """Get comprehensive validation report"""
        report = {
            'timestamp': time.time(),
            'required_keys': cls.validate_all_required(),
            'current_screen_valid': cls.validate_current_screen(),
            'processing_results_valid': cls.validate_processing_results() if cls.validate_key('processing_results', dict) else False,
            'hotspot_session_valid': cls.validate_hotspot_session() if cls.validate_key('hotspot_session', dict) else False,
            'recommendations': []
        }
        
        # Add recommendations based on validation results
        if not report['current_screen_valid']:
            report['recommendations'].append("Reset current_screen to 'landing'")
        
        missing_required = [k for k, v in report['required_keys'].items() if not v]
        if missing_required:
            report['recommendations'].append(f"Initialize missing required keys: {missing_required}")
        
        return report
    
    @classmethod
    def auto_repair_session_state(cls) -> List[str]:
        """Automatically repair common session state issues"""
        repairs_made = []
        
        # Repair current_screen if invalid
        if not cls.validate_current_screen():
            st.session_state.current_screen = 'landing'
            repairs_made.append("Reset current_screen to 'landing'")
        
        # Initialize missing required keys
        for key, expected_type in cls.REQUIRED_KEYS.items():
            if not cls.validate_key(key):
                cls.initialize_defaults()
                repairs_made.append(f"Initialized missing required keys")
                break
        
        # Validate screen access permissions
        current_screen = st.session_state.get('current_screen', 'landing')
        if current_screen == 'results' and not cls.validate_for_results_screen()[0]:
            st.session_state.current_screen = 'landing'
            repairs_made.append("Redirected from results to landing (no processing results)")
        
        if current_screen == 'hotspot_review' and not cls.validate_for_hotspot_review()[0]:
            if cls.validate_for_results_screen()[0]:
                st.session_state.current_screen = 'results'
                repairs_made.append("Redirected from hotspot_review to results")
            else:
                st.session_state.current_screen = 'landing'
                repairs_made.append("Redirected from hotspot_review to landing (no processing results)")
        
        return repairs_made


class ValidationError(Exception):
    """Custom exception for session state validation errors"""
    def __init__(self, message: str, key: Optional[str] = None, expected_type: Optional[type] = None):
        self.message = message
        self.key = key
        self.expected_type = expected_type
        super().__init__(self.message)


def validate_session_state_decorator(validation_func):
    """Decorator to validate session state before function execution"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            is_valid, error_msg = validation_func()
            if not is_valid:
                st.error(f"⚠️ **Session State Error**: {error_msg}")
                st.info("Redirecting to a safe screen...")
                SessionStateValidator.safe_navigate_to_screen('landing')
                st.rerun()
                return None
            return func(*args, **kwargs)
        return wrapper
    return decorator


# Convenient validation decorators for common use cases
def require_processing_results(func):
    """Decorator to require valid processing results"""
    return validate_session_state_decorator(SessionStateValidator.validate_for_results_screen)(func)


def require_hotspot_access(func):
    """Decorator to require valid hotspot review access"""
    return validate_session_state_decorator(SessionStateValidator.validate_for_hotspot_review)(func)