"""
Session State Test Validator

Comprehensive testing utility to validate all session state scenarios and ensure
no UI dead-ends exist in the transcription application.
"""

import streamlit as st
import time
from typing import Dict, List, Any, Optional
from utils.session_state_validator import SessionStateValidator


class SessionStateTestValidator:
    """Test validator for comprehensive session state scenario testing"""
    
    @classmethod
    def run_comprehensive_tests(cls) -> Dict[str, Any]:
        """Run comprehensive tests on all session state scenarios"""
        test_results = {
            'timestamp': time.time(),
            'tests_run': 0,
            'tests_passed': 0,
            'tests_failed': 0,
            'test_details': [],
            'critical_issues': [],
            'warnings': []
        }
        
        # Test scenarios
        test_scenarios = [
            ('test_fresh_session', cls._test_fresh_session),
            ('test_missing_current_screen', cls._test_missing_current_screen),
            ('test_corrupted_processing_results', cls._test_corrupted_processing_results),
            ('test_invalid_hotspot_session', cls._test_invalid_hotspot_session),
            ('test_results_screen_access', cls._test_results_screen_access),
            ('test_hotspot_review_access', cls._test_hotspot_review_access),
            ('test_navigation_validation', cls._test_navigation_validation),
            ('test_auto_repair_functionality', cls._test_auto_repair_functionality),
            ('test_emergency_recovery', cls._test_emergency_recovery)
        ]
        
        for test_name, test_func in test_scenarios:
            try:
                test_results['tests_run'] += 1
                result = test_func()
                
                if result['passed']:
                    test_results['tests_passed'] += 1
                else:
                    test_results['tests_failed'] += 1
                    if result.get('critical', False):
                        test_results['critical_issues'].append({
                            'test': test_name,
                            'issue': result.get('error', 'Unknown error')
                        })
                
                test_results['test_details'].append({
                    'test': test_name,
                    'passed': result['passed'],
                    'message': result.get('message', ''),
                    'details': result.get('details', {})
                })
                
            except Exception as e:
                test_results['tests_failed'] += 1
                test_results['critical_issues'].append({
                    'test': test_name,
                    'issue': f"Test execution failed: {str(e)}"
                })
        
        # Calculate success rate
        if test_results['tests_run'] > 0:
            test_results['success_rate'] = test_results['tests_passed'] / test_results['tests_run']
        else:
            test_results['success_rate'] = 0
        
        return test_results
    
    @classmethod
    def _test_fresh_session(cls) -> Dict[str, Any]:
        """Test behavior with fresh session state"""
        # Backup current state
        original_state = dict(st.session_state)
        
        try:
            # Clear session state
            st.session_state.clear()
            
            # Initialize and test
            SessionStateValidator.initialize_defaults()
            
            # Validate all required keys exist
            validation_results = SessionStateValidator.validate_all_required()
            missing_keys = [k for k, v in validation_results.items() if not v]
            
            if missing_keys:
                return {
                    'passed': False,
                    'message': f"Missing required keys after initialization: {missing_keys}",
                    'critical': True
                }
            
            # Test current screen validation
            if not SessionStateValidator.validate_current_screen():
                return {
                    'passed': False,
                    'message': "Current screen validation failed after initialization",
                    'critical': True
                }
            
            return {
                'passed': True,
                'message': "Fresh session initialization successful",
                'details': {'keys_validated': len(validation_results)}
            }
            
        finally:
            # Restore original state
            st.session_state.clear()
            st.session_state.update(original_state)
    
    @classmethod
    def _test_missing_current_screen(cls) -> Dict[str, Any]:
        """Test behavior when current_screen is missing"""
        # Backup and modify state
        original_screen = st.session_state.get('current_screen')
        
        try:
            # Remove current_screen
            if 'current_screen' in st.session_state:
                del st.session_state['current_screen']
            
            # Test auto-repair
            repairs = SessionStateValidator.auto_repair_session_state()
            
            # Check if screen was restored
            if not SessionStateValidator.validate_current_screen():
                return {
                    'passed': False,
                    'message': "Auto-repair failed to restore current_screen",
                    'critical': True
                }
            
            return {
                'passed': True,
                'message': "Missing current_screen auto-repaired successfully",
                'details': {'repairs_made': len(repairs)}
            }
            
        finally:
            # Restore original state
            if original_screen is not None:
                st.session_state.current_screen = original_screen
    
    @classmethod
    def _test_corrupted_processing_results(cls) -> Dict[str, Any]:
        """Test behavior with corrupted processing results"""
        # Backup original results
        original_results = st.session_state.get('processing_results')
        
        try:
            # Set corrupted results
            st.session_state.processing_results = {'incomplete': True}  # Missing required fields
            
            # Test validation
            is_valid = SessionStateValidator.validate_processing_results()
            
            if is_valid:
                return {
                    'passed': False,
                    'message': "Validator failed to detect corrupted processing results",
                    'critical': True
                }
            
            # Test safe navigation to results screen
            can_navigate = SessionStateValidator.safe_navigate_to_screen('results')
            
            if can_navigate:
                return {
                    'passed': False,
                    'message': "Safe navigation allowed access to results with corrupted data",
                    'critical': True
                }
            
            return {
                'passed': True,
                'message': "Corrupted processing results properly detected and blocked"
            }
            
        finally:
            # Restore original state
            if original_results is not None:
                st.session_state.processing_results = original_results
            elif 'processing_results' in st.session_state:
                del st.session_state['processing_results']
    
    @classmethod
    def _test_invalid_hotspot_session(cls) -> Dict[str, Any]:
        """Test behavior with invalid hotspot session"""
        # Backup original session
        original_session = st.session_state.get('hotspot_session')
        
        try:
            # Set invalid hotspot session
            st.session_state.hotspot_session = {'invalid': True}  # Missing required fields
            
            # Test validation
            is_valid = SessionStateValidator.validate_hotspot_session()
            
            if is_valid:
                return {
                    'passed': False,
                    'message': "Validator failed to detect invalid hotspot session",
                    'critical': True
                }
            
            return {
                'passed': True,
                'message': "Invalid hotspot session properly detected"
            }
            
        finally:
            # Restore original state
            if original_session is not None:
                st.session_state.hotspot_session = original_session
            elif 'hotspot_session' in st.session_state:
                del st.session_state['hotspot_session']
    
    @classmethod
    def _test_results_screen_access(cls) -> Dict[str, Any]:
        """Test results screen access validation"""
        # Test without processing results
        original_results = st.session_state.get('processing_results')
        
        try:
            # Clear processing results
            if 'processing_results' in st.session_state:
                del st.session_state['processing_results']
            
            # Test access validation
            can_access, error_msg = SessionStateValidator.validate_for_results_screen()
            
            if can_access:
                return {
                    'passed': False,
                    'message': "Results screen access allowed without processing results",
                    'critical': True
                }
            
            # Test with valid results
            st.session_state.processing_results = {
                'file_name': 'test.mp4',
                'transcript': 'Test transcript',
                'duration': '2:30',
                'speakers': ['Speaker 1']
            }
            
            can_access, _ = SessionStateValidator.validate_for_results_screen()
            
            if not can_access:
                return {
                    'passed': False,
                    'message': "Results screen access blocked with valid processing results",
                    'critical': True
                }
            
            return {
                'passed': True,
                'message': "Results screen access validation working correctly"
            }
            
        finally:
            # Restore original state
            if original_results is not None:
                st.session_state.processing_results = original_results
            elif 'processing_results' in st.session_state:
                del st.session_state['processing_results']
    
    @classmethod
    def _test_hotspot_review_access(cls) -> Dict[str, Any]:
        """Test hotspot review access validation"""
        # Test without processing results
        original_results = st.session_state.get('processing_results')
        
        try:
            # Clear processing results
            if 'processing_results' in st.session_state:
                del st.session_state['processing_results']
            
            # Test access validation
            can_access, error_msg = SessionStateValidator.validate_for_hotspot_review()
            
            if can_access:
                return {
                    'passed': False,
                    'message': "Hotspot review access allowed without processing results",
                    'critical': True
                }
            
            # Test with incomplete results
            st.session_state.processing_results = {
                'file_name': 'test.mp4',
                'transcript': 'Test transcript'
                # Missing full_results
            }
            
            can_access, _ = SessionStateValidator.validate_for_hotspot_review()
            
            if can_access:
                return {
                    'passed': False,
                    'message': "Hotspot review access allowed with incomplete results",
                    'critical': True
                }
            
            # Test with valid complete results
            st.session_state.processing_results = {
                'file_name': 'test.mp4',
                'transcript': 'Test transcript',
                'duration': '2:30',
                'speakers': ['Speaker 1'],
                'full_results': {'segments': []}
            }
            
            can_access, _ = SessionStateValidator.validate_for_hotspot_review()
            
            if not can_access:
                return {
                    'passed': False,
                    'message': "Hotspot review access blocked with valid complete results",
                    'critical': False
                }
            
            return {
                'passed': True,
                'message': "Hotspot review access validation working correctly"
            }
            
        finally:
            # Restore original state
            if original_results is not None:
                st.session_state.processing_results = original_results
            elif 'processing_results' in st.session_state:
                del st.session_state['processing_results']
    
    @classmethod
    def _test_navigation_validation(cls) -> Dict[str, Any]:
        """Test safe navigation functionality"""
        try:
            # Test invalid screen navigation
            result = SessionStateValidator.safe_navigate_to_screen('invalid_screen')
            
            if result:
                return {
                    'passed': False,
                    'message': "Safe navigation allowed invalid screen",
                    'critical': True
                }
            
            # Test valid screen navigation
            result = SessionStateValidator.safe_navigate_to_screen('landing', validate_access=False)
            
            if not result:
                return {
                    'passed': False,
                    'message': "Safe navigation blocked valid screen without validation",
                    'critical': True
                }
            
            return {
                'passed': True,
                'message': "Safe navigation working correctly"
            }
            
        except Exception as e:
            return {
                'passed': False,
                'message': f"Navigation validation failed with error: {str(e)}",
                'critical': True
            }
    
    @classmethod
    def _test_auto_repair_functionality(cls) -> Dict[str, Any]:
        """Test automatic repair functionality"""
        # Backup original state
        original_state = dict(st.session_state)
        
        try:
            # Introduce some issues
            if 'current_screen' in st.session_state:
                del st.session_state['current_screen']
            
            st.session_state.current_screen = 'invalid_screen'  # Invalid screen
            
            # Run auto-repair
            repairs = SessionStateValidator.auto_repair_session_state()
            
            if len(repairs) == 0:
                return {
                    'passed': False,
                    'message': "Auto-repair did not detect/fix any issues",
                    'critical': False
                }
            
            # Validate repairs worked
            if not SessionStateValidator.validate_current_screen():
                return {
                    'passed': False,
                    'message': "Auto-repair failed to fix current_screen issue",
                    'critical': True
                }
            
            return {
                'passed': True,
                'message': "Auto-repair functionality working correctly",
                'details': {'repairs_made': repairs}
            }
            
        finally:
            # Restore original state
            st.session_state.clear()
            st.session_state.update(original_state)
    
    @classmethod
    def _test_emergency_recovery(cls) -> Dict[str, Any]:
        """Test emergency recovery functionality"""
        # Backup original state
        original_state = dict(st.session_state)
        
        try:
            # Corrupt session state severely
            st.session_state.clear()
            st.session_state.update({'corrupted': True, 'invalid_data': None})
            
            # Test reset functionality
            SessionStateValidator.reset_session_state(preserve_keys=['app_initialized'])
            
            # Validate recovery
            validation_results = SessionStateValidator.validate_all_required()
            missing_keys = [k for k, v in validation_results.items() if not v]
            
            if missing_keys:
                return {
                    'passed': False,
                    'message': f"Emergency recovery failed - missing keys: {missing_keys}",
                    'critical': True
                }
            
            return {
                'passed': True,
                'message': "Emergency recovery working correctly"
            }
            
        finally:
            # Restore original state
            st.session_state.clear()
            st.session_state.update(original_state)
    
    @classmethod
    def generate_test_report(cls, test_results: Dict[str, Any]) -> str:
        """Generate a human-readable test report"""
        report = []
        report.append("# Session State Validation Test Report")
        report.append(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(test_results['timestamp']))}")
        report.append("")
        
        # Summary
        report.append("## Summary")
        report.append(f"- **Tests Run:** {test_results['tests_run']}")
        report.append(f"- **Tests Passed:** {test_results['tests_passed']}")
        report.append(f"- **Tests Failed:** {test_results['tests_failed']}")
        report.append(f"- **Success Rate:** {test_results['success_rate']:.1%}")
        report.append("")
        
        # Critical Issues
        if test_results['critical_issues']:
            report.append("## ⚠️ Critical Issues")
            for issue in test_results['critical_issues']:
                report.append(f"- **{issue['test']}:** {issue['issue']}")
            report.append("")
        
        # Test Details
        report.append("## Test Details")
        for test in test_results['test_details']:
            status = "✅ PASS" if test['passed'] else "❌ FAIL"
            report.append(f"### {test['test']} {status}")
            report.append(f"**Message:** {test['message']}")
            if test.get('details'):
                report.append(f"**Details:** {test['details']}")
            report.append("")
        
        return "\n".join(report)