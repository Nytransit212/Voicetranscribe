#!/usr/bin/env python3
"""
Production Readiness Verification for Multi-Provider Consensus System
======================================================================

This script verifies that the consensus system meets production reliability requirements:
1. Proper quorum gating with configurable thresholds
2. Comprehensive provider metadata tracking
3. Robust error handling for provider failures
4. Production-ready configuration validation

Usage:
    python verification_consensus_production_readiness.py
"""

import sys
import json
import time
import traceback
from typing import Dict, Any, List, Optional
from core.consensus_module import (
    ConsensusModule, 
    QuorumValidationError, 
    ProviderParticipation, 
    QuorumStatus,
    ConsensusResult
)

class ConsensusProductionVerifier:
    """Comprehensive verification of consensus system production readiness"""
    
    def __init__(self):
        self.test_results = []
        self.passed = 0
        self.failed = 0
        
    def log_test(self, test_name: str, passed: bool, details: str = ""):
        """Log test result"""
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {test_name}")
        if details:
            print(f"    {details}")
        
        self.test_results.append({
            'test': test_name,
            'passed': passed,
            'details': details,
            'timestamp': time.time()
        })
        
        if passed:
            self.passed += 1
        else:
            self.failed += 1
    
    def create_mock_candidates(self, count: int, providers: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Create mock candidates for testing"""
        if providers is None:
            providers = ['openai', 'faster_whisper', 'assemblyai', 'deepgram', 'google']
        
        candidates = []
        for i in range(count):
            provider = providers[i % len(providers)]
            confidence = 0.7 + (i * 0.05)  # Varying confidence scores
            
            candidate = {
                'candidate_id': f'candidate_{i}',
                'asr_data': {
                    'provider': provider,
                    'model_name': f'{provider}_model_v1',
                    'decode_mode': 'standard',
                    'processing_time': 1.0 + (i * 0.1),
                    'metadata': {'version': '1.0', 'region': 'us-east-1'}
                },
                'confidence_scores': {
                    'D_diarization': confidence,
                    'A_asr_alignment': confidence + 0.05,
                    'L_linguistic': confidence - 0.02,
                    'R_agreement': confidence + 0.01,
                    'O_overlap': confidence - 0.01,
                    'final_score': confidence
                },
                'aligned_segments': [
                    {
                        'start': 0.0,
                        'end': 5.0,
                        'text': f'Test transcript from {provider}',
                        'confidence': confidence
                    }
                ]
            }
            candidates.append(candidate)
        
        return candidates
    
    def test_quorum_gating(self):
        """Test 1: Verify quorum gating works correctly"""
        print("\n🔍 TESTING QUORUM GATING")
        
        # Test with quorum enabled
        consensus = ConsensusModule(minimum_candidates=3, enable_quorum_gating=True)
        
        # Test insufficient candidates
        try:
            candidates = self.create_mock_candidates(2)  # Only 2 candidates
            result = consensus.process_consensus(candidates)
            self.log_test("Quorum validation with insufficient candidates", 
                         False, "Should have raised QuorumValidationError")
        except QuorumValidationError as e:
            self.log_test("Quorum validation with insufficient candidates", 
                         True, f"Correctly rejected {e.quorum_status.participants_count}/{e.quorum_status.minimum_required}")
        except Exception as e:
            self.log_test("Quorum validation with insufficient candidates", 
                         False, f"Unexpected error: {e}")
        
        # Test sufficient candidates
        try:
            candidates = self.create_mock_candidates(5)  # 5 candidates
            result = consensus.process_consensus(candidates)
            quorum_met = result.quorum_status.quorum_met
            self.log_test("Quorum validation with sufficient candidates", 
                         quorum_met, f"Quorum status: {result.quorum_status.participants_count}/{result.quorum_status.minimum_required}")
        except Exception as e:
            self.log_test("Quorum validation with sufficient candidates", 
                         False, f"Unexpected error: {e}")
        
        # Test force consensus bypass
        try:
            candidates = self.create_mock_candidates(1)  # Only 1 candidate
            result = consensus.process_consensus(candidates, force_consensus=True)
            self.log_test("Force consensus bypass", 
                         True, "Successfully bypassed quorum with force_consensus=True")
        except Exception as e:
            self.log_test("Force consensus bypass", 
                         False, f"Failed to bypass: {e}")
    
    def test_provider_metadata_tracking(self):
        """Test 2: Verify comprehensive provider metadata tracking"""
        print("\n🔍 TESTING PROVIDER METADATA TRACKING")
        
        consensus = ConsensusModule(minimum_candidates=2)
        candidates = self.create_mock_candidates(4, ['openai', 'assemblyai', 'deepgram', 'google'])
        
        try:
            result = consensus.process_consensus(candidates)
            
            # Check provider participation tracking
            participation = result.provider_participation
            self.log_test("Provider participation tracking", 
                         len(participation) == 4, 
                         f"Tracked {len(participation)} providers")
            
            # Verify provider metadata completeness
            complete_metadata = all(
                p.provider_name and p.model_name and p.processing_time > 0
                for p in participation
            )
            self.log_test("Provider metadata completeness", 
                         complete_metadata, 
                         "All providers have complete metadata")
            
            # Check quorum status tracking
            quorum_status = result.quorum_status
            self.log_test("Quorum status tracking", 
                         hasattr(quorum_status, 'participating_providers'), 
                         f"Quorum: {quorum_status.participants_count}/{quorum_status.minimum_required}")
            
            # Verify unique providers
            unique_providers = set(p.provider_name for p in participation)
            self.log_test("Provider diversity tracking", 
                         len(unique_providers) == 4, 
                         f"Tracked {len(unique_providers)} unique providers")
            
        except Exception as e:
            self.log_test("Provider metadata tracking", False, f"Error: {e}")
    
    def test_provider_failure_handling(self):
        """Test 3: Verify provider failure handling"""
        print("\n🔍 TESTING PROVIDER FAILURE HANDLING")
        
        consensus = ConsensusModule(minimum_candidates=2, require_provider_diversity=True)
        
        # Create candidates with some missing provider info (simulating failures)
        candidates = self.create_mock_candidates(3)
        
        # Corrupt one candidate to simulate provider failure
        candidates[1]['asr_data'] = {}  # Missing provider info
        
        try:
            result = consensus.process_consensus(candidates)
            
            # Check that failed providers are tracked
            failed_providers = [p for p in result.provider_participation if not p.success]
            self.log_test("Failed provider tracking", 
                         len(failed_providers) > 0, 
                         f"Tracked {len(failed_providers)} failed providers")
            
            # Verify consensus still works with partial failures
            self.log_test("Consensus with partial failures", 
                         result.consensus_confidence > 0, 
                         f"Consensus confidence: {result.consensus_confidence:.3f}")
            
        except Exception as e:
            self.log_test("Provider failure handling", False, f"Error: {e}")
    
    def test_consensus_strategies(self):
        """Test 4: Verify consensus strategies work correctly"""
        print("\n🔍 TESTING CONSENSUS STRATEGIES")
        
        consensus = ConsensusModule(minimum_candidates=2)
        candidates = self.create_mock_candidates(4)
        
        strategies = [
            'best_single_candidate',
            'weighted_voting', 
            'multidimensional_consensus',
            'confidence_based'
        ]
        
        for strategy in strategies:
            try:
                result = consensus.process_consensus(candidates, strategy=strategy)
                
                success = (
                    result.winner_candidate is not None and
                    result.consensus_confidence > 0 and
                    result.consensus_method == strategy
                )
                
                self.log_test(f"Consensus strategy: {strategy}", 
                             success, 
                             f"Confidence: {result.consensus_confidence:.3f}")
                
            except Exception as e:
                self.log_test(f"Consensus strategy: {strategy}", 
                             False, f"Error: {e}")
    
    def test_error_handling_and_fallbacks(self):
        """Test 5: Verify error handling and fallback mechanisms"""
        print("\n🔍 TESTING ERROR HANDLING AND FALLBACKS")
        
        consensus = ConsensusModule(
            minimum_candidates=2, 
            fallback_strategy='best_single_candidate'
        )
        
        # Test fallback when strategy fails
        candidates = self.create_mock_candidates(3)
        
        try:
            # Try an invalid strategy (should fallback)
            result = consensus.process_consensus(candidates, strategy='invalid_strategy')
            self.log_test("Invalid strategy fallback", False, "Should have raised ValueError")
        except ValueError:
            self.log_test("Invalid strategy fallback", True, "Correctly rejected invalid strategy")
        except Exception as e:
            self.log_test("Invalid strategy fallback", False, f"Unexpected error: {e}")
        
        # Test metrics tracking
        metrics = consensus.get_consensus_metrics()
        self.log_test("Metrics tracking", 
                     'consensus_metrics' in metrics, 
                     "Consensus metrics are tracked")
        
        # Test configuration updates
        try:
            consensus.update_configuration(minimum_candidates=5)
            updated_min = consensus.minimum_candidates
            self.log_test("Configuration updates", 
                         updated_min == 5, 
                         f"Updated minimum_candidates to {updated_min}")
        except Exception as e:
            self.log_test("Configuration updates", False, f"Error: {e}")
    
    def test_production_configuration(self):
        """Test 6: Verify production-ready configuration"""
        print("\n🔍 TESTING PRODUCTION CONFIGURATION")
        
        # Test recommended production settings
        consensus = ConsensusModule(
            minimum_candidates=3,
            enable_quorum_gating=True,
            require_provider_diversity=True,
            fallback_strategy='best_single_candidate'
        )
        
        config = consensus.get_consensus_metrics()['configuration']
        
        # Verify production settings
        prod_ready = (
            config['minimum_candidates'] >= 3 and
            config['enable_quorum_gating'] and
            config['require_provider_diversity'] and
            config['fallback_strategy'] == 'best_single_candidate'
        )
        
        self.log_test("Production configuration", 
                     prod_ready, 
                     f"Config: min_candidates={config['minimum_candidates']}, quorum={config['enable_quorum_gating']}")
        
        # Test configuration validation
        try:
            consensus.update_configuration(minimum_candidates=0)  # Invalid
            updated_min = consensus.minimum_candidates
            self.log_test("Configuration validation", 
                         updated_min >= 1, 
                         f"Minimum candidates clamped to {updated_min}")
        except Exception as e:
            self.log_test("Configuration validation", False, f"Error: {e}")
    
    def run_all_tests(self):
        """Run all verification tests"""
        print("=" * 70)
        print("🚀 CONSENSUS SYSTEM PRODUCTION READINESS VERIFICATION")
        print("=" * 70)
        
        try:
            self.test_quorum_gating()
            self.test_provider_metadata_tracking()
            self.test_provider_failure_handling()
            self.test_consensus_strategies()
            self.test_error_handling_and_fallbacks()
            self.test_production_configuration()
            
            print("\n" + "=" * 70)
            print("📊 VERIFICATION SUMMARY")
            print("=" * 70)
            print(f"✅ Tests Passed: {self.passed}")
            print(f"❌ Tests Failed: {self.failed}")
            print(f"📈 Success Rate: {(self.passed / (self.passed + self.failed) * 100):.1f}%")
            
            if self.failed == 0:
                print("\n🎉 ALL TESTS PASSED - CONSENSUS SYSTEM IS PRODUCTION READY!")
                return True
            else:
                print(f"\n⚠️  {self.failed} TEST(S) FAILED - REVIEW REQUIRED")
                return False
                
        except Exception as e:
            print(f"\n💥 VERIFICATION FAILED: {e}")
            traceback.print_exc()
            return False
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate detailed verification report"""
        return {
            'verification_timestamp': time.time(),
            'total_tests': len(self.test_results),
            'passed_tests': self.passed,
            'failed_tests': self.failed,
            'success_rate': (self.passed / (self.passed + self.failed) * 100) if (self.passed + self.failed) > 0 else 0,
            'production_ready': self.failed == 0,
            'test_results': self.test_results,
            'recommendations': self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate production deployment recommendations"""
        recommendations = []
        
        if self.failed == 0:
            recommendations.extend([
                "✅ Consensus system is production ready",
                "✅ Deploy with minimum_candidates=3 for reliability",
                "✅ Enable quorum_gating=True for production safety",
                "✅ Use require_provider_diversity=True for redundancy",
                "✅ Monitor consensus metrics for ongoing reliability"
            ])
        else:
            recommendations.extend([
                "⚠️ Address test failures before production deployment",
                "⚠️ Review provider metadata tracking implementation",
                "⚠️ Validate quorum gating configuration",
                "⚠️ Test fallback mechanisms under load"
            ])
        
        return recommendations

def main():
    """Run consensus production readiness verification"""
    verifier = ConsensusProductionVerifier()
    
    try:
        success = verifier.run_all_tests()
        
        # Generate and save detailed report
        report = verifier.generate_report()
        
        with open('consensus_production_readiness_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Detailed report saved to: consensus_production_readiness_report.json")
        
        return 0 if success else 1
        
    except Exception as e:
        print(f"💥 Verification script failed: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())