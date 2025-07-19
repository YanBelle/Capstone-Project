"""
Practical Enhancement for ml_analyzer.py - Phase 1 Implementation
Replace rigid expert overrides with confidence adjustment approach
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

logger = logging.getLogger(__name__)


class KnowledgeGuidedConfidenceAdjuster:
    """
    Replaces binary expert overrides with confidence adjustment system
    This allows ML to learn while incorporating domain knowledge
    """
    
    def __init__(self):
        self.adjustment_rules = self.load_adjustment_rules()
        self.pattern_weights = self.load_pattern_weights()
        
    def load_adjustment_rules(self) -> Dict:
        """Load domain knowledge as confidence adjustment rules"""
        return {
            'normal_patterns': {
                'successful_withdrawal': {
                    'indicators': ['NOTES PRESENTED', 'NOTES TAKEN'],
                    'required_sequence': True,
                    'confidence_reduction': 0.15,  # Reduce anomaly confidence to 15%
                    'min_confidence': 0.05  # Never go below 5% (preserve learning signal)
                },
                'successful_inquiry': {
                    'indicators': ['CARD INSERTED', 'CARD TAKEN'],
                    'exclusions': ['UNABLE TO DISPENSE', 'ERROR'],
                    'confidence_reduction': 0.20,
                    'min_confidence': 0.05
                },
                'normal_maintenance': {
                    'indicators': ['POWER-UP/RESET', 'SYSTEM STARTUP'],
                    'confidence_reduction': 0.30,
                    'min_confidence': 0.10
                }
            },
            
            'error_patterns': {
                'dispense_failure': {
                    'indicators': ['UNABLE TO DISPENSE'],
                    'confidence_boost': 1.4,  # Increase confidence by 40%
                    'max_confidence': 0.95
                },
                'hardware_error': {
                    'indicators': ['DEVICE ERROR', 'HARDWARE FAULT', 'SENSOR ERROR'],
                    'confidence_boost': 1.3,
                    'max_confidence': 0.90
                },
                'security_issue': {
                    'indicators': ['SUPERVISOR MODE ENTRY', 'UNAUTHORIZED'],
                    'confidence_boost': 1.5,
                    'max_confidence': 0.98
                }
            },
            
            'contextual_adjustments': {
                'high_activity_period': {
                    'indicators': ['multiple_sessions_per_minute'],
                    'confidence_reduction': 0.90,  # Slight reduction during busy periods
                    'min_confidence': 0.20
                },
                'diagnostic_mode': {
                    'indicators': ['[000p', 'DIAGNOSTIC'],
                    'confidence_reduction': 0.70,
                    'min_confidence': 0.15
                }
            }
        }
    
    def load_pattern_weights(self) -> Dict:
        """Load weights for different types of evidence"""
        return {
            'sequence_match_weight': 0.8,      # Strong evidence from sequence matching
            'keyword_match_weight': 0.6,       # Moderate evidence from keywords
            'contextual_weight': 0.4,          # Weaker evidence from context
            'temporal_weight': 0.3,            # Temporal pattern evidence
            'frequency_weight': 0.5            # Pattern frequency evidence
        }
    
    def adjust_ml_confidence(self, session, ml_score: float) -> Dict:
        """
        Adjust ML confidence using domain knowledge
        Returns comprehensive adjustment information
        """
        
        adjustment_result = {
            'original_ml_score': ml_score,
            'adjusted_score': ml_score,
            'adjustment_factor': 1.0,
            'applied_rules': [],
            'confidence_in_adjustment': 0.5,
            'reasoning': [],
            'knowledge_signals': []
        }
        
        session_text = session.raw_text.upper()
        
        # Apply normal pattern adjustments
        normal_adjustment = self._apply_normal_pattern_adjustments(session_text)
        if normal_adjustment['applied']:
            adjustment_result['adjustment_factor'] *= normal_adjustment['factor']
            adjustment_result['applied_rules'].extend(normal_adjustment['rules'])
            adjustment_result['reasoning'].extend(normal_adjustment['reasoning'])
            adjustment_result['knowledge_signals'].extend(normal_adjustment['signals'])
        
        # Apply error pattern adjustments
        error_adjustment = self._apply_error_pattern_adjustments(session_text)
        if error_adjustment['applied']:
            adjustment_result['adjustment_factor'] *= error_adjustment['factor']
            adjustment_result['applied_rules'].extend(error_adjustment['rules'])
            adjustment_result['reasoning'].extend(error_adjustment['reasoning'])
            adjustment_result['knowledge_signals'].extend(error_adjustment['signals'])
        
        # Apply contextual adjustments
        contextual_adjustment = self._apply_contextual_adjustments(session, session_text)
        if contextual_adjustment['applied']:
            adjustment_result['adjustment_factor'] *= contextual_adjustment['factor']
            adjustment_result['applied_rules'].extend(contextual_adjustment['rules'])
            adjustment_result['reasoning'].extend(contextual_adjustment['reasoning'])
            adjustment_result['knowledge_signals'].extend(contextual_adjustment['signals'])
        
        # Calculate final adjusted score
        adjustment_result['adjusted_score'] = self._calculate_final_score(
            ml_score, 
            adjustment_result['adjustment_factor']
        )
        
        # Calculate confidence in our adjustment
        adjustment_result['confidence_in_adjustment'] = self._calculate_adjustment_confidence(
            adjustment_result['applied_rules'],
            adjustment_result['knowledge_signals']
        )
        
        return adjustment_result
    
    def _apply_normal_pattern_adjustments(self, session_text: str) -> Dict:
        """Apply adjustments for known normal patterns"""
        
        result = {
            'applied': False,
            'factor': 1.0,
            'rules': [],
            'reasoning': [],
            'signals': []
        }
        
        for pattern_name, pattern_config in self.adjustment_rules['normal_patterns'].items():
            match_strength = self._calculate_pattern_match_strength(session_text, pattern_config)
            
            if match_strength > 0.5:  # Threshold for pattern match
                confidence_reduction = pattern_config['confidence_reduction']
                min_confidence = pattern_config['min_confidence']
                
                # Calculate adjustment factor (never go below minimum)
                adjustment_factor = max(confidence_reduction, min_confidence)
                
                result['applied'] = True
                result['factor'] *= adjustment_factor
                result['rules'].append(pattern_name)
                result['reasoning'].append(
                    f"Matches normal pattern '{pattern_name}' with {match_strength:.2f} strength"
                )
                result['signals'].append({
                    'type': 'normal_pattern',
                    'pattern': pattern_name,
                    'strength': match_strength,
                    'adjustment': adjustment_factor
                })
                
                logger.info(f"Applied normal pattern adjustment: {pattern_name} "
                          f"(strength: {match_strength:.2f}, factor: {adjustment_factor:.2f})")
        
        return result
    
    def _apply_error_pattern_adjustments(self, session_text: str) -> Dict:
        """Apply adjustments for known error patterns"""
        
        result = {
            'applied': False,
            'factor': 1.0,
            'rules': [],
            'reasoning': [],
            'signals': []
        }
        
        for pattern_name, pattern_config in self.adjustment_rules['error_patterns'].items():
            match_strength = self._calculate_pattern_match_strength(session_text, pattern_config)
            
            if match_strength > 0.3:  # Lower threshold for error patterns
                confidence_boost = pattern_config['confidence_boost']
                max_confidence = pattern_config['max_confidence']
                
                # Calculate boost factor (never exceed maximum)
                boost_factor = min(confidence_boost, max_confidence / 0.5)  # Assuming base 0.5
                
                result['applied'] = True
                result['factor'] *= boost_factor
                result['rules'].append(pattern_name)
                result['reasoning'].append(
                    f"Matches error pattern '{pattern_name}' with {match_strength:.2f} strength"
                )
                result['signals'].append({
                    'type': 'error_pattern',
                    'pattern': pattern_name,
                    'strength': match_strength,
                    'adjustment': boost_factor
                })
                
                logger.info(f"Applied error pattern boost: {pattern_name} "
                          f"(strength: {match_strength:.2f}, factor: {boost_factor:.2f})")
        
        return result
    
    def _apply_contextual_adjustments(self, session, session_text: str) -> Dict:
        """Apply contextual adjustments based on session characteristics"""
        
        result = {
            'applied': False,
            'factor': 1.0,
            'rules': [],
            'reasoning': [],
            'signals': []
        }
        
        # Check for diagnostic mode
        diagnostic_count = session_text.count('[000P')
        if diagnostic_count > 20:  # High diagnostic activity
            context_config = self.adjustment_rules['contextual_adjustments']['diagnostic_mode']
            adjustment_factor = context_config['confidence_reduction']
            
            result['applied'] = True
            result['factor'] *= adjustment_factor
            result['rules'].append('diagnostic_mode')
            result['reasoning'].append(
                f"High diagnostic activity detected ({diagnostic_count} patterns)"
            )
            result['signals'].append({
                'type': 'contextual',
                'context': 'diagnostic_mode',
                'evidence': diagnostic_count,
                'adjustment': adjustment_factor
            })
        
        # Check for session length anomalies
        session_length = len(session.raw_text)
        if session_length > 50000:  # Very long session
            result['applied'] = True
            result['factor'] *= 0.80  # Slight reduction for very long sessions
            result['rules'].append('long_session')
            result['reasoning'].append(f"Very long session ({session_length} characters)")
            result['signals'].append({
                'type': 'contextual',
                'context': 'long_session',
                'evidence': session_length,
                'adjustment': 0.80
            })
        
        return result
    
    def _calculate_pattern_match_strength(self, session_text: str, pattern_config: Dict) -> float:
        """Calculate how strongly a session matches a pattern"""
        
        indicators = pattern_config.get('indicators', [])
        exclusions = pattern_config.get('exclusions', [])
        
        if not indicators:
            return 0.0
        
        # Check indicator presence
        indicators_found = sum(1 for indicator in indicators if indicator in session_text)
        indicator_ratio = indicators_found / len(indicators)
        
        # Check for exclusions (reduce strength if found)
        exclusion_penalty = 0.0
        if exclusions:
            exclusions_found = sum(1 for exclusion in exclusions if exclusion in session_text)
            exclusion_penalty = exclusions_found * 0.3  # 30% penalty per exclusion
        
        # Calculate final strength
        strength = max(0.0, indicator_ratio - exclusion_penalty)
        
        # Bonus for sequence matching if required
        if pattern_config.get('required_sequence', False):
            if self._check_sequence_order(session_text, indicators):
                strength *= 1.2  # 20% bonus for correct sequence
        
        return min(1.0, strength)
    
    def _check_sequence_order(self, session_text: str, indicators: List[str]) -> bool:
        """Check if indicators appear in the expected sequence"""
        
        positions = []
        for indicator in indicators:
            pos = session_text.find(indicator)
            if pos >= 0:
                positions.append(pos)
            else:
                return False  # Missing indicator
        
        # Check if positions are in ascending order
        return positions == sorted(positions)
    
    def _calculate_final_score(self, ml_score: float, adjustment_factor: float) -> float:
        """Calculate final adjusted score with bounds checking"""
        
        adjusted = ml_score * adjustment_factor
        
        # Ensure bounds
        adjusted = max(0.0, min(1.0, adjusted))
        
        return adjusted
    
    def _calculate_adjustment_confidence(self, applied_rules: List[str], 
                                       knowledge_signals: List[Dict]) -> float:
        """Calculate confidence in the adjustment decision"""
        
        if not applied_rules:
            return 0.5  # No knowledge applied - moderate confidence
        
        # Base confidence from number of rules applied
        base_confidence = min(0.9, 0.4 + len(applied_rules) * 0.2)
        
        # Boost confidence based on signal strength
        avg_signal_strength = 0.0
        if knowledge_signals:
            total_strength = sum(signal.get('strength', 0.5) for signal in knowledge_signals)
            avg_signal_strength = total_strength / len(knowledge_signals)
        
        final_confidence = min(0.95, base_confidence + avg_signal_strength * 0.3)
        
        return final_confidence


# Integration function to add to existing ml_analyzer.py
def enhance_existing_analyzer_with_confidence_adjustment():
    """
    Function to integrate confidence adjustment into existing ml_analyzer.py
    
    USAGE:
    1. Add this class to your ml_analyzer.py file
    2. In __init__, add: self.confidence_adjuster = KnowledgeGuidedConfidenceAdjuster()
    3. Replace apply_expert_override calls with confidence adjustment
    """
    
    enhancement_code = """
    
    # Add to __init__ method:
    self.confidence_adjuster = KnowledgeGuidedConfidenceAdjuster()
    
    # Replace in detect_anomalies_unsupervised method:
    # OLD CODE:
    # if self.apply_expert_override(session):
    #     session.is_anomaly = False
    #     session.anomaly_score = 0.0
    #     return
    
    # NEW CODE:
    adjustment_result = self.confidence_adjuster.adjust_ml_confidence(session, combined_ml_score)
    
    session.ml_anomaly_score = combined_ml_score  # Original ML score
    session.knowledge_adjusted_score = adjustment_result['adjusted_score']  # Adjusted score
    session.anomaly_score = adjustment_result['adjusted_score']  # Final score
    session.is_anomaly = adjustment_result['adjusted_score'] > 0.5
    
    # Store adjustment details for transparency
    session.knowledge_adjustment = {
        'applied_rules': adjustment_result['applied_rules'],
        'reasoning': adjustment_result['reasoning'],
        'confidence': adjustment_result['confidence_in_adjustment'],
        'adjustment_factor': adjustment_result['adjustment_factor']
    }
    
    # Log the adjustment for monitoring
    if adjustment_result['applied_rules']:
        logger.info(f"Session {session.session_id}: ML={combined_ml_score:.3f} -> "
                   f"Adjusted={adjustment_result['adjusted_score']:.3f} "
                   f"(Rules: {', '.join(adjustment_result['applied_rules'])})")
    """
    
    print("Integration code generated. Copy the code above into your ml_analyzer.py")
    return enhancement_code


if __name__ == "__main__":
    # Example usage
    adjuster = KnowledgeGuidedConfidenceAdjuster()
    
    # Simulate a session with successful withdrawal
    class MockSession:
        def __init__(self, text):
            self.raw_text = text
            self.session_id = "test_001"
    
    # Test normal pattern
    normal_session = MockSession("""
    CARD INSERTED
    PIN ENTERED
    NOTES PRESENTED
    NOTES TAKEN
    CARD TAKEN
    """)
    
    result = adjuster.adjust_ml_confidence(normal_session, 0.8)
    print("Normal Pattern Test:")
    print(f"Original ML Score: {result['original_ml_score']}")
    print(f"Adjusted Score: {result['adjusted_score']}")
    print(f"Reasoning: {result['reasoning']}")
    print()
    
    # Test error pattern
    error_session = MockSession("""
    CARD INSERTED
    PIN ENTERED
    UNABLE TO DISPENSE
    DEVICE ERROR
    """)
    
    result = adjuster.adjust_ml_confidence(error_session, 0.6)
    print("Error Pattern Test:")
    print(f"Original ML Score: {result['original_ml_score']}")
    print(f"Adjusted Score: {result['adjusted_score']}")
    print(f"Reasoning: {result['reasoning']}")
