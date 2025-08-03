# ML Analyzer Alias File - Re-exports from the actual implementation
"""
This file provides an alias to the ML analyzer implementations
to make imports work from the root directory.

Enhanced with improved false positive prevention for:
- Card initialization attempts (magstrip fallback)
- Customer cancellations (normal user behavior)
- Better detection of actual anomalies
"""

try:
    # Try to import from the anomaly detector service first
    from services.anomaly_detector.ml_analyzer import MLFirstAnomalyDetector, TransactionSession
    print("Imported ML analyzer from anomaly-detector service")
except ImportError:
    try:
        # Fallback to API service implementation
        from services.api.ml_analyzer import MLFirstAnomalyDetector, TransactionSession
        print("Imported ML analyzer from API service")
    except ImportError:
        raise ImportError("Could not import ML analyzer from either service. Please check the service implementations.")

import re
from typing import List, Dict, Any

# Create an enhanced alias class with improved false positive prevention
class MLAnomalyAnalyzer(MLFirstAnomalyDetector):
    """
    Enhanced ML analyzer with improved false positive prevention for ATM EJ logs.
    
    Key improvements:
    1. Card initialization attempts (3 attempts) are NOT treated as errors (magstrip fallback)
    2. Customer cancellations are NOT treated as errors (normal user behavior)
    3. Better detection of actual anomalies like incomplete transactions
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Override with enhanced false positive prevention rules
        self._setup_enhanced_false_positive_prevention()
    
    def _setup_enhanced_false_positive_prevention(self):
        """Setup enhanced false positive prevention rules"""
        
        # Enhanced normal patterns that should NOT be flagged as anomalies
        self.enhanced_normal_patterns = {
            # Card initialization attempts - normal for magstrip cards
            "card_init_attempts": {
                "pattern": re.compile(r'\*7235\*1\*D\*9,M-81,R-0\s+CARD INITIALISE ATTEMPT = [123]', re.MULTILINE),
                "max_attempts": 3,
                "description": "Normal magstrip card fallback - chip read attempts before magstrip",
                "override_reason": "Magstrip cards require 3 chip read attempts before fallback"
            },
            
            # Customer cancellations - normal user behavior
            "customer_cancelled": {
                "pattern": re.compile(r'CUSTOMER CANCELLED', re.IGNORECASE),
                "description": "Normal user-initiated transaction cancellation",
                "override_reason": "Customer cancellation is normal user behavior, not a system error"
            },
            
            # Successful transactions with normal flows
            "successful_withdrawal": {
                "pattern": re.compile(r'NOTES PRESENTED.*NOTES TAKEN', re.DOTALL | re.IGNORECASE),
                "description": "Successful cash withdrawal completion",
                "override_reason": "Notes presented and taken indicates successful transaction"
            }
        }
        
        # Enhanced anomaly patterns that SHOULD be flagged
        self.enhanced_anomaly_patterns = {
            # Critical operational issues
            "supervisor_mode_entry": {
                "pattern": re.compile(r'SUPERVISOR MODE ENTRY', re.IGNORECASE),
                "severity": "high",
                "description": "Supervisor intervention required during transaction"
            },
            
            "power_reset": {
                "pattern": re.compile(r'Power-Up/Reset', re.IGNORECASE),
                "severity": "high", 
                "description": "System power reset during or after transaction"
            },
            
            "unable_to_dispense": {
                "pattern": re.compile(r'UNABLE TO DISPENSE', re.IGNORECASE),
                "severity": "critical",
                "description": "Cash dispensing failure"
            },
            
            "deposit_error": {
                "pattern": re.compile(r'DEPOSIT ERROR', re.IGNORECASE),
                "severity": "high",
                "description": "Cash deposit processing error"
            },
            
            # Cash retract errors - Critical customer service issue
            "cash_deposit_retract": {
                "pattern": re.compile(r'(INIT BNA STARTED - RETRACT BIN|CASHIN RETRACT STARTED - RETRACT BIN|CIM-RESET CALLED - RETRACT BIN)', re.IGNORECASE),
                "severity": "critical",
                "description": "Cash deposit failed and money retained by ATM - customer funds not returned"
            },
            
            # Incomplete transaction patterns
            "notes_presented_not_taken": {
                "pattern": lambda text: (
                    re.search(r'NOTES PRESENTED', text, re.IGNORECASE) and 
                    not re.search(r'NOTES TAKEN', text, re.IGNORECASE)
                ),
                "severity": "critical",
                "description": "Cash presented but not taken by customer"
            },
            
            "card_inserted_not_taken": {
                "pattern": lambda text: (
                    re.search(r'CARD INSERTED', text, re.IGNORECASE) and 
                    not re.search(r'CARD TAKEN', text, re.IGNORECASE)
                ),
                "severity": "high", 
                "description": "Card inserted but not retrieved by customer"
            }
        }
    
    def _is_false_positive(self, session_text: str, anomaly_details: Dict[str, Any]) -> tuple[bool, str]:
        """
        Enhanced false positive detection with ATM domain knowledge.
        
        Returns:
            tuple: (is_false_positive: bool, override_reason: str)
        """
        
        # Check for card initialization attempts (not an error for magstrip cards)
        if self.enhanced_normal_patterns["card_init_attempts"]["pattern"].search(session_text):
            attempts = len(re.findall(r'CARD INITIALISE ATTEMPT = [123]', session_text))
            if attempts <= 3:
                return True, self.enhanced_normal_patterns["card_init_attempts"]["override_reason"]
        
        # Check for customer cancellation (normal user behavior)
        if self.enhanced_normal_patterns["customer_cancelled"]["pattern"].search(session_text):
            return True, self.enhanced_normal_patterns["customer_cancelled"]["override_reason"]
        
        # Check for successful transaction patterns
        if self.enhanced_normal_patterns["successful_withdrawal"]["pattern"].search(session_text):
            # If this was flagged as an anomaly but has successful completion, it's likely a false positive
            return True, self.enhanced_normal_patterns["successful_withdrawal"]["override_reason"]
        
        return False, ""
    
    def _detect_enhanced_anomalies(self, session_text: str) -> List[Dict[str, Any]]:
        """
        Detect specific anomalies that should be flagged based on ATM domain knowledge.
        
        Returns:
            List of anomaly dictionaries
        """
        anomalies = []
        
        for anomaly_type, config in self.enhanced_anomaly_patterns.items():
            pattern = config["pattern"]
            
            # Handle both regex patterns and callable patterns
            if callable(pattern):
                if pattern(session_text):
                    anomalies.append({
                        "anomaly_type": anomaly_type,
                        "confidence": 0.95,
                        "detection_method": "enhanced_expert_rule",
                        "description": config["description"],
                        "severity": config["severity"],
                        "details": {"pattern_matched": anomaly_type}
                    })
            else:
                if pattern.search(session_text):
                    anomalies.append({
                        "anomaly_type": anomaly_type, 
                        "confidence": 0.95,
                        "detection_method": "enhanced_expert_rule",
                        "description": config["description"],
                        "severity": config["severity"],
                        "details": {"pattern_matched": anomaly_type}
                    })
        
        # Additional specific check for cash retract patterns with enhanced detection
        self._detect_cash_retract_scenarios(session_text, anomalies)
        
        return anomalies
    
    def _detect_cash_retract_scenarios(self, session_text: str, anomalies: List[Dict[str, Any]]):
        """
        Enhanced detection for cash retract scenarios - critical customer service issue.
        
        These patterns indicate that customer deposited cash but the ATM retained it
        instead of returning it to the customer.
        """
        retract_patterns = [
            (r'INIT BNA STARTED - RETRACT BIN', "Bill Note Acceptor initialization retract"),
            (r'CASHIN RETRACT STARTED - RETRACT BIN', "Cash-in operation retract"),
            (r'CIM-RESET CALLED - RETRACT BIN', "CIM reset retract operation")
        ]
        
        for pattern, description in retract_patterns:
            if re.search(pattern, session_text, re.IGNORECASE):
                anomalies.append({
                    "anomaly_type": "cash_deposit_retract_critical",
                    "confidence": 0.98,  # Very high confidence for customer impact
                    "detection_method": "enhanced_expert_rule", 
                    "description": f"Critical: {description} - Customer cash retained by ATM",
                    "severity": "critical",
                    "details": {
                        "pattern_matched": pattern,
                        "customer_impact": "HIGH", 
                        "action_required": "IMMEDIATE_INVESTIGATION",
                        "retract_type": description,
                        "business_priority": "CRITICAL_CUSTOMER_FUNDS"
                    }
                })
                
                # Also add a financial impact flag
                anomalies.append({
                    "anomaly_type": "financial_impact_alert",
                    "confidence": 1.0,
                    "detection_method": "enhanced_expert_rule",
                    "description": "Customer funds at risk - deposit retracted to secure bin",
                    "severity": "critical",
                    "details": {
                        "impact_type": "CUSTOMER_FUNDS_RETAINED",
                        "requires_manual_reconciliation": True,
                        "escalation_required": True
                    }
                })

# Export all classes
__all__ = ['MLAnomalyAnalyzer', 'MLFirstAnomalyDetector', 'TransactionSession']
