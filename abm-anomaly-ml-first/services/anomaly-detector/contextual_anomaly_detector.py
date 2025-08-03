"""
Contextual Anomaly Detection Rules for EJ Logs
Works with enhanced BERT to provide domain-specific anomaly detection
"""

import logging
from typing import List, Dict, Any, Optional
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import re

from ej_contextual_labeler import (
    EJLogLabel, EventType, TransactionPhase, Severity, 
    OperationalMode, RecoveryType, ErrorCategory
)

logger = logging.getLogger(__name__)

class ContextualAnomalyDetector:
    """Advanced anomaly detection using EJ log context"""
    
    def __init__(self):
        self.anomaly_rules = [
            self._check_supervisor_mode_anomalies,
            self._check_recovery_anomalies,
            self._check_cash_reconciliation_anomalies,
            self._check_authentication_anomalies,
            self._check_transaction_flow_anomalies,
            self._check_temporal_anomalies,
            self._check_device_health_anomalies,
            self._check_security_anomalies,
            self._check_operational_anomalies
        ]
        
        # Thresholds for anomaly detection
        self.thresholds = {
            'max_recovery_attempts': 3,
            'max_auth_failures_per_session': 5,
            'max_transaction_duration_minutes': 15,
            'max_error_rate': 0.2,
            'max_supervisor_transaction_overlap': 0.1,
            'max_cash_rejection_rate': 0.15
        }
    
    def detect_anomalies(self, labels: List[EJLogLabel]) -> List[Dict[str, Any]]:
        """
        Main anomaly detection function
        
        Args:
            labels: List of contextual labels from EJ log
            
        Returns:
            List of detected anomalies with details
        """
        all_anomalies = []
        
        for rule in self.anomaly_rules:
            try:
                anomalies = rule(labels)
                if anomalies:
                    all_anomalies.extend(anomalies)
            except Exception as e:
                logger.error(f"Error in anomaly rule {rule.__name__}: {e}")
        
        # Deduplicate and prioritize anomalies
        return self._prioritize_anomalies(all_anomalies)
    
    def _check_supervisor_mode_anomalies(self, labels: List[EJLogLabel]) -> List[Dict]:
        """Detect supervisor mode related anomalies"""
        anomalies = []
        supervisor_active = False
        supervisor_transactions = 0
        total_transactions = 0
        
        for label in labels:
            if label.event_type == EventType.SUPERVISOR_ENTRY:
                supervisor_active = True
                logger.info(f"Supervisor mode entered at line {label.line_number}")
            elif label.event_type == EventType.SUPERVISOR_EXIT:
                supervisor_active = False
                logger.info(f"Supervisor mode exited at line {label.line_number}")
            
            # Check for customer transactions during supervisor mode
            if supervisor_active and label.event_type == EventType.TXN_START:
                supervisor_transactions += 1
                anomalies.append({
                    'type': 'supervisor_transaction',
                    'severity': 'HIGH',
                    'description': 'Customer transaction initiated during supervisor mode',
                    'line_number': label.line_number,
                    'timestamp': label.timestamp,
                    'recommendation': 'Review supervisor procedures - customers should not transact during maintenance',
                    'financial_impact': 'Potential regulatory compliance issue'
                })
            
            if label.event_type == EventType.TXN_START:
                total_transactions += 1
        
        # Check overall supervisor-customer overlap
        if total_transactions > 0:
            overlap_rate = supervisor_transactions / total_transactions
            if overlap_rate > self.thresholds['max_supervisor_transaction_overlap']:
                anomalies.append({
                    'type': 'high_supervisor_overlap',
                    'severity': 'MEDIUM',
                    'description': f'High supervisor-customer transaction overlap: {overlap_rate:.1%}',
                    'recommendation': 'Review ATM scheduling and maintenance procedures',
                    'metrics': {'overlap_rate': overlap_rate, 'threshold': self.thresholds['max_supervisor_transaction_overlap']}
                })
        
        return anomalies
    
    def _check_recovery_anomalies(self, labels: List[EJLogLabel]) -> List[Dict]:
        """Detect recovery-related anomalies indicating hardware issues"""
        anomalies = []
        recovery_counts = defaultdict(int)
        recovery_sequences = []
        
        for label in labels:
            if label.recovery_type:
                recovery_counts[label.recovery_type] += 1
                recovery_sequences.append({
                    'type': label.recovery_type,
                    'timestamp': label.timestamp,
                    'line_number': label.line_number
                })
        
        # Multiple recovery attempts indicate persistent issues
        for recovery_type, count in recovery_counts.items():
            if count > self.thresholds['max_recovery_attempts']:
                anomalies.append({
                    'type': 'repeated_recovery',
                    'severity': 'HIGH',
                    'description': f'Multiple {recovery_type.value} recovery attempts ({count})',
                    'line_numbers': [seq['line_number'] for seq in recovery_sequences if seq['type'] == recovery_type],
                    'recommendation': f'Schedule immediate maintenance for {recovery_type.value} component',
                    'financial_impact': 'High - potential for extended downtime and customer impact',
                    'metrics': {'attempt_count': count, 'threshold': self.thresholds['max_recovery_attempts']}
                })
        
        # Check for recovery cascades (multiple different recovery types)
        unique_recovery_types = len(set(recovery_counts.keys()))
        if unique_recovery_types >= 3:
            anomalies.append({
                'type': 'recovery_cascade',
                'severity': 'CRITICAL',
                'description': f'Multiple device recovery types in single session ({unique_recovery_types})',
                'recommendation': 'Immediate comprehensive system diagnosis required',
                'financial_impact': 'Critical - indicates systemic hardware failure',
                'affected_components': [rt.value for rt in recovery_counts.keys()]
            })
        
        return anomalies
    
    def _check_cash_reconciliation_anomalies(self, labels: List[EJLogLabel]) -> List[Dict]:
        """Detect cash handling and reconciliation anomalies"""
        anomalies = []
        cash_events = []
        total_dispensed = 0
        total_rejected = 0
        
        for label in labels:
            if label.denomination_data:
                cash_events.append(label)
                
                # Extract dispense and rejection data
                if 'dispensed' in label.metadata:
                    total_dispensed += label.metadata.get('dispensed', 0)
                if 'rejected' in label.metadata:
                    total_rejected += label.metadata.get('rejected', 0)
        
        # Check for high rejection rates
        if total_dispensed > 0:
            rejection_rate = total_rejected / (total_dispensed + total_rejected)
            if rejection_rate > self.thresholds['max_cash_rejection_rate']:
                anomalies.append({
                    'type': 'high_cash_rejection',
                    'severity': 'MEDIUM',
                    'description': f'High note rejection rate: {rejection_rate:.1%}',
                    'recommendation': 'Check note quality, cassette condition, and transport mechanism',
                    'financial_impact': 'Medium - impacts customer experience and cash availability',
                    'metrics': {
                        'rejection_rate': rejection_rate,
                        'total_dispensed': total_dispensed,
                        'total_rejected': total_rejected,
                        'threshold': self.thresholds['max_cash_rejection_rate']
                    }
                })
        
        # Check for denomination imbalances
        denomination_counts = defaultdict(int)
        for event in cash_events:
            if event.denomination_data:
                for denom, count in event.denomination_data.items():
                    denomination_counts[denom] += count
        
        if denomination_counts:
            total_notes = sum(denomination_counts.values())
            for denom, count in denomination_counts.items():
                if count / total_notes > 0.8:  # One denomination > 80%
                    anomalies.append({
                        'type': 'denomination_imbalance',
                        'severity': 'LOW',
                        'description': f'High usage of {denom}: {count/total_notes:.1%} of total notes',
                        'recommendation': 'Consider rebalancing cassette denomination mix',
                        'financial_impact': 'Low - may limit transaction options for customers'
                    })
        
        return anomalies
    
    def _check_authentication_anomalies(self, labels: List[EJLogLabel]) -> List[Dict]:
        """Detect authentication and security-related anomalies"""
        anomalies = []
        auth_failures = []
        auth_events = []
        
        for label in labels:
            if label.event_type == EventType.EXTERNAL_AUTH:
                auth_events.append(label)
            
            if label.auth_failure_type:
                auth_failures.append(label)
        
        # Check for excessive authentication failures
        if len(auth_failures) > self.thresholds['max_auth_failures_per_session']:
            failure_types = Counter(f.auth_failure_type for f in auth_failures)
            anomalies.append({
                'type': 'excessive_auth_failures',
                'severity': 'HIGH',
                'description': f'Excessive authentication failures ({len(auth_failures)})',
                'failure_breakdown': dict(failure_types),
                'recommendation': 'Investigate potential card skimming or EMV configuration issues',
                'financial_impact': 'High - potential fraud indicator or system configuration issue',
                'security_concern': True
            })
        
        # Check for specific authentication patterns
        no_arpc_failures = [f for f in auth_failures if f.auth_failure_type == 'no_arpc']
        if len(no_arpc_failures) > 2:
            anomalies.append({
                'type': 'emv_configuration_issue',
                'severity': 'MEDIUM',
                'description': f'Multiple NO ARPC failures ({len(no_arpc_failures)})',
                'recommendation': 'Check EMV configuration and key management',
                'financial_impact': 'Medium - affects chip card processing capability'
            })
        
        return anomalies
    
    def _check_transaction_flow_anomalies(self, labels: List[EJLogLabel]) -> List[Dict]:
        """Detect transaction flow and lifecycle anomalies"""
        anomalies = []
        transactions = defaultdict(list)
        
        # Group events by transaction ID
        current_txn_id = None
        for label in labels:
            if label.event_type == EventType.TXN_START:
                current_txn_id = label.transaction_id or f"txn_{label.line_number}"
            if current_txn_id:
                transactions[current_txn_id].append(label)
            if label.event_type == EventType.TXN_END:
                current_txn_id = None
        
        # Analyze each transaction
        for txn_id, txn_labels in transactions.items():
            # Check for incomplete transactions
            has_start = any(l.event_type == EventType.TXN_START for l in txn_labels)
            has_end = any(l.event_type == EventType.TXN_END for l in txn_labels)
            
            if has_start and not has_end:
                anomalies.append({
                    'type': 'incomplete_transaction',
                    'severity': 'MEDIUM',
                    'description': f'Transaction {txn_id} started but not completed',
                    'recommendation': 'Investigate customer abandonment or system timeout',
                    'financial_impact': 'Medium - potential revenue loss and customer dissatisfaction'
                })
            
            # Check transaction duration
            timestamps = [l.timestamp for l in txn_labels if l.timestamp]
            if len(timestamps) > 1:
                duration = (max(timestamps) - min(timestamps)).total_seconds() / 60
                if duration > self.thresholds['max_transaction_duration_minutes']:
                    anomalies.append({
                        'type': 'long_transaction_duration',
                        'severity': 'LOW',
                        'description': f'Transaction duration: {duration:.1f} minutes',
                        'recommendation': 'Review transaction flow optimization',
                        'metrics': {'duration_minutes': duration, 'threshold': self.thresholds['max_transaction_duration_minutes']}
                    })
            
            # Check for unusual phase sequences
            phases = [l.phase for l in txn_labels]
            if self._has_unusual_phase_sequence(phases):
                anomalies.append({
                    'type': 'unusual_transaction_flow',
                    'severity': 'MEDIUM',
                    'description': 'Unusual transaction phase sequence detected',
                    'phase_sequence': [p.value for p in phases],
                    'recommendation': 'Review transaction logic and error handling'
                })
        
        return anomalies
    
    def _check_temporal_anomalies(self, labels: List[EJLogLabel]) -> List[Dict]:
        """Detect temporal patterns and timing anomalies"""
        anomalies = []
        
        # Check for rapid repeated events
        event_timestamps = defaultdict(list)
        for label in labels:
            if label.timestamp:
                event_timestamps[label.event_type].append(label.timestamp)
        
        for event_type, timestamps in event_timestamps.items():
            if len(timestamps) > 1:
                # Check for events occurring too rapidly
                timestamps.sort()
                for i in range(1, len(timestamps)):
                    time_diff = (timestamps[i] - timestamps[i-1]).total_seconds()
                    if time_diff < 1.0:  # Same event within 1 second
                        anomalies.append({
                            'type': 'rapid_repeated_event',
                            'severity': 'MEDIUM',
                            'description': f'Rapid {event_type.value} events ({time_diff:.2f}s apart)',
                            'recommendation': 'Check for stuck buttons or sensor malfunction',
                            'timestamps': [timestamps[i-1], timestamps[i]]
                        })
        
        return anomalies
    
    def _check_device_health_anomalies(self, labels: List[EJLogLabel]) -> List[Dict]:
        """Detect device health and maintenance indicators"""
        anomalies = []
        
        # Check error rates by device component
        component_errors = defaultdict(list)
        total_events_by_component = defaultdict(int)
        
        for label in labels:
            if label.entity:
                total_events_by_component[label.entity] += 1
                if label.severity in [Severity.ERROR, Severity.CRITICAL]:
                    component_errors[label.entity].append(label)
        
        # Calculate error rates
        for component, total_events in total_events_by_component.items():
            error_count = len(component_errors[component])
            if total_events > 0:
                error_rate = error_count / total_events
                if error_rate > self.thresholds['max_error_rate']:
                    anomalies.append({
                        'type': 'high_component_error_rate',
                        'severity': 'HIGH',
                        'description': f'High error rate for {component}: {error_rate:.1%}',
                        'component': component,
                        'error_count': error_count,
                        'total_events': total_events,
                        'recommendation': f'Schedule maintenance for {component}',
                        'financial_impact': 'High - component reliability concern'
                    })
        
        return anomalies
    
    def _check_security_anomalies(self, labels: List[EJLogLabel]) -> List[Dict]:
        """Detect potential security-related anomalies"""
        anomalies = []
        
        # Check for security-related events
        security_events = [l for l in labels if l.error_category == ErrorCategory.SECURITY]
        
        if len(security_events) > 3:
            anomalies.append({
                'type': 'multiple_security_events',
                'severity': 'CRITICAL',
                'description': f'Multiple security events detected ({len(security_events)})',
                'events': [{'line': e.line_number, 'type': e.event_type.value} for e in security_events],
                'recommendation': 'Immediate security review required',
                'financial_impact': 'Critical - potential security breach',
                'security_concern': True
            })
        
        return anomalies
    
    def _check_operational_anomalies(self, labels: List[EJLogLabel]) -> List[Dict]:
        """Detect operational efficiency and performance anomalies"""
        anomalies = []
        
        # Check for excessive mode switching
        mode_changes = []
        current_mode = None
        
        for label in labels:
            if label.operational_mode != current_mode:
                mode_changes.append({
                    'from': current_mode,
                    'to': label.operational_mode,
                    'timestamp': label.timestamp,
                    'line_number': label.line_number
                })
                current_mode = label.operational_mode
        
        if len(mode_changes) > 10:  # Arbitrary threshold
            anomalies.append({
                'type': 'excessive_mode_switching',
                'severity': 'MEDIUM',
                'description': f'Excessive operational mode changes ({len(mode_changes)})',
                'recommendation': 'Review operational procedures and system stability',
                'mode_changes': len(mode_changes)
            })
        
        return anomalies
    
    def _has_unusual_phase_sequence(self, phases: List[TransactionPhase]) -> bool:
        """Check if transaction phase sequence is unusual"""
        if len(phases) < 2:
            return False
        
        # Check for backwards transitions (simplified check)
        phase_order = {
            TransactionPhase.INITIALIZATION: 0,
            TransactionPhase.CARD_AUTHENTICATION: 1,
            TransactionPhase.PIN_VERIFICATION: 2,
            TransactionPhase.ACCOUNT_SELECTION: 3,
            TransactionPhase.TRANSACTION_SELECTION: 4,
            TransactionPhase.AMOUNT_ENTRY: 5,
            TransactionPhase.PROCESSING: 6,
            TransactionPhase.CASH_DISPENSING: 7,
            TransactionPhase.RECEIPT_PRINTING: 8,
            TransactionPhase.COMPLETION: 9,
            TransactionPhase.ERROR_HANDLING: 10  # Can happen at any time
        }
        
        for i in range(1, len(phases)):
            current_order = phase_order.get(phases[i], 0)
            prev_order = phase_order.get(phases[i-1], 0)
            
            # Allow error handling at any time
            if phases[i] == TransactionPhase.ERROR_HANDLING:
                continue
            
            # Check for significant backwards movement
            if current_order < prev_order - 1:
                return True
        
        return False
    
    def _prioritize_anomalies(self, anomalies: List[Dict]) -> List[Dict]:
        """Prioritize and deduplicate anomalies"""
        if not anomalies:
            return []
        
        # Define severity priority
        severity_priority = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}
        
        # Sort by severity and add priority score
        for anomaly in anomalies:
            anomaly['priority'] = severity_priority.get(anomaly.get('severity', 'LOW'), 3)
            
            # Add financial impact score
            financial_impact = anomaly.get('financial_impact', '')
            if 'Critical' in financial_impact:
                anomaly['financial_priority'] = 0
            elif 'High' in financial_impact:
                anomaly['financial_priority'] = 1
            elif 'Medium' in financial_impact:
                anomaly['financial_priority'] = 2
            else:
                anomaly['financial_priority'] = 3
        
        # Sort by priority (lower is higher priority)
        anomalies.sort(key=lambda x: (x['priority'], x.get('financial_priority', 3)))
        
        return anomalies

class EJAnomalyAnalyzer:
    """Complete EJ anomaly analysis combining contextual and BERT analysis"""
    
    def __init__(self, enhanced_bert_analyzer, contextual_detector=None):
        self.bert_analyzer = enhanced_bert_analyzer
        self.contextual_detector = contextual_detector or ContextualAnomalyDetector()
    
    def analyze(self, ej_log_text: str) -> Dict[str, Any]:
        """
        Complete analysis combining BERT predictions with contextual anomaly detection
        
        Args:
            ej_log_text: Raw EJ log text
            
        Returns:
            Comprehensive analysis results
        """
        # Get BERT analysis with contextual enhancement
        bert_results = self.bert_analyzer.analyze_text(ej_log_text)
        
        # Extract contextual labels
        contextual_labels = [
            # Convert back to EJLogLabel objects for anomaly detection
            # This is a simplified conversion - in practice, we'd pass the original labels
        ]
        
        # Get contextual anomalies
        contextual_anomalies = self.contextual_detector.detect_anomalies(contextual_labels)
        
        # Combine results
        combined_results = {
            **bert_results,
            'contextual_anomalies': contextual_anomalies,
            'risk_assessment': self._assess_overall_risk(bert_results, contextual_anomalies),
            'recommendations': self._generate_recommendations(bert_results, contextual_anomalies),
            'financial_impact_assessment': self._assess_financial_impact(contextual_anomalies)
        }
        
        return combined_results
    
    def _assess_overall_risk(self, bert_results: Dict, contextual_anomalies: List[Dict]) -> Dict[str, Any]:
        """Assess overall risk level based on all analysis"""
        risk_factors = []
        risk_score = 0
        
        # BERT prediction risk
        if bert_results['prediction'] in ['Failed', 'Technical Fault']:
            risk_score += 3
            risk_factors.append(f"BERT classified as {bert_results['prediction']}")
        elif bert_results['prediction'] == 'Suspicious':
            risk_score += 2
            risk_factors.append("BERT classified as Suspicious")
        
        # Contextual anomaly risk
        critical_anomalies = [a for a in contextual_anomalies if a.get('severity') == 'CRITICAL']
        high_anomalies = [a for a in contextual_anomalies if a.get('severity') == 'HIGH']
        
        risk_score += len(critical_anomalies) * 3
        risk_score += len(high_anomalies) * 2
        
        risk_factors.extend([a['description'] for a in critical_anomalies + high_anomalies])
        
        # Determine risk level
        if risk_score >= 6:
            risk_level = 'CRITICAL'
        elif risk_score >= 4:
            risk_level = 'HIGH'
        elif risk_score >= 2:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'LOW'
        
        return {
            'risk_level': risk_level,
            'risk_score': risk_score,
            'risk_factors': risk_factors,
            'requires_immediate_attention': risk_level in ['CRITICAL', 'HIGH']
        }
    
    def _generate_recommendations(self, bert_results: Dict, contextual_anomalies: List[Dict]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Add recommendations from contextual anomalies
        for anomaly in contextual_anomalies:
            if 'recommendation' in anomaly:
                recommendations.append(anomaly['recommendation'])
        
        # Add BERT-based recommendations
        if bert_results['confidence'] < 0.7:
            recommendations.append("Low prediction confidence - consider manual review")
        
        # Remove duplicates and prioritize
        return list(dict.fromkeys(recommendations))  # Preserves order while removing duplicates
    
    def _assess_financial_impact(self, contextual_anomalies: List[Dict]) -> Dict[str, Any]:
        """Assess potential financial impact"""
        impact_levels = []
        security_concerns = False
        
        for anomaly in contextual_anomalies:
            financial_impact = anomaly.get('financial_impact', '')
            if financial_impact:
                impact_levels.append(financial_impact)
            
            if anomaly.get('security_concern', False):
                security_concerns = True
        
        # Determine overall financial impact
        if any('Critical' in impact for impact in impact_levels):
            overall_impact = 'CRITICAL'
        elif any('High' in impact for impact in impact_levels):
            overall_impact = 'HIGH'
        elif any('Medium' in impact for impact in impact_levels):
            overall_impact = 'MEDIUM'
        else:
            overall_impact = 'LOW'
        
        return {
            'overall_impact': overall_impact,
            'security_concerns': security_concerns,
            'impact_details': impact_levels,
            'requires_financial_review': overall_impact in ['CRITICAL', 'HIGH'] or security_concerns
        }
