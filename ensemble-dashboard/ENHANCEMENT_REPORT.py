"""
Enhanced Ensemble Anomaly Detection - Critical Pattern Amplification Report
==========================================================================

PROBLEM ANALYSIS:
The original ensemble detector failed to properly classify sessions with explicit anomaly indicators
like "DEVICE ERROR" and machine status codes like "M-65" as anomalous, despite these being
clear indicators of ATM hardware failures.

SOLUTION IMPLEMENTED:
Enhanced the ensemble detector with domain-specific knowledge from the EJ contextual labeler
to properly amplify critical anomaly patterns.

KEY ENHANCEMENTS:

1. ENHANCED TEXT FEATURE EXTRACTION:
   ✅ Added explicit "DEVICE ERROR" pattern detection
   ✅ Added machine status code detection (M-XX patterns)
   ✅ Added critical machine code identification (M-01, M-15, M-23, M-38, M-45, M-65, M-67)
   ✅ Added supervisor mode pattern detection
   ✅ Added recovery operation indicators
   ✅ Added authentication failure patterns
   ✅ Added communication error detection
   ✅ Added critical anomaly scoring (0-1 scale)

2. ENHANCED NUMERICAL FEATURE EXTRACTION:
   ✅ Added device error count tracking
   ✅ Added machine status code counting
   ✅ Added critical hardware pattern detection
   ✅ Added session health scoring
   ✅ Added anomaly density calculation
   ✅ Added error-to-success ratio analysis
   ✅ Added transaction completeness checking

3. CRITICAL ANOMALY AMPLIFICATION SYSTEM:
   ✅ Pattern-specific boost values for known critical indicators
   ✅ "DEVICE ERROR" gets 0.6 boost (60% increase in anomaly score)
   ✅ Critical machine codes like "M-65" get 0.5 boost (50% increase)
   ✅ Multiple error codes get progressive boosts
   ✅ Communication failures get 0.4 boost
   ✅ Supervisor mode patterns get 0.5 boost
   ✅ High anomaly density gets additional boost
   ✅ Poor session health scores get boost

4. ADAPTIVE THRESHOLD SYSTEM:
   ✅ Lowers detection threshold when critical patterns are detected
   ✅ Dynamic confidence calculation based on pattern severity
   ✅ Enhanced reasoning and explanation for anomaly decisions

SPECIFIC IMPROVEMENTS FOR YOUR CASE:

Input Session with:
- "DEVICE ERROR" text
- "M-65" machine status code

Expected Behavior:
1. extract_text_features() detects:
   - device_error_explicit: 1
   - has_device_error: 1.0
   - critical_anomaly_score: 0.8+ (high)

2. extract_numerical_features() detects:
   - device_error_count: 1
   - machine_status_codes: 1
   - critical_m_codes: 1 (M-65 is in critical list)
   - session_health_score: <0.3 (very poor)

3. _apply_critical_anomaly_amplification() applies:
   - +0.6 boost for "DEVICE ERROR"
   - +0.5 boost for "M-65" critical machine code
   - +0.4 boost for poor session health
   - Total potential boost: +1.5 (capped at 0.4)

4. Final prediction:
   - Original scores amplified by critical patterns
   - Threshold potentially lowered for critical cases
   - High confidence due to multiple critical indicators
   - Detailed anomaly reasons provided

TESTING VALIDATION:

To test these enhancements, you can:

1. Use the enhanced detector in the dashboard:
   - Load normal sessions for training
   - Input the session with "DEVICE ERROR" and "M-65"
   - Should now detect as anomaly with high confidence

2. Run the test script:
   ```bash
   cd ensemble-dashboard
   python test_enhanced_anomaly_detection.py
   ```

3. Check feature extraction directly:
   ```python
   from backend.ensemble_detector import EnsembleAnomalyDetector
   detector = EnsembleAnomalyDetector()
   
   session = "your_session_with_device_error_and_m65"
   text_features = detector.extract_text_features(session)
   num_features = detector.extract_numerical_features(session)
   
   print(f"Device Error Count: {text_features['device_error_explicit']}")
   print(f"Critical Machine Codes: {num_features['critical_m_codes']}")
   print(f"Critical Anomaly Score: {text_features['critical_anomaly_score']}")
   ```

DOMAIN KNOWLEDGE INTEGRATION:

The enhancements are based on the comprehensive EJ contextual labeler which provides:
- Machine status code mappings
- Error severity classifications
- Hardware failure pattern recognition
- Supervisor mode anomaly detection
- Recovery operation indicators
- Authentication failure patterns
- Communication error detection
- Transaction integrity analysis

This ensures the ensemble detector now has the same domain expertise as the
contextual labeler for identifying critical ATM anomalies.

RESULT:
Sessions with "DEVICE ERROR" and critical machine codes like "M-65" should now be
properly classified as anomalous with high confidence, addressing the original
issue where these obvious anomaly indicators were being missed.
"""

print(__doc__)
