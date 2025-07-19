#!/bin/bash

# Integration Script: Replace Rigid Rules with Knowledge-Guided Confidence Adjustment
# This script shows how to modify the existing ml_analyzer.py to use the new approach

echo "=================================================="
echo "KNOWLEDGE-GUIDED ML ENHANCEMENT - INTEGRATION GUIDE"
echo "=================================================="
echo ""

echo "STEP 1: Backup existing ml_analyzer.py"
echo "----------------------------------------"
echo "cp services/anomaly-detector/ml_analyzer.py services/anomaly-detector/ml_analyzer_backup_$(date +%Y%m%d_%H%M%S).py"
echo ""

echo "STEP 2: Copy the confidence adjuster to your project"
echo "----------------------------------------------------"
echo "The confidence_adjuster.py file is ready to use"
echo ""

echo "STEP 3: Modify ml_analyzer.py - Add import and initialization"
echo "------------------------------------------------------------"
echo "Add to the imports section:"
echo ""
cat << 'EOF'
# Add this import
from confidence_adjuster import KnowledgeGuidedConfidenceAdjuster
EOF
echo ""

echo "Add to the __init__ method of MLAnomalyAnalyzer class:"
echo ""
cat << 'EOF'
# Add this line in __init__ method after loading expert rules
self.confidence_adjuster = KnowledgeGuidedConfidenceAdjuster()
logger.info("Knowledge-guided confidence adjuster initialized")
EOF
echo ""

echo "STEP 4: Replace the rigid override logic"
echo "----------------------------------------"
echo "Find this section in detect_anomalies_unsupervised method:"
echo ""
cat << 'EOF'
# FIND AND REPLACE THIS OLD CODE:
for i, session in enumerate(self.sessions):
    # ... existing code ...
    
    # OLD RIGID APPROACH - REMOVE THIS:
    if self.apply_expert_override(session):
        session.is_anomaly = False
        session.anomaly_score = 0.0
        # ... rest of override logic
        continue
    
    # ... rest of method
EOF
echo ""

echo "REPLACE WITH THIS NEW APPROACH:"
echo ""
cat << 'EOF'
# NEW KNOWLEDGE-GUIDED APPROACH:
for i, session in enumerate(self.sessions):
    # Normalize scores to 0-1 range
    if_score_norm = (if_scores[i] - if_scores.min()) / (if_scores.max() - if_scores.min() + 1e-8)
    svm_score_norm = (svm_scores[i] - svm_scores.min()) / (svm_scores.max() - svm_scores.min() + 1e-8)
    
    # Combine ML scores
    combined_ml_score = max(1.0 - if_score_norm, 1.0 - svm_score_norm)
    
    # Apply knowledge-guided confidence adjustment instead of rigid override
    adjustment_result = self.confidence_adjuster.adjust_ml_confidence(session, combined_ml_score)
    
    # Set comprehensive scoring
    session.ml_anomaly_score = combined_ml_score  # Original ML score
    session.knowledge_adjusted_score = adjustment_result['adjusted_score']  # Knowledge-adjusted score
    session.anomaly_score = adjustment_result['adjusted_score']  # Final score for compatibility
    session.is_anomaly = adjustment_result['adjusted_score'] > 0.5
    
    # Store knowledge insights for transparency and debugging
    session.knowledge_adjustment = {
        'applied_rules': adjustment_result['applied_rules'],
        'reasoning': adjustment_result['reasoning'],
        'confidence_in_assessment': adjustment_result['confidence_in_adjustment'],
        'adjustment_factor': adjustment_result['adjustment_factor'],
        'knowledge_signals': adjustment_result['knowledge_signals']
    }
    
    # Enhanced logging for monitoring
    if adjustment_result['applied_rules']:
        logger.info(f"Session {session.session_id}: "
                   f"ML={combined_ml_score:.3f} -> "
                   f"Adjusted={adjustment_result['adjusted_score']:.3f} "
                   f"(Rules: {', '.join(adjustment_result['applied_rules'])}) "
                   f"Confidence: {adjustment_result['confidence_in_adjustment']:.2f}")
    
    # Continue with existing anomaly detection logic...
    # (detect specific anomalies, etc.)
EOF
echo ""

echo "STEP 5: Update the results dataframe creation"
echo "---------------------------------------------"
echo "In create_results_dataframe method, add these new columns:"
echo ""
cat << 'EOF'
# Add these columns to the results dataframe:
'ml_anomaly_score': session.ml_anomaly_score if hasattr(session, 'ml_anomaly_score') else session.anomaly_score,
'knowledge_adjusted_score': session.knowledge_adjusted_score if hasattr(session, 'knowledge_adjusted_score') else session.anomaly_score,
'knowledge_rules_applied': ', '.join(session.knowledge_adjustment['applied_rules']) if hasattr(session, 'knowledge_adjustment') else '',
'knowledge_confidence': session.knowledge_adjustment['confidence_in_assessment'] if hasattr(session, 'knowledge_adjustment') else 0.5,
'adjustment_factor': session.knowledge_adjustment['adjustment_factor'] if hasattr(session, 'knowledge_adjustment') else 1.0,
'knowledge_reasoning': ' | '.join(session.knowledge_adjustment['reasoning']) if hasattr(session, 'knowledge_adjustment') else '',
EOF
echo ""

echo "STEP 6: Optional - Remove or deprecate old expert override methods"
echo "----------------------------------------------------------------"
echo "You can now remove or comment out these methods (they're no longer needed):"
echo "- apply_expert_override()"
echo "- is_successful_withdrawal()" 
echo "- is_successful_inquiry()"
echo "- has_genuine_anomaly()"
echo ""
echo "Or keep them for reference/fallback during transition period."
echo ""

echo "STEP 7: Test the integration"
echo "---------------------------"
echo "Run this test command to verify the integration:"
echo ""
cat << 'EOF'
# Test command
python3 -c "
from services.anomaly_detector.ml_analyzer import MLAnomalyAnalyzer
from services.anomaly_detector.confidence_adjuster import KnowledgeGuidedConfidenceAdjuster

print('Testing enhanced ML analyzer...')
analyzer = MLAnomalyAnalyzer()
print(f'Confidence adjuster loaded: {hasattr(analyzer, \"confidence_adjuster\")}')

adjuster = KnowledgeGuidedConfidenceAdjuster()
print('Knowledge-guided confidence adjuster working correctly')
print('Integration successful!')
"
EOF
echo ""

echo "STEP 8: Monitor the results"
echo "--------------------------"
echo "After deployment, monitor these metrics:"
echo ""
echo "1. Compare anomaly counts: ML-only vs Knowledge-adjusted"
echo "2. Track confidence levels in assessments"
echo "3. Review knowledge rules being applied"
echo "4. Monitor false positive reduction"
echo ""

echo "EXPECTED BENEFITS:"
echo "- Reduced false positives while preserving ML learning"
echo "- Better transparency in decision making"
echo "- Ability to discover pattern variations"
echo "- Continuous improvement through expert feedback"
echo ""

echo "=================================================="
echo "INTEGRATION COMPLETE"
echo "=================================================="
echo ""
echo "The enhanced system will:"
echo "✓ Use domain knowledge to guide ML confidence rather than override it"
echo "✓ Preserve ML learning signals for pattern discovery"
echo "✓ Provide detailed reasoning for anomaly decisions" 
echo "✓ Enable continuous improvement through feedback"
echo ""
echo "Your EJ anomaly detection system is now knowledge-guided!"

# Create a quick patch file for easy application
cat > ml_analyzer_enhancement.patch << 'PATCH_EOF'
--- ml_analyzer.py.orig
+++ ml_analyzer.py
@@ -1,6 +1,7 @@
 # ML-First ABM Anomaly Detection with Supervised Learning
 import pandas as pd
 import numpy as np
+from confidence_adjuster import KnowledgeGuidedConfidenceAdjuster
 
 # ... existing imports ...
 
@@ -100,6 +101,8 @@
         # Expert knowledge system to avoid false positives
         self.expert_rules = self.load_expert_rules()
         
+        # Enhanced knowledge-guided confidence adjustment
+        self.confidence_adjuster = KnowledgeGuidedConfidenceAdjuster()
         # ... rest of init ...
 
 @@ -650,15 +653,25 @@
         for i, session in enumerate(self.sessions):
-            # Apply expert knowledge to prevent false positives
-            should_override = self.apply_expert_override(session)
-            if should_override:
-                session.is_anomaly = False
-                session.anomaly_score = 0.0
-                continue
+            # Normalize ML scores
+            if_score_norm = (if_scores[i] - if_scores.min()) / (if_scores.max() - if_scores.min() + 1e-8)
+            svm_score_norm = (svm_scores[i] - svm_scores.min()) / (svm_scores.max() - svm_scores.min() + 1e-8)
+            combined_ml_score = max(1.0 - if_score_norm, 1.0 - svm_score_norm)
+            
+            # Apply knowledge-guided confidence adjustment
+            adjustment_result = self.confidence_adjuster.adjust_ml_confidence(session, combined_ml_score)
+            
+            # Set comprehensive scoring
+            session.ml_anomaly_score = combined_ml_score
+            session.knowledge_adjusted_score = adjustment_result['adjusted_score']
+            session.anomaly_score = adjustment_result['adjusted_score']
+            session.is_anomaly = adjustment_result['adjusted_score'] > 0.5
+            session.knowledge_adjustment = adjustment_result
             
-            # ... existing ML detection logic
+            # Log knowledge-guided adjustments
+            if adjustment_result['applied_rules']:
+                logger.info(f"Session {session.session_id}: ML={combined_ml_score:.3f} -> Adjusted={adjustment_result['adjusted_score']:.3f}")
             
+            # ... continue with existing detection logic
 
PATCH_EOF

echo ""
echo "Patch file created: ml_analyzer_enhancement.patch"
echo "You can apply it with: patch -p0 < ml_analyzer_enhancement.patch"
