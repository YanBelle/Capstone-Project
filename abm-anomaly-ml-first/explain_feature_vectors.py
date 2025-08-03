#!/usr/bin/env python3
"""
Feature Vector Analysis - Understanding the Bridge Between Contextual Labeler and ML
"""

import sys
import os
sys.path.append('services/anomaly-detector')
sys.path.append('services/api')

from ej_contextual_labeler import EJLogLabeler, EJLogLabel

def explain_feature_vectors():
    """Explain feature vectors and their relationship to contextual labeler"""
    
    print("🔢 FEATURE VECTORS: THE BRIDGE BETWEEN CONTEXTUAL LABELER AND ML")
    print("=" * 80)
    
    print("\n📋 1. WHAT ARE FEATURE VECTORS?")
    print("   Feature Vectors are numerical representations (arrays of numbers) that")
    print("   translate the rich contextual information from your EJ Contextual Labeler")
    print("   into mathematical format that ML models can process and learn from.")
    
    print("\n🎯 2. THE TRANSFORMATION PROCESS:")
    print("   Raw EJ Log → Contextual Labeler → Feature Vectors → ML Models")
    print("   (Text)        (35 Event Types)   (50+ Numbers)    (Predictions)")
    
    # Create example transformation
    labeler = EJLogLabeler()
    
    # Example log entry
    test_log = "07:45:12 CIM-DEPOSIT ACTIVATED A/C OPERATION OK ESC: 2 VAL: 0 REF: 0 REJECTS: 1 JMD$5000: 2"
    
    print(f"\n📝 3. EXAMPLE TRANSFORMATION:")
    print(f"   Raw Log: '{test_log}'")
    
    # Get contextual labels
    labels = labeler.label_log(test_log)
    if labels:
        label = labels[0]
        print(f"\n   🏷️ Contextual Labeler Output:")
        print(f"      • Event Type: {label.event_type.value}")
        print(f"      • Phase: {label.phase.value}")
        print(f"      • Operational Mode: {label.operational_mode.value}")
        print(f"      • Severity: {label.severity.value}")
        if label.cim_status:
            print(f"      • CIM Status: {label.cim_status}")
        if label.deposit_amount:
            print(f"      • Deposit Amount: ${label.deposit_amount}")
        
        print(f"\n   🔢 Feature Vector Creation (50+ dimensions):")
        
        # Demonstrate feature extraction
        feature_dimensions = [
            ("Basic Text Stats", [
                f"text_length: {len(test_log)}",
                f"line_count: {test_log.count('\\n') + 1}",
                f"word_count: {len(test_log.split())}"
            ]),
            ("Event Type Features", [
                f"is_cim_deposit: 1.0",
                f"is_supervisor_mode: 0.0", 
                f"is_error_event: 0.0",
                f"is_cash_operation: 1.0"
            ]),
            ("Financial Features", [
                f"deposit_amount: 10000.0",
                f"rejection_rate: 0.33",
                f"escrow_count: 2.0",
                f"currency_jmd: 1.0"
            ]),
            ("Contextual Features", [
                f"confidence_score: {label.confidence_score}",
                f"customer_present: {1.0 if label.customer_present else 0.0}",
                f"operational_mode_normal: 1.0",
                f"phase_cash_depositing: 1.0"
            ]),
            ("Anomaly Indicators", [
                f"anomaly_count: {len(label.metadata.get('contextual_anomalies', []))}",
                f"high_rejection_rate: 1.0",
                f"quality_issues: 0.0",
                f"validation_failure: 1.0"
            ])
        ]
        
        total_features = 0
        for category, features in feature_dimensions:
            print(f"      📊 {category}:")
            for feature in features:
                print(f"         {feature}")
                total_features += 1
        
        print(f"\n      📈 Total Feature Vector Size: {total_features}+ dimensions")

def explain_ml_integration():
    """Explain how feature vectors enable ML integration"""
    
    print(f"\n🧠 4. HOW FEATURE VECTORS ENABLE ML LEARNING:")
    
    integration_examples = [
        ("Isolation Forest", [
            "Takes 50+ dimensional feature vectors",
            "Learns normal vs anomalous patterns",
            "Example: High rejection_rate + CIM deposit = potential anomaly",
            "Identifies outliers in multi-dimensional space"
        ]),
        ("BERT Embeddings", [
            "768-dimensional semantic vectors from enhanced text",
            "Contextual labeler enriches text before embedding",
            "Example: 'CIM-DEPOSIT [ESCROW:2, REJECTS:1]' → semantic vector",
            "Captures meaning beyond simple keyword matching"
        ]),
        ("Supervised Learning", [
            "Expert labels become training targets",
            "Feature vectors become training inputs",
            "Example: rejection_rate=0.33 + deposit_amount=10000 → 'high_risk_deposit'",
            "Model learns expert reasoning patterns"
        ]),
        ("Ensemble Voting", [
            "Multiple ML models vote using same feature vectors",
            "Contextual rules + Unsupervised + Supervised agreement",
            "Example: All models agree high rejection rate = anomaly",
            "Reduces false positives through consensus"
        ])
    ]
    
    for ml_type, explanations in integration_examples:
        print(f"\n   🎯 {ml_type}:")
        for explanation in explanations:
            print(f"      ✅ {explanation}")

def explain_adaptive_learning():
    """Explain the adaptive learning cycle"""
    
    print(f"\n🔄 5. THE ADAPTIVE LEARNING CYCLE:")
    
    cycle_steps = [
        ("Step 1: Feature Extraction", "Contextual Labeler converts logs → 50+ feature vectors"),
        ("Step 2: ML Detection", "Models process vectors → identify potential anomalies"),
        ("Step 3: Expert Review", "Domain experts label detected anomalies as true/false"),
        ("Step 4: Feature Learning", "ML learns which feature combinations = real anomalies"),
        ("Step 5: Enhanced Detection", "Updated models better recognize similar patterns"),
        ("Step 6: Continuous Improvement", "System adapts to new operational patterns automatically")
    ]
    
    for step, description in cycle_steps:
        print(f"   📍 {step}: {description}")

def explain_real_world_benefits():
    """Explain real-world benefits of this approach"""
    
    print(f"\n✨ 6. REAL-WORLD BENEFITS:")
    
    benefits = [
        ("🎯 Precision", "Feature vectors capture subtle patterns humans miss"),
        ("📚 Learning", "Every expert interaction improves future detection"),
        ("🔍 Discovery", "ML finds unknown anomaly patterns in feature space"),
        ("⚡ Speed", "No manual rule updates - system learns automatically"),
        ("🌐 Scalability", "Same feature extraction works across different ATM models"),
        ("🧠 Intelligence", "Combines domain expertise with ML pattern recognition"),
        ("📊 Explainability", "Can trace predictions back to specific features")
    ]
    
    for benefit, description in benefits:
        print(f"   {benefit} {description}")

def explain_technical_implementation():
    """Explain technical implementation details"""
    
    print(f"\n🔧 7. TECHNICAL IMPLEMENTATION:")
    
    print(f"\n   📦 Feature Categories (50+ total dimensions):")
    categories = [
        "Basic Statistics (3): text length, line count, word count",
        "Event Counts (11): CARD, PIN, NOTES, ERROR, TIMEOUT, etc.",
        "Financial Metrics (8): amounts, rejection rates, denomination data",
        "Timing Features (5): timestamp patterns, session duration",
        "Transaction Flow (6): phase transitions, sequence validation",
        "Quality Metrics (4): CAT1-CAT5 distributions, serial failures", 
        "Contextual Intelligence (8): confidence scores, operational modes",
        "Anomaly Indicators (5): error patterns, supervisor mode frequency",
        "CIM Status Features (6): escrow, validation, refusal counts"
    ]
    
    for category in categories:
        print(f"      🔹 {category}")
    
    print(f"\n   🎛️ Feature Engineering Techniques:")
    techniques = [
        "Categorical → One-Hot Encoding (event_type → binary features)",
        "Continuous → Normalization (amounts → 0-1 scale)",
        "Text → Count Vectorization (error patterns → frequencies)",
        "Sequences → Pattern Encoding (phase transitions → validity scores)",
        "Domain Knowledge → Expert Features (rejection rates → risk scores)"
    ]
    
    for technique in techniques:
        print(f"      ⚙️ {technique}")

if __name__ == "__main__":
    try:
        explain_feature_vectors()
        explain_ml_integration()
        explain_adaptive_learning()
        explain_real_world_benefits()
        explain_technical_implementation()
        
        print(f"\n🎉 SUMMARY:")
        print("Feature Vectors are the mathematical bridge that enables your EJ Contextual")
        print("Labeler to power machine learning. They transform domain expertise into")
        print("numerical format that ML models can learn from, creating an adaptive system")
        print("that improves with every expert interaction!")
        
    except Exception as e:
        print(f"\n💥 Analysis failed with error: {e}")
        import traceback
        traceback.print_exc()
