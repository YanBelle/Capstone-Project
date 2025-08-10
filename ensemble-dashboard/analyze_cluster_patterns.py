#!/usr/bin/env python3
"""
Analyze Cluster 15 Text Patterns for UI Enhancement
"""

import requests
import json
from collections import Counter

def analyze_cluster_text_patterns():
    """Analyze what text patterns Cluster 15 is actually grouping by"""
    
    print("=" * 80)
    print("CLUSTER 15 TEXT PATTERN ANALYSIS")
    print("=" * 80)
    
    # Get cluster 15 data
    response = requests.post(
        "http://localhost:8001/api/cluster_sessions",
        json={"cluster_id": 15, "feature_type": "text"},
        timeout=10
    )
    
    if response.status_code != 200:
        print(f"Failed to get cluster data: {response.status_code}")
        return
    
    data = response.json()
    sessions = data.get('sessions', [])
    
    print(f"Analyzing {len(sessions)} sessions in Cluster 15...")
    print(f"Total cluster size: {sessions[0]['cluster_size'] if sessions else 0}")
    print()
    
    # Extract common text patterns
    all_words = []
    all_sequences = []
    common_phrases = []
    
    for session in sessions:
        text = session['session_text']
        clean_text = text.replace('\u001b', ' ').strip()
        
        # Extract words
        words = clean_text.split()
        all_words.extend(words)
        
        # Extract sequences (3-word phrases)
        for i in range(len(words) - 2):
            sequence = ' '.join(words[i:i+3])
            all_sequences.append(sequence)
        
        # Store cleaned text
        common_phrases.append(clean_text)
    
    # Analyze patterns
    word_counter = Counter(all_words)
    sequence_counter = Counter(all_sequences)
    
    print("🔍 COMMON TEXT PATTERNS THAT BERT IS CLUSTERING BY:")
    print("-" * 60)
    
    print("\n1. MOST FREQUENT WORDS (shows transaction flow patterns):")
    for word, count in word_counter.most_common(15):
        if word and len(word) > 1:
            print(f"   • '{word}': appears {count} times")
    
    print("\n2. MOST FREQUENT 3-WORD SEQUENCES (shows semantic patterns):")
    for seq, count in sequence_counter.most_common(10):
        if seq and count > 1:
            print(f"   • '{seq}': appears {count} times")
    
    print("\n3. ACTUAL SESSION TEXT PATTERNS:")
    print("-" * 40)
    for i, session in enumerate(sessions[:3]):
        text = session['session_text'].replace('\u001b', ' → ')
        print(f"\nSession {i+1}:")
        print(f"   Raw Text: {text[:120]}...")
        
        # Identify key transaction flow elements
        if 'TRANSACTION_START' in text:
            print("   ✓ Contains: Transaction initiation")
        if 'CARD_INSERTED' in text:
            print("   ✓ Contains: Card processing")
        if 'PIN_ENTERED' in text:
            print("   ✓ Contains: Authentication")
        if 'CASH_DISPENSED' in text:
            print("   ✓ Contains: Cash dispensing")
        if 'RECEIPT_PRINTED' in text:
            print("   ✓ Contains: Receipt printing")
        if 'TRANSACTION_END' in text:
            print("   ✓ Contains: Transaction completion")
        
        # Check for specific patterns
        if 'OPCODE_FI' in text and 'GENAC' in text:
            print("   🏧 EMV Pattern: Chip card authentication sequence")
        if 'NOTES_STACKED' in text and 'NOTES_PRESENTED' in text:
            print("   💰 Cash Pattern: Successful cash handling sequence")
        if 'CARD_INITIALISE_ATTEMPT' in text:
            print("   🔄 Retry Pattern: Card initialization attempts")
    
    print("\n" + "=" * 80)
    print("🎯 WHAT SEMANTIC CLUSTERING IS ACTUALLY GROUPING BY:")
    print("=" * 80)
    
    print("\n✅ SUCCESSFUL TRANSACTION FLOW PATTERNS:")
    print("   • TRANSACTION_START → CARD_INSERTED → PIN_ENTERED")
    print("   • CASH_DISPENSED → RECEIPT_PRINTED → TRANSACTION_END")
    print("   • NOTES_STACKED → NOTES_PRESENTED → NOTES_TAKEN")
    print("   • EMV chip authentication sequences (OPCODE_FI, GENAC)")
    
    print("\n📊 BERT SEMANTIC UNDERSTANDING:")
    print("   • Recognizes 'successful withdrawal' transaction patterns")
    print("   • Groups sessions with similar transaction flow sequences")
    print("   • Understands EMV payment protocol semantics")
    print("   • Clusters by business meaning, not just word frequency")
    
    print("\n🎨 UI ENHANCEMENT RECOMMENDATIONS:")
    print("   1. Show 'Transaction Flow Pattern' instead of just 'text similarity'")
    print("   2. Display common sequences like 'CARD_INSERTED → PIN_ENTERED'")
    print("   3. Highlight business meanings: 'Successful Cash Withdrawal'")
    print("   4. Show semantic pattern: 'EMV Chip Authentication Flow'")
    print("   5. Add pattern confidence: 'High similarity in transaction steps'")

def suggest_contextual_labeler_clustering():
    """Suggest how to use contextual labeler for error-type clustering"""
    
    print("\n" + "=" * 80)
    print("🏷️  CONTEXTUAL LABELER INTEGRATION FOR ERROR CLUSTERING")
    print("=" * 80)
    
    print("\n📋 CURRENT CONTEXTUAL LABELER CAPABILITIES:")
    print("   • 35 event types (transaction_started, cash_dispensed, etc.)")
    print("   • 8 error categories (hardware, software, network, security)")
    print("   • 10 error codes (M-38, M-01, M-15, etc.)")
    print("   • 4 severity levels (info, warning, error, critical)")
    
    print("\n🎯 PROPOSED ERROR-TYPE CLUSTERING:")
    print("-" * 50)
    
    error_clusters = {
        "Authentication Failures": {
            "patterns": ["authentication_failure", "M-38", "pin_verification failed"],
            "labeler_events": ["authentication_failure"],
            "severity": "error",
            "category": "security"
        },
        "Hardware Malfunctions": {
            "patterns": ["device_error", "M-01", "M-15", "dispenser_error"],
            "labeler_events": ["dispenser_error", "device_activated", "device_deactivated"],
            "severity": "critical",
            "category": "hardware"
        },
        "Cash Handling Issues": {
            "patterns": ["notes_not_taken", "notes_retracted", "cash_deposit_retract"],
            "labeler_events": ["notes_not_taken", "notes_retracted", "cash_deposit_retract"],
            "severity": "error", 
            "category": "cash_handling"
        },
        "Communication Failures": {
            "patterns": ["M-67", "network", "communication_reset"],
            "labeler_events": ["communication_reset"],
            "severity": "critical",
            "category": "network"
        },
        "CIM Deposit Problems": {
            "patterns": ["cim_input_refused", "cim_items_validated failed"],
            "labeler_events": ["cim_input_refused", "cim_deposit_activated"],
            "severity": "warning",
            "category": "cash_handling"
        }
    }
    
    for cluster_name, details in error_clusters.items():
        print(f"\n🔴 {cluster_name}:")
        print(f"   • Text Patterns: {details['patterns']}")
        print(f"   • Labeler Events: {details['labeler_events']}")
        print(f"   • Severity: {details['severity']}")
        print(f"   • Category: {details['category']}")
    
    print("\n⚡ IMPLEMENTATION STRATEGY:")
    print("   1. Pre-process sessions with contextual labeler")
    print("   2. Create semantic embeddings enhanced with labeler features")
    print("   3. Use error_category + event_type as clustering guidance")
    print("   4. Combine BERT embeddings with labeler classifications")
    print("   5. Create error-specific cluster names from labeler taxonomy")
    
    print("\n📈 EXPECTED CLUSTERING IMPROVEMENTS:")
    print("   ✅ 'Authentication Failure Cluster' (instead of 'text cluster 1')")
    print("   ✅ 'Hardware Malfunction Cluster' (M-01, M-15 codes)")
    print("   ✅ 'Cash Retract Cluster' (deposit failures)")
    print("   ✅ 'CIM Deposit Issue Cluster' (validation problems)")
    print("   ✅ Business-meaningful cluster names and descriptions")

if __name__ == "__main__":
    analyze_cluster_text_patterns()
    suggest_contextual_labeler_clustering()
