#!/usr/bin/env python3
"""
Direct test of BERT visualization without containers
"""

import sys
sys.path.append('/Users/christopherpearson/Projects/abm_ej_exporter_docker_cleaned/EJAnomalyDetectionV3/abm-anomaly-ml-first/services/api')

from bertviz_analyzer import BertVizAnalyzer
import json

def test_bert_visualization():
    try:
        print("Testing BERT visualization...")
        
        # Initialize analyzer
        analyzer = BertVizAnalyzer()
        
        # Test sample text
        text = "This is a test transaction for anomaly detection analysis"
        
        print(f"Analyzing text: {text}")
        
        # Get visualization
        result = analyzer.get_visualization(text)
        
        print("Result keys:", list(result.keys()))
        
        if 'attention_heatmap' in result:
            print("✅ Attention heatmap generated")
            print(f"Heatmap type: {type(result['attention_heatmap'])}")
            
        if 'token_importance' in result:
            print("✅ Token importance generated") 
            print(f"Token importance type: {type(result['token_importance'])}")
            
        if 'tokens' in result:
            print("✅ Tokens extracted")
            print(f"Tokens: {result['tokens']}")
            
        print("BERT visualization test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error in BERT visualization: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_bert_visualization()
    sys.exit(0 if success else 1)
