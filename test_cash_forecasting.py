#!/usr/bin/env python
"""
Test Cash Forecasting System
============================

Quick test to verify the cash forecasting system works correctly.
This uses synthetic data to demonstrate the functionality.
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_forecasting_system():
    """Test the cash forecasting system with basic functionality"""
    print("Testing Cash Forecasting System...")
    print("=" * 50)
    
    try:
        # Import the forecasting system
        from cash_forecasting_system import CashForecastingSystem
        
        # Initialize system
        forecaster = CashForecastingSystem()
        print("✓ System initialized successfully")
        
        # Load synthetic data
        df = forecaster.load_cassette_data()
        print(f"✓ Generated {len(df)} synthetic transactions")
        
        # Prepare features
        df_features = forecaster.prepare_features(df)
        print(f"✓ Prepared features for {len(df_features)} records")
        
        # Check data quality
        terminals = df_features['terminal_id'].unique()
        print(f"✓ Found {len(terminals)} terminals: {list(terminals)}")
        
        # Train models (subset for quick test)
        print("Training models (this may take a few minutes)...")
        forecaster.train_models(df_features)
        
        trained_terminals = len(forecaster.models)
        print(f"✓ Trained models for {trained_terminals} terminals")
        
        # Check model performance
        if forecaster.performance_metrics:
            avg_mae = sum(m['ensemble_mae'] for m in forecaster.performance_metrics.values()) / len(forecaster.performance_metrics)
            avg_r2 = sum(m['ensemble_r2'] for m in forecaster.performance_metrics.values()) / len(forecaster.performance_metrics)
            print(f"✓ Average MAE: ${avg_mae:.2f}")
            print(f"✓ Average R²: {avg_r2:.3f}")
        
        # Test prediction functionality
        if terminals:
            sample_terminal = terminals[0]
            prediction = forecaster.predict_cash_depletion(sample_terminal)
            if prediction:
                print(f"✓ Sample prediction for Terminal {sample_terminal}:")
                print(f"  Days until depletion: {prediction['days_until_depletion']:.1f}")
        
        print("\n" + "=" * 50)
        print("CASH FORECASTING SYSTEM TEST PASSED! ✓")
        print("=" * 50)
        
        return True
        
    except ImportError as e:
        print(f"✗ Import Error: {e}")
        print("Missing dependencies. Install with:")
        print("pip install -r cash_forecasting_requirements.txt")
        return False
        
    except Exception as e:
        print(f"✗ Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_forecasting_system()
    if not success:
        sys.exit(1)
