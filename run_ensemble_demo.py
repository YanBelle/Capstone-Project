#!/usr/bin/env python3
"""
Quick ensemble training results demo
"""

import sys
import os
sys.path.append('/Users/christopherpearson/Projects/abm_ej_exporter_docker_cleaned/EJAnomalyDetectionV3')

from basic_ensemble_trainer import BasicEnsembleTrainer

def main():
    print("🚀 Running Ensemble Training Demo")
    print("=" * 50)
    
    # Create trainer
    trainer = BasicEnsembleTrainer()
    
    # Load data
    trainer.load_ej_sessions()
    
    # Extract features
    trainer.prepare_training_data()
    
    # Train models
    trainer.simple_anomaly_detection()
    
    # Generate report
    trainer.generate_detailed_report()
    
    print("\n🎉 Demo complete!")
    print(f"📊 Visualization saved at: {trainer.output_dir}/ensemble_training_results.png")

if __name__ == "__main__":
    main()
