#!/usr/bin/env python3
"""
Test script for the real-time monitoring system
This script simulates parsing, sessionization, and ML training activities
"""
import sys
import os
import time
import random
import asyncio
from datetime import datetime

# Add the API directory to the path
sys.path.append('/Users/christopherpearson/Projects/abm_ej_exporter_docker_cleaned/EJAnomalyDetectionV3/abm-anomaly-ml-first/services/api')

try:
    from monitoring_utils import (
        update_parsing_stats, update_sessionization_stats, 
        update_ml_training_stats, log_component_activity
    )
    monitoring_available = True
    print("✅ Monitoring utilities imported successfully")
except ImportError as e:
    print(f"❌ Failed to import monitoring utilities: {e}")
    monitoring_available = False

def simulate_parsing_activity():
    """Simulate parsing activity for 30 seconds"""
    print("\n🔄 Simulating parsing activity...")
    
    for i in range(30):
        if monitoring_available:
            # Simulate processing 10-50 transactions per second
            processed = random.randint(10, 50)
            errors = random.randint(0, 2)
            rate = processed / 1.0  # per second, convert to per minute for display
            
            update_parsing_stats(
                processed_count=processed,
                error_count=errors,
                rate=rate * 60,  # convert to per minute
                status="active"
            )
            
            log_component_activity(
                component="parsing",
                activity=f"Processed {processed} transactions",
                details={"rate": rate, "errors": errors}
            )
        
        print(f"  📊 Second {i+1}: Processed {processed} transactions, {errors} errors")
        time.sleep(1)
    
    # Mark parsing as idle
    if monitoring_available:
        update_parsing_stats(status="idle")
    
    print("✅ Parsing simulation complete")

def simulate_sessionization_activity():
    """Simulate sessionization activity"""
    print("\n🔄 Simulating sessionization activity...")
    
    for i in range(20):
        if monitoring_available:
            # Simulate creating 1-5 sessions per iteration
            sessions_created = random.randint(1, 5)
            active_sessions = random.randint(10, 50)
            
            update_sessionization_stats(
                sessions_created=sessions_created,
                active_sessions=active_sessions,
                status="active"
            )
            
            log_component_activity(
                component="sessionization",
                activity=f"Created {sessions_created} new sessions",
                details={"active_sessions": active_sessions}
            )
        
        print(f"  📊 Iteration {i+1}: Created {sessions_created} sessions, {active_sessions} active")
        time.sleep(2)
    
    # Mark sessionization as idle
    if monitoring_available:
        update_sessionization_stats(status="idle")
    
    print("✅ Sessionization simulation complete")

def simulate_ml_training_activity():
    """Simulate ML training activity"""
    print("\n🔄 Simulating ML training activity...")
    
    training_start = time.time()
    
    # Simulate training initialization
    if monitoring_available:
        update_ml_training_stats(status="training", model_type="isolation_forest")
        log_component_activity(
            component="ml_training",
            activity="Started Isolation Forest training"
        )
    
    print("  🤖 Training Isolation Forest...")
    time.sleep(5)
    
    # Simulate training progress
    for epoch in range(10):
        accuracy = random.uniform(0.85, 0.95)
        
        if monitoring_available:
            update_ml_training_stats(
                accuracy=accuracy,
                training_time=time.time() - training_start,
                status="training"
            )
        
        print(f"  📈 Epoch {epoch+1}: Accuracy = {accuracy:.3f}")
        time.sleep(1)
    
    # Simulate training completion
    final_accuracy = random.uniform(0.90, 0.95)
    training_time = time.time() - training_start
    
    if monitoring_available:
        update_ml_training_stats(
            accuracy=final_accuracy,
            models_trained=1,
            training_time=training_time,
            status="idle",
            model_type="isolation_forest"
        )
        
        log_component_activity(
            component="ml_training",
            activity="Completed Isolation Forest training",
            details={
                "final_accuracy": final_accuracy,
                "training_time": training_time,
                "model_type": "isolation_forest"
            }
        )
    
    print(f"✅ ML training complete: Accuracy = {final_accuracy:.3f}, Time = {training_time:.1f}s")

def main():
    """Main test function"""
    print("🚀 Starting Real-time Monitoring System Test")
    print("=" * 50)
    
    if not monitoring_available:
        print("❌ Monitoring system not available. Please check Redis connection and dependencies.")
        return
    
    try:
        # Test parsing simulation
        simulate_parsing_activity()
        time.sleep(2)
        
        # Test sessionization simulation
        simulate_sessionization_activity()
        time.sleep(2)
        
        # Test ML training simulation
        simulate_ml_training_activity()
        time.sleep(2)
        
        print("\n" + "=" * 50)
        print("✅ All monitoring simulations completed successfully!")
        print("\n📋 Test Results:")
        print("  - Parsing activity: ✅ Simulated 30 seconds of transaction processing")
        print("  - Sessionization: ✅ Simulated 20 iterations of session creation")
        print("  - ML Training: ✅ Simulated complete training cycle")
        print("\n💡 You can now check the monitoring dashboard to see the real-time data!")
        
    except Exception as e:
        print(f"❌ Error during monitoring test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
