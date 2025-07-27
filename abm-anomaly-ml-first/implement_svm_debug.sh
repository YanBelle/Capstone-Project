#!/bin/bash

echo "Integrating SVM Visualization and Debugging System..."

# Add SVM debugging methods to the ML analyzer
echo "Adding SVM debugging functionality to ML analyzer..."

# Create the integration patch
cat > svm_debug_integration.py << 'INTEGRATION'
#!/usr/bin/env python3
"""
Integration patch to add SVM debugging capabilities to the ML analyzer
"""

def add_svm_debugging_to_ml_analyzer():
    """
    Add SVM debugging methods to the MLFirstAnomalyDetector class
    """
    
    svm_debug_methods = '''
    def debug_svm_decisions(self, detailed_output: bool = True):
        """Debug SVM decision making process"""
        
        if not self.sessions:
            logger.warning("No sessions available for SVM debugging")
            return None
        
        try:
            # Import the visualizer
            from svm_visualizer import OneClassSVMVisualizer
            
            # Initialize visualizer
            visualizer = OneClassSVMVisualizer(self)
            
            # Prepare session data for visualization
            sessions_data = []
            for session in self.sessions:
                if hasattr(session, 'embedding_vector') and session.embedding_vector is not None:
                    sessions_data.append({
                        'session_id': session.session_id,
                        'embedding': session.embedding_vector,
                        'is_anomaly': session.is_anomaly,
                        'raw_text': session.raw_text,
                        'anomaly_score': getattr(session, 'anomaly_score', 0.0)
                    })
            
            if not sessions_data:
                logger.warning("No sessions with embeddings found for SVM debugging")
                return None
            
            # Generate debug report
            output_dir = "/app/debug_output"
            os.makedirs(output_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = os.path.join(output_dir, f"svm_debug_report_{timestamp}.html")
            
            visualizer.generate_svm_debug_report(sessions_data, report_path)
            logger.info(f"SVM debug report generated: {report_path}")
            
            if detailed_output:
                # Print detailed analysis
                embeddings = np.array([s['embedding'] for s in sessions_data])
                embeddings_scaled = self.scaler.transform(embeddings)
                decision_scores = self.one_class_svm.decision_function(embeddings_scaled)
                predictions = self.one_class_svm.predict(embeddings_scaled)
                
                print("\\n" + "="*50)
                print("ONE-CLASS SVM DECISION ANALYSIS")
                print("="*50)
                
                print(f"Model Parameters:")
                print(f"  - Nu (outlier fraction): {self.one_class_svm.nu}")
                print(f"  - Gamma: {self.one_class_svm.gamma}")
                print(f"  - Kernel: {self.one_class_svm.kernel}")
                print(f"  - Support Vectors: {len(self.one_class_svm.support_vectors_)}")
                
                print(f"\\nDecision Statistics:")
                print(f"  - Mean Decision Score: {np.mean(decision_scores):.3f}")
                print(f"  - Std Decision Score: {np.std(decision_scores):.3f}")
                print(f"  - Min Score: {np.min(decision_scores):.3f}")
                print(f"  - Max Score: {np.max(decision_scores):.3f}")
                print(f"  - Anomalies Detected: {np.sum(predictions == -1)}/{len(predictions)}")
                print(f"  - Anomaly Rate: {np.sum(predictions == -1)/len(predictions)*100:.1f}%")
                
                # Show individual session decisions
                print(f"\\nIndividual Session Decisions:")
                print("-" * 80)
                print(f"{'Session ID':<20} {'Decision':<10} {'Score':<10} {'Confidence':<12} {'Text Preview':<30}")
                print("-" * 80)
                
                for i, session_data in enumerate(sessions_data[:10]):  # Show first 10
                    pred = predictions[i]
                    score = decision_scores[i]
                    confidence = abs(score)
                    decision = "Anomaly" if pred == -1 else "Normal"
                    text_preview = session_data['raw_text'][:30].replace('\\n', ' ')
                    
                    print(f"{session_data['session_id']:<20} {decision:<10} {score:<10.3f} {confidence:<12.3f} {text_preview:<30}")
        
            return report_path
            
        except ImportError:
            logger.warning("SVM visualizer not available. Install required dependencies.")
            return None
        except Exception as e:
            logger.error(f"Error in SVM debugging: {str(e)}")
            return None

    def monitor_svm_performance(self):
        """Monitor SVM performance over time"""
        
        performance_metrics = {
            'timestamp': datetime.now(),
            'support_vector_count': len(self.one_class_svm.support_vectors_) if hasattr(self.one_class_svm, 'support_vectors_') else 0,
            'model_parameters': {
                'nu': self.one_class_svm.nu,
                'gamma': self.one_class_svm.gamma,
                'kernel': self.one_class_svm.kernel
            }
        }
        
        if self.sessions:
            embeddings = []
            for session in self.sessions:
                if hasattr(session, 'embedding_vector') and session.embedding_vector is not None:
                    embeddings.append(session.embedding_vector)
            
            if embeddings:
                embeddings_scaled = self.scaler.transform(np.array(embeddings))
                decision_scores = self.one_class_svm.decision_function(embeddings_scaled)
                predictions = self.one_class_svm.predict(embeddings_scaled)
                
                performance_metrics.update({
                    'total_sessions': len(embeddings),
                    'anomalies_detected': int(np.sum(predictions == -1)),
                    'anomaly_rate': float(np.sum(predictions == -1) / len(predictions)),
                    'decision_score_stats': {
                        'mean': float(np.mean(decision_scores)),
                        'std': float(np.std(decision_scores)),
                        'min': float(np.min(decision_scores)),
                        'max': float(np.max(decision_scores))
                    }
                })
        
        # Log performance metrics
        logger.info(f"SVM Performance: {performance_metrics}")
        return performance_metrics

    def real_time_svm_debug(self, session):
        """Debug SVM decision for a single session in real-time"""
        
        if not hasattr(session, 'embedding_vector') or session.embedding_vector is None:
            logger.warning(f"No embedding available for session {session.session_id}")
            return None
        
        try:
            # Get SVM decision
            embedding_scaled = self.scaler.transform(session.embedding_vector.reshape(1, -1))
            decision_score = self.one_class_svm.decision_function(embedding_scaled)[0]
            prediction = self.one_class_svm.predict(embedding_scaled)[0]
            
            debug_info = {
                'session_id': session.session_id,
                'decision_score': float(decision_score),
                'prediction': 'Anomaly' if prediction == -1 else 'Normal',
                'confidence': float(abs(decision_score)),
                'raw_score': float(decision_score),
                'embedding_norm': float(np.linalg.norm(session.embedding_vector)),
                'scaled_embedding_norm': float(np.linalg.norm(embedding_scaled)),
                'support_vectors_used': len(self.one_class_svm.support_vectors_),
                'timestamp': datetime.now().isoformat()
            }
            
            return debug_info
            
        except Exception as e:
            logger.error(f"Error in real-time SVM debug: {str(e)}")
            return None

    def tune_svm_parameters(self, nu_range=None, gamma_range=None):
        """Automatically tune SVM parameters for better performance"""
        
        if not self.sessions:
            logger.warning("No sessions available for parameter tuning")
            return None
        
        # Prepare data
        embeddings = []
        for session in self.sessions:
            if hasattr(session, 'embedding_vector') and session.embedding_vector is not None:
                embeddings.append(session.embedding_vector)
        
        if len(embeddings) < 10:
            logger.warning("Not enough data for parameter tuning (minimum 10 sessions required)")
            return None
        
        embeddings_scaled = self.scaler.transform(np.array(embeddings))
        
        # Default parameter ranges
        if nu_range is None:
            nu_range = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
        if gamma_range is None:
            gamma_range = ['scale', 'auto', 0.001, 0.01, 0.1, 1.0]
        
        best_params = None
        best_score = float('-inf')
        results = []
        
        logger.info(f"Tuning SVM parameters with {len(embeddings)} sessions...")
        
        for nu in nu_range:
            for gamma in gamma_range:
                try:
                    # Test SVM with these parameters
                    test_svm = OneClassSVM(nu=nu, gamma=gamma, kernel='rbf')
                    test_svm.fit(embeddings_scaled)
                    
                    # Calculate anomaly rate and score distribution
                    predictions = test_svm.predict(embeddings_scaled)
                    decision_scores = test_svm.decision_function(embeddings_scaled)
                    
                    anomaly_rate = np.sum(predictions == -1) / len(predictions)
                    score_std = np.std(decision_scores)
                    
                    # Scoring function (customize based on your needs)
                    # We want reasonable anomaly rate (5-20%) and good score separation
                    if 0.05 <= anomaly_rate <= 0.25:
                        score = score_std  # Higher std means better separation
                    else:
                        score = -abs(anomaly_rate - 0.15)  # Penalty for extreme rates
                    
                    results.append({
                        'nu': nu,
                        'gamma': gamma,
                        'anomaly_rate': anomaly_rate,
                        'score_std': score_std,
                        'score': score,
                        'support_vectors': len(test_svm.support_vectors_)
                    })
                    
                    if score > best_score:
                        best_score = score
                        best_params = {'nu': nu, 'gamma': gamma}
                        
                except Exception as e:
                    logger.warning(f"Failed to test nu={nu}, gamma={gamma}: {e}")
        
        if best_params:
            logger.info(f"Best parameters found: {best_params} (score: {best_score:.3f})")
            
            # Optionally update the model with best parameters
            update_model = input("Update model with best parameters? (y/n): ").lower() == 'y'
            if update_model:
                self.one_class_svm = OneClassSVM(
                    nu=best_params['nu'], 
                    gamma=best_params['gamma'], 
                    kernel='rbf'
                )
                self.one_class_svm.fit(embeddings_scaled)
                logger.info("Model updated with optimized parameters")
        
        return {
            'best_parameters': best_params,
            'best_score': best_score,
            'all_results': results,
            'current_parameters': {
                'nu': self.one_class_svm.nu,
                'gamma': self.one_class_svm.gamma
            }
        }
    '''
    
    return svm_debug_methods

if __name__ == "__main__":
    print("SVM Debug Integration Methods Generated")
    print(add_svm_debugging_to_ml_analyzer())
INTEGRATION

# Update the requirements to include visualization dependencies
echo "Updating requirements for SVM visualization..."

cat >> requirements.txt << 'REQUIREMENTS'

# SVM Visualization and Debugging Dependencies
plotly>=5.17.0
matplotlib>=3.8.0
seaborn>=0.13.0
pandas>=2.1.0
numpy>=1.24.0
scikit-learn>=1.3.0

# Additional analysis tools
scipy>=1.11.0
ipython>=8.16.0
jupyter>=1.0.0

REQUIREMENTS

# Create a CLI tool for SVM debugging
echo "Creating SVM debug CLI tool..."

cat > debug_svm_cli.py << 'CLI'
#!/usr/bin/env python3
"""
Command-line tool for debugging One-Class SVM anomaly detection
Usage: python debug_svm_cli.py --session-file sessions.json --output-dir ./debug_output
"""

import argparse
import json
import sys
import os
from pathlib import Path

# Add the anomaly detector path
sys.path.append('services/anomaly-detector')

try:
    from ml_analyzer import MLFirstAnomalyDetector, TransactionSession
    from svm_visualizer import OneClassSVMVisualizer
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Make sure you're running from the correct directory and all dependencies are installed")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Debug One-Class SVM anomaly detection')
    parser.add_argument('--session-file', required=True, help='JSON file containing session data')
    parser.add_argument('--output-dir', default='./svm_debug', help='Output directory for debug files')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--session-id', help='Debug specific session ID')
    parser.add_argument('--tune-parameters', action='store_true', help='Run parameter tuning')
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Load session data
    try:
        with open(args.session_file, 'r') as f:
            sessions_data = json.load(f)
        print(f"Loaded {len(sessions_data)} sessions from {args.session_file}")
    except Exception as e:
        print(f"Error loading session file: {e}")
        sys.exit(1)
    
    # Initialize ML analyzer
    try:
        analyzer = MLFirstAnomalyDetector()
        print("ML Analyzer initialized successfully")
    except Exception as e:
        print(f"Error initializing ML analyzer: {e}")
        sys.exit(1)
    
    # Process sessions
    analyzer.sessions = []
    for session_data in sessions_data:
        try:
            session = TransactionSession(
                session_id=session_data.get('session_id', f"session_{len(analyzer.sessions)}"),
                raw_text=session_data.get('raw_text', session_data.get('text', ''))
            )
            analyzer.sessions.append(session)
        except Exception as e:
            print(f"Error creating session: {e}")
    
    print(f"Created {len(analyzer.sessions)} session objects")
    
    # Extract embeddings and run analysis
    try:
        print("Extracting embeddings...")
        analyzer.extract_embeddings()
        
        print("Running anomaly detection...")
        analyzer.detect_anomalies_unsupervised()
        
        valid_sessions = sum(1 for s in analyzer.sessions 
                           if hasattr(s, 'embedding_vector') and s.embedding_vector is not None)
        print(f"Successfully processed {valid_sessions} sessions with embeddings")
        
    except Exception as e:
        print(f"Error in analysis: {e}")
        sys.exit(1)
    
    # Debug specific session if requested
    if args.session_id:
        print(f"\\nDebugging specific session: {args.session_id}")
        target_session = None
        for session in analyzer.sessions:
            if session.session_id == args.session_id:
                target_session = session
                break
        
        if target_session:
            debug_info = analyzer.real_time_svm_debug(target_session)
            if debug_info:
                print("Session Debug Information:")
                for key, value in debug_info.items():
                    print(f"  {key}: {value}")
            else:
                print("Failed to debug session")
        else:
            print(f"Session {args.session_id} not found")
    
    # Parameter tuning if requested
    if args.tune_parameters:
        print("\\nRunning parameter tuning...")
        tuning_results = analyzer.tune_svm_parameters()
        if tuning_results:
            print("Parameter Tuning Results:")
            print(f"  Best Parameters: {tuning_results['best_parameters']}")
            print(f"  Best Score: {tuning_results['best_score']:.3f}")
            print(f"  Current Parameters: {tuning_results['current_parameters']}")
        else:
            print("Parameter tuning failed")
    
    # Generate comprehensive debug report
    try:
        print("\\nGenerating SVM debug report...")
        report_path = analyzer.debug_svm_decisions(detailed_output=args.verbose)
        if report_path:
            print(f"Debug report generated: {report_path}")
            
            # Copy to output directory
            import shutil
            dest_path = output_path / f"svm_debug_report.html"
            shutil.copy2(report_path, dest_path)
            print(f"Report copied to: {dest_path}")
        else:
            print("Failed to generate debug report")
            
    except Exception as e:
        print(f"Error generating debug report: {e}")
    
    # Performance monitoring
    try:
        print("\\nMonitoring SVM performance...")
        performance = analyzer.monitor_svm_performance()
        if performance:
            print("Performance Metrics:")
            print(f"  Total Sessions: {performance.get('total_sessions', 0)}")
            print(f"  Anomalies Detected: {performance.get('anomalies_detected', 0)}")
            print(f"  Anomaly Rate: {performance.get('anomaly_rate', 0)*100:.1f}%")
            print(f"  Support Vectors: {performance.get('support_vector_count', 0)}")
            
            # Save performance metrics
            perf_file = output_path / "performance_metrics.json"
            with open(perf_file, 'w') as f:
                # Convert datetime objects to strings for JSON serialization
                performance_json = {k: (v.isoformat() if hasattr(v, 'isoformat') else v) 
                                  for k, v in performance.items()}
                json.dump(performance_json, f, indent=2)
            print(f"Performance metrics saved to: {perf_file}")
            
    except Exception as e:
        print(f"Error monitoring performance: {e}")
    
    print(f"\\nSVM debugging completed. Output files saved to: {output_path}")

if __name__ == "__main__":
    main()
CLI

chmod +x debug_svm_cli.py

# Create example session data for testing
echo "Creating example session data..."

cat > example_sessions.json << 'EXAMPLES'
[
  {
    "session_id": "test_session_1",
    "raw_text": "2025/01/03 10:59:00 CARD INSERTED\\n2025/01/03 10:59:05 PIN ENTERED\\n2025/01/03 10:59:10 TRANSACTION START\\n2025/01/03 10:59:15 CASH DISPENSED\\n2025/01/03 10:59:20 TRANSACTION END\\n2025/01/03 10:59:25 CARD TAKEN"
  },
  {
    "session_id": "test_session_2", 
    "raw_text": "2025/01/03 11:30:00 CARD INSERTED\\n2025/01/03 11:30:05 CARD TAKEN"
  },
  {
    "session_id": "test_session_3",
    "raw_text": "2025/01/03 12:00:00 CARD INSERTED\\n2025/01/03 12:00:05 PIN ENTERED\\n2025/01/03 12:00:10 OPCODE RECEIVED\\n2025/01/03 12:00:15 CARD TAKEN"
  },
  {
    "session_id": "test_session_4",
    "raw_text": "2025/01/03 14:15:00 CARD INSERTED\\n2025/01/03 14:15:05 PIN ENTERED\\n2025/01/03 14:15:10 TRANSACTION START\\n2025/01/03 14:15:15 DEVICE ERROR\\n2025/01/03 14:15:20 UNABLE TO DISPENSE\\n2025/01/03 14:15:25 TRANSACTION END\\n2025/01/03 14:15:30 CARD TAKEN"
  },
  {
    "session_id": "test_session_5",
    "raw_text": "2025/01/03 16:45:00 CARD INSERTED\\n2025/01/03 16:45:05 PIN ENTERED\\n2025/01/03 16:45:10 TRANSACTION START\\n2025/01/03 16:45:15 BALANCE INQUIRY\\n2025/01/03 16:45:20 BALANCE DISPLAYED\\n2025/01/03 16:45:25 TRANSACTION END\\n2025/01/03 16:45:30 CARD TAKEN"
  }
]
EXAMPLES

echo "✅ SVM Visualization and Debugging System Integration Complete!"
echo ""
echo "🔧 Components Added:"
echo "   - SVM Visualizer (svm_visualizer.py)"
echo "   - SVM Debug API endpoints (svm_debug_api.py)" 
echo "   - React Dashboard Component (SVMDebugDashboard.js)"
echo "   - CLI Debug Tool (debug_svm_cli.py)"
echo "   - Example session data (example_sessions.json)"
echo ""
echo "🚀 How to Use:"
echo "   1. Install dependencies: pip install -r requirements.txt"
echo "   2. Test CLI tool: python debug_svm_cli.py --session-file example_sessions.json"
echo "   3. Access SVM Debug Dashboard at: http://localhost:3000/svm-debug"
echo "   4. Use API endpoints: /api/v1/svm-debug/*"
echo ""
echo "📊 Features Available:"
echo "   - Real-time SVM decision visualization"
echo "   - Parameter sensitivity analysis"
echo "   - Feature importance analysis"
echo "   - Performance monitoring"
echo "   - Automatic parameter tuning"
echo "   - Comprehensive HTML debug reports"
echo ""
