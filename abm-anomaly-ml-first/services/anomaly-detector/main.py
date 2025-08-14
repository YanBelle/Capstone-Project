# ML-First ABM Anomaly Detection Service
import os
import sys
import time
import schedule
from loguru import logger
from dotenv import load_dotenv
from datetime import datetime
import pandas as pd
from sqlalchemy import create_engine, text
import redis
import json
import numpy as np
from typing import List, Dict, Tuple, Any
import re
from sklearn.preprocessing import LabelEncoder

# Import the ML-first anomaly detector - Use unified analyzer
try:
    # Try to import unified analyzer from shared directory
    import sys
    import os
    
    # Try multiple paths for the shared directory (dev vs container environments)
    shared_paths = [
        os.path.join(os.path.dirname(os.path.dirname(__file__)), 'shared'),  # Development
        '/app/shared',  # Container path
        '/app/../shared',  # Container relative path
        os.path.abspath(os.path.join(os.path.dirname(__file__), '../../shared'))  # Absolute dev path
    ]
    
    unified_imported = False
    for shared_path in shared_paths:
        try:
            if os.path.exists(shared_path):
                sys.path.insert(0, shared_path)
                from ml_analyzer_unified import UnifiedMLAnomalyDetector as MLFirstAnomalyDetector
                logger.info(f"Using Unified ML Analyzer from {shared_path}")
                unified_imported = True
                break
        except ImportError:
            continue
    
    if not unified_imported:
        raise ImportError("Unified analyzer not found in any path")
        
except ImportError as e:
    # Fallback to original analyzer
    from ml_analyzer import MLFirstAnomalyDetector
    logger.info(f"Using original ML Analyzer (fallback): {e}")

load_dotenv()

logger.add("/app/logs/anomaly_detector_{time}.log", rotation="100 MB")


class MLFirstEJProcessor:
    """Main processor for ML-first anomaly detection"""
    
    def __init__(self):
        # Database connection
        self.db_engine = create_engine(
            f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}"
            f"@{os.getenv('POSTGRES_HOST', 'postgres')}:5432/{os.getenv('POSTGRES_DB')}"
        )
        
        # Redis connection
        self.redis_client = redis.Redis(
            host=os.getenv('REDIS_HOST', 'redis'),
            port=6379,
            password=os.getenv('REDIS_PASSWORD'),
            decode_responses=True
        )
        
        # Initialize ML detector with database connection and service mode
        try:
            # Try unified analyzer first
            self.detector = MLFirstAnomalyDetector(
                model_name='bert-base-uncased', 
                db_engine=self.db_engine, 
                service_mode='anomaly-detector'
            )
            logger.info("Successfully initialized unified ML analyzer")
        except TypeError:
            # Fallback to original analyzer constructor (no service_mode parameter)
            self.detector = MLFirstAnomalyDetector(
                model_name='bert-base-uncased', 
                db_engine=self.db_engine
            )
            logger.info("Successfully initialized original ML analyzer (fallback)")
        
        # Load existing models if available
        self.load_models()
    
    def load_models(self):
        """Load pre-trained models if they exist"""
        model_dir = "/app/models"
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        if os.path.exists(os.path.join(model_dir, "isolation_forest.pkl")):
            logger.info("Loading existing ML models...")
            try:
                import joblib
                self.detector.isolation_forest = joblib.load(
                    os.path.join(model_dir, "isolation_forest.pkl")
                )
                self.detector.one_class_svm = joblib.load(
                    os.path.join(model_dir, "one_class_svm.pkl")
                )
                # Load DBSCAN model if available
                if os.path.exists(os.path.join(model_dir, "dbscan.pkl")):
                    self.detector.dbscan = joblib.load(
                        os.path.join(model_dir, "dbscan.pkl")
                    )
                    logger.info("Loaded DBSCAN model")
                
                self.detector.scaler = joblib.load(
                    os.path.join(model_dir, "scaler.pkl")
                )
                if os.path.exists(os.path.join(model_dir, "pca.pkl")):
                    self.detector.pca = joblib.load(
                        os.path.join(model_dir, "pca.pkl")
                    )
                logger.info("Models loaded successfully")
            except Exception as e:
                logger.warning(f"Error loading models: {str(e)}. Will train new models.")
                # Continue with default models instead of treating this as an error
        else:
            logger.info("No existing models found. Will train on first batch.")
    
    def process_ej_file(self, file_path: str):
        """Process an EJ log file with mode detection (training vs production)"""
        logger.info(f"Processing EJ file: {file_path}")
        
        # Check if file was already processed recently to avoid duplicates
        if self.should_skip_file(file_path):
            logger.info(f"Skipping {file_path} - already processed recently")
            return
        
        # Track current source file for cassette counter storage
        self.current_source_file = os.path.basename(file_path)
        
        # Determine processing mode based on model availability and system state
        processing_mode = self.determine_processing_mode()
        logger.info(f"Processing mode: {processing_mode}")
        
        try:
            if processing_mode == 'production':
                # Use production pipeline with trained supervised models
                result = self.process_production_ej_file(file_path)
                logger.info(f"Production processing result: {result['status']}")
                
            else:
                # Use training/development pipeline (original flow)
                logger.info("Using training/development pipeline")
                
                # Run ML-first detection pipeline
                results_df = self.detector.process_ej_logs(file_path)
                
                # Store sessions in database
                self.store_sessions(results_df)
                
                # Store anomalies
                anomalies_df = results_df[results_df['is_anomaly']]
                if len(anomalies_df) > 0:
                    self.store_anomalies(anomalies_df)
                    self.generate_alerts(anomalies_df)
                
                # Publish real-time updates
                self.publish_updates(results_df)
                
                # Save updated models if method exists
                try:
                    if hasattr(self.detector, 'save_models'):
                        self.detector.save_models("/app/models")
                    else:
                        logger.debug("Detector does not have save_models method")
                except Exception as e:
                    logger.warning(f"Failed to save models: {e}")
                
                logger.info(f"Training mode processing complete. Found {len(anomalies_df)} anomalies.")
                
                # Generate report
                self.generate_anomaly_report(anomalies_df)
            
        except Exception as e:
            logger.error(f"Error processing EJ file: {str(e)}")
            raise
    
    def determine_processing_mode(self) -> str:
        """
        Determine whether to use production or training mode based on:
        1. Availability of trained supervised models
        2. System configuration
        3. Model performance metrics
        """
        # Check if supervised model is available and trained
        has_supervised_model = (
            hasattr(self.detector, 'supervised_classifier') and 
            self.detector.supervised_classifier is not None
        )
        
        # Check environment variable for mode override
        force_mode = os.getenv('ANOMALY_DETECTION_MODE', '').lower()
        if force_mode in ['production', 'training']:
            logger.info(f"Processing mode forced via environment: {force_mode}")
            return force_mode
        
        # Auto-determine based on model availability
        if has_supervised_model:
            # Check if we have enough confidence in the supervised model
            model_confidence = self.assess_supervised_model_confidence()
            if model_confidence > 0.7:
                return 'production'
            else:
                logger.info(f"Supervised model confidence ({model_confidence:.2f}) below threshold, using training mode")
                return 'training'
        else:
            logger.info("No supervised model available, using training mode")
            return 'training'
    
    def assess_supervised_model_confidence(self) -> float:
        """Assess confidence in the supervised model based on training metrics"""
        try:
            # Check for model performance metrics file
            metrics_path = "/app/models/supervised_model_metrics.json"
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                return metrics.get('accuracy', 0.0)
            
            # If no metrics file, assume moderate confidence
            return 0.75
            
        except Exception as e:
            logger.warning(f"Could not assess model confidence: {e}")
            # Default to medium confidence
            return 0.5  # Default to medium confidence
    
    def train_supervised_models_from_labels(self) -> dict:
        """
        Train supervised models from expert-labeled anomalies in the database.
        This method is triggered after experts have labeled anomalies.
        """
        logger.info("Starting supervised model training from expert labels")
        
        try:
            # Query labeled anomalies from database
            labeled_data = self.get_labeled_training_data()
            
            if len(labeled_data) < 20:
                logger.warning(f"Insufficient labeled data for training: {len(labeled_data)} samples")
                return {
                    'status': 'insufficient_data',
                    'labeled_samples': len(labeled_data),
                    'minimum_required': 20
                }
            
            # Prepare training data
            X_train, y_train = self.prepare_supervised_training_data(labeled_data)
            
            # Train the supervised classifier
            training_results = self.detector.train_supervised_classifier(X_train, y_train)
            
            # Evaluate model performance
            performance_metrics = self.evaluate_supervised_model(X_train, y_train)
            
            # Save model and metrics if method exists
            try:
                if hasattr(self.detector, 'save_models'):
                    self.detector.save_models("/app/models")
                else:
                    logger.debug("Detector does not have save_models method")
            except Exception as e:
                logger.warning(f"Failed to save models: {e}")
            self.save_model_performance_metrics(performance_metrics)
            
            logger.info(f"Supervised training complete. Accuracy: {performance_metrics['accuracy']:.3f}")
            
            return {
                'status': 'success',
                'training_samples': len(labeled_data),
                'performance_metrics': performance_metrics,
                'model_path': '/app/models/supervised_classifier.pkl'
            }
            
        except Exception as e:
            logger.error(f"Error in supervised training: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def get_labeled_training_data(self) -> pd.DataFrame:
        """Retrieve labeled training data from database"""
        query = """
        SELECT 
            s.session_id,
            s.embedding_vector,
            s.raw_text,
            s.detected_patterns,
            s.critical_events,
            a.expert_label,
            a.expert_confidence,
            a.expert_notes,
            s.anomaly_type
        FROM ml_sessions s
        JOIN ml_anomaly_labels a ON s.session_id = a.session_id
        WHERE a.expert_label IS NOT NULL
        AND s.embedding_vector IS NOT NULL
        ORDER BY a.labeled_at DESC
        """
        
        with self.db_engine.connect() as conn:
            labeled_data = pd.read_sql(query, conn)
        
        logger.info(f"Retrieved {len(labeled_data)} labeled samples for training")
        return labeled_data
    
    def prepare_supervised_training_data(self, labeled_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data for supervised learning"""
        X_train = []
        y_train = []
        
        for _, row in labeled_data.iterrows():
            # Convert embedding from bytes back to array
            if row['embedding_vector'] is not None:
                embedding = np.frombuffer(row['embedding_vector'], dtype=np.float32)
                X_train.append(embedding)
                
                # Use expert label as ground truth
                y_train.append(row['expert_label'])
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        logger.info(f"Prepared training data: {X_train.shape[0]} samples, {len(np.unique(y_train))} classes")
        return X_train, y_train
    
    def evaluate_supervised_model(self, X_train: np.ndarray, y_train: np.ndarray) -> dict:
        """Evaluate supervised model performance using cross-validation"""
        from sklearn.model_selection import cross_val_score, train_test_split
        from sklearn.metrics import classification_report, confusion_matrix
        
        # Split data for evaluation
        X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        # Scale the data
        X_train_scaled = self.detector.scaler.transform(X_train_split)
        X_test_scaled = self.detector.scaler.transform(X_test_split)
        
        # Apply PCA if available
        if hasattr(self.detector.pca, 'components_'):
            X_train_scaled = self.detector.pca.transform(X_train_scaled)
            X_test_scaled = self.detector.pca.transform(X_test_scaled)
        
        # Fit on training split
        if self.detector.label_encoder is None:
            self.detector.label_encoder = LabelEncoder()
        
        y_train_encoded = self.detector.label_encoder.fit_transform(y_train_split)
        y_test_encoded = self.detector.label_encoder.transform(y_test_split)
        
        # Train and evaluate
        self.detector.supervised_classifier.fit(X_train_scaled, y_train_encoded)
        
        # Predictions
        y_pred = self.detector.supervised_classifier.predict(X_test_scaled)
        y_pred_proba = self.detector.supervised_classifier.predict_proba(X_test_scaled)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        
        accuracy = accuracy_score(y_test_encoded, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test_encoded, y_pred, average='weighted')
        
        # Cross-validation score
        cv_scores = cross_val_score(self.detector.supervised_classifier, X_train_scaled, y_train_encoded, cv=5)
        
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'cv_mean': float(cv_scores.mean()),
            'cv_std': float(cv_scores.std()),
            'training_samples': len(X_train),
            'test_samples': len(X_test_split),
            'unique_labels': len(np.unique(y_train)),
            'evaluation_timestamp': datetime.now().isoformat()
        }
        
        return metrics
    
    def save_model_performance_metrics(self, metrics: dict):
        """Save model performance metrics to file"""
        metrics_path = "/app/models/supervised_model_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Model performance metrics saved to {metrics_path}")
    
    def trigger_supervised_training(self) -> dict:
        """
        API endpoint method to trigger supervised training.
        Called by experts after they have labeled sufficient anomalies.
        """
        logger.info("Supervised training triggered by expert")
        
        # Check prerequisites
        if not hasattr(self.detector, 'scaler') or self.detector.scaler is None:
            logger.error("No feature scaler available - run unsupervised training first")
            return {
                'status': 'error',
                'message': 'No feature scaler available. Process some EJ files first to train base models.'
            }
        
        # Train the supervised models
        training_result = self.train_supervised_models_from_labels()
        
        # Update system mode if training was successful
        if training_result['status'] == 'success':
            # Set environment variable to use production mode
            os.environ['ANOMALY_DETECTION_MODE'] = 'production'
            logger.info("System switched to production mode after successful supervised training")
        
        return training_result
    
    def store_sessions(self, results_df: pd.DataFrame):
        """Store all sessions in database with embeddings and multi-anomaly support"""
        sessions_data = []
        cassette_data_list = []
        
        for i, (_, row) in enumerate(results_df.iterrows()):
            # Get the embedding for this session
            embedding = self.detector.sessions[i].embedding
            
            # Store texts on file system instead of database
            session_id = row['session_id']
            raw_text = self.detector.sessions[i].raw_text
            cleaned_text = self.detector.sessions[i].cleaned_text
            self.store_session_texts(session_id, raw_text, cleaned_text)
            
            session_data = {
                'session_id': session_id,
                'timestamp': row['start_time'] if pd.notna(row['start_time']) else datetime.now(),
                'session_length': row['session_length'],
                'is_anomaly': row['is_anomaly'],
                'anomaly_score': row['anomaly_score'],
                'anomaly_type': row['anomaly_type'] if row['anomaly_type'] else None,
                'detected_patterns': json.dumps(row['detected_patterns']),
                'critical_events': json.dumps(row['critical_events']),
                'embedding_vector': embedding.tobytes() if embedding is not None else None,
                # Removed raw_text from database storage - now stored on file system
                'terminal_id': self.detector.sessions[i].terminal_id,
                
                # Multi-anomaly fields
                'anomaly_count': row.get('anomaly_count', 0),
                'anomaly_types': json.dumps(row.get('anomaly_types', [])),
                'max_severity': row.get('max_severity', 'normal'),
                'overall_anomaly_score': row.get('overall_anomaly_score', 0.0),
                'critical_anomalies_count': row.get('critical_anomalies_count', 0),
                'high_severity_anomalies_count': row.get('high_severity_anomalies_count', 0),
                'detection_methods': json.dumps(row.get('detection_methods', [])),
                'anomalies_detail': json.dumps(row.get('anomalies_detail', [])),
                
                'created_at': datetime.now()
            }
            sessions_data.append(session_data)
            
            # Parse and collect cassette counter data for cash forecasting
            try:
                cassette_data = self.detector.parse_cassette_counters(self.detector.sessions[i])
                if cassette_data:
                    # Add source file information
                    cassette_data['source_file'] = getattr(self, 'current_source_file', 'unknown')
                    cassette_data_list.append(cassette_data)
                    logger.debug(f"Collected cassette data for session {session_id}")
            except Exception as e:
                logger.warning(f"Failed to parse cassette data for session {session_id}: {str(e)}")
        
        # Store sessions in database with conflict resolution
        logger.info(f"Storing {len(sessions_data)} sessions with conflict resolution and multi-anomaly support...")
        result = self.store_sessions_with_conflict_resolution(sessions_data)
        logger.info(f"Session storage complete - New: {result['success_count']}, Updated: {result['duplicate_count']}, Errors: {result['error_count']}")
        
        # Store cassette counter data for cash forecasting
        if cassette_data_list:
            cassette_result = self.store_cassette_counters(cassette_data_list)
            logger.info(f"Cassette counter storage complete - New: {cassette_result['success_count']}, Errors: {cassette_result['error_count']}")
        else:
            logger.info("No cassette counter data found in sessions")
    
    def store_session_texts(self, session_id: str, raw_text: str, cleaned_text: str = None):
        """Store raw and cleaned text for a session on file system"""
        # Store in file system with session_id prefix directories for better organization
        output_dir = f"/app/data/sessions/{session_id[:2]}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Store raw text
        with open(f"{output_dir}/{session_id}_raw.txt", 'w', encoding='utf-8') as f:
            f.write(raw_text)
        
        # Store cleaned text if provided
        if cleaned_text:
            with open(f"{output_dir}/{session_id}_cleaned.txt", 'w', encoding='utf-8') as f:
                f.write(cleaned_text)
    
    def store_session_raw_text(self, session_id: str, raw_text: str):
        """Store raw text for a session - deprecated, use store_session_texts"""
        logger.warning("store_session_raw_text is deprecated, use store_session_texts instead")
        self.store_session_texts(session_id, raw_text)
    
    def get_session_raw_text(self, session_id: str) -> str:
        """Retrieve raw text for a session from file system"""
        file_path = f"/app/data/sessions/{session_id[:2]}/{session_id}_raw.txt"
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            # Fallback to old file naming convention
            old_file_path = f"/app/data/sessions/{session_id[:2]}/{session_id}.txt"
            try:
                with open(old_file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except FileNotFoundError:
                logger.warning(f"Raw text file not found for session {session_id}")
                return ""
        except Exception as e:
            logger.error(f"Error reading raw text for session {session_id}: {e}")
            return ""
    
    def get_session_cleaned_text(self, session_id: str) -> str:
        """Retrieve cleaned text for a session from file system"""
        file_path = f"/app/data/sessions/{session_id[:2]}/{session_id}_cleaned.txt"
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            logger.warning(f"Cleaned text file not found for session {session_id}")
            return ""
        except Exception as e:
            logger.error(f"Error reading cleaned text for session {session_id}: {e}")
            return ""
    
    def get_session_texts(self, session_id: str) -> dict:
        """Retrieve both raw and cleaned text for a session from file system"""
        return {
            'raw_text': self.get_session_raw_text(session_id),
            'cleaned_text': self.get_session_cleaned_text(session_id)
        }
    
    def store_anomalies(self, anomalies_df: pd.DataFrame):
        """Store detected anomalies with ML-based details"""
        stored_count = 0
        error_count = 0
        
        for _, anomaly in anomalies_df.iterrows():
            try:
                session_id = anomaly['session_id']
                
                # First verify the session exists in ml_sessions table
                check_query = text("SELECT COUNT(*) FROM ml_sessions WHERE session_id = :session_id")
                
                with self.db_engine.connect() as conn:
                    result = conn.execute(check_query, {"session_id": session_id})
                    session_exists = result.scalar() > 0
                
                if not session_exists:
                    logger.warning(f"Session {session_id} not found in ml_sessions table, skipping anomaly insertion")
                    error_count += 1
                    continue
                
                # Insert anomaly record
                insert_query = text("""
                    INSERT INTO ml_anomalies 
                    (session_id, anomaly_type, anomaly_score, detected_patterns, 
                     critical_events, model_name, detected_at)
                    VALUES 
                    (:session_id, :anomaly_type, :anomaly_score, :detected_patterns,
                     :critical_events, :model_name, :detected_at)
                """)
                
                anomaly_data = {
                    'session_id': session_id,
                    'anomaly_type': anomaly['anomaly_type'] if anomaly['anomaly_type'] else 'unknown',
                    'anomaly_score': float(anomaly['anomaly_score']),
                    'detected_patterns': json.dumps(anomaly['detected_patterns']),
                    'critical_events': json.dumps(anomaly['critical_events']),
                    'model_name': 'ml_ensemble',
                    'detected_at': datetime.now()
                }
                
                with self.db_engine.connect() as conn:
                    conn.execute(insert_query, anomaly_data)
                    conn.commit()
                    stored_count += 1
                    logger.debug(f"Stored anomaly for session {session_id}")
                    
            except Exception as e:
                error_count += 1
                logger.error(f"Failed to store anomaly for session {session_id}: {e}")
        
        logger.info(f"Anomaly storage complete - Stored: {stored_count}, Errors: {error_count}")
    
    def generate_alerts(self, anomalies_df: pd.DataFrame):
        """Generate alerts for detected anomalies"""
        for _, anomaly in anomalies_df.iterrows():
            # Determine alert level
            alert_level = 'LOW'
            if anomaly['anomaly_score'] > 0.8:
                alert_level = 'HIGH'
            elif anomaly['anomaly_score'] > 0.6:
                alert_level = 'MEDIUM'
            
            # Check for critical patterns
            critical_patterns = [
                'unable_to_dispense', 
                'device_error', 
                'power_reset',
                'cash_retract',
                'recovery_failed'
            ]
            
            if any(pattern in anomaly['detected_patterns'] for pattern in critical_patterns):
                alert_level = 'HIGH'
            
            alert_data = {
                'alert_level': alert_level,
                'message': json.dumps({
                    'session_id': anomaly['session_id'],
                    'anomaly_type': anomaly['anomaly_type'],
                    'anomaly_score': float(anomaly['anomaly_score']),
                    'patterns': anomaly['detected_patterns'],
                    'critical_events': anomaly['critical_events'],
                    'description': self.generate_alert_description(anomaly)
                }),
                'is_resolved': False,
                'created_at': datetime.now()
            }
            
            pd.DataFrame([alert_data]).to_sql(
                'alerts', 
                self.db_engine, 
                if_exists='append', 
                index=False
            )
            
            # Publish real-time alert
            self.redis_client.publish(
                'anomaly_alerts',
                json.dumps({
                    'session_id': anomaly['session_id'],
                    'alert_level': alert_level,
                    'anomaly_score': float(anomaly['anomaly_score']),
                    'patterns': anomaly['detected_patterns'],
                    'critical_events': anomaly['critical_events'],
                    'timestamp': datetime.now().isoformat()
                })
            )
    
    def generate_alert_description(self, anomaly):
        """Generate human-readable description of the anomaly"""
        descriptions = []
        
        # Map patterns to descriptions
        pattern_descriptions = {
            'supervisor_mode': 'Supervisor mode activity detected',
            'unable_to_dispense': 'ATM unable to dispense cash',
            'device_error': 'Hardware device error occurred',
            'power_reset': 'Power reset or restart detected',
            'cash_retract': 'Cash retraction initiated',
            'no_dispense': 'Cash dispensing failed',
            'notes_issue': 'Issue with note handling',
            'note_error': 'Note processing error',
            'recovery_failed': 'Recovery operation failed'
        }
        
        for pattern in anomaly['detected_patterns']:
            if pattern in pattern_descriptions:
                descriptions.append(pattern_descriptions[pattern])
        
        # Add critical events
        for event in anomaly['critical_events']:
            descriptions.append(event)
        
        return '; '.join(descriptions) if descriptions else 'Anomalous pattern detected'
    
    def publish_updates(self, results_df: pd.DataFrame):
        """Publish dashboard updates via Redis"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_sessions': len(results_df),
            'total_anomalies': int(results_df['is_anomaly'].sum()),
            'anomaly_rate': float(results_df['is_anomaly'].mean()),
            'anomaly_types': {},
            'pattern_summary': {},
            'processing_mode': 'ml_first'
        }
        
        # Count anomaly types
        anomaly_types = results_df[results_df['is_anomaly']]['anomaly_type'].value_counts()
        summary['anomaly_types'] = anomaly_types.to_dict() if len(anomaly_types) > 0 else {}
        
        # Pattern frequency
        all_patterns = []
        for patterns in results_df[results_df['is_anomaly']]['detected_patterns']:
            all_patterns.extend(patterns)
        
        if all_patterns:
            pattern_counts = pd.Series(all_patterns).value_counts().head(5)
            summary['pattern_summary'] = pattern_counts.to_dict()
        
        # Publish to Redis
        self.redis_client.publish('dashboard_updates', json.dumps(summary))
        self.redis_client.setex('latest_ml_summary', 3600, json.dumps(summary))
    
    def generate_anomaly_report(self, anomalies_df: pd.DataFrame):
        """Generate detailed anomaly report"""
        if len(anomalies_df) == 0:
            return
        
        report = {
            'report_timestamp': datetime.now().isoformat(),
            'total_anomalies': len(anomalies_df),
            'anomaly_breakdown': {},
            'critical_findings': [],
            'pattern_analysis': {},
            'recommendations': []
        }
        
        # Anomaly type breakdown
        type_counts = anomalies_df['anomaly_type'].value_counts()
        report['anomaly_breakdown'] = type_counts.to_dict()
        
        # Critical findings
        for _, anomaly in anomalies_df.iterrows():
            if anomaly['anomaly_score'] > 0.8:
                finding = {
                    'session_id': anomaly['session_id'],
                    'score': float(anomaly['anomaly_score']),
                    'events': anomaly['critical_events']
                }
                report['critical_findings'].append(finding)
        
        # Pattern analysis
        all_patterns = []
        for patterns in anomalies_df['detected_patterns']:
            all_patterns.extend(patterns)
        
        pattern_counts = pd.Series(all_patterns).value_counts()
        report['pattern_analysis'] = pattern_counts.to_dict()
        
        # Generate recommendations
        if 'device_error' in pattern_counts:
            report['recommendations'].append(
                f"Hardware maintenance recommended - {pattern_counts['device_error']} device errors detected"
            )
        
        if 'power_reset' in pattern_counts:
            report['recommendations'].append(
                f"Power stability check needed - {pattern_counts['power_reset']} unexpected resets"
            )
        
        if 'unable_to_dispense' in pattern_counts:
            report['recommendations'].append(
                f"Cash handling mechanism inspection required - {pattern_counts['unable_to_dispense']} dispense failures"
            )
        
        # Add comprehensive anomaly summary if available
        try:
            if hasattr(self, 'detector') and self.detector:
                anomaly_summary = self.detector.generate_anomaly_summary_report()
                if anomaly_summary:
                    report['comprehensive_analysis'] = anomaly_summary
                    logger.info("Added comprehensive anomaly analysis to report")
        except Exception as e:
            logger.warning(f"Could not generate comprehensive anomaly summary: {e}")
        
        # Save report
        report_path = f"/app/output/anomaly_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Anomaly report generated: {report_path}")
    
    def scan_input_directory(self):
        """Scan for new EJ log files"""
        input_dir = "/app/input"
        processed_dir = "/app/input/processed"
        
        os.makedirs(processed_dir, exist_ok=True)
        
        for filename in os.listdir(input_dir):
            if filename.endswith('.txt') or filename.endswith('.log'):
                file_path = os.path.join(input_dir, filename)
                
                try:
                    # Check if file should be skipped
                    if self.should_skip_file(file_path):
                        continue
                    
                    # Process the file
                    self.process_ej_file(file_path)
                    
                    # Move to processed directory
                    os.rename(
                        file_path,
                        os.path.join(processed_dir, filename)
                    )
                    
                    logger.info(f"Successfully processed {filename}")
                    
                except Exception as e:
                    logger.error(f"Failed to process {filename}: {str(e)}")
    
    def process_production_ej_file(self, file_path: str) -> dict:
        """
        Production-mode processing for new EJ files after supervised training.
        Uses trained supervised models as primary detection method with unsupervised as fallback.
        """
        logger.info(f"Processing EJ file in PRODUCTION mode: {file_path}")
        
        try:
            # Process the EJ file using the trained models
            results_df = self.detector.process_ej_logs(file_path)
            
            # Enhanced production analysis
            production_summary = self.analyze_production_results(results_df)
            
            # Store sessions with production metadata
            self.store_production_sessions(results_df, file_path)
            
            # Generate production alerts for high-confidence anomalies
            alerts_generated = self.generate_production_alerts(results_df)
            
            # Create production report
            production_report = self.generate_production_report(results_df, production_summary, file_path)
            
            # Publish real-time updates for production dashboard
            self.publish_production_updates(results_df, production_summary)
            
            logger.info(f"Production processing complete. Analyzed {len(results_df)} sessions, "
                       f"found {len(results_df[results_df['is_anomaly']])} anomalies, "
                       f"generated {alerts_generated} alerts")
            
            return {
                'status': 'success',
                'file_path': file_path,
                'total_sessions': len(results_df),
                'anomalies_detected': len(results_df[results_df['is_anomaly']]),
                'alerts_generated': alerts_generated,
                'production_summary': production_summary,
                'report_path': production_report
            }
            
        except Exception as e:
            logger.error(f"Error in production processing: {str(e)}")
            return {
                'status': 'error',
                'file_path': file_path,
                'error': str(e)
            }
    
    def analyze_production_results(self, results_df: pd.DataFrame) -> dict:
        """Analyze production results for quality metrics and confidence levels"""
        if len(results_df) == 0:
            return {'status': 'no_data'}
        
        analysis = {
            'total_sessions': len(results_df),
            'anomalies_detected': len(results_df[results_df['is_anomaly']]),
            'anomaly_rate': float(results_df['is_anomaly'].mean()),
            'supervised_coverage': 0,
            'high_confidence_anomalies': 0,
            'model_agreement': 0,
            'detection_quality': 'unknown'
        }
        
        # Check supervised model coverage
        if 'supervised_label' in results_df.columns:
            supervised_predictions = results_df['supervised_label'].notna().sum()
            analysis['supervised_coverage'] = supervised_predictions / len(results_df)
            
            # High confidence anomalies (supervised model confident)
            if 'supervised_confidence' in results_df.columns:
                high_conf = results_df[
                    (results_df['supervised_confidence'] > 0.8) & 
                    (results_df['is_anomaly'] == True)
                ]
                analysis['high_confidence_anomalies'] = len(high_conf)
        
        # Determine detection quality based on coverage and confidence
        if analysis['supervised_coverage'] > 0.8:
            analysis['detection_quality'] = 'high'  # Mostly using trained supervised models
        elif analysis['supervised_coverage'] > 0.5:
            analysis['detection_quality'] = 'medium'  # Mixed supervised/unsupervised
        else:
            analysis['detection_quality'] = 'low'  # Mostly unsupervised fallback
        
        return analysis
    
    def store_production_sessions(self, results_df: pd.DataFrame, file_path: str):
        """Store sessions with production-specific metadata"""
        sessions_data = []
        cassette_data_list = []
        
        for i, (_, row) in enumerate(results_df.iterrows()):
            session_data = {
                'session_id': row['session_id'],
                'timestamp': row['start_time'] if pd.notna(row['start_time']) else datetime.now(),
                'session_length': row['session_length'],
                'is_anomaly': row['is_anomaly'],
                'anomaly_score': row['anomaly_score'],
                'anomaly_type': row['anomaly_type'] if row['anomaly_type'] else None,
                'detected_patterns': json.dumps(row['detected_patterns']),
                'critical_events': json.dumps(row['critical_events']),
                'processing_mode': 'production',
                'source_file': os.path.basename(file_path),
                'terminal_id': self.detector.sessions[i].terminal_id if i < len(self.detector.sessions) else None,  # Include terminal ID
                'created_at': datetime.now()
            }
            
            # Add supervised learning fields if available
            if 'supervised_label' in row:
                session_data['supervised_label'] = row['supervised_label']
                session_data['supervised_confidence'] = row.get('supervised_confidence', 0.0)
                session_data['detection_method'] = 'supervised' if row['supervised_confidence'] > 0.7 else 'ensemble'
            else:
                session_data['detection_method'] = 'unsupervised'
            
            # Add embedding if available
            if i < len(self.detector.sessions):
                embedding = self.detector.sessions[i].embedding
                session_data['embedding_vector'] = embedding.tobytes() if embedding is not None else None
                session_data['raw_text'] = self.detector.sessions[i].raw_text
            
            sessions_data.append(session_data)
            
            # Parse and collect cassette counter data for cash forecasting
            try:
                if i < len(self.detector.sessions):
                    cassette_data = self.detector.parse_cassette_counters(self.detector.sessions[i])
                    if cassette_data:
                        # Add source file information
                        cassette_data['source_file'] = os.path.basename(file_path)
                        cassette_data_list.append(cassette_data)
                        logger.debug(f"Collected cassette data for production session {session_data['session_id']}")
            except Exception as e:
                logger.warning(f"Failed to parse cassette data for production session {session_data['session_id']}: {str(e)}")
        
        # Store with conflict resolution
        result = self.store_sessions_with_conflict_resolution(sessions_data)
        logger.info(f"Production session storage: {result['success_count']} new, {result['duplicate_count']} updated")
        
        # Store cassette counter data for cash forecasting
        if cassette_data_list:
            cassette_result = self.store_cassette_counters(cassette_data_list)
            logger.info(f"Production cassette counter storage - New: {cassette_result['success_count']}, Errors: {cassette_result['error_count']}")
        else:
            logger.info("No cassette counter data found in production sessions")
    
    def generate_production_alerts(self, results_df: pd.DataFrame) -> int:
        """Generate production alerts for high-confidence anomalies"""
        alerts_generated = 0
        
        # Only generate alerts for high-confidence anomalies in production
        for _, anomaly in results_df[results_df['is_anomaly']].iterrows():
            # Determine if this is a high-confidence detection
            is_high_confidence = False
            
            # Check supervised confidence
            if 'supervised_confidence' in anomaly and anomaly['supervised_confidence'] > 0.8:
                is_high_confidence = True
            # Check unsupervised score if no supervised prediction
            elif anomaly['anomaly_score'] > 0.9:
                is_high_confidence = True
            
            if not is_high_confidence:
                continue  # Skip low-confidence anomalies in production
            
            # Determine alert level based on confidence and patterns
            alert_level = self.determine_production_alert_level(anomaly)
            
            alert_data = {
                'alert_level': alert_level,
                'message': json.dumps({
                    'session_id': anomaly['session_id'],
                    'anomaly_type': anomaly['anomaly_type'],
                    'anomaly_score': float(anomaly['anomaly_score']),
                    'supervised_confidence': anomaly.get('supervised_confidence', 0.0),
                    'detection_method': anomaly.get('detection_method', 'unknown'),
                    'patterns': anomaly['detected_patterns'],
                    'critical_events': anomaly['critical_events'],
                    'description': self.generate_production_alert_description(anomaly),
                    'production_mode': True
                }),
                'is_resolved': False,
                'created_at': datetime.now()
            }
            
            # Store alert
            pd.DataFrame([alert_data]).to_sql('alerts', self.db_engine, if_exists='append', index=False)
            
            # Publish real-time alert
            self.redis_client.publish('production_anomaly_alerts', json.dumps({
                'session_id': anomaly['session_id'],
                'alert_level': alert_level,
                'anomaly_score': float(anomaly['anomaly_score']),
                'supervised_confidence': anomaly.get('supervised_confidence', 0.0),
                'detection_method': anomaly.get('detection_method', 'unknown'),
                'patterns': anomaly['detected_patterns'],
                'critical_events': anomaly['critical_events'],
                'timestamp': datetime.now().isoformat(),
                'production_mode': True
            }))
            
            alerts_generated += 1
        
        return alerts_generated
    
    def determine_production_alert_level(self, anomaly) -> str:
        """Determine alert level for production anomalies"""
        # Start with supervised confidence if available
        if 'supervised_confidence' in anomaly and anomaly['supervised_confidence'] > 0.9:
            base_level = 'HIGH'
        elif anomaly['anomaly_score'] > 0.95:
            base_level = 'HIGH'
        elif anomaly['anomaly_score'] > 0.8:
            base_level = 'MEDIUM'
        else:
            base_level = 'LOW'
        
        # Escalate based on critical patterns
        critical_patterns = [
            'unable_to_dispense', 'device_error', 'power_reset',
            'cash_retract', 'recovery_failed', 'hardware_fault'
        ]
        
        if any(pattern in anomaly['detected_patterns'] for pattern in critical_patterns):
            base_level = 'HIGH'
        
        return base_level
    
    def generate_production_alert_description(self, anomaly) -> str:
        """Generate production-specific alert descriptions"""
        parts = []
        
        # Add detection method info
        detection_method = anomaly.get('detection_method', 'unknown')
        if detection_method == 'supervised':
            confidence = anomaly.get('supervised_confidence', 0.0)
            parts.append(f"Supervised model detection (confidence: {confidence:.2f})")
        elif detection_method == 'ensemble':
            parts.append(f"Ensemble detection (score: {anomaly['anomaly_score']:.2f})")
        else:
            parts.append(f"Unsupervised detection (score: {anomaly['anomaly_score']:.2f})")
        
        # Add pattern descriptions
        pattern_descriptions = {
            'supervisor_mode': 'Supervisor access detected',
            'unable_to_dispense': 'Cash dispensing failure',
            'device_error': 'Hardware malfunction',
            'power_reset': 'Unexpected system restart',
            'cash_retract': 'Cash retraction event',
            'recovery_failed': 'Recovery procedure failed'
        }
        
        for pattern in anomaly['detected_patterns']:
            if pattern in pattern_descriptions:
                parts.append(pattern_descriptions[pattern])
        
        return '; '.join(parts) if parts else 'Anomalous transaction pattern detected'
    
    def generate_production_report(self, results_df: pd.DataFrame, production_summary: dict, file_path: str) -> str:
        """Generate comprehensive production analysis report"""
        report = {
            'report_type': 'production_analysis',
            'timestamp': datetime.now().isoformat(),
            'source_file': os.path.basename(file_path),
            'processing_summary': production_summary,
            'anomaly_analysis': {},
            'model_performance': {},
            'recommendations': [],
            'next_actions': []
        }
        
        if len(results_df) > 0:
            # Anomaly breakdown
            anomalies = results_df[results_df['is_anomaly']]
            if len(anomalies) > 0:
                report['anomaly_analysis'] = {
                    'total_anomalies': len(anomalies),
                    'anomaly_types': anomalies['anomaly_type'].value_counts().to_dict(),
                    'severity_distribution': self.analyze_anomaly_severity(anomalies),
                    'detection_methods': anomalies.get('detection_method', pd.Series()).value_counts().to_dict()
                }
        
        # Model performance assessment
        report['model_performance'] = {
            'supervised_model_availability': self.detector.supervised_classifier is not None,
            'supervised_coverage': production_summary.get('supervised_coverage', 0),
            'detection_quality': production_summary.get('detection_quality', 'unknown'),
            'high_confidence_rate': production_summary.get('high_confidence_anomalies', 0) / max(1, production_summary.get('anomalies_detected', 1))
        }
        
        # Generate recommendations
        self.add_production_recommendations(report, production_summary, results_df)
        
        # Save report
        report_path = f"/app/output/production_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Production report generated: {report_path}")
        return report_path
    
    def analyze_anomaly_severity(self, anomalies: pd.DataFrame) -> dict:
        """Analyze severity distribution of detected anomalies"""
        severity_counts = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
        
        for _, anomaly in anomalies.iterrows():
            if anomaly['anomaly_score'] > 0.95:
                severity_counts['critical'] += 1
            elif anomaly['anomaly_score'] > 0.8:
                severity_counts['high'] += 1
            elif anomaly['anomaly_score'] > 0.6:
                severity_counts['medium'] += 1
            else:
                severity_counts['low'] += 1
        
        return severity_counts
    
    def add_production_recommendations(self, report: dict, summary: dict, results_df: pd.DataFrame):
        """Add production-specific recommendations to report"""
        # Model quality recommendations
        if summary['detection_quality'] == 'low':
            report['recommendations'].append("Consider retraining supervised models with more labeled data")
            report['next_actions'].append("Schedule expert review session for recent anomalies")
        
        # Anomaly pattern recommendations
        if len(results_df[results_df['is_anomaly']]) > 0:
            anomalies = results_df[results_df['is_anomaly']]
            pattern_counts = {}
            for patterns in anomalies['detected_patterns']:
                for pattern in patterns:
                    pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
            
            if pattern_counts.get('device_error', 0) > 5:
                report['recommendations'].append("Urgent: Multiple device errors detected - schedule hardware inspection")
                report['next_actions'].append("Contact maintenance team for device diagnostics")
            
            if pattern_counts.get('unable_to_dispense', 0) > 3:
                report['recommendations'].append("Cash dispensing issues detected - check cash levels and mechanism")
                report['next_actions'].append("Verify cash cassette status and dispenser calibration")
        
        # Performance recommendations
        if summary['supervised_coverage'] < 0.5:
            report['recommendations'].append("Low supervised model coverage - consider model retraining")
            report['next_actions'].append("Collect more labeled examples for supervised learning")
    
    def publish_production_updates(self, results_df: pd.DataFrame, production_summary: dict):
        """Publish production analysis updates to dashboard"""
        update = {
            'timestamp': datetime.now().isoformat(),
            'mode': 'production',
            'summary': production_summary,
            'anomaly_distribution': {},
            'quality_metrics': {
                'supervised_coverage': production_summary.get('supervised_coverage', 0),
                'detection_quality': production_summary.get('detection_quality', 'unknown'),
                'high_confidence_rate': production_summary.get('high_confidence_anomalies', 0) / max(1, production_summary.get('anomalies_detected', 1))
            }
        }
        
        # Add anomaly distribution
        if len(results_df[results_df['is_anomaly']]) > 0:
            anomaly_types = results_df[results_df['is_anomaly']]['anomaly_type'].value_counts()
            update['anomaly_distribution'] = anomaly_types.to_dict()
        
        # Publish updates
        self.redis_client.publish('production_dashboard_updates', json.dumps(update))
        self.redis_client.setex('latest_production_summary', 3600, json.dumps(update))
    
    def process_realtime_session(self, session_text: str, terminal_id: str = None) -> dict:
        """Process a single session in real-time (enhanced for production)"""
        try:
            # Create a temporary session - Import from unified analyzer
            try:
                from ml_analyzer_unified import TransactionSession
            except ImportError:
                from ml_analyzer import TransactionSession
            
            session = TransactionSession(
                session_id=f"realtime_{datetime.now().timestamp()}",
                raw_text=session_text,
                start_time=datetime.now(),
                end_time=None,
                terminal_id=terminal_id  # Include terminal ID if provided
            )
            
            # Get embedding
            embeddings = self.detector.convert_to_embeddings([session])
            
            # Priority 1: Use supervised model if available and trained
            if (hasattr(self.detector, 'supervised_classifier') and 
                self.detector.supervised_classifier is not None and
                hasattr(self.detector, 'scaler') and self.detector.scaler is not None):
                
                embeddings_scaled = self.detector.scaler.transform(embeddings)
                if hasattr(self.detector.pca, 'components_'):
                    embeddings_scaled = self.detector.pca.transform(embeddings_scaled)
                
                # Supervised prediction
                sup_pred = self.detector.supervised_classifier.predict(embeddings_scaled)[0]
                sup_proba = self.detector.supervised_classifier.predict_proba(embeddings_scaled)[0]
                sup_confidence = sup_proba.max()
                
                if self.detector.label_encoder:
                    sup_label = self.detector.label_encoder.inverse_transform([sup_pred])[0]
                else:
                    sup_label = str(sup_pred)
                
                result = {
                    'session_id': session.session_id,
                    'detection_method': 'supervised',
                    'is_anomaly': sup_label != 'normal',
                    'anomaly_type': sup_label if sup_label != 'normal' else None,
                    'supervised_confidence': float(sup_confidence),
                    'timestamp': datetime.now().isoformat()
                }
                
                # Also run unsupervised for comparison if confident enough
                if sup_confidence > 0.8:
                    result['primary_detection'] = 'supervised'
                else:
                    # Use ensemble approach for low confidence
                    if_score = self.detector.isolation_forest.score_samples(embeddings_scaled)[0]
                    if_pred = self.detector.isolation_forest.predict(embeddings_scaled)[0]
                    unsup_anomaly_score = max(0, min(1, (if_score - self.detector.isolation_forest.offset_) / -self.detector.isolation_forest.offset_))
                    
                    result['unsupervised_score'] = float(unsup_anomaly_score)
                    result['ensemble_decision'] = sup_confidence > 0.6 or (if_pred == -1 and unsup_anomaly_score > 0.8)
                    result['primary_detection'] = 'ensemble'
                    
                    if result['ensemble_decision']:
                        result['is_anomaly'] = True
                
            # Fallback: Use unsupervised ensemble if supervised not available
            elif hasattr(self.detector, 'scaler') and self.detector.scaler is not None:
                embeddings_scaled = self.detector.scaler.transform(embeddings)
                
                # Isolation Forest
                if_score = self.detector.isolation_forest.score_samples(embeddings_scaled)[0]
                if_pred = self.detector.isolation_forest.predict(embeddings_scaled)[0]
                if_anomaly_score = max(0, min(1, (if_score - self.detector.isolation_forest.offset_) / -self.detector.isolation_forest.offset_))
                
                # One-Class SVM
                svm_pred = self.detector.one_class_svm.predict(embeddings_scaled)[0]
                svm_score = self.detector.one_class_svm.decision_function(embeddings_scaled)[0]
                svm_anomaly_score = max(0, min(1, (svm_score + 1) / 2))  # Normalize to 0-1
                
                # DBSCAN (if available)
                dbscan_anomaly_score = 0.0
                dbscan_is_anomaly = False
                if hasattr(self.detector, 'dbscan') and self.detector.dbscan is not None:
                    try:
                        # For single point, we need to check against existing clusters
                        # This is a simplified approach for real-time processing
                        dbscan_pred = self.detector.dbscan.fit_predict(embeddings_scaled)
                        dbscan_is_anomaly = dbscan_pred[0] == -1
                        dbscan_anomaly_score = 1.0 if dbscan_is_anomaly else 0.0
                    except Exception as e:
                        logger.warning(f"DBSCAN processing failed: {e}")
                
                # Ensemble decision - majority voting with weighted scores
                ensemble_score = max(if_anomaly_score, svm_anomaly_score, dbscan_anomaly_score)
                is_anomaly = (if_pred == -1) or (svm_pred == -1) or dbscan_is_anomaly
                
                result = {
                    'session_id': session.session_id,
                    'detection_method': 'unsupervised_ensemble',
                    'is_anomaly': bool(is_anomaly),
                    'anomaly_score': float(ensemble_score),
                    'if_score': float(if_anomaly_score),
                    'svm_score': float(svm_anomaly_score),
                    'dbscan_score': float(dbscan_anomaly_score),
                    'timestamp': datetime.now().isoformat(),
                    'primary_detection': 'unsupervised_ensemble'
                }
            else:
                # No models trained yet
                return {
                    'session_id': session.session_id,
                    'detection_method': 'none',
                    'is_anomaly': False,
                    'anomaly_score': 0.0,
                    'message': 'No trained models available',
                    'timestamp': datetime.now().isoformat()
                }
            
            # Extract explanations if anomaly detected
            if result['is_anomaly']:
                session.is_anomaly = True
                session.anomaly_score = result.get('anomaly_score', result.get('supervised_confidence', 0.0))
                extracted = self.detector.extract_anomaly_reasons(session)
                result['patterns'] = extracted['detected_patterns']
                result['critical_events'] = extracted['critical_events']
            
            return result
                
        except Exception as e:
            logger.error(f"Error processing realtime session: {str(e)}")
            raise
    
    def store_sessions_with_conflict_resolution(self, sessions_data: List[Dict]):
        """Store sessions individually with conflict resolution"""
        
        success_count = 0
        duplicate_count = 0
        error_count = 0
        
        for session_data in sessions_data:
            try:
                # First check if session already exists
                check_query = text("SELECT COUNT(*) FROM ml_sessions WHERE session_id = :session_id")
                
                with self.db_engine.connect() as conn:
                    result = conn.execute(check_query, {"session_id": session_data['session_id']})
                    exists = result.scalar() > 0
                    
                if exists:
                    # Update existing session with new data
                    update_query = text("""
                        UPDATE ml_sessions SET 
                            timestamp = :timestamp,
                            session_length = :session_length,
                            is_anomaly = :is_anomaly,
                            anomaly_score = :anomaly_score,
                            anomaly_type = :anomaly_type,
                            detected_patterns = :detected_patterns,
                            critical_events = :critical_events,
                            embedding_vector = :embedding_vector,
                            terminal_id = :terminal_id,
                            created_at = :created_at
                        WHERE session_id = :session_id
                    """)
                    
                    # Prepare data for database - only include columns that exist in the schema
                    db_data = {
                        'session_id': session_data['session_id'],
                        'timestamp': session_data['timestamp'],
                        'session_length': session_data['session_length'],
                        'is_anomaly': session_data['is_anomaly'],
                        'anomaly_score': session_data['anomaly_score'],
                        'anomaly_type': session_data['anomaly_type'],
                        'detected_patterns': session_data['detected_patterns'],
                        'critical_events': session_data['critical_events'],
                        'embedding_vector': session_data['embedding_vector'],
                        'terminal_id': session_data.get('terminal_id'),
                        'created_at': session_data['created_at']
                    }
                    
                    with self.db_engine.connect() as conn:
                        conn.execute(update_query, db_data)
                        conn.commit()
                        duplicate_count += 1
                        logger.debug(f"Updated existing session: {session_data['session_id']}")
                        
                    # Store raw text to file system
                    if 'raw_text' in session_data:
                        self.store_session_texts(session_data['session_id'], session_data['raw_text'])
                        
                else:
                    # Insert new session
                    insert_query = text("""
                        INSERT INTO ml_sessions 
                        (session_id, timestamp, session_length, is_anomaly, anomaly_score, 
                         anomaly_type, detected_patterns, critical_events, embedding_vector, terminal_id, created_at)
                        VALUES 
                        (:session_id, :timestamp, :session_length, :is_anomaly, :anomaly_score,
                         :anomaly_type, :detected_patterns, :critical_events, :embedding_vector, :terminal_id, :created_at)
                    """)
                    
                    # Prepare data for database - only include columns that exist in the schema
                    db_data = {
                        'session_id': session_data['session_id'],
                        'timestamp': session_data['timestamp'],
                        'session_length': session_data['session_length'],
                        'is_anomaly': session_data['is_anomaly'],
                        'anomaly_score': session_data['anomaly_score'],
                        'anomaly_type': session_data['anomaly_type'],
                        'detected_patterns': session_data['detected_patterns'],
                        'critical_events': session_data['critical_events'],
                        'embedding_vector': session_data['embedding_vector'],
                        'terminal_id': session_data.get('terminal_id'),
                        'created_at': session_data['created_at']
                    }
                    
                    with self.db_engine.connect() as conn:
                        conn.execute(insert_query, db_data)
                        conn.commit()
                        success_count += 1
                        logger.debug(f"Inserted new session: {session_data['session_id']}")
                        
                    # Store raw text to file system
                    if 'raw_text' in session_data:
                        self.store_session_texts(session_data['session_id'], session_data['raw_text'])
                        
            except Exception as e:
                error_count += 1
                logger.error(f"Failed to store session {session_data['session_id']}: {e}")
        
        logger.info(f"Session storage complete - New: {success_count}, Updated: {duplicate_count}, Errors: {error_count}")
        
        return {
            "success_count": success_count,
            "duplicate_count": duplicate_count, 
            "error_count": error_count
        }
    
    def store_cassette_counters(self, cassette_data_list: List[Dict[str, Any]]) -> Dict[str, int]:
        """Store cassette counter data for cash forecasting"""
        success_count = 0
        error_count = 0
        
        logger.info(f"Storing {len(cassette_data_list)} cassette counter records")
        
        for cassette_data in cassette_data_list:
            try:
                # Check if cassette data already exists for this session
                check_query = text("""
                    SELECT COUNT(*) FROM cassette_counters 
                    WHERE session_id = :session_id
                """)
                
                with self.db_engine.connect() as conn:
                    result = conn.execute(check_query, {"session_id": cassette_data['session_id']})
                    exists = result.scalar() > 0
                
                if exists:
                    # Update existing record
                    update_query = text("""
                        UPDATE cassette_counters SET
                            terminal_id = :terminal_id,
                            transaction_datetime = :transaction_datetime,
                            cassette_1_remaining = :cassette_1_remaining,
                            cassette_2_remaining = :cassette_2_remaining,
                            cassette_3_remaining = :cassette_3_remaining,
                            cassette_4_remaining = :cassette_4_remaining,
                            cassette_1_denomination = :cassette_1_denomination,
                            cassette_2_denomination = :cassette_2_denomination,
                            cassette_3_denomination = :cassette_3_denomination,
                            cassette_4_denomination = :cassette_4_denomination,
                            cassette_1_dispensed = :cassette_1_dispensed,
                            cassette_2_dispensed = :cassette_2_dispensed,
                            cassette_3_dispensed = :cassette_3_dispensed,
                            cassette_4_dispensed = :cassette_4_dispensed,
                            cassette_1_rejected = :cassette_1_rejected,
                            cassette_2_rejected = :cassette_2_rejected,
                            cassette_3_rejected = :cassette_3_rejected,
                            cassette_4_rejected = :cassette_4_rejected,
                            total_dispensed_amount = :total_dispensed_amount,
                            total_rejected_amount = :total_rejected_amount,
                            withdrawal_successful = :withdrawal_successful,
                            source_file = :source_file,
                            raw_cassette_data = :raw_cassette_data,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE session_id = :session_id
                    """)
                    
                    with self.db_engine.connect() as conn:
                        conn.execute(update_query, cassette_data)
                        conn.commit()
                        success_count += 1
                        logger.debug(f"Updated cassette data for session: {cassette_data['session_id']}")
                else:
                    # Insert new record
                    insert_query = text("""
                        INSERT INTO cassette_counters 
                        (session_id, terminal_id, transaction_datetime,
                         cassette_1_remaining, cassette_2_remaining, cassette_3_remaining, cassette_4_remaining,
                         cassette_1_denomination, cassette_2_denomination, cassette_3_denomination, cassette_4_denomination,
                         cassette_1_dispensed, cassette_2_dispensed, cassette_3_dispensed, cassette_4_dispensed,
                         cassette_1_rejected, cassette_2_rejected, cassette_3_rejected, cassette_4_rejected,
                         total_dispensed_amount, total_rejected_amount, withdrawal_successful,
                         source_file, raw_cassette_data)
                        VALUES 
                        (:session_id, :terminal_id, :transaction_datetime,
                         :cassette_1_remaining, :cassette_2_remaining, :cassette_3_remaining, :cassette_4_remaining,
                         :cassette_1_denomination, :cassette_2_denomination, :cassette_3_denomination, :cassette_4_denomination,
                         :cassette_1_dispensed, :cassette_2_dispensed, :cassette_3_dispensed, :cassette_4_dispensed,
                         :cassette_1_rejected, :cassette_2_rejected, :cassette_3_rejected, :cassette_4_rejected,
                         :total_dispensed_amount, :total_rejected_amount, :withdrawal_successful,
                         :source_file, :raw_cassette_data)
                    """)
                    
                    with self.db_engine.connect() as conn:
                        conn.execute(insert_query, cassette_data)
                        conn.commit()
                        success_count += 1
                        logger.debug(f"Inserted cassette data for session: {cassette_data['session_id']}")
                        
            except Exception as e:
                error_count += 1
                logger.error(f"Failed to store cassette data for session {cassette_data['session_id']}: {e}")
        
        logger.info(f"Cassette counter storage complete - New/Updated: {success_count}, Errors: {error_count}")
        
        return {
            "success_count": success_count,
            "error_count": error_count
        }

    def should_skip_file(self, file_path: str) -> bool:
        """Check if file has already been processed recently"""
        file_name = os.path.basename(file_path)
        
        # Force processing for development/testing - temporarily disable skip logic
        # TODO: Re-enable file skip logic for production
        logger.info(f"Force processing enabled for {file_name}")
        return False
        
        # Check if we have a record of processing this file in the last 24 hours
        try:
            check_query = text("""
                SELECT COUNT(*) FROM ml_sessions 
                WHERE session_id LIKE :file_pattern 
                AND created_at > NOW() - INTERVAL '24 hours'
            """)
            
            # Extract file identifier for pattern matching
            file_match = re.search(r'ABM(\d+)EJ_(\d{8})_(\d{8})', file_name)
            if file_match:
                abm_num = file_match.group(1)
                start_date = file_match.group(2)
                file_pattern = f"ABM{abm_num}_{start_date}%"
                
                with self.db_engine.connect() as conn:
                    result = conn.execute(check_query, {"file_pattern": file_pattern})
                    count = result.scalar()
                    
                if count > 0:
                    logger.info(f"Skipping {file_name} - already processed {count} sessions in last 24 hours")
                    return True
                    
        except Exception as e:
            logger.warning(f"Could not check file processing status: {e}")
            
        return False
def run_ml_anomaly_detection():
    """Run the ML-first anomaly detection process"""
    processor = MLFirstEJProcessor()
    processor.scan_input_directory()


def main():
    logger.info("ML-First ABM Anomaly Detector Service Started")
    
    # Schedule periodic runs
    interval = int(os.getenv('MODEL_UPDATE_INTERVAL', 3600))
    schedule.every(interval).seconds.do(run_ml_anomaly_detection)
    
    # Run once on startup
    run_ml_anomaly_detection()
    
    # Keep running
    while True:
        schedule.run_pending()
        time.sleep(60)


if __name__ == "__main__":
    main()
