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
from typing import List, Dict
import re

# Import the ML-first anomaly detector
from ml_analyzer import MLFirstAnomalyDetector

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
        
        # Initialize ML detector with database connection
        self.detector = MLFirstAnomalyDetector(db_engine=self.db_engine)
        
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
        """Process an EJ log file using ML-first approach"""
        logger.info(f"Processing EJ file: {file_path}")
        
        # Check if file was already processed recently to avoid duplicates
        if self.should_skip_file(file_path):
            logger.info(f"Skipping {file_path} - already processed recently")
            return
        
        try:
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
            
            # Save updated models
            self.detector.save_models("/app/models")
            
            logger.info(f"Processing complete. Found {len(anomalies_df)} anomalies.")
            
            # Generate report
            self.generate_anomaly_report(anomalies_df)
            
        except Exception as e:
            logger.error(f"Error processing EJ file: {str(e)}")
            raise
    
    def store_sessions(self, results_df: pd.DataFrame):
        """Store all sessions in database with embeddings and multi-anomaly support"""
        sessions_data = []
        
        for i, (_, row) in enumerate(results_df.iterrows()):
            # Get the embedding for this session
            embedding = self.detector.sessions[i].embedding
            
            # Store raw text
            session_id = row['session_id']
            raw_text = self.detector.sessions[i].raw_text
            self.store_session_raw_text(session_id, raw_text)
            
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
        
        # Store in database with conflict resolution - always use individual inserts
        logger.info(f"Storing {len(sessions_data)} sessions with conflict resolution and multi-anomaly support...")
        result = self.store_sessions_with_conflict_resolution(sessions_data)
        logger.info(f"Storage complete - New: {result['success_count']}, Updated: {result['duplicate_count']}, Errors: {result['error_count']}")
    
    def store_session_raw_text(self, session_id: str, raw_text: str):
        """Store raw text for a session"""
        # Store in file system
        output_dir = f"/app/data/sessions/{session_id[:2]}"
        os.makedirs(output_dir, exist_ok=True)
        
        with open(f"{output_dir}/{session_id}.txt", 'w') as f:
            f.write(raw_text)
    
    def store_anomalies(self, anomalies_df: pd.DataFrame):
        """Store detected anomalies with ML-based details"""
        for _, anomaly in anomalies_df.iterrows():
            anomaly_data = {
                'session_id': anomaly['session_id'],
                'anomaly_type': anomaly['anomaly_type'] if anomaly['anomaly_type'] else 'unknown',
                'anomaly_score': float(anomaly['anomaly_score']),
                'detected_patterns': json.dumps(anomaly['detected_patterns']),
                'critical_events': json.dumps(anomaly['critical_events']),
                'model_name': 'ml_ensemble',
                'detected_at': datetime.now()
            }
            
            pd.DataFrame([anomaly_data]).to_sql(
                'ml_anomalies', 
                self.db_engine, 
                if_exists='append', 
                index=False
            )
    
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
    
    def process_realtime_session(self, session_text: str) -> dict:
        """Process a single session in real-time"""
        try:
            # Create a temporary session
            from ml_analyzer import TransactionSession
            
            session = TransactionSession(
                session_id=f"realtime_{datetime.now().timestamp()}",
                raw_text=session_text,
                start_time=datetime.now(),
                end_time=None
            )
            
            # Get embedding
            embeddings = self.detector.convert_to_embeddings([session])
            
            # Check if anomaly using existing models
            if hasattr(self.detector, 'scaler') and self.detector.scaler is not None:
                embeddings_scaled = self.detector.scaler.transform(embeddings)
                
                # Get predictions
                if_score = self.detector.isolation_forest.score_samples(embeddings_scaled)[0]
                if_pred = self.detector.isolation_forest.predict(embeddings_scaled)[0]
                
                # Normalize score
                anomaly_score = (if_score - self.detector.isolation_forest.offset_) / -self.detector.isolation_forest.offset_
                anomaly_score = max(0, min(1, anomaly_score))
                
                is_anomaly = if_pred == -1
                
                result = {
                    'session_id': session.session_id,
                    'is_anomaly': bool(is_anomaly),
                    'anomaly_score': float(anomaly_score),
                    'timestamp': datetime.now().isoformat()
                }
                
                # If anomaly, extract reasons
                if is_anomaly:
                    session.is_anomaly = True
                    session.anomaly_score = anomaly_score
                    extracted = self.detector.extract_anomaly_reasons(session)
                    result['patterns'] = extracted['detected_patterns']
                    result['critical_events'] = extracted['critical_events']
                
                return result
            else:
                # Models not trained yet
                return {
                    'session_id': session.session_id,
                    'is_anomaly': False,
                    'anomaly_score': 0.0,
                    'message': 'ML models not yet trained'
                }
                
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
                            created_at = :created_at
                        WHERE session_id = :session_id
                    """)
                    
                    with self.db_engine.connect() as conn:
                        conn.execute(update_query, session_data)
                        conn.commit()
                        duplicate_count += 1
                        logger.debug(f"Updated existing session: {session_data['session_id']}")
                else:
                    # Insert new session
                    insert_query = text("""
                        INSERT INTO ml_sessions 
                        (session_id, timestamp, session_length, is_anomaly, anomaly_score, 
                         anomaly_type, detected_patterns, critical_events, embedding_vector, created_at)
                        VALUES 
                        (:session_id, :timestamp, :session_length, :is_anomaly, :anomaly_score,
                         :anomaly_type, :detected_patterns, :critical_events, :embedding_vector, :created_at)
                    """)
                    
                    with self.db_engine.connect() as conn:
                        conn.execute(insert_query, session_data)
                        conn.commit()
                        success_count += 1
                        logger.debug(f"Inserted new session: {session_data['session_id']}")
                        
            except Exception as e:
                error_count += 1
                logger.error(f"Failed to store session {session_data['session_id']}: {e}")
        
        logger.info(f"Session storage complete - New: {success_count}, Updated: {duplicate_count}, Errors: {error_count}")
        
        return {
            "success_count": success_count,
            "duplicate_count": duplicate_count, 
            "error_count": error_count
        }
    
    def should_skip_file(self, file_path: str) -> bool:
        """Check if file has already been processed recently"""
        file_name = os.path.basename(file_path)
        
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
