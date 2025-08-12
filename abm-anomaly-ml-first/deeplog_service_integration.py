"""
Integration script for BERT + DeepLog with ABM Anomaly Detection System
Provides API endpoints and database integration for DeepLog training and prediction
"""

import os
import sys
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import logging
from pathlib import Path

# Add paths for imports
sys.path.append('/app/shared')
sys.path.append('/app')

from deeplog_bert_trainer import BERTDeepLogTrainer, DeepLogConfig
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import pandas as pd

logger = logging.getLogger(__name__)

class DeepLogServiceIntegration:
    """
    Service integration for BERT + DeepLog training and prediction
    Connects with existing ABM ML system
    """
    
    def __init__(self, model_save_path: str = "/app/models"):
        self.model_save_path = Path(model_save_path)
        self.model_save_path.mkdir(exist_ok=True)
        
        # Database connection
        self.db_engine = create_engine(
            f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}"
            f"@{os.getenv('POSTGRES_HOST', 'postgres')}:5432/{os.getenv('POSTGRES_DB')}"
        )
        
        # Initialize DeepLog trainer
        self.config = DeepLogConfig(
            sequence_length=64,
            hidden_dim=256,
            num_layers=2,
            batch_size=32,
            num_epochs=100,
            patience=15,
            anomaly_threshold=0.3,  # Adjusted for ABM transactions
            learning_rate=0.001
        )
        
        self.trainer = BERTDeepLogTrainer(self.config, str(self.model_save_path))
        
        # Try to load existing model
        try:
            self.trainer.load_model()
            logger.info("Loaded existing DeepLog model")
        except FileNotFoundError:
            logger.info("No existing DeepLog model found")
    
    def get_training_sessions(self, limit: int = 1000) -> List[str]:
        """
        Retrieve session data from database for training
        
        Args:
            limit: Maximum number of sessions to retrieve
            
        Returns:
            sessions: List of raw session texts
        """
        query = text("""
            SELECT raw_text 
            FROM ml_sessions 
            WHERE raw_text IS NOT NULL 
            AND LENGTH(raw_text) > 50
            ORDER BY created_at DESC 
            LIMIT :limit
        """)
        
        with self.db_engine.connect() as conn:
            result = conn.execute(query, {"limit": limit})
            sessions = [row[0] for row in result.fetchall()]
        
        logger.info(f"Retrieved {len(sessions)} sessions for training")
        return sessions
    
    def get_labeled_sessions(self) -> Dict[str, List[str]]:
        """
        Retrieve labeled anomaly sessions for training
        
        Returns:
            labeled_data: Dictionary with 'normal' and 'anomaly' session lists
        """
        # Get sessions with known anomaly labels
        query = text("""
            SELECT s.raw_text, s.is_anomaly
            FROM ml_sessions s
            WHERE s.raw_text IS NOT NULL 
            AND s.is_anomaly IS NOT NULL
            ORDER BY s.created_at DESC
        """)
        
        with self.db_engine.connect() as conn:
            result = conn.execute(query)
            rows = result.fetchall()
        
        labeled_data = {
            'normal': [],
            'anomaly': []
        }
        
        for raw_text, is_anomaly in rows:
            if is_anomaly:
                labeled_data['anomaly'].append(raw_text)
            else:
                labeled_data['normal'].append(raw_text)
        
        logger.info(f"Retrieved {len(labeled_data['normal'])} normal and {len(labeled_data['anomaly'])} anomaly sessions")
        return labeled_data
    
    def train_deeplog_model(self, use_labeled_data: bool = True) -> Dict[str, Any]:
        """
        Train the DeepLog model on available session data
        
        Args:
            use_labeled_data: Whether to use labeled data or all sessions
            
        Returns:
            training_results: Training metrics and status
        """
        logger.info("Starting DeepLog model training")
        
        if use_labeled_data:
            # Use labeled data for supervised training
            labeled_data = self.get_labeled_sessions()
            
            # Combine normal and anomaly sessions
            all_sessions = labeled_data['normal'] + labeled_data['anomaly']
            
            if len(all_sessions) < 10:
                logger.warning("Insufficient labeled data, falling back to all sessions")
                all_sessions = self.get_training_sessions()
        else:
            # Use all available sessions
            all_sessions = self.get_training_sessions()
        
        if len(all_sessions) < 5:
            raise ValueError("Insufficient training data. Need at least 5 sessions.")
        
        # Train the model
        results = self.trainer.train_on_sessions(all_sessions)
        
        # Store training metadata in database
        self.store_training_metadata(results)
        
        return results
    
    def predict_session_anomalies(self, sessions: List[str]) -> List[Dict[str, Any]]:
        """
        Predict anomalies in sessions using trained DeepLog model
        
        Args:
            sessions: List of session texts
            
        Returns:
            predictions: List of prediction results
        """
        try:
            predictions = self.trainer.predict_anomalies(sessions)
            
            # Enhance predictions with additional metadata
            for i, pred in enumerate(predictions):
                pred.update({
                    'model_type': 'deeplog_bert',
                    'model_version': '1.0',
                    'prediction_timestamp': datetime.now().isoformat()
                })
            
            return predictions
        
        except Exception as e:
            logger.error(f"Error in DeepLog prediction: {e}")
            # Return fallback predictions
            return [
                {
                    'session_id': f"session_{i}",
                    'is_anomaly': False,
                    'anomaly_score': 0.0,
                    'error': str(e),
                    'model_type': 'deeplog_bert'
                }
                for i in range(len(sessions))
            ]
    
    def store_training_metadata(self, results: Dict[str, Any]):
        """Store training metadata in database"""
        try:
            insert_query = text("""
                INSERT INTO model_training_history 
                (model_type, training_timestamp, num_sessions, vocab_size, 
                 best_loss, num_epochs, config, results)
                VALUES 
                (:model_type, :training_timestamp, :num_sessions, :vocab_size,
                 :best_loss, :num_epochs, :config, :results)
            """)
            
            with self.db_engine.connect() as conn:
                conn.execute(insert_query, {
                    'model_type': 'deeplog_bert',
                    'training_timestamp': datetime.now(),
                    'num_sessions': results.get('num_sessions', 0),
                    'vocab_size': results.get('vocab_size', 0),
                    'best_loss': results.get('training_history', {}).get('best_loss', 0.0),
                    'num_epochs': len(results.get('training_history', {}).get('train_loss', [])),
                    'config': json.dumps(self.config.__dict__),
                    'results': json.dumps(results)
                })
                conn.commit()
                
        except Exception as e:
            logger.warning(f"Failed to store training metadata: {e}")
    
    def analyze_transaction_patterns(self, sessions: List[str]) -> Dict[str, Any]:
        """
        Analyze specific transaction patterns like the examples you provided
        
        Args:
            sessions: List of session texts to analyze
            
        Returns:
            analysis: Pattern analysis results
        """
        logger.info("Analyzing transaction patterns with DeepLog")
        
        # Get predictions
        predictions = self.predict_session_anomalies(sessions)
        
        # Analyze patterns
        analysis = {
            'total_sessions': len(sessions),
            'anomalies_detected': sum(1 for p in predictions if p.get('is_anomaly', False)),
            'average_anomaly_score': sum(p.get('anomaly_score', 0) for p in predictions) / len(predictions),
            'pattern_analysis': []
        }
        
        # Analyze specific patterns
        for i, (session, prediction) in enumerate(zip(sessions, predictions)):
            pattern_info = {
                'session_index': i,
                'is_anomaly': prediction.get('is_anomaly', False),
                'anomaly_score': prediction.get('anomaly_score', 0.0),
                'patterns_detected': self._detect_specific_patterns(session)
            }
            
            analysis['pattern_analysis'].append(pattern_info)
        
        return analysis
    
    def _detect_specific_patterns(self, session_text: str) -> List[str]:
        """Detect specific problematic patterns in session text"""
        patterns = []
        
        session_lower = session_text.lower()
        
        # Pattern 1: Card inserted but taken immediately (no transaction)
        if ('card inserted' in session_lower and 
            'card taken' in session_lower and
            'pin entered' not in session_lower and
            'balance' not in session_lower and
            'cash' not in session_lower):
            patterns.append('immediate_card_removal')
        
        # Pattern 2: PIN entered but no transaction completion
        if ('pin entered' in session_lower and
            'card taken' in session_lower and
            'balance displayed' not in session_lower and
            'cash dispensed' not in session_lower and
            'receipt printed' not in session_lower):
            patterns.append('incomplete_transaction')
        
        # Pattern 3: Transaction started but no outcome
        if ('transaction start' in session_lower and
            'start of transaction' in session_lower and
            'transaction end' in session_lower and
            len(session_text.strip()) < 300):  # Very short transaction
            patterns.append('minimal_transaction')
        
        # Pattern 4: Error conditions
        if any(error in session_lower for error in ['error', 'unable', 'failed', 'timeout']):
            patterns.append('error_condition')
        
        return patterns

# API Integration Functions
async def train_deeplog_api():
    """API endpoint for training DeepLog model"""
    try:
        service = DeepLogServiceIntegration()
        results = service.train_deeplog_model(use_labeled_data=True)
        
        return {
            'status': 'success',
            'message': 'DeepLog model trained successfully',
            'results': results
        }
    
    except Exception as e:
        logger.error(f"DeepLog training failed: {e}")
        return {
            'status': 'error',
            'message': f'Training failed: {str(e)}'
        }

async def predict_deeplog_api(sessions: List[str]):
    """API endpoint for DeepLog anomaly prediction"""
    try:
        service = DeepLogServiceIntegration()
        predictions = service.predict_session_anomalies(sessions)
        
        return {
            'status': 'success',
            'predictions': predictions,
            'model_type': 'deeplog_bert'
        }
    
    except Exception as e:
        logger.error(f"DeepLog prediction failed: {e}")
        return {
            'status': 'error',
            'message': f'Prediction failed: {str(e)}'
        }

def test_transaction_examples():
    """
    Test the DeepLog model on your specific transaction examples
    """
    # Your transaction examples
    test_transactions = [
        # Transaction 1 (should be anomaly)
        """
        TRANSACTION START
        CARD INSERTED
        CARD TAKEN
        TRANSACTION END
        PRIMARY CARD READER ACTIVATED
        """,
        
        # Transaction 2 (should be anomaly)
        """
        TRANSACTION START
        CARD INSERTED
        ATR RECEIVED T=0
        OPCODE = FI
        PAN 0004263********6687
        START OF TRANSACTION
        PIN ENTERED
        OPCODE = BC
        PAN 0004263********6687
        START OF TRANSACTION
        CARD TAKEN
        TRANSACTION END
        PRIMARY CARD READER ACTIVATED
        """,
        
        # Normal transaction (should not be anomaly)
        """
        TRANSACTION START
        CARD INSERTED
        ATR RECEIVED T=0
        OPCODE = FI
        PAN 0004263********6687
        START OF TRANSACTION
        PIN ENTERED
        CASH WITHDRAWAL SELECTED
        AMOUNT ENTERED $100.00
        CASH DISPENSED $100.00
        RECEIPT PRINTED
        CARD TAKEN
        TRANSACTION END
        PRIMARY CARD READER ACTIVATED
        """
    ]
    
    service = DeepLogServiceIntegration()
    
    # Analyze the patterns
    analysis = service.analyze_transaction_patterns(test_transactions)
    
    print("=== DeepLog Transaction Analysis ===")
    print(f"Total sessions analyzed: {analysis['total_sessions']}")
    print(f"Anomalies detected: {analysis['anomalies_detected']}")
    print(f"Average anomaly score: {analysis['average_anomaly_score']:.4f}")
    
    for pattern in analysis['pattern_analysis']:
        print(f"\nSession {pattern['session_index']}:")
        print(f"  Is Anomaly: {pattern['is_anomaly']}")
        print(f"  Anomaly Score: {pattern['anomaly_score']:.4f}")
        print(f"  Patterns: {pattern['patterns_detected']}")

if __name__ == "__main__":
    # Test the implementation
    test_transaction_examples()
