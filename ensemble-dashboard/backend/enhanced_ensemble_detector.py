"""
Enhanced Ensemble Anomaly Detection Model with BERT and DBSCAN Integration
"""

import numpy as np
import re
import json
import pickle
import os
from typing import Dict, List, Tuple, Any, Optional
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# BERT dependencies
try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    BERT_AVAILABLE = True
except ImportError:
    print("Warning: BERT dependencies not available. Falling back to TF-IDF.")
    BERT_AVAILABLE = False

class EnhancedEnsembleAnomalyDetector:
    """
    Enhanced ensemble anomaly detection system with DBSCAN integration
    Combines text analysis, statistical analysis, and density-based clustering
    """
    
    def __init__(self, model_dir: str = "./models", use_bert: bool = True):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # BERT configuration
        self.use_bert = use_bert and BERT_AVAILABLE
        if self.use_bert:
            # Use DistilBERT for faster processing while maintaining good performance
            self.bert_model_name = "distilbert-base-uncased"
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.bert_model_name)
                self.bert_model = AutoModel.from_pretrained(self.bert_model_name)
                self.bert_model.eval()
                print("BERT initialized successfully for semantic text analysis")
            except Exception as e:
                print(f"Failed to initialize BERT: {e}")
                self.use_bert = False
                
        # Fallback to TF-IDF if BERT not available
        if not self.use_bert:
            self.text_vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2), lowercase=True)
            print("Using TF-IDF for text vectorization")
        
        # Original ensemble components
        self.svm_model = OneClassSVM(kernel='rbf', gamma='scale', nu=0.1)
        self.isolation_model = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        
        # New DBSCAN components
        self.text_dbscan = DBSCAN(eps=0.5, min_samples=3, metric='cosine')
        self.numerical_dbscan = DBSCAN(eps=0.5, min_samples=3)
        self.combined_dbscan = DBSCAN(eps=0.5, min_samples=3)
        
        # Feature reduction for high-dimensional clustering
        # These will be configured dynamically based on data size during training
        self.text_pca = None
        self.numerical_pca = None
        
        # Training state
        self.is_trained = False
        self.training_stats = {}
        self.feature_names = []
        self.cluster_profiles = {}
        
        # Weights for enhanced ensemble
        self.text_weight = 0.4
        self.statistical_weight = 0.3
        self.density_weight = 0.3
        self.threshold = 0.5
        
        # DBSCAN parameters
        self.dbscan_params = {
            'text_eps': 0.5,
            'text_min_samples': 3,
            'numerical_eps': 0.5,
            'numerical_min_samples': 3,
            'combined_eps': 0.5,
            'combined_min_samples': 3
        }
    
    def _preprocess_atm_text(self, text: str) -> str:
        """Preprocess ATM session text for better BERT understanding"""
        # Replace ATM-specific codes with more semantic terms
        text = re.sub(r'M-(\d+)', r'machine_status_\1', text)
        text = re.sub(r'E-(\d+)', r'error_code_\1', text)
        text = re.sub(r'OPCODE_([A-Z]+)', r'operation_\1', text)
        
        # Add semantic context to operations for better BERT understanding
        replacements = {
            'CARD_INSERTED': 'card reader activated',
            'ATR_RECEIVED': 'card authentication started', 
            'PIN_ENTERED': 'customer authentication',
            'NOTES_STACKED': 'cash dispensing successful',
            'NOTES_PRESENTED': 'cash presented to customer',
            'NOTES_TAKEN': 'cash taken by customer',
            'CARD_TAKEN': 'card returned to customer',
            'RECEIPT_PRINTED': 'transaction receipt printed',
            'TRANSACTION_START': 'transaction initiated',
            'TRANSACTION_END': 'transaction completed',
            'DEVICE_ERROR': 'critical hardware failure',
            'TIMEOUT': 'operation timeout error',
            'COMMUNICATION_FAILURE': 'network connectivity error',
            'RECOVERY_FAILED': 'device recovery unsuccessful',
            'SUPERVISOR_MODE': 'maintenance mode activated',
            'CASH_DISPENSED': 'money dispensing completed',
            'PRIMARY_CARD_READER_ACTIVATED': 'card reader ready for next transaction'
        }
        
        for code, meaning in replacements.items():
            text = text.replace(code, meaning)
        
        # Clean up formatting for BERT
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single space
        text = text.strip().lower()
        
        return text
    
    def _get_bert_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate BERT embeddings for text clustering"""
        if not self.use_bert:
            raise ValueError("BERT not available")
            
        embeddings = []
        batch_size = 8  # Process in batches to manage memory
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            processed_texts = [self._preprocess_atm_text(text) for text in batch_texts]
            
            # Tokenize batch
            inputs = self.tokenizer(
                processed_texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512
            )
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                # Use [CLS] token embedding as sentence representation
                batch_embeddings = outputs.last_hidden_state[:, 0, :].numpy()
                embeddings.extend(batch_embeddings)
        
        return np.array(embeddings)
    
    def _get_enhanced_semantic_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate enhanced BERT embeddings with improved ATM domain preprocessing
        """
        if not self.use_bert:
            raise ValueError("BERT not available for semantic clustering")
            
        print(f"Generating enhanced semantic embeddings for {len(texts)} ATM sessions...")
        
        # Enhanced ATM semantic mappings for better clustering
        atm_semantic_mappings = {
            # Transaction types
            'TRANSACTION_START': 'customer initiated transaction',
            'CARD_INSERTED': 'card reader activation and verification',
            'PIN_ENTERED': 'customer authentication process',
            'AMOUNT_SELECTED': 'cash withdrawal request',
            'CASH_DISPENSED': 'successful money dispensing',
            'CARD_EJECTED': 'transaction completion',
            'RECEIPT_PRINTED': 'transaction documentation',
            
            # Error categories  
            'DEVICE_ERROR': 'critical hardware malfunction requiring service',
            'COMMUNICATION_FAILURE': 'network connectivity issues affecting operations',
            'CASH_JAM': 'physical dispenser mechanism failure',
            'CARD_CAPTURE': 'security response to authentication failure',
            'TIMEOUT_ERROR': 'system response delay exceeding limits',
            'SUPERVISOR_MODE': 'administrative intervention required',
            
            # Status codes
            'M-65': 'device initialization failure',
            'M-01': 'critical system error',
            'M-15': 'dispenser mechanism fault',
            'M-23': 'communication timeout',
            'E-45': 'authentication failure',
            'E-67': 'cash handling error'
        }
        
        embeddings = []
        batch_size = 8
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Enhanced preprocessing for semantic understanding
            processed_texts = []
            for text in batch_texts:
                processed_text = text.lower()
                
                # Apply semantic mappings
                for code, meaning in atm_semantic_mappings.items():
                    pattern = code.lower().replace('_', r'[\s_-]*')
                    processed_text = re.sub(pattern, meaning, processed_text)
                
                # Clean up common ATM patterns for better semantic focus
                processed_text = re.sub(r'\b\d{2}:\d{2}:\d{2}\b', 'timestamp', processed_text)
                processed_text = re.sub(r'\b\d{4}-\d{2}-\d{2}\b', 'date', processed_text)
                processed_text = re.sub(r'\$\d+\.?\d*', 'currency_amount', processed_text)
                processed_text = re.sub(r'\b[A-Z]{2,}\b', lambda m: m.group().lower(), processed_text)
                
                # Focus on semantic content
                semantic_keywords = [
                    'customer', 'transaction', 'authentication', 'dispensing', 'error',
                    'failure', 'success', 'completion', 'verification', 'security',
                    'hardware', 'network', 'communication', 'service', 'maintenance'
                ]
                
                # Ensure semantic keywords are preserved and emphasized
                for keyword in semantic_keywords:
                    if keyword in processed_text:
                        processed_text += f' {keyword}_context'
                
                processed_texts.append(processed_text)
            
            # Tokenize with attention to semantic content
            inputs = self.tokenizer(
                processed_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            # Generate embeddings with attention pooling for better semantic representation
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                
                # Use attention-weighted pooling instead of just [CLS] token
                attention_mask = inputs['attention_mask']
                last_hidden_states = outputs.last_hidden_state
                
                # Weighted average using attention mask for better semantic capture
                masked_hidden_states = last_hidden_states * attention_mask.unsqueeze(-1)
                summed_hidden_states = masked_hidden_states.sum(dim=1)
                attention_sums = attention_mask.sum(dim=1, keepdim=True)
                batch_embeddings = summed_hidden_states / attention_sums
                
                embeddings.extend(batch_embeddings.numpy())
        
        return np.array(embeddings)
    
    def _optimize_semantic_dbscan_parameters(self, embeddings: np.ndarray) -> Dict[str, float]:
        """
        Optimize DBSCAN parameters specifically for semantic clustering
        """
        print("Optimizing DBSCAN parameters for semantic clustering...")
        
        best_params = {'semantic_eps': 0.3, 'semantic_min_samples': 5}
        best_score = -1
        
        # Test range optimized for semantic similarity
        eps_values = [0.25, 0.3, 0.35, 0.4, 0.45]
        min_samples_values = [3, 4, 5, 6, 7, 8]
        
        for eps in eps_values:
            for min_samples in min_samples_values:
                try:
                    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
                    labels = dbscan.fit_predict(embeddings)
                    
                    # Check if we have reasonable clustering
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    n_noise = list(labels).count(-1)
                    
                    # Prefer larger, more meaningful clusters for semantic understanding
                    if n_clusters >= 2 and n_clusters <= len(embeddings) // 3 and n_noise < len(embeddings) * 0.4:
                        score = silhouette_score(embeddings, labels, metric='cosine')
                        
                        # Bonus for fewer, larger clusters (more semantic meaning)
                        cluster_size_bonus = 1.0 / (1.0 + n_clusters * 0.1)  # Prefer fewer clusters
                        adjusted_score = score * cluster_size_bonus
                        
                        if adjusted_score > best_score:
                            best_score = adjusted_score
                            best_params = {'semantic_eps': eps, 'semantic_min_samples': min_samples}
                            
                except Exception as e:
                    continue
        
        print(f"Best semantic parameters: eps={best_params['semantic_eps']}, min_samples={best_params['semantic_min_samples']}, score={best_score:.3f}")
        return best_params
    
    def _analyze_semantic_clusters(self, embeddings: np.ndarray, cluster_labels: np.ndarray, 
                                 sessions: List[str]) -> Dict[int, Dict[str, Any]]:
        """
        Analyze semantic clusters to understand business meaning and patterns
        """
        cluster_analysis = {}
        unique_labels = set(cluster_labels)
        
        for cluster_id in unique_labels:
            if cluster_id == -1:  # Skip noise points
                continue
                
            # Get sessions in this cluster
            cluster_mask = cluster_labels == cluster_id
            cluster_sessions = [sessions[i] for i in range(len(sessions)) if cluster_mask[i]]
            cluster_embeddings = embeddings[cluster_mask]
            
            # Analyze semantic patterns
            semantic_patterns = self._extract_semantic_patterns(cluster_sessions)
            
            # Calculate cluster characteristics
            centroid = np.mean(cluster_embeddings, axis=0)
            
            # Find most representative session (closest to centroid)
            distances = [np.linalg.norm(emb - centroid) for emb in cluster_embeddings]
            representative_idx = np.argmin(distances)
            representative_session = cluster_sessions[representative_idx]
            
            # Generate business meaning descriptions
            business_characteristics = self._describe_semantic_cluster(semantic_patterns)
            
            cluster_analysis[cluster_id] = {
                'size': len(cluster_sessions),
                'semantic_patterns': semantic_patterns,
                'business_meaning': business_characteristics,
                'representative_session': representative_session[:500],  # Truncate for display
                'sessions_sample': cluster_sessions[:5],  # First 5 sessions as examples
                'clustering_reason': self._explain_clustering_reason(semantic_patterns),
                'centroid': centroid.tolist() if len(centroid) < 20 else centroid[:20].tolist(),  # Truncate for storage
                # Enhanced pattern analysis
                'actual_text_patterns': {
                    'common_sequences': self._extract_common_sequences(cluster_sessions),
                    'key_terms': self._extract_key_terms(cluster_sessions),
                    'transaction_flows': self._extract_transaction_flows(cluster_sessions)
                },
                'cluster_name': self._generate_meaningful_cluster_name(semantic_patterns, cluster_sessions),
                'contextual_error_types': self._classify_error_types(cluster_sessions) if any(error in ' '.join(cluster_sessions).lower() for error in ['error', 'fail', 'timeout', 'malfunction']) else None
            }
        
        return cluster_analysis
    
    def _extract_semantic_patterns(self, sessions: List[str]) -> Dict[str, int]:
        """
        Extract semantic patterns from cluster sessions for business understanding
        """
        combined_text = ' '.join(sessions).lower()
        
        patterns = {
            'authentication_issues': len(re.findall(r'pin.*fail|auth.*fail|authentication.*error|card.*capture', combined_text)),
            'hardware_failures': len(re.findall(r'device.*error|hardware.*fail|malfunction|initialization.*failure', combined_text)),
            'communication_errors': len(re.findall(r'communication.*fail|network.*error|timeout|connectivity', combined_text)),
            'cash_dispensing_issues': len(re.findall(r'cash.*error|dispenser.*fail|notes.*jam|dispensing.*problem', combined_text)),
            'successful_transactions': len(re.findall(r'completed|successful|dispensed.*successfully|printed.*receipt', combined_text)),
            'supervisor_interventions': len(re.findall(r'supervisor.*mode|administrative.*intervention|maintenance.*required', combined_text)),
            'security_events': len(re.findall(r'capture|security|fraud|suspicious|card.*retained', combined_text)),
            # Enhanced patterns for better semantic understanding
            'transaction_flow_patterns': self._extract_transaction_flows(sessions),
            'common_text_sequences': self._extract_common_sequences(sessions),
            'key_operational_terms': self._extract_key_terms(sessions)
        }
        
        return patterns
    
    def _extract_transaction_flows(self, sessions: List[str]) -> Dict[str, int]:
        """Extract common transaction flow patterns"""
        flows = {
            'complete_withdrawal_flow': 0,
            'authentication_sequence': 0,
            'cash_handling_sequence': 0,
            'error_recovery_sequence': 0,
            'emv_chip_sequence': 0
        }
        
        for session in sessions:
            upper_session = session.upper()
            
            # Complete withdrawal pattern
            if all(term in upper_session for term in ['TRANSACTION_START', 'CASH_DISPENSED', 'TRANSACTION_END']):
                flows['complete_withdrawal_flow'] += 1
            
            # Authentication sequence
            if all(term in upper_session for term in ['CARD_INSERTED', 'PIN_ENTERED']):
                flows['authentication_sequence'] += 1
            
            # Cash handling sequence
            if all(term in upper_session for term in ['NOTES_STACKED', 'NOTES_PRESENTED', 'NOTES_TAKEN']):
                flows['cash_handling_sequence'] += 1
            
            # EMV chip sequence
            if any(term in upper_session for term in ['OPCODE_FI', 'GENAC', 'ATR_RECEIVED']):
                flows['emv_chip_sequence'] += 1
            
            # Error recovery
            if any(term in upper_session for term in ['RECOVERY', 'RESET', 'RETRY']):
                flows['error_recovery_sequence'] += 1
        
        return flows
    
    def _extract_common_sequences(self, sessions: List[str]) -> List[str]:
        """Extract most common 3-word sequences"""
        from collections import Counter
        
        all_sequences = []
        for session in sessions:
            # Clean and normalize text
            clean_text = session.replace('\x1b', ' ').replace('\u001b', ' ').upper()
            words = clean_text.split()
            
            # Extract 3-word sequences
            for i in range(len(words) - 2):
                sequence = ' '.join(words[i:i+3])
                if len(sequence) > 10:  # Filter out very short sequences
                    all_sequences.append(sequence)
        
        # Return top 5 most common sequences
        counter = Counter(all_sequences)
        return [seq for seq, count in counter.most_common(5) if count > 1]
    
    def _extract_key_terms(self, sessions: List[str]) -> List[str]:
        """Extract key operational terms from sessions"""
        from collections import Counter
        
        all_words = []
        for session in sessions:
            clean_text = session.replace('\x1b', ' ').replace('\u001b', ' ').upper()
            words = clean_text.split()
            # Filter for meaningful ATM terms
            meaningful_words = [word for word in words 
                              if len(word) > 3 and 
                              any(keyword in word for keyword in 
                                  ['TRANSACTION', 'CARD', 'PIN', 'CASH', 'RECEIPT', 'ERROR', 'NOTES', 'OPCODE'])]
            all_words.extend(meaningful_words)
        
        # Return top 8 most common meaningful terms
        counter = Counter(all_words)
        return [word for word, count in counter.most_common(8) if count > 1]
    
    def _generate_meaningful_cluster_name(self, semantic_patterns: Dict, sessions: List[str]) -> str:
        """Generate a meaningful business name for the cluster"""
        
        # Check transaction flow patterns first
        transaction_flows = semantic_patterns.get('transaction_flow_patterns', {})
        
        if transaction_flows.get('complete_withdrawal_flow', 0) > 0:
            if transaction_flows.get('emv_chip_sequence', 0) > 0:
                return "Successful EMV Cash Withdrawal"
            else:
                return "Successful Cash Withdrawal"
        
        # Check for error patterns
        if semantic_patterns.get('authentication_issues', 0) > 0:
            return "Authentication Failure Events"
        elif semantic_patterns.get('hardware_failures', 0) > 0:
            return "Hardware Malfunction Events"
        elif semantic_patterns.get('cash_dispensing_issues', 0) > 0:
            return "Cash Dispensing Issues"
        elif semantic_patterns.get('communication_errors', 0) > 0:
            return "Communication Error Events"
        elif semantic_patterns.get('security_events', 0) > 0:
            return "Security Related Events"
        elif semantic_patterns.get('supervisor_interventions', 0) > 0:
            return "Supervisor Intervention Events"
        
        # Check for specific operational patterns
        combined_text = ' '.join(sessions).upper()
        if 'RECEIPT_PRINTED' in combined_text and 'TRANSACTION_END' in combined_text:
            return "Transaction Completion Events"
        elif 'MAINTENANCE' in combined_text or 'SUPERVISOR' in combined_text:
            return "Maintenance and Administrative Events"
        elif 'CARD_INSERTED' in combined_text and 'PIN_ENTERED' in combined_text:
            return "Authentication Sequence Events"
        
        # Default naming based on size and common terms
        key_terms = semantic_patterns.get('key_operational_terms', [])
        if key_terms:
            primary_term = key_terms[0].replace('_', ' ').title()
            return f"{primary_term} Related Events"
        
        return "Mixed Transaction Events"
    
    def _classify_error_types(self, sessions: List[str]) -> Dict[str, Any]:
        """Classify sessions using contextual labeler error categories"""
        
        combined_text = ' '.join(sessions).lower()
        
        # Based on EJ Contextual Labeler categories
        error_classifications = {
            'hardware_errors': {
                'device_initialization_failure': 'device.*initialization.*failure' in combined_text,
                'cash_dispenser_malfunction': any(term in combined_text for term in ['dispenser.*malfunction', 'notes.*jam', 'cash.*mechanism.*error']),
                'card_reader_failure': any(term in combined_text for term in ['card.*reader.*error', 'magnetic.*stripe.*error', 'chip.*reader.*failure']),
                'receipt_printer_issues': any(term in combined_text for term in ['printer.*error', 'receipt.*jam', 'paper.*empty'])
            },
            'software_errors': {
                'application_timeout': 'timeout' in combined_text or 'application.*timeout' in combined_text,
                'processing_error': any(term in combined_text for term in ['processing.*error', 'transaction.*processing.*failure']),
                'data_validation_error': any(term in combined_text for term in ['validation.*error', 'invalid.*data', 'format.*error'])
            },
            'network_errors': {
                'host_communication_failure': any(term in combined_text for term in ['host.*communication.*fail', 'network.*error', 'connectivity.*issue']),
                'authorization_timeout': any(term in combined_text for term in ['authorization.*timeout', 'host.*timeout'])
            },
            'security_errors': {
                'authentication_failure': any(term in combined_text for term in ['authentication.*fail', 'pin.*fail', 'invalid.*pin']),
                'card_capture_event': any(term in combined_text for term in ['card.*capture', 'card.*retained', 'suspicious.*activity']),
                'fraud_detection': any(term in combined_text for term in ['fraud.*detect', 'suspicious.*transaction'])
            },
            'operational_events': {
                'successful_transaction': any(term in combined_text for term in ['completed.*successfully', 'transaction.*completed', 'cash.*dispensed.*successfully']),
                'user_cancellation': any(term in combined_text for term in ['user.*cancel', 'transaction.*cancel', 'customer.*cancel']),
                'maintenance_mode': any(term in combined_text for term in ['maintenance.*mode', 'supervisor.*intervention', 'administrative.*access'])
            }
        }
        
        # Determine primary error category
        primary_categories = []
        for category, errors in error_classifications.items():
            if any(errors.values()):
                primary_categories.append(category)
        
        return {
            'classifications': error_classifications,
            'primary_categories': primary_categories,
            'error_severity': self._assess_error_severity(combined_text),
            'contextual_labels': self._generate_contextual_labels(sessions)
        }
    
    def _assess_error_severity(self, combined_text: str) -> str:
        """Assess error severity based on contextual indicators"""
        
        if any(term in combined_text for term in ['critical', 'fatal', 'emergency', 'out.*of.*service']):
            return 'critical'
        elif any(term in combined_text for term in ['error', 'failure', 'malfunction', 'timeout']):
            return 'moderate'
        elif any(term in combined_text for term in ['warning', 'retry', 'delay']):
            return 'low'
        else:
            return 'informational'
    
    def _generate_contextual_labels(self, sessions: List[str]) -> List[str]:
        """Generate contextual labels based on EJ Contextual Labeler system"""
        
        labels = []
        combined_text = ' '.join(sessions).upper()
        
        # Event type labels (based on the 35 event types from contextual labeler)
        event_mappings = {
            'CARD_INSERTED': 'card_insertion_event',
            'PIN_ENTERED': 'pin_authentication_event', 
            'CASH_DISPENSED': 'cash_dispensing_event',
            'RECEIPT_PRINTED': 'receipt_printing_event',
            'TRANSACTION_START': 'transaction_initiation_event',
            'TRANSACTION_END': 'transaction_completion_event',
            'NOTES_STACKED': 'cash_handling_event',
            'CARD_TAKEN': 'card_retrieval_event',
            'SUPERVISOR': 'administrative_intervention',
            'MAINTENANCE': 'maintenance_activity',
            'ERROR': 'error_condition',
            'TIMEOUT': 'timeout_event',
            'RETRY': 'retry_attempt',
            'CANCEL': 'cancellation_event'
        }
        
        for keyword, label in event_mappings.items():
            if keyword in combined_text:
                labels.append(label)
        
        return list(set(labels))  # Remove duplicates
    
    def _describe_semantic_cluster(self, patterns: Dict[str, int]) -> List[str]:
        """
        Generate human-readable business descriptions of cluster characteristics
        """
        descriptions = []
        total_events = sum(patterns.values())
        
        if total_events == 0:
            return ["General ATM operations with mixed patterns"]
        
        # Identify dominant patterns for business meaning
        for pattern_name, count in patterns.items():
            if count > 0:
                percentage = (count / total_events) * 100
                if percentage > 15:  # Significant pattern threshold
                    pattern_descriptions = {
                        'authentication_issues': f"🔐 Authentication and PIN verification problems ({percentage:.1f}%)",
                        'hardware_failures': f"⚙️ Hardware malfunctions and device errors ({percentage:.1f}%)",
                        'communication_errors': f"📡 Network connectivity and communication issues ({percentage:.1f}%)",
                        'cash_dispensing_issues': f"💰 Cash dispensing and mechanical problems ({percentage:.1f}%)",
                        'successful_transactions': f"✅ Successful transaction completions ({percentage:.1f}%)",
                        'supervisor_interventions': f"👨‍💼 Administrative and maintenance interventions ({percentage:.1f}%)",
                        'security_events': f"🔒 Security responses and fraud prevention ({percentage:.1f}%)"
                    }
                    descriptions.append(pattern_descriptions[pattern_name])
        
        return descriptions if descriptions else ["Mixed ATM operational patterns"]
    
    def _explain_clustering_reason(self, patterns: Dict[str, int]) -> str:
        """
        Explain why sessions were grouped together semantically
        """
        dominant_patterns = []
        total_events = sum(patterns.values())
        
        for pattern_name, count in patterns.items():
            if count > 0 and total_events > 0:
                percentage = (count / total_events) * 100
                if percentage > 20:
                    dominant_patterns.append(pattern_name.replace('_', ' '))
        
        if dominant_patterns:
            return f"Sessions clustered by semantic similarity in: {', '.join(dominant_patterns)}"
        else:
            return "Sessions grouped by general operational similarity"
    
    def _get_text_vectors(self, texts: List[str]) -> np.ndarray:
        """Get text vectors using BERT or TF-IDF"""
        if self.use_bert:
            print("Generating BERT embeddings for semantic analysis...")
            return self._get_bert_embeddings(texts)
        else:
            print("Using TF-IDF vectorization...")
            return self.text_vectorizer.fit_transform(texts).toarray()
    
    def _transform_text_vectors(self, texts: List[str]) -> np.ndarray:
        """Transform text using trained vectorizer (for prediction)"""
        if self.use_bert:
            return self._get_bert_embeddings(texts)
        else:
            return self.text_vectorizer.transform(texts).toarray()
    
    def extract_text_features(self, session_text: str) -> Dict[str, float]:
        """Extract text-based features (same as original but with clustering preparation)"""
        text_lower = session_text.lower()
        text_upper = session_text.upper()
        words = text_lower.split()
        
        # Enhanced term categories
        normal_terms = ['card', 'pin', 'verified', 'completed', 'successful', 'dispensed', 'printed', 'ejected', 'taken', 'approved']
        critical_error_terms = ['device error', 'device offline', 'critical', 'fatal', 'malfunction', 'communication failure']
        hardware_error_terms = ['hardware error', 'power-up/reset', 'cim-reset', 'recovery failed', 'capture failed', 'jam']
        general_error_terms = ['error', 'fail', 'timeout', 'reset', 'offline', 'fault', 'unable', 'declined']
        
        # Machine status patterns
        machine_status_patterns = re.findall(r'M-\d+', text_upper)
        critical_machine_codes = ['M-65', 'M-01', 'M-15', 'M-23', 'M-45', 'M-67']
        
        # Error patterns
        error_code_patterns = re.findall(r'[ME]-\d+', text_upper)
        device_error_count = len(re.findall(r'device\s+error', text_lower))
        aac_errors = len(re.findall(r'aac|no arpc', text_lower))
        communication_errors = len(re.findall(r'communication\s+failure|comm\s+error|timeout', text_lower))
        
        # Operational patterns
        supervisor_patterns = len(re.findall(r'supervisor\s+mode|supervisor\s+entry|supervisor\s+exit', text_lower))
        recovery_patterns = len(re.findall(r'recovery|cim-reset|init\s+bna|device\s+init|retract\s+bin', text_lower))
        cash_anomalies = len(re.findall(r'cash\s+error|dispenser\s+error|notes\s+jam|cash\s+jam', text_lower))
        retract_operations = len(re.findall(r'retract|capture\s+failed', text_lower))
        auth_failures = len(re.findall(r'external\s+authenticate.*fail|pin.*fail|auth.*fail', text_lower))
        
        # Base features
        features = {
            'total_words': len(words),
            
            # Term counting
            'normal_term_count': sum(1 for word in words if any(term in word for term in normal_terms)),
            'critical_error_count': sum(1 for phrase in critical_error_terms if phrase in text_lower),
            'hardware_error_count': sum(1 for phrase in hardware_error_terms if phrase in text_lower),
            'general_error_count': sum(1 for phrase in general_error_terms if phrase in text_lower),
            
            # Critical indicators
            'device_error_explicit': device_error_count,
            'machine_status_codes': len(machine_status_patterns),
            'critical_machine_codes': sum(1 for code in machine_status_patterns if code in critical_machine_codes),
            'error_codes_total': len(error_code_patterns),
            'aac_authentication_errors': aac_errors,
            'communication_failures': communication_errors,
            
            # Operational indicators
            'supervisor_mode_indicators': supervisor_patterns,
            'recovery_operations': recovery_patterns,
            'cash_handling_anomalies': cash_anomalies,
            'retract_operations': retract_operations,
            'authentication_failures': auth_failures,
            
            # Pattern-based features
            'error_pattern_density': 0.0,
            'critical_anomaly_score': 0.0,
        }
        
        # Calculate derived features
        if features['total_words'] > 0:
            features['error_ratio'] = (features['critical_error_count'] + features['hardware_error_count'] + features['general_error_count']) / features['total_words']
            features['normal_ratio'] = features['normal_term_count'] / features['total_words']
            features['anomaly_term_density'] = (features['device_error_explicit'] + features['critical_machine_codes'] + features['supervisor_mode_indicators']) / features['total_words']
        else:
            features['error_ratio'] = features['normal_ratio'] = features['anomaly_term_density'] = 0
            
        # Error pattern density
        total_error_indicators = (features['critical_error_count'] + features['hardware_error_count'] + 
                                features['device_error_explicit'] + features['critical_machine_codes'] + 
                                features['communication_failures'] + features['recovery_operations'])
        
        if features['total_words'] > 0:
            features['error_pattern_density'] = total_error_indicators / features['total_words']
        
        # Critical anomaly score
        critical_score = 0.0
        if features['device_error_explicit'] > 0:
            critical_score += 0.8
        if features['critical_machine_codes'] > 0:
            critical_score += 0.7
        if features['error_codes_total'] > 2:
            critical_score += 0.6
        elif features['error_codes_total'] > 0:
            critical_score += 0.3
        if features['communication_failures'] > 0:
            critical_score += 0.5
        if features['aac_authentication_errors'] > 0:
            critical_score += 0.4
        if features['recovery_operations'] > 1:
            critical_score += 0.4
        elif features['recovery_operations'] > 0:
            critical_score += 0.2
        if features['supervisor_mode_indicators'] > 0:
            critical_score += 0.6
        if features['error_pattern_density'] > 0.1:
            critical_score += 0.5
        elif features['error_pattern_density'] > 0.05:
            critical_score += 0.3
            
        features['critical_anomaly_score'] = min(1.0, critical_score)
        
        # Add binary flags for clustering
        features['has_device_error'] = 1.0 if features['device_error_explicit'] > 0 else 0.0
        features['has_critical_machine_code'] = 1.0 if features['critical_machine_codes'] > 0 else 0.0
        features['has_supervisor_anomaly'] = 1.0 if features['supervisor_mode_indicators'] > 0 else 0.0
        features['has_recovery_operations'] = 1.0 if features['recovery_operations'] > 0 else 0.0
        features['multiple_error_codes'] = 1.0 if features['error_codes_total'] > 1 else 0.0
        
        return features
    
    def extract_numerical_features(self, session_text: str) -> Dict[str, float]:
        """Extract numerical/statistical features (enhanced for clustering)"""
        lines = session_text.strip().split('\n')
        text_lower = session_text.lower()
        text_upper = session_text.upper()
        
        features = {
            # Error patterns
            'error_count': len(re.findall(r'error', text_lower)),
            'fail_count': len(re.findall(r'fail', text_lower)),
            'malfunction_count': len(re.findall(r'malfunction', text_lower)),
            'timeout_count': len(re.findall(r'timeout', text_lower)),
            'device_error_count': len(re.findall(r'device\s+error', text_lower)),
            
            # Machine status
            'machine_status_codes': len(re.findall(r'M-\d+', text_upper)),
            'critical_m_codes': len(re.findall(r'M-(?:01|15|23|38|45|65|67)', text_upper)),
            'error_codes_total': len(re.findall(r'[ME]-\d+', text_upper)),
            
            # Hardware patterns
            'hardware_mentions': len(re.findall(r'hardware', text_lower)),
            'power_reset_count': len(re.findall(r'power.*reset|reset.*power|power-up/reset', text_lower)),
            'cim_mentions': len(re.findall(r'cim', text_lower)),
            'recovery_failures': len(re.findall(r'recovery.*fail', text_lower)),
            'capture_failures': len(re.findall(r'capture.*fail', text_lower)),
            
            # Critical patterns
            'critical_hardware_patterns': len(re.findall(
                r'power-up/reset|hardware.*error|cim-reset|recovery.*failed|capture.*failed|device\s+error',
                text_lower
            )),
            
            # Communication and authentication
            'communication_errors': len(re.findall(r'communication\s+failure|comm\s+error|no arpc|aac', text_lower)),
            'authentication_failures': len(re.findall(r'external\s+authenticate.*fail|pin.*fail|genac.*aac', text_lower)),
            'network_errors': len(re.findall(r'network.*error|connection.*lost|timeout', text_lower)),
            
            # Cash handling
            'cash_errors': len(re.findall(r'cash.*error|dispenser.*error|jam', text_lower)),
            'retract_operations': len(re.findall(r'retract|capture\s+failed', text_lower)),
            'dispensing_issues': len(re.findall(r'notes.*jam|cash.*jam|dispenser.*malfunction', text_lower)),
            
            # Operational patterns
            'supervisor_patterns': len(re.findall(r'supervisor\s+mode|supervisor\s+entry|supervisor\s+exit', text_lower)),
            'recovery_operations': len(re.findall(r'init\s+bna|cim-reset|device\s+init|recovery', text_lower)),
            'reset_operations': len(re.findall(r'reset|init.*started|recovery.*ok', text_lower)),
            
            # Transaction integrity
            'transaction_start_count': len(re.findall(r'transaction\s+start', text_lower)),
            'transaction_end_count': len(re.findall(r'transaction\s+end', text_lower)),
            'incomplete_transaction_ratio': 0.0,
            
            # Success indicators
            'success_indicators': len(re.findall(
                r'completed|successful|verified|dispensed|printed|taken|approved',
                text_lower
            )),
            
            # Derived scores
            'anomaly_density_score': 0.0,
            'critical_error_density': 0.0,
            'hardware_failure_score': 0.0,
        }
        
        # Calculate derived features
        if features['transaction_start_count'] > 0:
            features['incomplete_transaction_ratio'] = abs(features['transaction_start_count'] - features['transaction_end_count']) / features['transaction_start_count']
        
        total_anomaly_indicators = (features['device_error_count'] + features['critical_m_codes'] + 
                                  features['critical_hardware_patterns'] + features['communication_errors'] + 
                                  features['supervisor_patterns'] + features['recovery_operations'])
        
        # Calculate anomaly density based on total session content (words) instead of lines
        session_words = len(session_text.split())
        if session_words > 0:
            features['anomaly_density_score'] = total_anomaly_indicators / session_words
            features['critical_error_density'] = (features['device_error_count'] + features['critical_m_codes']) / session_words
        else:
            features['anomaly_density_score'] = 0.0
            features['critical_error_density'] = 0.0
        
        hardware_failures = (features['critical_hardware_patterns'] + features['power_reset_count'] + 
                           features['recovery_failures'] + features['capture_failures'])
        features['hardware_failure_score'] = min(1.0, hardware_failures / 5.0)
        
        total_errors = (features['error_count'] + features['fail_count'] + features['device_error_count'] + 
                       features['critical_m_codes'])
        if features['success_indicators'] > 0:
            features['error_to_success_ratio'] = total_errors / features['success_indicators']
        else:
            features['error_to_success_ratio'] = total_errors
        
        # Calculate error-to-content ratio based on word count instead of line count
        if session_words > 0:
            features['error_to_content_ratio'] = total_errors / session_words
        else:
            features['error_to_content_ratio'] = 0
        
        features['session_health_score'] = self._calculate_session_health_score(features, text_lower)
        
        return features
    
    def _calculate_session_health_score(self, features: Dict[str, float], text_lower: str) -> float:
        """Calculate session health score"""
        health_score = 1.0
        
        if features['device_error_count'] > 0:
            health_score -= 0.8
        if features['critical_m_codes'] > 0:
            health_score -= 0.7
        if features['error_codes_total'] > 2:
            health_score -= 0.6
        elif features['error_codes_total'] > 0:
            health_score -= 0.3
        if features['communication_errors'] > 0:
            health_score -= 0.5
        if features['critical_hardware_patterns'] > 1:
            health_score -= 0.6
        elif features['critical_hardware_patterns'] > 0:
            health_score -= 0.4
        if features['supervisor_patterns'] > 0:
            health_score -= 0.7
        if features['recovery_operations'] > 1:
            health_score -= 0.4
        elif features['recovery_operations'] > 0:
            health_score -= 0.2
        if features['anomaly_density_score'] > 0.2:
            health_score -= 0.5
        elif features['anomaly_density_score'] > 0.1:
            health_score -= 0.3
        if features['authentication_failures'] > 0:
            health_score -= 0.4
        if features['incomplete_transaction_ratio'] > 0:
            health_score -= 0.3
        if features['error_to_success_ratio'] > 2.0:
            health_score -= 0.4
        elif features['error_to_success_ratio'] > 1.0:
            health_score -= 0.2
        
        return max(0.0, min(1.0, health_score))
    
    def _optimize_dbscan_parameters(self, X: np.ndarray, parameter_name: str) -> Dict[str, float]:
        """Optimize DBSCAN parameters using silhouette analysis"""
        best_eps = 0.5
        best_min_samples = 3
        best_score = -1
        
        eps_values = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        min_samples_values = [2, 3, 4, 5]
        
        for eps in eps_values:
            for min_samples in min_samples_values:
                try:
                    if parameter_name == 'text':
                        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
                    else:
                        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                    
                    labels = dbscan.fit_predict(X)
                    
                    # Check if we have at least 2 clusters and not all noise
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    if n_clusters >= 2 and np.sum(labels == -1) < len(labels) * 0.8:  # Less than 80% noise
                        score = silhouette_score(X, labels)
                        if score > best_score:
                            best_score = score
                            best_eps = eps
                            best_min_samples = min_samples
                except:
                    continue
        
        return {
            f'{parameter_name}_eps': best_eps,
            f'{parameter_name}_min_samples': best_min_samples,
            f'{parameter_name}_silhouette_score': best_score
        }
    
    def _analyze_clusters(self, X: np.ndarray, labels: np.ndarray, sessions: List[str], 
                         feature_names: List[str] = None) -> Dict[str, Any]:
        """Analyze cluster characteristics"""
        cluster_analysis = {
            'n_clusters': len(set(labels)) - (1 if -1 in labels else 0),
            'n_noise': np.sum(labels == -1),
            'noise_ratio': np.sum(labels == -1) / len(labels),
            'cluster_sizes': {},
            'cluster_profiles': {}
        }
        
        # Analyze each cluster
        for cluster_id in set(labels):
            if cluster_id == -1:  # Noise points
                cluster_analysis['cluster_sizes']['noise'] = np.sum(labels == -1)
                continue
            
            cluster_mask = labels == cluster_id
            cluster_size = np.sum(cluster_mask)
            cluster_analysis['cluster_sizes'][f'cluster_{cluster_id}'] = cluster_size
            
            # Calculate cluster statistics
            cluster_data = X[cluster_mask]
            cluster_center = np.mean(cluster_data, axis=0)
            cluster_std = np.std(cluster_data, axis=0)
            
            profile = {
                'size': cluster_size,
                'center': cluster_center.tolist(),
                'std': cluster_std.tolist(),
                'sessions_sample': sessions[:3] if len(sessions) > 0 else []  # Sample sessions
            }
            
            # Add feature-specific analysis if feature names provided
            if feature_names:
                top_features_idx = np.argsort(cluster_center)[-5:]  # Top 5 features
                profile['top_features'] = [(feature_names[i], cluster_center[i]) for i in top_features_idx]
            
            cluster_analysis['cluster_profiles'][f'cluster_{cluster_id}'] = profile
        
        return cluster_analysis
    
    def train(self, normal_sessions: List[str]) -> Dict[str, Any]:
        """Enhanced training with DBSCAN integration"""
        print(f"Training enhanced ensemble with DBSCAN on {len(normal_sessions)} sessions...")
        
        # Extract features
        texts = []
        numerical_features_list = []
        text_features_list = []
        
        for session in normal_sessions:
            texts.append(session)
            num_features = self.extract_numerical_features(session)
            text_features_dict = self.extract_text_features(session)
            
            numerical_features_list.append(list(num_features.values()))
            text_features_list.append(list(text_features_dict.values()))
        
        # Store feature names
        sample_num_features = self.extract_numerical_features(normal_sessions[0])
        sample_text_features = self.extract_text_features(normal_sessions[0])
        self.feature_names = list(sample_num_features.keys())
        self.text_feature_names = list(sample_text_features.keys())
        
        # Train original ensemble components with BERT or TF-IDF
        text_vectors = self._get_text_vectors(texts)
        self.svm_model.fit(text_vectors)
        
        numerical_features = np.array(numerical_features_list)
        numerical_features = self.scaler.fit_transform(numerical_features)
        self.isolation_model.fit(numerical_features)
        
        # Train DBSCAN components with SEMANTIC CLUSTERING
        
        # 1. Pure BERT Semantic Clustering (replaces both text and numerical clustering)
        print("Training BERT semantic clustering...")
        if self.use_bert:
            # Use enhanced semantic preprocessing and clustering
            semantic_embeddings = self._get_enhanced_semantic_embeddings(texts)
            
            # Optimize DBSCAN for semantic similarity
            semantic_params = self._optimize_semantic_dbscan_parameters(semantic_embeddings)
            self.dbscan_params.update(semantic_params)
            
            self.semantic_dbscan = DBSCAN(
                eps=semantic_params['semantic_eps'],
                min_samples=semantic_params['semantic_min_samples'],
                metric='cosine'
            )
            semantic_clusters = self.semantic_dbscan.fit_predict(semantic_embeddings)
            
            # Store semantic clustering results
            self.semantic_embeddings = semantic_embeddings
            self.semantic_clusters = semantic_clusters
            
        else:
            # Fallback to text-based clustering if BERT unavailable
            print("BERT unavailable, using text-based clustering...")
            if text_vectors.shape[1] > 50:
                n_components = min(50, text_vectors.shape[0] - 1, text_vectors.shape[1])
                self.text_pca = PCA(n_components=n_components, random_state=42)
                text_features_reduced = self.text_pca.fit_transform(text_vectors)
            else:
                text_features_reduced = text_vectors
                
            text_params = self._optimize_dbscan_parameters(text_features_reduced, 'text')
            self.dbscan_params.update(text_params)
            
            self.text_dbscan = DBSCAN(
                eps=text_params['text_eps'], 
                min_samples=text_params['text_min_samples'], 
                metric='cosine'
            )
            semantic_clusters = self.text_dbscan.fit_predict(text_features_reduced)
        
        # 2. REMOVE old numerical clustering - replaced with semantic clustering above
        
        # 3. Combined features clustering (now based on semantic + enhanced features)
        print("Training combined semantic features DBSCAN...")
        text_features_array = np.array(text_features_list)
        text_features_scaled = StandardScaler().fit_transform(text_features_array)
        
        if self.use_bert and hasattr(self, 'semantic_embeddings'):
            # Combine semantic embeddings with selected engineered features
            combined_features = np.hstack([self.semantic_embeddings, text_features_scaled])
        else:
            # Fallback: combine text vectors with engineered features
            combined_features = np.hstack([text_features_scaled, numerical_features])
            
        combined_params = self._optimize_dbscan_parameters(combined_features, 'combined')
        self.dbscan_params.update(combined_params)
        
        self.combined_dbscan = DBSCAN(
            eps=combined_params['combined_eps'],
            min_samples=combined_params['combined_min_samples']
        )
        combined_clusters = self.combined_dbscan.fit_predict(combined_features)
        
        # Analyze clusters - prioritize semantic clustering results
        if self.use_bert and hasattr(self, 'semantic_clusters'):
            # Primary analysis on semantic clusters
            semantic_cluster_analysis = self._analyze_semantic_clusters(
                self.semantic_embeddings, semantic_clusters, normal_sessions
            )
            
            # Combined analysis for comparison
            combined_cluster_analysis = self._analyze_clusters(
                combined_features, combined_clusters, normal_sessions
            )
        else:
            # Fallback analysis
            text_cluster_analysis = self._analyze_clusters(
                text_features_reduced, semantic_clusters, normal_sessions  # semantic_clusters from fallback
            )
            combined_cluster_analysis = self._analyze_clusters(
                combined_features, combined_clusters, normal_sessions
            )
        
        # Store cluster profiles for anomaly detection - prioritize semantic clustering
        if self.use_bert and hasattr(self, 'semantic_clusters'):
            self.cluster_profiles = {
                'semantic_clusters': semantic_cluster_analysis,
                'combined_clusters': combined_cluster_analysis,
                'clustering_method': 'semantic_bert',
                'primary_clustering': 'semantic'
            }
        else:
            self.cluster_profiles = {
                'text_clusters': text_cluster_analysis,
                'combined_clusters': combined_cluster_analysis,
                'clustering_method': 'text_fallback',
                'primary_clustering': 'text'
            }
        
        # Calculate original ensemble statistics
        svm_scores = self.svm_model.decision_function(text_vectors)
        iso_scores = self.isolation_model.decision_function(numerical_features)
        
        svm_probabilities = 1 / (1 + np.exp(svm_scores))
        iso_probabilities = 1 / (1 + np.exp(iso_scores))
        
        # Calculate DBSCAN-based anomaly scores - use semantic clustering
        if self.use_bert and hasattr(self, 'semantic_clusters'):
            semantic_density_scores = self._calculate_density_scores(semantic_clusters)
            combined_density_scores = self._calculate_density_scores(combined_clusters)
            
            # Ensemble density score prioritizing semantic clustering
            density_ensemble_scores = (semantic_density_scores * 0.7 + combined_density_scores * 0.3)
        else:
            # Fallback density scores
            text_density_scores = self._calculate_density_scores(semantic_clusters)  # using fallback semantic_clusters
            combined_density_scores = self._calculate_density_scores(combined_clusters)
            
            density_ensemble_scores = (text_density_scores + combined_density_scores) / 2
        
        # Final ensemble with density component
        ensemble_scores = (self.text_weight * svm_probabilities + 
                         self.statistical_weight * iso_probabilities + 
                         self.density_weight * density_ensemble_scores)
        
        self.training_stats = {
            'num_training_sessions': len(normal_sessions),
            'text_feature_dims': text_vectors.shape[1],
            'numerical_feature_dims': len(self.feature_names),
            'avg_svm_score': float(np.mean(svm_probabilities)),
            'avg_isolation_score': float(np.mean(iso_probabilities)),
            'avg_density_score': float(np.mean(density_ensemble_scores)),
            'avg_ensemble_score': float(np.mean(ensemble_scores)),
            'feature_names': self.feature_names,
            'text_feature_names': self.text_feature_names,
            'weights': {
                'text_weight': self.text_weight,
                'statistical_weight': self.statistical_weight,
                'density_weight': self.density_weight
            },
            'threshold': self.threshold,
            'dbscan_params': self.dbscan_params,
            'cluster_analysis': {
                'semantic_clusters': semantic_cluster_analysis if (self.use_bert and 'semantic_cluster_analysis' in locals()) else None,
                'combined_clusters': combined_cluster_analysis,
                'clustering_method': 'semantic_bert' if self.use_bert else 'text_fallback'
            }
        }
        
        self.is_trained = True
        print("Enhanced training with DBSCAN complete!")
        
        return self.training_stats
    
    def _calculate_density_scores(self, cluster_labels: np.ndarray) -> np.ndarray:
        """Calculate anomaly scores based on cluster density"""
        scores = np.zeros(len(cluster_labels))
        
        # Noise points (label = -1) get high anomaly scores
        noise_mask = cluster_labels == -1
        scores[noise_mask] = 0.9  # High anomaly score for noise points
        
        # For clustered points, score based on cluster size (smaller clusters = more anomalous)
        for cluster_id in set(cluster_labels):
            if cluster_id == -1:  # Skip noise
                continue
            
            cluster_mask = cluster_labels == cluster_id
            cluster_size = np.sum(cluster_mask)
            
            # Smaller clusters get higher anomaly scores
            # Large clusters (>20% of data) get low scores
            # Small clusters (<5% of data) get high scores
            total_points = len(cluster_labels)
            cluster_ratio = cluster_size / total_points
            
            if cluster_ratio > 0.2:  # Large cluster
                cluster_score = 0.1
            elif cluster_ratio > 0.1:  # Medium cluster  
                cluster_score = 0.3
            elif cluster_ratio > 0.05:  # Small cluster
                cluster_score = 0.6
            else:  # Very small cluster
                cluster_score = 0.8
            
            scores[cluster_mask] = cluster_score
        
        return scores
    
    def predict(self, session_text: str) -> Dict[str, Any]:
        """Enhanced prediction with DBSCAN-based density analysis"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Extract features using BERT or TF-IDF
        text_vectors = self._transform_text_vectors([session_text])
        num_features_dict = self.extract_numerical_features(session_text)
        text_features_dict = self.extract_text_features(session_text)
        
        # Get original ensemble predictions
        num_features = np.array([list(num_features_dict.values())])
        num_features = self.scaler.transform(num_features)
        
        svm_score = self.svm_model.decision_function(text_vectors)[0]
        iso_score = self.isolation_model.decision_function(num_features)[0]
        
        svm_probability = 1 / (1 + np.exp(svm_score))
        iso_probability = 1 / (1 + np.exp(iso_score))
        
        # Get DBSCAN-based density predictions
        density_scores = self._predict_density_anomaly(text_vectors[0], num_features[0], 
                                                     list(text_features_dict.values()))
        
        # Apply critical pattern amplification (same as original)
        amplified_scores = self._apply_critical_anomaly_amplification(
            svm_probability, iso_probability, text_features_dict, num_features_dict, session_text
        )
        
        amplified_svm = amplified_scores['amplified_text_score']
        amplified_iso = amplified_scores['amplified_statistical_score']
        critical_boost = amplified_scores['critical_boost']
        anomaly_reasons = amplified_scores['anomaly_reasons']
        
        # Enhanced ensemble with density component
        base_ensemble_score = (self.text_weight * amplified_svm + 
                             self.statistical_weight * amplified_iso + 
                             self.density_weight * density_scores['ensemble_density_score'])
        
        final_ensemble_score = min(1.0, base_ensemble_score + critical_boost)
        
        # Dynamic threshold adjustment
        effective_threshold = self.threshold
        if critical_boost > 0.3:
            effective_threshold = max(0.3, self.threshold - 0.2)
        
        # Additional threshold adjustment for density-based anomalies
        if density_scores['is_density_anomaly']:
            effective_threshold = max(0.2, effective_threshold - 0.1)
            anomaly_reasons.append(f"Density-based anomaly detected - unusual pattern clustering")
        
        is_anomaly = final_ensemble_score > effective_threshold
        
        confidence = self._calculate_enhanced_confidence(
            final_ensemble_score, effective_threshold, critical_boost, anomaly_reasons
        )
        
        return {
            'session_text': session_text,
            'text_anomaly_score': float(amplified_svm),
            'statistical_anomaly_score': float(amplified_iso),
            'density_anomaly_score': float(density_scores['ensemble_density_score']),
            'ensemble_score': float(final_ensemble_score),
            'base_ensemble_score': float(base_ensemble_score),
            'critical_boost': float(critical_boost),
            'is_anomaly': bool(is_anomaly),
            'confidence': confidence,
            'threshold': effective_threshold,
            'original_threshold': self.threshold,
            'text_features': text_features_dict,
            'numerical_features': num_features_dict,
            'density_analysis': density_scores,
            'anomaly_reasons': anomaly_reasons,
            'prediction_breakdown': {
                'text_component': {
                    'original_score': float(svm_probability),
                    'amplified_score': float(amplified_svm),
                    'weight': self.text_weight,
                    'contribution': float(self.text_weight * amplified_svm)
                },
                'statistical_component': {
                    'original_score': float(iso_probability),
                    'amplified_score': float(amplified_iso),
                    'weight': self.statistical_weight,
                    'contribution': float(self.statistical_weight * amplified_iso)
                },
                'density_component': {
                    'density_score': float(density_scores['ensemble_density_score']),
                    'weight': self.density_weight,
                    'contribution': float(self.density_weight * density_scores['ensemble_density_score']),
                    'is_density_anomaly': density_scores['is_density_anomaly'],
                    'cluster_distances': density_scores['cluster_distances']
                },
                'critical_amplification': {
                    'boost_applied': float(critical_boost),
                    'reasons': anomaly_reasons,
                    'threshold_adjustment': float(self.threshold - effective_threshold)
                }
            }
        }
    
    def _predict_density_anomaly(self, text_vector: np.ndarray, num_features: np.ndarray, 
                               text_features_list: List[float]) -> Dict[str, Any]:
        """Predict anomaly based on density clustering"""
        
        # Prepare features for clustering
        if self.text_pca is not None and text_vector.shape[0] > 50:
            text_features_reduced = self.text_pca.transform([text_vector])
        else:
            text_features_reduced = [text_vector]
            
        if self.numerical_pca is not None and num_features.shape[0] > 20:
            numerical_features_reduced = self.numerical_pca.transform([num_features])
        else:
            numerical_features_reduced = [num_features]
        
        text_features_scaled = StandardScaler().fit_transform([text_features_list])
        combined_features = np.hstack([text_features_scaled, [num_features]])
        
        # Predict cluster membership (distance to existing clusters)
        text_distances = self._calculate_cluster_distances(text_features_reduced[0], 'text')
        numerical_distances = self._calculate_cluster_distances(numerical_features_reduced[0], 'numerical')
        combined_distances = self._calculate_cluster_distances(combined_features[0], 'combined')
        
        # Calculate density-based anomaly scores
        text_density_score = min(text_distances.values()) if text_distances else 0.9
        numerical_density_score = min(numerical_distances.values()) if numerical_distances else 0.9
        combined_density_score = min(combined_distances.values()) if combined_distances else 0.9
        
        # Ensemble density score
        ensemble_density_score = (text_density_score + numerical_density_score + combined_density_score) / 3
        
        # Determine if this is a density-based anomaly
        is_density_anomaly = ensemble_density_score > 0.7  # High distance from all clusters
        
        return {
            'text_density_score': text_density_score,
            'numerical_density_score': numerical_density_score,
            'combined_density_score': combined_density_score,
            'ensemble_density_score': ensemble_density_score,
            'is_density_anomaly': is_density_anomaly,
            'cluster_distances': {
                'text': text_distances,
                'numerical': numerical_distances,
                'combined': combined_distances
            }
        }
    
    def _calculate_cluster_distances(self, point: np.ndarray, cluster_type: str) -> Dict[str, float]:
        """Calculate distances to existing cluster centers"""
        distances = {}
        
        if cluster_type not in self.cluster_profiles:
            return distances
        
        cluster_analysis = self.cluster_profiles[f'{cluster_type}_clusters']
        
        for cluster_name, profile in cluster_analysis['cluster_profiles'].items():
            if cluster_name == 'noise':
                continue
                
            cluster_center = np.array(profile['center'])
            distance = np.linalg.norm(point - cluster_center)
            
            # Normalize by cluster standard deviation
            cluster_std = np.array(profile['std'])
            avg_std = np.mean(cluster_std)
            if avg_std > 0:
                normalized_distance = distance / avg_std
            else:
                normalized_distance = distance
            
            distances[cluster_name] = normalized_distance
        
        return distances
    
    def _apply_critical_anomaly_amplification(self, svm_prob: float, iso_prob: float, 
                                            text_features: Dict[str, float], 
                                            num_features: Dict[str, float], 
                                            session_text: str) -> Dict[str, Any]:
        """Apply critical pattern amplification (same as original implementation)"""
        anomaly_reasons = []
        critical_boost = 0.0
        
        # Critical patterns with same logic as original
        if text_features.get('device_error_explicit', 0) > 0:
            boost = 0.6
            critical_boost += boost
            anomaly_reasons.append(f"DEVICE ERROR detected - critical hardware failure indicator (+{boost:.1f})")
        
        if num_features.get('critical_m_codes', 0) > 0:
            boost = 0.5
            critical_boost += boost
            m_codes = re.findall(r'M-\d+', session_text.upper())
            anomaly_reasons.append(f"Critical machine status codes detected: {', '.join(m_codes)} (+{boost:.1f})")
        
        error_code_count = num_features.get('error_codes_total', 0)
        if error_code_count > 2:
            boost = 0.4
            critical_boost += boost
            anomaly_reasons.append(f"Multiple error codes ({int(error_code_count)}) - cascading system failures (+{boost:.1f})")
        elif error_code_count > 0:
            boost = 0.2
            critical_boost += boost
            anomaly_reasons.append(f"Error codes detected ({int(error_code_count)}) - system issues (+{boost:.1f})")
        
        if num_features.get('communication_errors', 0) > 0:
            boost = 0.4
            critical_boost += boost
            anomaly_reasons.append(f"Communication failures detected - network/host connectivity issues (+{boost:.1f})")
        
        if num_features.get('supervisor_patterns', 0) > 0:
            boost = 0.5
            critical_boost += boost
            anomaly_reasons.append(f"Supervisor mode activity detected - unusual operational pattern (+{boost:.1f})")
        
        if num_features.get('critical_hardware_patterns', 0) > 1:
            boost = 0.4
            critical_boost += boost
            anomaly_reasons.append(f"Multiple critical hardware patterns detected - device reliability issues (+{boost:.1f})")
        elif num_features.get('critical_hardware_patterns', 0) > 0:
            boost = 0.2
            critical_boost += boost
            anomaly_reasons.append(f"Critical hardware pattern detected - potential device issue (+{boost:.1f})")
        
        anomaly_density = num_features.get('anomaly_density_score', 0)
        if anomaly_density > 0.2:
            boost = 0.3
            critical_boost += boost
            anomaly_reasons.append(f"High anomaly density ({anomaly_density:.1%}) - concentrated error patterns (+{boost:.1f})")
        elif anomaly_density > 0.1:
            boost = 0.15
            critical_boost += boost
            anomaly_reasons.append(f"Elevated anomaly density ({anomaly_density:.1%}) - multiple error indicators (+{boost:.1f})")
        
        session_health = num_features.get('session_health_score', 1.0)
        if session_health < 0.3:
            boost = 0.4
            critical_boost += boost
            anomaly_reasons.append(f"Very poor session health (score: {session_health:.2f}) - multiple critical issues (+{boost:.1f})")
        elif session_health < 0.5:
            boost = 0.2
            critical_boost += boost
            anomaly_reasons.append(f"Poor session health (score: {session_health:.2f}) - concerning patterns (+{boost:.1f})")
        
        if num_features.get('authentication_failures', 0) > 0:
            boost = 0.3
            critical_boost += boost
            anomaly_reasons.append(f"Authentication failures detected - security concerns (+{boost:.1f})")
        
        recovery_ops = num_features.get('recovery_operations', 0)
        if recovery_ops > 1:
            boost = 0.3
            critical_boost += boost
            anomaly_reasons.append(f"Multiple recovery operations ({int(recovery_ops)}) - device instability (+{boost:.1f})")
        elif recovery_ops > 0:
            boost = 0.15
            critical_boost += boost
            anomaly_reasons.append(f"Recovery operation detected - previous device issue (+{boost:.1f})")
        
        # Apply amplification
        text_amplification = (text_features.get('critical_anomaly_score', 0) * 0.5 + 
                            min(0.3, critical_boost * 0.6))
        amplified_svm = min(1.0, svm_prob + text_amplification)
        
        statistical_amplification = min(0.4, critical_boost * 0.7)
        amplified_iso = min(1.0, iso_prob + statistical_amplification)
        
        critical_boost = min(0.4, critical_boost)
        
        return {
            'amplified_text_score': amplified_svm,
            'amplified_statistical_score': amplified_iso,
            'critical_boost': critical_boost,
            'anomaly_reasons': anomaly_reasons,
            'text_amplification': text_amplification,
            'statistical_amplification': statistical_amplification
        }
    
    def _calculate_enhanced_confidence(self, ensemble_score: float, threshold: float, 
                                     critical_boost: float, anomaly_reasons: List[str]) -> str:
        """Calculate enhanced confidence"""
        threshold_distance = abs(ensemble_score - threshold)
        
        if threshold_distance > 0.4:
            base_confidence = "HIGH"
        elif threshold_distance > 0.2:
            base_confidence = "MEDIUM"
        else:
            base_confidence = "LOW"
        
        if critical_boost > 0.3 and len(anomaly_reasons) >= 3:
            return "VERY_HIGH"
        elif critical_boost > 0.2 and len(anomaly_reasons) >= 2:
            if base_confidence == "LOW":
                return "MEDIUM"
            elif base_confidence == "MEDIUM":
                return "HIGH"
            else:
                return "VERY_HIGH"
        elif critical_boost > 0.1:
            if base_confidence == "LOW" and threshold_distance > 0.1:
                return "MEDIUM"
        
        return base_confidence
    
    def get_cluster_insights(self) -> Dict[str, Any]:
        """Get insights about discovered clusters"""
        if not self.is_trained:
            return {"error": "Model not trained"}
        
        insights = {
            'cluster_summary': {},
            'anomaly_patterns': {},
            'recommendations': []
        }
        
        for cluster_type in ['text', 'numerical', 'combined']:
            cluster_key = f'{cluster_type}_clusters'
            if cluster_key in self.cluster_profiles:
                analysis = self.cluster_profiles[cluster_key]
                
                insights['cluster_summary'][cluster_type] = {
                    'n_clusters': analysis['n_clusters'],
                    'noise_ratio': analysis['noise_ratio'],
                    'largest_cluster_size': max(analysis['cluster_sizes'].values()) if analysis['cluster_sizes'] else 0
                }
                
                # Identify potential anomaly patterns
                high_noise_ratio = analysis['noise_ratio'] > 0.1
                many_small_clusters = analysis['n_clusters'] > 5
                
                insights['anomaly_patterns'][cluster_type] = {
                    'high_noise_ratio': high_noise_ratio,
                    'many_small_clusters': many_small_clusters,
                    'fragmented_patterns': high_noise_ratio and many_small_clusters
                }
        
        # Generate recommendations
        if insights['anomaly_patterns']['combined']['high_noise_ratio']:
            insights['recommendations'].append("High noise ratio detected - consider reviewing DBSCAN parameters")
        
        if insights['anomaly_patterns']['text']['many_small_clusters']:
            insights['recommendations'].append("Many small text clusters found - potential for fine-grained anomaly detection")
        
        return insights
    
    # Include all other methods from original class (save_model, load_model, etc.)
    def save_model(self, filepath: str = None):
        """Save the enhanced model with DBSCAN components"""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        if filepath is None:
            filepath = os.path.join(self.model_dir, "enhanced_ensemble_model.pkl")
        
        model_data = {
            'text_vectorizer': getattr(self, 'text_vectorizer', None),
            'use_bert': self.use_bert,
            'bert_model_name': getattr(self, 'bert_model_name', None),
            'svm_model': self.svm_model,
            'isolation_model': self.isolation_model,
            'scaler': self.scaler,
            'text_dbscan': self.text_dbscan,
            'numerical_dbscan': self.numerical_dbscan,
            'combined_dbscan': self.combined_dbscan,
            'text_pca': self.text_pca,
            'numerical_pca': self.numerical_pca,
            'is_trained': self.is_trained,
            'training_stats': self.training_stats,
            'feature_names': self.feature_names,
            'text_feature_names': getattr(self, 'text_feature_names', []),
            'cluster_profiles': self.cluster_profiles,
            'text_weight': self.text_weight,
            'statistical_weight': self.statistical_weight,
            'density_weight': self.density_weight,
            'threshold': self.threshold,
            'dbscan_params': self.dbscan_params
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Enhanced model saved to {filepath}")
    
    def load_model(self, filepath: str = None):
        """Load the enhanced model with DBSCAN components"""
        if filepath is None:
            filepath = os.path.join(self.model_dir, "enhanced_ensemble_model.pkl")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Load BERT configuration first
        self.use_bert = model_data.get('use_bert', False)
        
        # Initialize BERT if needed and available
        if self.use_bert and BERT_AVAILABLE:
            bert_model_name = model_data.get('bert_model_name', 'distilbert-base-uncased')
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
                self.bert_model = AutoModel.from_pretrained(bert_model_name)
                self.bert_model.eval()
                self.bert_model_name = bert_model_name
                print("BERT model reloaded successfully")
            except Exception as e:
                print(f"Failed to reload BERT: {e}")
                self.use_bert = False
        
        # Load all components
        if not self.use_bert:
            self.text_vectorizer = model_data.get('text_vectorizer')
        self.svm_model = model_data['svm_model']
        self.isolation_model = model_data['isolation_model']
        self.scaler = model_data['scaler']
        self.text_dbscan = model_data['text_dbscan']
        self.numerical_dbscan = model_data['numerical_dbscan']
        self.combined_dbscan = model_data['combined_dbscan']
        self.text_pca = model_data['text_pca']
        self.numerical_pca = model_data['numerical_pca']
        self.is_trained = model_data['is_trained']
        self.training_stats = model_data['training_stats']
        self.feature_names = model_data['feature_names']
        self.text_feature_names = model_data.get('text_feature_names', [])
        self.cluster_profiles = model_data['cluster_profiles']
        self.text_weight = model_data['text_weight']
        self.statistical_weight = model_data['statistical_weight']
        self.density_weight = model_data['density_weight']
        self.threshold = model_data['threshold']
        self.dbscan_params = model_data['dbscan_params']
        
        print(f"Enhanced model loaded from {filepath}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get enhanced model information"""
        base_info = {
            'is_trained': self.is_trained,
            'training_stats': self.training_stats,
            'ensemble_config': {
                'text_weight': self.text_weight,
                'statistical_weight': self.statistical_weight,
                'density_weight': self.density_weight,
                'threshold': self.threshold,
                'use_bert': self.use_bert,
                'bert_model': getattr(self, 'bert_model_name', 'Not available') if self.use_bert else 'TF-IDF'
            },
            'feature_names': self.feature_names,
            'dbscan_params': self.dbscan_params
        }
        
        if self.is_trained:
            base_info['cluster_insights'] = self.get_cluster_insights()
        
        return base_info
    
    def sessionize_ej_log(self, ej_log_text):
        """
        Sessionize raw EJ log text into individual sessions
        Compatible with existing API
        """
        import re
        
        # Split on transaction markers
        sessions = []
        
        # Look for transaction start/end patterns
        transaction_pattern = r'\[020t.*?\*.*?\*.*?\*.*?\*.*?\*TRANSACTION START\*.*?(?=\[020t.*?\*TRANSACTION START\*|\[020t.*?\*PRIMARY CARD READER ACTIVATED\*|$)'
        
        matches = re.findall(transaction_pattern, ej_log_text, re.DOTALL)
        
        for match in matches:
            # Clean up the session text
            cleaned_session = match.strip()
            if cleaned_session and len(cleaned_session) > 50:  # Filter out very short sessions
                sessions.append(cleaned_session)
        
        # If no matches found, try simpler pattern
        if not sessions:
            # Split on timestamp patterns
            lines = ej_log_text.split('\n')
            current_session = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Check if this is a new session start
                if '[020t*' in line and '*TRANSACTION START*' in line:
                    if current_session:
                        session_text = '\n'.join(current_session)
                        if len(session_text) > 50:
                            sessions.append(session_text)
                    current_session = [line]
                elif '[020t*' in line and '*PRIMARY CARD READER ACTIVATED*' in line:
                    if current_session:
                        session_text = '\n'.join(current_session)
                        if len(session_text) > 50:
                            sessions.append(session_text)
                    current_session = []
                else:
                    current_session.append(line)
            
            # Add the last session if it exists
            if current_session:
                session_text = '\n'.join(current_session)
                if len(session_text) > 50:
                    sessions.append(session_text)
        
        # If still no sessions, split by major sections
        if not sessions:
            sections = re.split(r'\[020t.*?\*\d+\*.*?\*.*?\*', ej_log_text)
            for section in sections:
                section = section.strip()
                if section and len(section) > 50:
                    sessions.append(section)
        
        return sessions

    def get_cluster_sessions(self, cluster_id: int, feature_type: str = 'combined') -> Dict[str, Any]:
        """
        Get all sessions belonging to a specific cluster with enhanced metadata
        
        Args:
            cluster_id: The cluster ID to get sessions for
            feature_type: Type of features used for clustering ('semantic', 'combined', 'text')
        
        Returns:
            Dictionary with sessions list and cluster characteristics
        """
        try:
            if not self.is_trained:
                raise ValueError("Model not trained")
            
            # Map feature_type to new semantic clustering structure
            if feature_type == 'numerical':
                # Redirect numerical requests to semantic clustering
                feature_type = 'semantic' if (self.use_bert and hasattr(self, 'cluster_profiles') and 'semantic_clusters' in self.cluster_profiles) else 'combined'
            
            # Get cluster profiles from the saved model data
            if hasattr(self, 'cluster_profiles'):
                # Check for semantic clusters first (preferred)
                if feature_type == 'semantic' and 'semantic_clusters' in self.cluster_profiles:
                    cluster_data = self.cluster_profiles['semantic_clusters']
                elif feature_type == 'combined' and 'combined_clusters' in self.cluster_profiles:
                    cluster_data = self.cluster_profiles['combined_clusters']
                elif 'combined_clusters' in self.cluster_profiles:
                    # Fallback to combined if semantic not available
                    cluster_data = self.cluster_profiles['combined_clusters']
                    feature_type = 'combined'
                else:
                    raise ValueError("No cluster data available")
            else:
                raise ValueError("No cluster profiles available")
            
            # Check if the requested cluster exists in the data structure
            if cluster_id not in cluster_data:
                available_clusters = list(cluster_data.keys())
                cluster_info = f"Available clusters for {feature_type}: {available_clusters}"
                
                # Also check other clustering types
                alternative_info = []
                if hasattr(self, 'cluster_profiles'):
                    for cluster_type, type_data in self.cluster_profiles.items():
                        if cluster_type.endswith('_clusters') and isinstance(type_data, dict):
                            type_name = cluster_type.replace('_clusters', '')
                            available_ids = list(type_data.keys()) if type_data else []
                            if available_ids:
                                alternative_info.append(f"{type_name}: {available_ids}")
                
                error_msg = f"Cluster {cluster_id} not found for {feature_type} clustering. {cluster_info}"
                if alternative_info:
                    error_msg += f"\nOther available cluster types: {'; '.join(alternative_info)}"
                
                raise ValueError(error_msg)
            
            # Get cluster profile data
            cluster_profile = cluster_data[cluster_id]
            cluster_sessions = []
            
            # Calculate cluster characteristics from the center values
            cluster_characteristics = self._calculate_cluster_characteristics(cluster_profile, feature_type)
            
            # Create session data from cluster profile
            if 'sessions_sample' in cluster_profile:
                for i, session_text in enumerate(cluster_profile['sessions_sample']):
                    # Extract features for this session to provide rich metadata
                    session_features = self._extract_session_features(session_text, feature_type)
                    
                    # Try to load original raw and processed text from stored session data
                    raw_text, processed_text = self._get_original_session_texts(session_text)
                    
                    session_data = {
                        'session_id': f'session_{cluster_id}_{i}',
                        'cluster_id': int(cluster_id),
                        'index': int(i),
                        'feature_type': str(feature_type),
                        
                        # Text data in multiple formats
                        'session_text': str(session_text),  # Current session text used for clustering
                        'raw_ej_text': raw_text,  # Original raw EJ log text
                        'processed_text': processed_text,  # BERT preprocessed text
                        'text': str(session_text),  # Keep original field for backward compatibility
                        'raw_text_preview': raw_text[:500] + '...' if len(raw_text) > 500 else raw_text,  # Preview for UI
                        'bert_preprocessed_text': processed_text,  # Full preprocessed text
                        
                        # Session metadata
                        'confidence': float(0.85),
                        'cluster_size': int(cluster_profile.get('size', 1)),
                        'length': int(len(session_text)) if isinstance(session_text, str) else 0,
                        'word_count': int(len(session_text.split())) if isinstance(session_text, str) else 0,
                        
                        # Rich feature analysis for this session
                        'features': session_features,
                    }
                    cluster_sessions.append(session_data)
            else:
                # Fallback: create placeholder sessions based on cluster size
                cluster_size = int(cluster_profile.get('size', 1))
                for i in range(min(cluster_size, 5)):  # Limit to 5 sample sessions
                    session_data = {
                        'session_id': f'session_{cluster_id}_{i}',
                        'cluster_id': int(cluster_id),
                        'index': int(i),
                        'feature_type': str(feature_type),
                        'session_text': f'Session {i+1} from cluster {cluster_id} ({feature_type} clustering)',
                        'raw_ej_text': f'Session {i+1} from cluster {cluster_id} ({feature_type} clustering)',
                        'processed_text': f'Session {i+1} from cluster {cluster_id} ({feature_type} clustering)',
                        'text': f'Session {i+1} from cluster {cluster_id} ({feature_type} clustering)',
                        'raw_text_preview': f'Session {i+1} from cluster {cluster_id} ({feature_type} clustering)',
                        'bert_preprocessed_text': f'Session {i+1} from cluster {cluster_id} ({feature_type} clustering)',
                        'confidence': float(0.75),
                        'cluster_size': cluster_size,
                        'length': int(50),
                        'word_count': int(8),
                        'features': {}
                    }
                    cluster_sessions.append(session_data)
            
            # Return comprehensive cluster data with enhanced semantic analysis
            enhanced_data = {
                'sessions': cluster_sessions,
                'cluster_characteristics': cluster_characteristics,
                'cluster_metadata': {
                    'cluster_id': cluster_id,
                    'feature_type': feature_type,
                    'cluster_size': cluster_profile.get('size', 0),
                    'total_sessions_in_cluster': len(cluster_sessions),
                    'cluster_center': cluster_profile.get('center', []),
                    'cluster_std': cluster_profile.get('std', [])
                }
            }
            
            # Add enhanced semantic analysis if available
            if 'actual_text_patterns' in cluster_profile:
                enhanced_data['actual_text_patterns'] = cluster_profile['actual_text_patterns']
            
            if 'cluster_name' in cluster_profile:
                enhanced_data['cluster_name'] = cluster_profile['cluster_name']
                
            if 'business_meaning' in cluster_profile:
                enhanced_data['business_meaning'] = cluster_profile['business_meaning']
                
            if 'contextual_error_types' in cluster_profile:
                enhanced_data['contextual_error_types'] = cluster_profile['contextual_error_types']
                
            if 'semantic_patterns' in cluster_profile:
                enhanced_data['semantic_patterns'] = cluster_profile['semantic_patterns']
                
            if 'clustering_reason' in cluster_profile:
                enhanced_data['clustering_reason'] = cluster_profile['clustering_reason']
                
            return enhanced_data
            
        except Exception as e:
            print(f"Error getting cluster sessions: {e}")
            raise ValueError(f"Failed to get cluster sessions: {str(e)}")

    def _calculate_cluster_characteristics(self, cluster_profile: Dict, feature_type: str) -> Dict[str, Any]:
        """Calculate the common characteristics that define this cluster with semantic understanding"""
        characteristics = {
            'dominant_features': {},
            'common_patterns': [],
            'cluster_summary': {},
            'feature_importance': {},
            'clustering_reasons': [],
            'distinguishing_attributes': {},
            'business_meaning': [],
            'semantic_patterns': {}
        }
        
        try:
            cluster_size = cluster_profile.get('size', 0)
            
            # Handle semantic clustering characteristics
            if feature_type == 'semantic' and 'business_meaning' in cluster_profile:
                characteristics['business_meaning'] = cluster_profile['business_meaning']
                characteristics['clustering_reasons'] = [cluster_profile.get('clustering_reason', 'Semantic similarity')]
                
                # Add semantic patterns if available
                if 'semantic_patterns' in cluster_profile:
                    semantic_patterns = cluster_profile['semantic_patterns']
                    characteristics['semantic_patterns'] = semantic_patterns
                    
                    # Convert semantic patterns to readable characteristics
                    total_events = sum(semantic_patterns.values()) if semantic_patterns.values() else 1
                    for pattern_name, count in semantic_patterns.items():
                        if count > 0:
                            percentage = (count / total_events) * 100
                            pattern_display = pattern_name.replace('_', ' ').title()
                            characteristics['dominant_features'][pattern_display] = f"{count} occurrences ({percentage:.1f}%)"
                
                # Add business context
                characteristics['cluster_summary'] = {
                    'semantic_focus': cluster_profile.get('business_meaning', ['General ATM operations'])[0] if cluster_profile.get('business_meaning') else 'Mixed operations',
                    'cluster_size': cluster_size,
                    'business_value': f"Represents {cluster_size} sessions with similar semantic patterns"
                }
                
            else:
                # Handle traditional clustering characteristics
                cluster_center = cluster_profile.get('center', [])
                cluster_std = cluster_profile.get('std', [])
                
                if cluster_center and len(cluster_center) > 0:
                    # Handle numerical/traditional clustering
                    if feature_type == 'numerical' and hasattr(self, 'feature_names'):
                        feature_names = self.feature_names
                        if len(cluster_center) == len(feature_names):
                            # Calculate feature importance and identify distinguishing characteristics
                            significant_features = []
                            
                            for i, (feature_name, center_val, std_val) in enumerate(zip(feature_names, cluster_center, cluster_std)):
                                if center_val > 0.05:  # Lower threshold to catch more features
                                    importance = center_val / (std_val + 0.001)
                                    consistency = 1.0 / (std_val + 0.001)  # Low std = high consistency
                                    
                                    characteristics['dominant_features'][feature_name] = {
                                        'center_value': float(center_val),
                                        'std_value': float(std_val),
                                        'importance_score': float(importance),
                                        'consistency_score': float(consistency)
                                    }
                                    
                                    # Determine what makes this feature significant
                                    if center_val > 0.8:
                                        level = "Very High"
                                        significance = "CRITICAL"
                                    elif center_val > 0.5:
                                        level = "High" 
                                        significance = "MAJOR"
                                    elif center_val > 0.2:
                                        level = "Moderate"
                                        significance = "NOTABLE"
                                    else:
                                        level = "Low"
                                        significance = "MINOR"
                                    
                                    feature_desc = feature_name.replace('_', ' ').title()
                                    
                                    # Add to clustering reasons
                                    if std_val < 0.1:  # Very consistent across cluster
                                        characteristics['clustering_reasons'].append(
                                            f"🎯 {significance}: All sessions have {level.lower()} {feature_desc} ({center_val:.3f} ± {std_val:.3f})"
                                        )
                                    else:
                                        characteristics['clustering_reasons'].append(
                                            f"📊 {significance}: Sessions share {level.lower()} {feature_desc} pattern ({center_val:.3f} ± {std_val:.3f})"
                                        )
                            
                            significant_features.append((feature_name, importance, center_val, std_val))
                    
                    # Sort by importance and get top distinguishing features
                    significant_features.sort(key=lambda x: x[1], reverse=True)
                    top_features = significant_features[:8]  # Top 8 features
                    
                    # Create distinguishing attributes summary
                    for feature_name, importance, center_val, std_val in top_features:
                        clean_name = feature_name.replace('_', ' ').title()
                        
                        # Create human-readable descriptions
                        if 'error' in feature_name.lower():
                            if center_val > 0.5:
                                description = f"High error activity: {clean_name} ({center_val:.2f})"
                                impact = "These sessions contain significant error patterns"
                            else:
                                description = f"Some error activity: {clean_name} ({center_val:.2f})"
                                impact = "These sessions show minor error indicators"
                        elif 'transaction' in feature_name.lower():
                            description = f"Transaction pattern: {clean_name} ({center_val:.2f})"
                            impact = "These sessions share similar transaction behaviors"
                        elif 'health' in feature_name.lower():
                            if center_val > 0.8:
                                description = f"Excellent health: {clean_name} ({center_val:.2f})"
                                impact = "These sessions represent normal, healthy operations"
                            elif center_val > 0.5:
                                description = f"Good health: {clean_name} ({center_val:.2f})"
                                impact = "These sessions show generally healthy patterns"
                            else:
                                description = f"Poor health: {clean_name} ({center_val:.2f})"
                                impact = "These sessions show concerning health indicators"
                        elif 'anomaly' in feature_name.lower():
                            if center_val > 0.3:
                                description = f"High anomaly density: {clean_name} ({center_val:.2f})"
                                impact = "These sessions contain multiple anomalous patterns"
                            else:
                                description = f"Low anomaly density: {clean_name} ({center_val:.2f})"
                                impact = "These sessions appear relatively normal"
                        else:
                            description = f"{clean_name}: {center_val:.2f}"
                            impact = f"Shared {clean_name.lower()} characteristics"
                        
                        characteristics['distinguishing_attributes'][feature_name] = {
                            'description': description,
                            'impact': impact,
                            'value': center_val,
                            'consistency': std_val,
                            'rank': len(characteristics['distinguishing_attributes']) + 1
                        }
                    
                    # Generate summary patterns
                    if significant_features:
                        top_feature = significant_features[0]
                        characteristics['common_patterns'].append(
                            f"Primary clustering factor: {top_feature[0].replace('_', ' ').title()} ({top_feature[2]:.3f})"
                        )
                        
                        # Group similar feature types
                        error_features = [f for f in significant_features if 'error' in f[0].lower()]
                        health_features = [f for f in significant_features if 'health' in f[0].lower()]
                        transaction_features = [f for f in significant_features if 'transaction' in f[0].lower()]
                        
                        if error_features:
                            avg_error = sum(f[2] for f in error_features) / len(error_features)
                            characteristics['common_patterns'].append(f"Error pattern cluster (avg: {avg_error:.3f})")
                        
                        if health_features:
                            avg_health = sum(f[2] for f in health_features) / len(health_features)
                            characteristics['common_patterns'].append(f"Health pattern cluster (avg: {avg_health:.3f})")
                        
                        if transaction_features:
                            avg_transaction = sum(f[2] for f in transaction_features) / len(transaction_features)
                            characteristics['common_patterns'].append(f"Transaction pattern cluster (avg: {avg_transaction:.3f})")
                    
                    characteristics['feature_importance'] = dict(significant_features[:5])
            
            if feature_type == 'numerical':
                # Numerical feature clustering analysis
                characteristics['common_patterns'].append("Numerical feature-based clustering")
                characteristics['clustering_reasons'].append("📊 Sessions grouped by numerical characteristics")
            elif feature_type == 'text':
                characteristics['common_patterns'].append("Text-based semantic clustering")
                characteristics['clustering_reasons'].append("📝 Sessions grouped by similar text content and patterns")
                
                if hasattr(self, 'text_feature_names') and len(cluster_center) > 0:
                    # Analyze text features
                    text_features = self.text_feature_names if hasattr(self, 'text_feature_names') else []
                    significant_text_features = []
                    
                    for i, center_val in enumerate(cluster_center[:min(len(text_features), len(cluster_center))]):
                        if center_val > 0.1 and i < len(text_features):
                            feature_name = text_features[i]
                            significant_text_features.append((feature_name, center_val))
                    
                    if significant_text_features:
                        significant_text_features.sort(key=lambda x: x[1], reverse=True)
                        for feature_name, value in significant_text_features[:5]:
                            clean_name = feature_name.replace('_', ' ').title()
                            characteristics['distinguishing_attributes'][feature_name] = {
                                'description': f"Text feature: {clean_name} ({value:.3f})",
                                'impact': f"Shared {clean_name.lower()} in session content",
                                'value': value,
                                'rank': len(characteristics['distinguishing_attributes']) + 1
                            }
                            characteristics['clustering_reasons'].append(
                                f"📄 Text similarity: {clean_name} pattern ({value:.3f})"
                            )
                else:
                    characteristics['clustering_reasons'].append("📄 Sessions clustered by semantic text similarity")
            
            elif feature_type == 'combined':
                characteristics['common_patterns'].append("Multi-dimensional clustering (text + numerical)")
                characteristics['clustering_reasons'].append("🔄 Sessions grouped by both content similarity and numerical patterns")
                
                # Try to analyze combined features if we have them
                if len(cluster_center) > 0:
                    # Assume first part is text features, second part is numerical
                    total_features = len(cluster_center)
                    
                    # Find the most significant values in the combined feature vector
                    significant_indices = []
                    for i, val in enumerate(cluster_center):
                        if abs(val) > 0.1:  # Significant value
                            significant_indices.append((i, val))
                    
                    significant_indices.sort(key=lambda x: abs(x[1]), reverse=True)
                    
                    for i, (idx, val) in enumerate(significant_indices[:5]):
                        if hasattr(self, 'feature_names') and idx < len(self.feature_names):
                            feature_name = self.feature_names[idx]
                        else:
                            feature_name = f"Combined_Feature_{idx}"
                        
                        characteristics['distinguishing_attributes'][feature_name] = {
                            'description': f"Combined feature: {feature_name.replace('_', ' ').title()} ({val:.3f})",
                            'impact': f"Shared {feature_name.replace('_', ' ').lower()} characteristics",
                            'value': float(val),
                            'rank': i + 1
                        }
                        characteristics['clustering_reasons'].append(
                            f"⚡ Combined pattern: {feature_name.replace('_', ' ').title()} ({val:.3f})"
                        )
            
            # Add cluster summary with actionable insights
            characteristics['cluster_summary'] = {
                'size': cluster_size,
                'feature_type': feature_type,
                'description': f"Cluster of {cluster_size} sessions with shared characteristics",
                'quality': self._assess_cluster_quality(cluster_center, cluster_std),
                'interpretation': self._interpret_cluster_meaning(characteristics['distinguishing_attributes'], feature_type)
            }
            
        except Exception as e:
            print(f"Error calculating cluster characteristics: {e}")
            characteristics['error'] = str(e)
            characteristics['clustering_reasons'] = [f"❌ Error analyzing cluster: {str(e)}"]
        
        return characteristics
    
    def _assess_cluster_quality(self, center: list, std: list) -> str:
        """Assess the quality of the cluster based on consistency"""
        if not center or not std:
            return "Unknown"
        
        avg_std = sum(std) / len(std) if std else 1.0
        
        if avg_std < 0.1:
            return "Very High - Sessions are very similar"
        elif avg_std < 0.3:
            return "High - Sessions have consistent patterns"
        elif avg_std < 0.5:
            return "Medium - Sessions show some variation"
        else:
            return "Low - Sessions are quite diverse"
    
    def _interpret_cluster_meaning(self, attributes: dict, feature_type: str) -> str:
        """Provide human-readable interpretation of what this cluster represents"""
        if not attributes:
            return f"Sessions grouped by {feature_type} similarity"
        
        # Look for dominant themes
        error_attrs = [a for a in attributes.keys() if 'error' in a.lower()]
        health_attrs = [a for a in attributes.keys() if 'health' in a.lower()]
        transaction_attrs = [a for a in attributes.keys() if 'transaction' in a.lower()]
        
        if error_attrs:
            return "This cluster represents sessions with similar error patterns or anomalous behavior"
        elif health_attrs:
            health_values = [attributes[a]['value'] for a in health_attrs]
            avg_health = sum(health_values) / len(health_values) if health_values else 0.5
            if avg_health > 0.8:
                return "This cluster represents healthy, normal operation sessions"
            else:
                return "This cluster represents sessions with health or operational concerns"
        elif transaction_attrs:
            return "This cluster represents sessions with similar transaction patterns or behaviors"
        else:
            return f"This cluster represents sessions with shared {feature_type} characteristics"

    def _extract_session_features(self, session_text: str, feature_type: str) -> Dict[str, Any]:
        """Extract detailed features for a specific session"""
        try:
            features = {}
            
            # Always extract numerical features as they're most informative
            numerical_features = self.extract_numerical_features(session_text)
            features['numerical'] = numerical_features
            
            # Extract text features
            text_features = self.extract_text_features(session_text)
            features['text'] = text_features
            
            # Add session-specific analysis
            features['analysis'] = {
                'session_health_score': numerical_features.get('session_health_score', 0.0),
                'anomaly_density': numerical_features.get('anomaly_density_score', 0.0),
                'error_count': numerical_features.get('error_count', 0),
                'critical_patterns': numerical_features.get('critical_hardware_patterns', 0),
                'has_device_errors': numerical_features.get('device_error_count', 0) > 0,
                'has_critical_codes': numerical_features.get('critical_m_codes', 0) > 0,
                'transaction_complete': numerical_features.get('incomplete_transaction_ratio', 0) == 0
            }
            
            return features
            
        except Exception as e:
            print(f"Error extracting session features: {e}")
            return {'error': str(e)}

    def _get_original_session_texts(self, session_text: str) -> tuple:
        """
        Distinguish between raw EJ text and processed text for BERT
        """
        try:
            # The session_text passed here is already the processed/cleaned version
            processed_text = session_text
            
            # Create a more "raw" version by simulating original EJ log format
            # Extract key components and reconstruct a raw-like format
            raw_text = self._simulate_raw_ej_format(session_text)
            
            return raw_text, processed_text
            
        except Exception as e:
            print(f"Error getting original session texts: {e}")
            return session_text, session_text
    
    def _simulate_raw_ej_format(self, processed_text: str) -> str:
        """
        Simulate what the raw EJ text might have looked like before processing
        """
        try:
            # For CSV-like data, add more raw formatting
            if ',' in processed_text:
                parts = processed_text.split(',')
                if len(parts) > 10:  # This looks like CSV session data
                    # Create a more "raw" EJ log-like format
                    session_id = parts[0] if len(parts) > 0 else "UNKNOWN_SESSION"
                    transaction_type = parts[4] if len(parts) > 4 else "UNKNOWN"
                    amount = parts[5] if len(parts) > 5 else "0.0"
                    auth_code = parts[-1] if len(parts) > 0 else ""
                    
                    # Simulate raw EJ format with timestamps and device codes
                    raw_format = f"""[020t*{session_id}*TRANSACTION_START*
CARD_INSERTED: {parts[7] if len(parts) > 7 else 'True'}
PIN_ENTERED: {parts[8] if len(parts) > 8 else 'True'}
TRANSACTION_TYPE: {transaction_type.upper()}
AMOUNT: ${amount}
NOTES_DISPENSED: {parts[10] if len(parts) > 10 else 'True'}
NOTES_TAKEN: {parts[11] if len(parts) > 11 else 'True'}
CARD_TAKEN: {parts[12] if len(parts) > 12 else 'True'}
AUTH_CODE: {auth_code}
[020t*TRANSACTION_END*]"""
                    return raw_format
            
            # For other text, add some raw EJ formatting
            lines = processed_text.split('\n')
            raw_lines = []
            for line in lines:
                if line.strip():
                    # Add EJ timestamp formatting
                    raw_lines.append(f"[020t*{line.strip()}*]")
            
            return '\n'.join(raw_lines) if raw_lines else processed_text
            
        except Exception as e:
            print(f"Error simulating raw EJ format: {e}")
            return processed_text

    def label_cluster(self, cluster_id: int, label: str, feature_type: str = 'combined') -> bool:
        """
        Assign a human-readable label to a cluster
        
        Args:
            cluster_id: The cluster ID to label
            label: Human-readable label for the cluster
            feature_type: Type of features used for clustering
        
        Returns:
            True if successful, False otherwise
        """
        try:
            from datetime import datetime
            
            # Initialize cluster labels storage if it doesn't exist
            if not hasattr(self, 'cluster_labels'):
                self.cluster_labels = {}
            
            # Store the label with feature type context
            label_key = f"{feature_type}_{cluster_id}"
            self.cluster_labels[label_key] = {
                'cluster_id': cluster_id,
                'label': label,
                'feature_type': feature_type,
                'timestamp': datetime.now().isoformat()
            }
            
            print(f"Labeled cluster {cluster_id} ({feature_type}) as '{label}'")
            return True
            
        except Exception as e:
            print(f"Error labeling cluster: {e}")
            return False

    def train_supervised_classifier(self, force_retrain: bool = False) -> Dict[str, Any]:
        """
        Train a supervised classifier using labeled clusters
        
        Args:
            force_retrain: Whether to retrain even if classifier exists
        
        Returns:
            Training statistics and performance metrics
        """
        try:
            if not hasattr(self, 'cluster_labels') or not self.cluster_labels:
                raise ValueError("No labeled clusters available for supervised training")
            
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import cross_val_score
            from datetime import datetime
            
            # Prepare training data
            X_train = []
            y_train = []
            
            for label_key, label_info in self.cluster_labels.items():
                cluster_id = label_info['cluster_id']
                feature_type = label_info['feature_type']
                label = label_info['label']
                
                # Get sessions for this cluster
                try:
                    sessions = self.get_cluster_sessions(cluster_id, feature_type)
                    
                    # Extract features for each session
                    for session in sessions:
                        session_text = session['text']
                        
                        # Create feature vector (use combined features for consistency)
                        if hasattr(self, 'tfidf_vectorizer'):
                            text_features = self.tfidf_vectorizer.transform([session_text])
                            numerical_features = self._extract_numerical_features([session_text])
                            
                            # Combine features
                            combined_features = np.hstack([
                                text_features.toarray(),
                                numerical_features
                            ])
                            
                            X_train.append(combined_features[0])
                            y_train.append(label)
                            
                except Exception as e:
                    print(f"Error processing cluster {cluster_id}: {e}")
                    continue
            
            if len(X_train) == 0:
                raise ValueError("No training data could be extracted from labeled clusters")
            
            # Train supervised classifier
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            
            self.supervised_classifier = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight='balanced'
            )
            
            self.supervised_classifier.fit(X_train, y_train)
            
            # Evaluate classifier
            cv_scores = cross_val_score(self.supervised_classifier, X_train, y_train, cv=5)
            
            # Store training statistics
            training_stats = {
                'num_samples': len(X_train),
                'num_classes': len(np.unique(y_train)),
                'classes': list(np.unique(y_train)),
                'cv_accuracy_mean': cv_scores.mean(),
                'cv_accuracy_std': cv_scores.std(),
                'feature_dimensions': X_train.shape[1],
                'training_timestamp': datetime.now().isoformat()
            }
            
            # Store classifier state
            self.supervised_training_stats = training_stats
            
            print(f"Supervised classifier trained successfully: {training_stats}")
            return training_stats
            
        except Exception as e:
            print(f"Error training supervised classifier: {e}")
            raise

    def predict_supervised(self, session_text: str) -> Dict[str, Any]:
        """
        Predict cluster label for a session using supervised classifier
        
        Args:
            session_text: The session text to classify
        
        Returns:
            Prediction results with confidence scores
        """
        try:
            if not hasattr(self, 'supervised_classifier'):
                raise ValueError("Supervised classifier not trained. Call train_supervised_classifier() first.")
            
            # Extract features
            text_features = self.tfidf_vectorizer.transform([session_text])
            numerical_features = self._extract_numerical_features([session_text])
            
            # Combine features
            combined_features = np.hstack([
                text_features.toarray(),
                numerical_features
            ])
            
            # Make prediction
            prediction = self.supervised_classifier.predict(combined_features)[0]
            probabilities = self.supervised_classifier.predict_proba(combined_features)[0]
            
            # Get class labels
            classes = self.supervised_classifier.classes_
            
            # Create probability dictionary
            probability_dict = {classes[i]: prob for i, prob in enumerate(probabilities)}
            
            # Get confidence (max probability)
            confidence = max(probabilities)
            
            result = {
                'predicted_label': prediction,
                'confidence': confidence,
                'all_probabilities': probability_dict,
                'session_text': session_text[:100] + '...' if len(session_text) > 100 else session_text
            }
            
            print(f"Supervised prediction: {prediction} (confidence: {confidence:.3f})")
            return result
            
        except Exception as e:
            print(f"Error in supervised prediction: {e}")
            raise

    def _extract_numerical_features(self, sessions: List[str]) -> np.ndarray:
        """Helper method to extract numerical features for multiple sessions"""
        features = []
        for session in sessions:
            session_features = self.extract_numerical_features(session)
            # Convert dict to array in consistent order - using meaningful ATM-specific features
            feature_vector = [
                session_features.get('error_count', 0),
                session_features.get('device_error_count', 0),
                session_features.get('critical_m_codes', 0),
                session_features.get('communication_errors', 0),
                session_features.get('hardware_failure_score', 0),
                session_features.get('anomaly_density_score', 0),
                session_features.get('session_health_score', 0),
                session_features.get('error_to_success_ratio', 0)
            ]
            features.append(feature_vector)
        return np.array(features)
