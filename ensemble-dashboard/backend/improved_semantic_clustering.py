#!/usr/bin/env python3
"""
Improved Semantic Clustering for ATM Transaction Analysis
Focus on meaningful BERT-based semantic clustering with better interpretability
"""

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModel
import torch
import re
from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

class ImprovedSemanticClustering:
    """
    Enhanced BERT-based semantic clustering specifically designed for ATM transactions
    with better parameter optimization and semantic meaningfulness
    """
    
    def __init__(self):
        """Initialize with BERT and optimized DBSCAN parameters"""
        self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        self.bert_model = AutoModel.from_pretrained('distilbert-base-uncased')
        self.scaler = StandardScaler()
        
        # ATM-specific semantic mappings for better clustering
        self.atm_semantic_mappings = {
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
        
        self.cluster_labels = None
        self.embeddings = None
        self.sessions = None
        
    def preprocess_atm_text(self, text: str) -> str:
        """
        Enhanced ATM text preprocessing for better semantic understanding
        """
        processed_text = text.lower()
        
        # Apply semantic mappings
        for code, meaning in self.atm_semantic_mappings.items():
            pattern = code.lower().replace('_', r'[\s_-]*')
            processed_text = re.sub(pattern, meaning, processed_text)
        
        # Clean up common ATM patterns
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
        
        # Ensure semantic keywords are preserved
        for keyword in semantic_keywords:
            if keyword in processed_text:
                processed_text += f' {keyword}_context'
        
        return processed_text
    
    def get_bert_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate BERT embeddings optimized for ATM domain clustering
        """
        embeddings = []
        batch_size = 8
        
        print(f"Generating BERT embeddings for {len(texts)} ATM sessions...")
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            processed_texts = [self.preprocess_atm_text(text) for text in batch_texts]
            
            # Tokenize with attention to semantic content
            inputs = self.tokenizer(
                processed_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            # Generate embeddings with attention pooling
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                
                # Use attention-weighted pooling instead of just [CLS] token
                attention_mask = inputs['attention_mask']
                last_hidden_states = outputs.last_hidden_state
                
                # Weighted average using attention mask
                masked_hidden_states = last_hidden_states * attention_mask.unsqueeze(-1)
                summed_hidden_states = masked_hidden_states.sum(dim=1)
                attention_sums = attention_mask.sum(dim=1, keepdim=True)
                batch_embeddings = summed_hidden_states / attention_sums
                
                embeddings.extend(batch_embeddings.numpy())
        
        return np.array(embeddings)
    
    def optimize_dbscan_parameters(self, embeddings: np.ndarray) -> Dict[str, float]:
        """
        Optimize DBSCAN parameters specifically for semantic clustering
        """
        print("Optimizing DBSCAN parameters for semantic clustering...")
        
        best_params = {'eps': 0.3, 'min_samples': 2}
        best_score = -1
        
        # Test range optimized for semantic similarity
        eps_values = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        min_samples_values = [2, 3, 4, 5]
        
        for eps in eps_values:
            for min_samples in min_samples_values:
                try:
                    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
                    labels = dbscan.fit_predict(embeddings)
                    
                    # Check if we have reasonable clustering
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    n_noise = list(labels).count(-1)
                    
                    if n_clusters >= 2 and n_noise < len(labels) * 0.5:
                        score = silhouette_score(embeddings, labels, metric='cosine')
                        
                        if score > best_score:
                            best_score = score
                            best_params = {'eps': eps, 'min_samples': min_samples}
                            
                except Exception as e:
                    continue
        
        print(f"Best parameters: eps={best_params['eps']}, min_samples={best_params['min_samples']}, score={best_score:.3f}")
        return best_params
    
    def perform_semantic_clustering(self, sessions: List[str]) -> Dict[str, Any]:
        """
        Perform semantic clustering using optimized BERT embeddings
        """
        print(f"Starting semantic clustering for {len(sessions)} ATM sessions...")
        
        # Store sessions for later analysis
        self.sessions = sessions
        
        # Generate BERT embeddings
        self.embeddings = self.get_bert_embeddings(sessions)
        
        # Optimize DBSCAN parameters
        optimal_params = self.optimize_dbscan_parameters(self.embeddings)
        
        # Perform clustering
        dbscan = DBSCAN(
            eps=optimal_params['eps'],
            min_samples=optimal_params['min_samples'],
            metric='cosine'
        )
        
        self.cluster_labels = dbscan.fit_predict(self.embeddings)
        
        # Analyze results
        n_clusters = len(set(self.cluster_labels)) - (1 if -1 in self.cluster_labels else 0)
        n_noise = list(self.cluster_labels).count(-1)
        
        print(f"Clustering Results:")
        print(f"  - Number of clusters: {n_clusters}")
        print(f"  - Number of noise points: {n_noise}")
        print(f"  - Silhouette score: {silhouette_score(self.embeddings, self.cluster_labels, metric='cosine'):.3f}")
        
        # Generate cluster analysis
        cluster_analysis = self.analyze_semantic_clusters()
        
        return {
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'cluster_labels': self.cluster_labels.tolist(),
            'optimal_params': optimal_params,
            'cluster_analysis': cluster_analysis
        }
    
    def analyze_semantic_clusters(self) -> Dict[int, Dict[str, Any]]:
        """
        Analyze each cluster to understand semantic patterns
        """
        cluster_analysis = {}
        unique_labels = set(self.cluster_labels)
        
        for cluster_id in unique_labels:
            if cluster_id == -1:  # Skip noise
                continue
                
            # Get sessions in this cluster
            cluster_mask = self.cluster_labels == cluster_id
            cluster_sessions = [self.sessions[i] for i in range(len(self.sessions)) if cluster_mask[i]]
            cluster_embeddings = self.embeddings[cluster_mask]
            
            # Analyze semantic patterns
            semantic_patterns = self.extract_semantic_patterns(cluster_sessions)
            
            # Calculate cluster characteristics
            centroid = np.mean(cluster_embeddings, axis=0)
            
            # Find most representative session (closest to centroid)
            distances = [np.linalg.norm(emb - centroid) for emb in cluster_embeddings]
            representative_idx = np.argmin(distances)
            representative_session = cluster_sessions[representative_idx]
            
            cluster_analysis[cluster_id] = {
                'size': len(cluster_sessions),
                'semantic_patterns': semantic_patterns,
                'representative_session': representative_session[:500],  # Truncate for display
                'sessions_sample': cluster_sessions[:3],  # First 3 sessions as examples
                'characteristics': self.describe_cluster_semantics(semantic_patterns)
            }
        
        return cluster_analysis
    
    def extract_semantic_patterns(self, sessions: List[str]) -> Dict[str, int]:
        """
        Extract semantic patterns from cluster sessions
        """
        combined_text = ' '.join(sessions).lower()
        
        patterns = {
            'authentication_issues': len(re.findall(r'pin.*fail|auth.*fail|authentication.*error', combined_text)),
            'hardware_failures': len(re.findall(r'device.*error|hardware.*fail|malfunction', combined_text)),
            'communication_errors': len(re.findall(r'communication.*fail|network.*error|timeout', combined_text)),
            'cash_dispensing_issues': len(re.findall(r'cash.*error|dispenser.*fail|notes.*jam', combined_text)),
            'successful_transactions': len(re.findall(r'completed|successful|dispensed|printed', combined_text)),
            'supervisor_interventions': len(re.findall(r'supervisor.*mode|administrative|maintenance', combined_text)),
            'security_events': len(re.findall(r'capture|security|fraud|suspicious', combined_text))
        }
        
        return patterns
    
    def describe_cluster_semantics(self, patterns: Dict[str, int]) -> List[str]:
        """
        Generate human-readable descriptions of cluster characteristics
        """
        descriptions = []
        total_events = sum(patterns.values())
        
        if total_events == 0:
            return ["General ATM operations"]
        
        # Identify dominant patterns
        for pattern_name, count in patterns.items():
            if count > 0:
                percentage = (count / total_events) * 100
                if percentage > 20:  # Significant pattern
                    pattern_descriptions = {
                        'authentication_issues': f"🔐 Authentication problems ({percentage:.1f}%)",
                        'hardware_failures': f"⚙️ Hardware malfunctions ({percentage:.1f}%)",
                        'communication_errors': f"📡 Network/communication issues ({percentage:.1f}%)",
                        'cash_dispensing_issues': f"💰 Cash dispensing problems ({percentage:.1f}%)",
                        'successful_transactions': f"✅ Successful operations ({percentage:.1f}%)",
                        'supervisor_interventions': f"👨‍💼 Administrative interventions ({percentage:.1f}%)",
                        'security_events': f"🔒 Security-related events ({percentage:.1f}%)"
                    }
                    descriptions.append(pattern_descriptions[pattern_name])
        
        return descriptions if descriptions else ["Mixed ATM operations"]
    
    def visualize_clusters(self, save_path: str = None):
        """
        Create visualization of semantic clusters
        """
        if self.embeddings is None or self.cluster_labels is None:
            print("No clustering data available. Run perform_semantic_clustering first.")
            return
        
        # Reduce dimensionality for visualization
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(self.embeddings)-1))
        embeddings_2d = tsne.fit_transform(self.embeddings)
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                             c=self.cluster_labels, cmap='tab10', alpha=0.7)
        plt.colorbar(scatter)
        plt.title('Semantic Clustering of ATM Transactions (BERT + DBSCAN)')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        
        # Add cluster labels
        unique_labels = set(self.cluster_labels)
        for label in unique_labels:
            if label != -1:  # Skip noise
                cluster_mask = self.cluster_labels == label
                cluster_center = np.mean(embeddings_2d[cluster_mask], axis=0)
                plt.annotate(f'Cluster {label}', cluster_center, 
                           xytext=(5, 5), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

def demonstrate_improved_clustering():
    """
    Demonstration of improved semantic clustering
    """
    # Sample ATM transaction data
    sample_sessions = [
        "TRANSACTION_START CARD_INSERTED PIN_ENTERED CASH_DISPENSED 100 RECEIPT_PRINTED CARD_EJECTED successful",
        "TRANSACTION_START CARD_INSERTED PIN_ENTERED DEVICE_ERROR M-65 SUPERVISOR_MODE maintenance required",
        "TRANSACTION_START CARD_INSERTED PIN_VERIFICATION_FAILED CARD_CAPTURE security response",
        "TRANSACTION_START CARD_INSERTED PIN_ENTERED COMMUNICATION_FAILURE timeout network error",
        "TRANSACTION_START CARD_INSERTED PIN_ENTERED CASH_JAM dispenser mechanism failure service needed",
        "TRANSACTION_START CARD_INSERTED PIN_ENTERED AMOUNT_SELECTED CASH_DISPENSED 200 successful completion",
        "DEVICE_ERROR critical hardware malfunction immediate service required M-01",
        "COMMUNICATION_FAILURE network timeout unable to process authentication server unreachable"
    ]
    
    # Initialize clustering system
    semantic_cluster = ImprovedSemanticClustering()
    
    # Perform clustering
    results = semantic_cluster.perform_semantic_clustering(sample_sessions)
    
    # Display results
    print("\n" + "="*60)
    print("SEMANTIC CLUSTERING ANALYSIS")
    print("="*60)
    
    for cluster_id, analysis in results['cluster_analysis'].items():
        print(f"\n🔍 CLUSTER {cluster_id} ({analysis['size']} sessions)")
        print("-" * 40)
        print("Semantic Characteristics:")
        for characteristic in analysis['characteristics']:
            print(f"  • {characteristic}")
        
        print(f"\nRepresentative Session:")
        print(f"  {analysis['representative_session']}")
        
        print(f"\nSemantic Patterns:")
        for pattern, count in analysis['semantic_patterns'].items():
            if count > 0:
                print(f"  • {pattern}: {count}")
    
    return semantic_cluster, results

if __name__ == "__main__":
    demonstrate_improved_clustering()
