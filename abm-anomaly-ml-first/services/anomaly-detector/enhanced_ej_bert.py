"""
Enhanced BERT Model with EJ Log Contextual Understanding
Combines domain-specific labeling with BERT's language understanding
"""

import torch
import torch.nn as nn
import numpy as np
from transformers import BertTokenizer, BertModel, BertConfig
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
import json

from ej_contextual_labeler import EJLogLabeler, EJLogLabel, EventType, TransactionPhase, Severity

logger = logging.getLogger(__name__)

@dataclass
class EJBertConfig:
    """Configuration for EJ-enhanced BERT model"""
    bert_model_name: str = "bert-base-uncased"
    max_sequence_length: int = 512
    num_classes: int = 4  # Normal, Failed, Suspicious, Technical Fault
    
    # EJ-specific configuration
    use_contextual_features: bool = True
    contextual_feature_dim: int = 64
    phase_embedding_dim: int = 32
    event_embedding_dim: int = 32
    
    # Training parameters
    dropout_rate: float = 0.3
    learning_rate: float = 2e-5
    warmup_steps: int = 500

class EJContextualFeatureExtractor(nn.Module):
    """Extract and encode EJ-specific contextual features"""
    
    def __init__(self, config: EJBertConfig):
        super().__init__()
        self.config = config
        
        # Define feature dimensions
        self.num_phases = len(TransactionPhase)
        self.num_events = len(EventType) 
        self.num_severities = len(Severity)
        
        # Embedding layers for categorical features
        self.phase_embedding = nn.Embedding(self.num_phases, config.phase_embedding_dim)
        self.event_embedding = nn.Embedding(self.num_events, config.event_embedding_dim)
        self.severity_embedding = nn.Embedding(self.num_severities, 16)
        
        # Feature projection layers
        self.contextual_features_dim = (config.phase_embedding_dim + 
                                       config.event_embedding_dim + 
                                       16 +  # severity embedding
                                       32)   # additional numerical features
        
        self.feature_projector = nn.Sequential(
            nn.Linear(self.contextual_features_dim, config.contextual_feature_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.contextual_feature_dim, config.contextual_feature_dim)
        )
        
        # Attention mechanism for feature importance
        self.feature_attention = nn.MultiheadAttention(
            embed_dim=config.contextual_feature_dim,
            num_heads=4,
            dropout=config.dropout_rate
        )
    
    def forward(self, contextual_labels: List[List[EJLogLabel]], 
                sequence_lengths: List[int]) -> torch.Tensor:
        """
        Extract contextual features from EJ labels
        
        Args:
            contextual_labels: List of label sequences for each input
            sequence_lengths: Length of each sequence
            
        Returns:
            Tensor of contextual features [batch_size, contextual_feature_dim]
        """
        batch_size = len(contextual_labels)
        device = next(self.parameters()).device
        
        # Initialize batch features
        batch_features = []
        
        for labels in contextual_labels:
            if not labels:
                # Handle empty labels
                features = torch.zeros(self.contextual_features_dim, device=device)
            else:
                features = self._extract_sequence_features(labels, device)
            
            batch_features.append(features)
        
        # Stack and project features
        batch_features = torch.stack(batch_features)  # [batch_size, feature_dim]
        projected_features = self.feature_projector(batch_features)
        
        # Apply attention to focus on important features
        attended_features, _ = self.feature_attention(
            projected_features.unsqueeze(0),
            projected_features.unsqueeze(0), 
            projected_features.unsqueeze(0)
        )
        
        return attended_features.squeeze(0)
    
    def _extract_sequence_features(self, labels: List[EJLogLabel], device) -> torch.Tensor:
        """Extract features from a sequence of EJ labels"""
        if not labels:
            return torch.zeros(self.contextual_features_dim, device=device)
        
        # Extract categorical features (use mode for sequence-level features)
        phases = [self._phase_to_idx(label.phase) for label in labels]
        events = [self._event_to_idx(label.event_type) for label in labels]
        severities = [self._severity_to_idx(label.severity) for label in labels]
        
        # Get most common values
        phase_mode = max(set(phases), key=phases.count) if phases else 0
        event_mode = max(set(events), key=events.count) if events else 0
        severity_mode = max(set(severities), key=severities.count) if severities else 0
        
        # Create embeddings
        phase_emb = self.phase_embedding(torch.tensor(phase_mode, device=device))
        event_emb = self.event_embedding(torch.tensor(event_mode, device=device))
        severity_emb = self.severity_embedding(torch.tensor(severity_mode, device=device))
        
        # Extract numerical features
        numerical_features = self._extract_numerical_features(labels, device)
        
        # Concatenate all features
        features = torch.cat([phase_emb, event_emb, severity_emb, numerical_features])
        
        return features
    
    def _extract_numerical_features(self, labels: List[EJLogLabel], device) -> torch.Tensor:
        """Extract numerical features from labels"""
        features = torch.zeros(32, device=device)  # 32 numerical features
        
        if not labels:
            return features
        
        # Transaction flow features
        has_start = any(l.event_type == EventType.TXN_START for l in labels)
        has_end = any(l.event_type == EventType.TXN_END for l in labels)
        features[0] = float(has_start and has_end)  # Complete transaction
        
        # Error analysis
        error_count = sum(1 for l in labels if l.severity in [Severity.ERROR, Severity.CRITICAL])
        features[1] = min(error_count / len(labels), 1.0)  # Error rate
        
        # Operational mode features
        supervisor_time = sum(1 for l in labels if l.operational_mode.value == 'supervisor')
        features[2] = supervisor_time / len(labels) if labels else 0.0
        
        recovery_time = sum(1 for l in labels if l.recovery_type is not None)
        features[3] = recovery_time / len(labels) if labels else 0.0
        
        # Authentication features
        auth_failures = sum(1 for l in labels if l.auth_failure_type is not None)
        features[4] = auth_failures / len(labels) if labels else 0.0
        
        # Cash handling features
        cash_operations = sum(1 for l in labels if l.denomination_data is not None)
        features[5] = cash_operations / len(labels) if labels else 0.0
        
        # Temporal features
        timestamps = [l.timestamp for l in labels if l.timestamp]
        if len(timestamps) > 1:
            duration = (timestamps[-1] - timestamps[0]).total_seconds()
            features[6] = min(duration / 3600, 1.0)  # Duration in hours (capped at 1)
        
        # Confidence features
        avg_confidence = sum(l.confidence_score for l in labels) / len(labels)
        features[7] = avg_confidence
        
        # Phase distribution (remaining 24 features for phase coverage)
        phase_counts = {}
        for label in labels:
            phase_idx = self._phase_to_idx(label.phase)
            phase_counts[phase_idx] = phase_counts.get(phase_idx, 0) + 1
        
        for i in range(min(24, len(TransactionPhase))):
            features[8 + i] = phase_counts.get(i, 0) / len(labels)
        
        return features
    
    def _phase_to_idx(self, phase: TransactionPhase) -> int:
        """Convert phase to index"""
        phase_mapping = {p: i for i, p in enumerate(TransactionPhase)}
        return phase_mapping.get(phase, 0)
    
    def _event_to_idx(self, event: EventType) -> int:
        """Convert event to index"""
        event_mapping = {e: i for i, e in enumerate(EventType)}
        return event_mapping.get(event, 0)
    
    def _severity_to_idx(self, severity: Severity) -> int:
        """Convert severity to index"""
        severity_mapping = {s: i for i, s in enumerate(Severity)}
        return severity_mapping.get(severity, 0)

class EnhancedEJBertModel(nn.Module):
    """Enhanced BERT model with EJ Log contextual understanding"""
    
    def __init__(self, config: EJBertConfig):
        super().__init__()
        self.config = config
        
        # Initialize BERT
        self.bert_config = BertConfig.from_pretrained(config.bert_model_name)
        self.bert = BertModel.from_pretrained(config.bert_model_name)
        
        # EJ contextual feature extractor
        self.contextual_extractor = EJContextualFeatureExtractor(config)
        
        # Fusion layers
        bert_dim = self.bert_config.hidden_size
        total_dim = bert_dim + config.contextual_feature_dim
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(total_dim, bert_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(bert_dim, bert_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(bert_dim // 2, config.num_classes),
            nn.Dropout(config.dropout_rate)
        )
        
        # Attention mechanism for BERT-contextual fusion
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=bert_dim,
            num_heads=8,
            dropout=config.dropout_rate
        )
    
    def forward(self, input_ids, attention_mask, contextual_labels=None):
        """
        Forward pass with EJ contextual enhancement
        
        Args:
            input_ids: BERT input token IDs
            attention_mask: BERT attention mask
            contextual_labels: List of EJ contextual labels for each sequence
            
        Returns:
            logits: Classification logits
            attention_weights: Attention weights for interpretability
        """
        # BERT forward pass
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True
        )
        
        sequence_output = bert_outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        pooled_output = bert_outputs.pooler_output        # [batch_size, hidden_size]
        attention_weights = bert_outputs.attentions
        
        # Extract contextual features if available
        if contextual_labels and self.config.use_contextual_features:
            sequence_lengths = [len(labels) for labels in contextual_labels]
            contextual_features = self.contextual_extractor(contextual_labels, sequence_lengths)
            
            # Fuse BERT and contextual features
            fused_features = torch.cat([pooled_output, contextual_features], dim=1)
            enhanced_representation = self.fusion_layer(fused_features)
        else:
            # Use only BERT features
            enhanced_representation = self.fusion_layer(
                torch.cat([pooled_output, torch.zeros_like(pooled_output[:, :self.config.contextual_feature_dim])], dim=1)
            )
        
        # Classification
        logits = self.classifier(enhanced_representation)
        
        return {
            'logits': logits,
            'attention_weights': attention_weights,
            'contextual_features': contextual_features if contextual_labels else None,
            'bert_features': pooled_output
        }

class EJBertAnalyzer:
    """Main analyzer class combining BERT with EJ contextual understanding"""
    
    def __init__(self, model_path: str = None, config: EJBertConfig = None):
        self.config = config or EJBertConfig()
        self.tokenizer = BertTokenizer.from_pretrained(self.config.bert_model_name)
        self.labeler = EJLogLabeler()
        
        # Initialize model
        self.model = EnhancedEJBertModel(self.config)
        
        if model_path:
            self.load_model(model_path)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Class labels
        self.class_labels = ['Normal', 'Failed', 'Suspicious', 'Technical Fault']
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze EJ log text with enhanced contextual understanding
        
        Args:
            text: EJ log text
            
        Returns:
            Analysis results with predictions, confidence, and explanations
        """
        self.model.eval()
        
        with torch.no_grad():
            # Get contextual labels
            contextual_labels = self.labeler.label_log(text)
            
            # Tokenize input
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding=True,
                max_length=self.config.max_sequence_length,
                return_tensors='pt'
            )
            
            # Move to device
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                contextual_labels=[contextual_labels]
            )
            
            # Get predictions
            logits = outputs['logits']
            probabilities = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            predicted_class = np.argmax(probabilities)
            confidence = float(probabilities[predicted_class])
            
            # Generate explanation
            explanation = self._generate_explanation(
                text, contextual_labels, probabilities, outputs
            )
            
            return {
                'prediction': self.class_labels[predicted_class],
                'confidence': confidence,
                'probabilities': {
                    label: float(prob) for label, prob in zip(self.class_labels, probabilities)
                },
                'contextual_labels': [
                    {
                        'line_number': label.line_number,
                        'phase': label.phase.value,
                        'event_type': label.event_type.value,
                        'severity': label.severity.value,
                        'operational_mode': label.operational_mode.value,
                        'confidence_score': label.confidence_score
                    } for label in contextual_labels
                ],
                'explanation': explanation,
                'technical_details': self._extract_technical_details(contextual_labels),
                'anomaly_indicators': self._extract_anomaly_indicators(contextual_labels)
            }
    
    def _generate_explanation(self, text: str, labels: List[EJLogLabel], 
                            probabilities: np.ndarray, outputs: Dict) -> str:
        """Generate human-readable explanation of the analysis"""
        explanation_parts = []
        
        # Primary prediction explanation
        predicted_class = np.argmax(probabilities)
        confidence = probabilities[predicted_class]
        
        explanation_parts.append(
            f"Predicted class: {self.class_labels[predicted_class]} "
            f"(confidence: {confidence:.2%})"
        )
        
        # Contextual insights
        if labels:
            # Transaction flow analysis
            transaction_phases = set(label.phase for label in labels)
            if len(transaction_phases) > 1:
                explanation_parts.append(
                    f"Transaction spans {len(transaction_phases)} phases: "
                    f"{', '.join(phase.value for phase in transaction_phases)}"
                )
            
            # Error analysis
            errors = [label for label in labels if label.severity in [Severity.ERROR, Severity.CRITICAL]]
            if errors:
                explanation_parts.append(
                    f"Detected {len(errors)} error(s) with severities: "
                    f"{', '.join(error.severity.value for error in errors)}"
                )
            
            # Operational context
            operational_modes = set(label.operational_mode for label in labels)
            if OperationalMode.SUPERVISOR in operational_modes:
                explanation_parts.append("Supervisor mode detected - indicates maintenance activity")
            
            recovery_types = set(label.recovery_type for label in labels if label.recovery_type)
            if recovery_types:
                explanation_parts.append(
                    f"Recovery operations detected: "
                    f"{', '.join(rt.value for rt in recovery_types)}"
                )
            
            # Authentication issues
            auth_failures = [label.auth_failure_type for label in labels if label.auth_failure_type]
            if auth_failures:
                explanation_parts.append(
                    f"Authentication issues detected: {', '.join(set(auth_failures))}"
                )
        
        return " | ".join(explanation_parts)
    
    def _extract_technical_details(self, labels: List[EJLogLabel]) -> Dict[str, Any]:
        """Extract technical details for debugging and analysis"""
        if not labels:
            return {}
        
        return {
            'total_events': len(labels),
            'unique_phases': len(set(label.phase for label in labels)),
            'error_rate': sum(1 for l in labels if l.severity in [Severity.ERROR, Severity.CRITICAL]) / len(labels),
            'average_confidence': sum(label.confidence_score for label in labels) / len(labels),
            'supervisor_mode_duration': sum(1 for l in labels if l.operational_mode == OperationalMode.SUPERVISOR),
            'recovery_operations': len([l for l in labels if l.recovery_type is not None]),
            'cash_operations': len([l for l in labels if l.denomination_data is not None]),
            'authentication_events': len([l for l in labels if l.event_type == EventType.EXTERNAL_AUTH])
        }
    
    def _extract_anomaly_indicators(self, labels: List[EJLogLabel]) -> List[Dict[str, Any]]:
        """Extract specific anomaly indicators from contextual analysis"""
        anomalies = []
        
        if not labels:
            return anomalies
        
        # Check for specific anomaly patterns
        for label in labels:
            if 'contextual_anomalies' in label.metadata:
                for anomaly in label.metadata['contextual_anomalies']:
                    anomalies.append({
                        'type': 'contextual',
                        'description': anomaly,
                        'line_number': label.line_number,
                        'severity': label.severity.value,
                        'confidence': label.confidence_score
                    })
            
            if 'flow_anomalies' in label.metadata:
                for anomaly in label.metadata['flow_anomalies']:
                    anomalies.append({
                        'type': 'flow',
                        'description': anomaly,
                        'line_number': label.line_number,
                        'severity': 'warning',
                        'confidence': label.confidence_score
                    })
        
        return anomalies
    
    def load_model(self, model_path: str):
        """Load trained model weights"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def save_model(self, model_path: str):
        """Save model weights"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'class_labels': self.class_labels
        }, model_path)
        logger.info(f"Model saved to {model_path}")

# Import necessary modules for OperationalMode
from ej_contextual_labeler import OperationalMode
