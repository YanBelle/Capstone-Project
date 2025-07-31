import numpy as np
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                           confusion_matrix, roc_curve, auc, precision_recall_curve)
from typing import List, Dict, Tuple

class MetricsCalculator:
    def __init__(self):
        self.predictions = []
        self.true_labels = []
        self.probabilities = []
        
    def add_batch(self, predictions: List[int], true_labels: List[int], 
                  probabilities: List[List[float]]):
        """Add a batch of predictions for metric calculation"""
        self.predictions.extend(predictions)
        self.true_labels.extend(true_labels)
        self.probabilities.extend(probabilities)
    
    def calculate_metrics(self) -> Dict:
        """Calculate comprehensive metrics"""
        if not self.predictions:
            return {}
            
        predictions = np.array(self.predictions)
        true_labels = np.array(self.true_labels)
        probabilities = np.array(self.probabilities)
        
        # Basic metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='weighted', zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        
        # ROC and PR curves (for binary classification)
        roc_data = {}
        pr_data = {}
        
        if probabilities.shape[1] == 2:  # Binary classification
            fpr, tpr, _ = roc_curve(true_labels, probabilities[:, 1])
            roc_auc = auc(fpr, tpr)
            
            precision_curve, recall_curve, _ = precision_recall_curve(
                true_labels, probabilities[:, 1]
            )
            
            roc_data = {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "auc": roc_auc
            }
            
            pr_data = {
                "precision": precision_curve.tolist(),
                "recall": recall_curve.tolist()
            }
        
        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "confusion_matrix": cm.tolist(),
            "roc_curve": roc_data,
            "pr_curve": pr_data
        }
    
    def get_misclassifications(self) -> List[Dict]:
        """Get misclassified examples"""
        misclassified = []
        for i, (pred, true) in enumerate(zip(self.predictions, self.true_labels)):
            if pred != true:
                misclassified.append({
                    "index": i,
                    "predicted": pred,
                    "true_label": true,
                    "confidence": max(self.probabilities[i])
                })
        return misclassified
