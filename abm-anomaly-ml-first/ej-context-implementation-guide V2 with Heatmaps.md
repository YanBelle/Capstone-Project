# EJ Log Contextual Labeling Implementation Guide

## Overview
This guide will help you modify your existing code to incorporate contextual labeling for EJ logs, enabling better anomaly detection and understanding of ATM transaction flows.

## New Context Discovered from Your Logs

### 1. **Supervisor Mode Operations**
```
[05pSUPERVISOR MODE ENTRY[00p
- Indicates manual intervention/maintenance
- All operations during this mode are administrative
- Should not occur during customer transactions
```

### 2. **Device Recovery Operations**
```
INIT BNA STARTED - RETRACT BIN
CASHIN RETRACT STARTED
CIM-RESET CALLED
- Recovery sequences indicate previous failure
- Important for tracking device reliability
```

### 3. **Multi-Note Denomination Tracking**
```
NOTES PRESENTED 1,1,1,1
CASH TOTAL TYPE1 TYPE2 TYPE3 TYPE4
- Critical for cash reconciliation
- Can detect denomination-specific issues
```

### 4. **External Authentication Failures**
```
EXTERNAL AUTHENTICATE: NO ARPC
GENAC 2 : AAC
- Specific chip card authentication failure
- Different from generic declines
```

## Implementation Steps

### Step 1: Extend the Labeling Schema

```python
# Add to your existing enums in your code

class EventType(Enum):
    # Add these new event types
    SUPERVISOR_ENTRY = "supervisor_entry"
    SUPERVISOR_EXIT = "supervisor_exit"
    DEVICE_RECOVERY = "device_recovery"
    CASH_RECONCILIATION = "cash_reconciliation"
    EXTERNAL_AUTH = "external_auth"
    NOTES_PRESENT = "notes_present"
    NOTES_TAKEN = "notes_taken"
    
class OperationalMode(Enum):
    """Track operational context"""
    NORMAL = "normal"
    SUPERVISOR = "supervisor"
    RECOVERY = "recovery"
    MAINTENANCE = "maintenance"

class RecoveryType(Enum):
    """Types of recovery operations"""
    CIM_RESET = "cim_reset"
    BNA_INIT = "bna_init"
    CHEQUE_RECOVERY = "cheque_recovery"
    CASHIN_RETRACT = "cashin_retract"

# Update your EJLogLabel dataclass
@dataclass
class EJLogLabel:
    # Add these fields to your existing dataclass
    operational_mode: OperationalMode = OperationalMode.NORMAL
    recovery_type: Optional[RecoveryType] = None
    denomination_data: Optional[Dict[str, int]] = None
    auth_failure_type: Optional[str] = None
```

### Step 2: Update Pattern Recognition

```python
class EJLogLabeler:
    def __init__(self):
        # Add these patterns to your existing patterns dictionary
        self.patterns.update({
            # Supervisor mode patterns
            r'SUPERVISOR MODE ENTRY': (None, EventType.SUPERVISOR_ENTRY),
            r'SUPERVISOR MODE EXIT': (None, EventType.SUPERVISOR_EXIT),
            
            # Recovery patterns
            r'INIT BNA STARTED': (None, EventType.DEVICE_RECOVERY),
            r'CIM-RESET CALLED': (None, EventType.DEVICE_RECOVERY),
            r'CASHIN RECOVERY OK': (None, EventType.DEVICE_RECOVERY),
            r'CHEQUE RECOVERY': (None, EventType.DEVICE_RECOVERY),
            
            # Cash handling patterns
            r'NOTES PRESENTED': (TransactionPhase.COMPLETION, EventType.NOTES_PRESENT),
            r'NOTES TAKEN': (TransactionPhase.COMPLETION, EventType.NOTES_TAKEN),
            r'NOTES STACKED': (TransactionPhase.PROCESSING, EventType.CASH_DISPENSE),
            
            # Authentication patterns
            r'EXTERNAL AUTHENTICATE': (TransactionPhase.AUTHENTICATION, EventType.EXTERNAL_AUTH),
        })
        
        # Add recovery type mappings
        self.recovery_patterns = {
            'INIT BNA STARTED': RecoveryType.BNA_INIT,
            'CIM-RESET CALLED': RecoveryType.CIM_RESET,
            'CHEQUE RECOVERY': RecoveryType.CHEQUE_RECOVERY,
            'CASHIN RETRACT STARTED': RecoveryType.CASHIN_RETRACT,
        }
        
        # Add error code for new pattern
        self.error_codes.update({
            'M-38': ('External authentication failure', Severity.ERROR, ErrorCategory.SECURITY),
        })
```

### Step 3: Enhance Context Tracking

```python
def label_log(self, log_text: str) -> List[EJLogLabel]:
    """Enhanced labeling with new context awareness"""
    lines = log_text.split('\n')
    labels = []
    
    # Enhanced state tracking
    current_phase = None
    transaction_active = False
    supervisor_mode = False
    recovery_active = False
    current_recovery_type = None
    transaction_context = {}
    
    for line_num, line in enumerate(lines):
        if not line.strip():
            continue
        
        # Extract base information (existing code)
        timestamp = self._extract_timestamp(line)
        phase, event_type = self._determine_phase_and_event(line, current_phase)
        
        # Track supervisor mode
        if event_type == EventType.SUPERVISOR_ENTRY:
            supervisor_mode = True
        elif event_type == EventType.SUPERVISOR_EXIT:
            supervisor_mode = False
        
        # Track recovery operations
        if event_type == EventType.DEVICE_RECOVERY:
            recovery_active = True
            current_recovery_type = self._identify_recovery_type(line)
        elif 'RECOVERY OK' in line or 'END' in line:
            recovery_active = False
        
        # Determine operational mode
        if supervisor_mode:
            operational_mode = OperationalMode.SUPERVISOR
        elif recovery_active:
            operational_mode = OperationalMode.RECOVERY
        else:
            operational_mode = OperationalMode.NORMAL
        
        # Extract denomination data if present
        denomination_data = self._extract_denomination_data(line)
        
        # Check for authentication failures
        auth_failure = None
        if 'NO ARPC' in line:
            auth_failure = 'no_arpc'
        elif 'GENAC 2 : AAC' in line and 'NO ARPC' in log_text[max(0, line_num-5):line_num]:
            auth_failure = 'external_auth_failed'
        
        # Create enhanced label
        label = EJLogLabel(
            line_number=line_num,
            timestamp=timestamp,
            phase=current_phase or TransactionPhase.INITIALIZATION,
            event_type=event_type,
            severity=severity,
            error_category=error_category,
            error_code=error_code,
            entity=entity,
            amount=amount,
            metadata=metadata,
            operational_mode=operational_mode,
            recovery_type=current_recovery_type if recovery_active else None,
            denomination_data=denomination_data,
            auth_failure_type=auth_failure
        )
        
        # Add context-specific metadata
        if supervisor_mode and transaction_active:
            label.metadata['anomaly'] = 'Transaction during supervisor mode'
            label.severity = Severity.WARNING
        
        if recovery_active and event_type == EventType.TXN_START:
            label.metadata['anomaly'] = 'Transaction started during recovery'
            label.severity = Severity.CRITICAL
        
        labels.append(label)
    
    return labels

def _extract_denomination_data(self, line: str) -> Optional[Dict[str, int]]:
    """Extract cash denomination data"""
    if 'NOTES PRESENTED' in line:
        # Parse "NOTES PRESENTED 1,1,1,1"
        match = re.search(r'NOTES PRESENTED ([\d,]+)', line)
        if match:
            counts = match.group(1).split(',')
            return {f'type_{i+1}': int(count) for i, count in enumerate(counts)}
    
    elif 'DENOMINATION' in line:
        # Parse denomination table
        # This is complex - would need several lines context
        return self._parse_denomination_table(line)
    
    return None

def _identify_recovery_type(self, line: str) -> Optional[RecoveryType]:
    """Identify specific recovery type"""
    for pattern, recovery_type in self.recovery_patterns.items():
        if pattern in line:
            return recovery_type
    return None
```

### Step 4: Add Contextual Anomaly Rules

```python
class ContextAwareAnomalyDetector:
    def __init__(self):
        # Add these rules to your existing anomaly_rules
        self.anomaly_rules.extend([
            # Supervisor mode anomalies
            self._check_supervisor_mode_anomalies,
            
            # Recovery operation anomalies
            self._check_recovery_anomalies,
            
            # Cash reconciliation anomalies
            self._check_cash_reconciliation,
            
            # Authentication anomalies
            self._check_auth_anomalies,
        ])
    
    def _check_supervisor_mode_anomalies(self, labels: List[EJLogLabel]) -> List[Dict]:
        """Check for supervisor mode related anomalies"""
        anomalies = []
        supervisor_active = False
        
        for label in labels:
            if label.event_type == EventType.SUPERVISOR_ENTRY:
                supervisor_active = True
            elif label.event_type == EventType.SUPERVISOR_EXIT:
                supervisor_active = False
            
            # Check for customer transactions during supervisor mode
            if supervisor_active and label.event_type == EventType.TXN_START:
                anomalies.append({
                    'severity': 'HIGH',
                    'type': 'supervisor_transaction',
                    'description': 'Customer transaction initiated during supervisor mode',
                    'line': label.line_number
                })
        
        return anomalies
    
    def _check_recovery_anomalies(self, labels: List[EJLogLabel]) -> List[Dict]:
        """Check for recovery-related anomalies"""
        anomalies = []
        recovery_counts = defaultdict(int)
        
        for label in labels:
            if label.recovery_type:
                recovery_counts[label.recovery_type] += 1
        
        # Multiple recovery attempts indicate persistent issues
        for recovery_type, count in recovery_counts.items():
            if count > 2:
                anomalies.append({
                    'severity': 'MEDIUM',
                    'type': 'repeated_recovery',
                    'description': f'Multiple {recovery_type.value} recovery attempts ({count})',
                    'recommendation': 'Schedule maintenance for affected component'
                })
        
        return anomalies
    
    def _check_cash_reconciliation(self, labels: List[EJLogLabel]) -> List[Dict]:
        """Check cash dispensing anomalies"""
        anomalies = []
        
        for i, label in enumerate(labels):
            if label.denomination_data:
                # Check for high rejection rates
                if 'rejected' in label.metadata:
                    rejected = label.metadata['rejected']
                    dispensed = label.metadata.get('dispensed', 0)
                    if dispensed > 0 and rejected / dispensed > 0.1:  # >10% rejection
                        anomalies.append({
                            'severity': 'MEDIUM',
                            'type': 'high_note_rejection',
                            'description': f'High note rejection rate: {rejected}/{dispensed}',
                            'recommendation': 'Check note quality and cassette condition'
                        })
        
        return anomalies
```

### Step 5: Integration with Your Existing Code

```python
# Modify your existing analyze_text function
def analyze_text(self, text: str) -> Dict[str, Any]:
    """Enhanced analysis with contextual labeling"""
    
    # Get contextual labels
    labeler = EJLogLabeler()
    labels = labeler.label_log(text)
    
    # Run existing BERT analysis
    bert_result = self.bert_detector.predict(text)
    
    # Enhance with contextual information
    enhanced_result = {
        **bert_result,  # Keep existing results
        'contextual_labels': labels,
        'operational_context': self._extract_operational_context(labels),
        'transaction_summary': self._create_transaction_summary(labels),
        'maintenance_indicators': self._check_maintenance_needs(labels),
    }
    
    # Check for context-specific anomalies
    context_anomalies = []
    for rule in self.anomaly_rules:
        anomalies = rule(labels)
        if anomalies:
            context_anomalies.extend(anomalies)
    
    enhanced_result['contextual_anomalies'] = context_anomalies
    
    # Generate actionable insights
    enhanced_result['insights'] = self._generate_insights(labels, context_anomalies)
    
    return enhanced_result

def _extract_operational_context(self, labels: List[EJLogLabel]) -> Dict:
    """Extract operational context from labels"""
    modes = [label.operational_mode for label in labels]
    recovery_types = [label.recovery_type for label in labels if label.recovery_type]
    
    return {
        'primary_mode': max(set(modes), key=modes.count),
        'had_supervisor_intervention': OperationalMode.SUPERVISOR in modes,
        'had_recovery_operations': len(recovery_types) > 0,
        'recovery_types': list(set(recovery_types)),
        'duration_seconds': self._calculate_duration(labels),
    }

def _generate_insights(self, labels: List[EJLogLabel], anomalies: List[Dict]) -> List[str]:
    """Generate actionable insights"""
    insights = []
    
    # Check patterns
    if any(l.auth_failure_type == 'no_arpc' for l in labels):
        insights.append("Chip authentication failing - possible EMV configuration issue")
    
    if any(l.recovery_type == RecoveryType.CIM_RESET for l in labels):
        insights.append("Cash deposit module required reset - monitor for hardware degradation")
    
    # Add more pattern-based insights
    return insights
```

### Step 6: Training Data Preparation

```python
# Prepare your labeled training data
def prepare_training_data(log_files: List[str]) -> List[Tuple[str, List[EJLogLabel]]]:
    """Prepare contextually labeled training data"""
    labeler = EJLogLabeler()
    training_data = []
    
    for log_file in log_files:
        with open(log_file, 'r') as f:
            log_text = f.read()
        
        # Get contextual labels
        labels = labeler.label_log(log_text)
        
        # Review and adjust labels if needed
        labels = review_and_adjust_labels(log_text, labels)
        
        training_data.append((log_text, labels))
    
    return training_data

# Train your model with the new context
trainer = ContextualBERTTrainer()
dataset = trainer.prepare_training_data(training_data)
trainer.train(dataset, epochs=10, learning_rate=2e-5)
```

## Step 7: Visualization Components

### 7.1 SHAP-based Interpretation

```python
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from transformers import pipeline
import torch

class EJLogVisualizer:
    """Visualization tools for EJ log model interpretation"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize SHAP explainer
        self.explainer = None
        self._initialize_shap_explainer()
        
    def _initialize_shap_explainer(self):
        """Initialize SHAP explainer with model wrapper"""
        def model_predict(texts):
            """Wrapper for SHAP compatibility"""
            inputs = self.tokenizer(texts, return_tensors="pt", 
                                  padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
            
            return probs.cpu().numpy()
        
        # Use partition explainer for efficiency
        self.explainer = shap.Explainer(
            model_predict,
            self.tokenizer,
            output_names=["Normal", "Failed", "Suspicious", "Technical Fault"]
        )
    
    def create_shap_visualization(self, log_text: str, save_path: str = None):
        """Create SHAP visualization for log interpretation"""
        # Get SHAP values
        shap_values = self.explainer([log_text])
        
        # Create multiple visualization types
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        # 1. Waterfall plot for top prediction
        plt.sca(axes[0, 0])
        prediction = self.model.predict([log_text])[0]
        shap.plots.waterfall(shap_values[0, :, prediction.argmax()], max_display=15)
        axes[0, 0].set_title(f"SHAP Waterfall - Predicted: {prediction.argmax()}")
        
        # 2. Force plot style visualization
        plt.sca(axes[0, 1])
        self._create_custom_force_plot(log_text, shap_values[0], axes[0, 1])
        
        # 3. Token importance heatmap
        plt.sca(axes[1, 0])
        self._create_token_importance_heatmap(log_text, shap_values[0], axes[1, 0])
        
        # 4. Multi-class SHAP comparison
        plt.sca(axes[1, 1])
        self._create_multiclass_comparison(log_text, shap_values[0], axes[1, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def _create_token_importance_heatmap(self, text: str, shap_values, ax):
        """Create heatmap of token importance across classes"""
        tokens = self.tokenizer.tokenize(text)[:50]  # Limit for visibility
        
        # Create importance matrix
        importance_matrix = np.abs(shap_values[:len(tokens), :]).T
        
        # Create heatmap
        sns.heatmap(importance_matrix, 
                   xticklabels=tokens,
                   yticklabels=["Normal", "Failed", "Suspicious", "Technical"],
                   cmap='YlOrRd',
                   ax=ax,
                   cbar_kws={'label': 'SHAP Value'})
        
        ax.set_title("Token Importance Across Classes")
        ax.set_xlabel("Tokens")
        ax.set_ylabel("Classes")
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    def _create_custom_force_plot(self, text: str, shap_values, ax):
        """Create custom force-style plot for EJ logs"""
        tokens = self.tokenizer.tokenize(text)[:30]
        predicted_class = np.argmax(np.sum(np.abs(shap_values), axis=0))
        values = shap_values[:len(tokens), predicted_class]
        
        # Color tokens by importance
        colors = ['red' if v < 0 else 'green' for v in values]
        positions = np.arange(len(tokens))
        
        ax.barh(positions, values, color=colors, alpha=0.6)
        ax.set_yticks(positions)
        ax.set_yticklabels(tokens)
        ax.set_xlabel('SHAP Value')
        ax.set_title(f'Token Contributions to Prediction')
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Highlight critical tokens
        critical_tokens = ['ERROR', 'FAILED', 'AAC', 'RESET', 'M-']
        for i, token in enumerate(tokens):
            if any(crit in token.upper() for crit in critical_tokens):
                ax.get_yticklabels()[i].set_weight('bold')
                ax.get_yticklabels()[i].set_color('darkred')
```

### 7.2 Attention Visualization with Rollout

```python
class AttentionVisualizer:
    """Visualize attention patterns using Attention Rollout and other techniques"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def get_attention_rollout(self, text: str, layer: int = -1):
        """Compute attention rollout for better attention flow visualization"""
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="pt", 
                               truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get attention weights
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
        
        attentions = outputs.attentions  # List of attention matrices
        
        # Attention rollout computation
        # Average attention heads
        averaged_attentions = []
        for attn in attentions:
            avg_attn = attn.mean(dim=1)  # Average over heads
            averaged_attentions.append(avg_attn)
        
        # Compute rollout
        rollout = averaged_attentions[0]
        for i in range(1, len(averaged_attentions)):
            if i <= layer or layer == -1:
                rollout = torch.matmul(averaged_attentions[i], rollout)
        
        return rollout.squeeze().cpu().numpy()
    
    def visualize_attention_patterns(self, log_text: str, labels: List[EJLogLabel], 
                                   save_path: str = None):
        """Comprehensive attention visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        # 1. Attention Rollout Heatmap
        rollout = self.get_attention_rollout(log_text)
        tokens = self.tokenizer.tokenize(log_text)[:50]
        
        ax = axes[0, 0]
        self._plot_attention_heatmap(rollout[:len(tokens), :len(tokens)], 
                                    tokens, tokens, ax, "Attention Rollout")
        
        # 2. Layer-wise Attention Evolution
        ax = axes[0, 1]
        self._plot_layer_attention_evolution(log_text, ax)
        
        # 3. Contextual Label Attention
        ax = axes[1, 0]
        self._plot_contextual_attention(log_text, labels, ax)
        
        # 4. Critical Token Focus
        ax = axes[1, 1]
        self._plot_critical_token_attention(log_text, labels, ax)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def _plot_attention_heatmap(self, attention_matrix, x_labels, y_labels, ax, title):
        """Plot attention heatmap with EJ log specific formatting"""
        # Create mask for special tokens
        mask = np.zeros_like(attention_matrix)
        for i, token in enumerate(x_labels):
            if token in ['[CLS]', '[SEP]', '[PAD]']:
                mask[i, :] = 1
                mask[:, i] = 1
        
        # Plot heatmap
        sns.heatmap(attention_matrix,
                   xticklabels=x_labels,
                   yticklabels=y_labels,
                   cmap='Blues',
                   mask=mask,
                   ax=ax,
                   cbar_kws={'label': 'Attention Weight'})
        
        # Highlight important tokens
        important_tokens = ['ERROR', 'FAILED', 'M-', 'RESET', 'AAC', 'DEVICE']
        for i, token in enumerate(x_labels):
            if any(imp in token.upper() for imp in important_tokens):
                ax.get_xticklabels()[i].set_weight('bold')
                ax.get_xticklabels()[i].set_color('red')
        
        ax.set_title(title)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    def _plot_contextual_attention(self, text: str, labels: List[EJLogLabel], ax):
        """Visualize attention aligned with contextual labels"""
        # Get attention for each labeled segment
        attention_by_phase = defaultdict(list)
        attention_by_severity = defaultdict(list)
        
        # Process each label
        for label in labels:
            if label.line_number < len(text.split('\n')):
                line_text = text.split('\n')[label.line_number]
                if line_text.strip():
                    # Get attention for this line
                    line_attention = self._get_line_attention(line_text)
                    attention_by_phase[label.phase.value].append(np.mean(line_attention))
                    attention_by_severity[label.severity.value].append(np.mean(line_attention))
        
        # Create grouped bar plot
        phases = list(attention_by_phase.keys())
        severities = list(attention_by_severity.keys())
        
        x = np.arange(len(phases))
        width = 0.35
        
        phase_means = [np.mean(attention_by_phase[p]) if attention_by_phase[p] else 0 
                      for p in phases]
        
        ax.bar(x, phase_means, width, label='Attention by Phase', alpha=0.8)
        ax.set_xlabel('Transaction Phase')
        ax.set_ylabel('Average Attention')
        ax.set_title('Attention Distribution Across Transaction Phases')
        ax.set_xticks(x)
        ax.set_xticklabels(phases, rotation=45)
        
        # Add severity overlay
        severity_colors = {'INFO': 'green', 'WARNING': 'yellow', 
                          'ERROR': 'orange', 'CRITICAL': 'red'}
        
        for i, phase in enumerate(phases):
            # Color bars by dominant severity in that phase
            phase_labels = [l for l in labels if l.phase.value == phase]
            if phase_labels:
                dominant_severity = max(phase_labels, key=lambda l: l.severity.value).severity.name
                ax.patches[i].set_facecolor(severity_colors.get(dominant_severity, 'blue'))
```

### 7.3 Integrated Visualization Dashboard

```python
class EJLogAnalysisDashboard:
    """Complete visualization dashboard for EJ log analysis"""
    
    def __init__(self, model, tokenizer, labeler):
        self.model = model
        self.tokenizer = tokenizer
        self.labeler = labeler
        self.shap_viz = EJLogVisualizer(model, tokenizer)
        self.attention_viz = AttentionVisualizer(model, tokenizer)
    
    def create_analysis_report(self, log_text: str, output_dir: str = "./analysis_output"):
        """Generate comprehensive analysis report with visualizations"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Get contextual labels
        labels = self.labeler.label_log(log_text)
        
        # 2. Get model predictions
        result = self.model.analyze_text(log_text)
        
        # 3. Create visualizations
        print("Generating SHAP visualizations...")
        shap_fig = self.shap_viz.create_shap_visualization(
            log_text, 
            os.path.join(output_dir, "shap_analysis.png")
        )
        
        print("Generating attention visualizations...")
        attention_fig = self.attention_viz.visualize_attention_patterns(
            log_text, 
            labels,
            os.path.join(output_dir, "attention_analysis.png")
        )
        
        # 4. Create summary visualization
        self._create_summary_dashboard(log_text, labels, result, output_dir)
        
        # 5. Generate HTML report
        self._generate_html_report(log_text, labels, result, output_dir)
        
        print(f"Analysis complete. Report saved to {output_dir}")
        
        return {
            'labels': labels,
            'predictions': result,
            'visualizations': {
                'shap': shap_fig,
                'attention': attention_fig
            }
        }
    
    def _create_summary_dashboard(self, log_text: str, labels: List[EJLogLabel], 
                                 result: Dict, output_dir: str):
        """Create summary dashboard visualization"""
        fig = plt.figure(figsize=(20, 12))
        
        # Create grid
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Transaction Flow Timeline
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_transaction_timeline(labels, ax1)
        
        # 2. Severity Distribution
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_severity_distribution(labels, ax2)
        
        # 3. Event Type Distribution
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_event_distribution(labels, ax3)
        
        # 4. Anomaly Score Timeline
        ax4 = fig.add_subplot(gs[1, 2])
        self._plot_anomaly_timeline(labels, result, ax4)
        
        # 5. Model Confidence
        ax5 = fig.add_subplot(gs[2, 0])
        self._plot_model_confidence(result, ax5)
        
        # 6. Top Contributing Tokens
        ax6 = fig.add_subplot(gs[2, 1:])
        self._plot_top_tokens(log_text, result, ax6)
        
        plt.suptitle('EJ Log Analysis Summary Dashboard', fontsize=16)
        plt.savefig(os.path.join(output_dir, 'summary_dashboard.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_transaction_timeline(self, labels: List[EJLogLabel], ax):
        """Plot transaction events timeline"""
        events = [(l.timestamp, l.event_type.value, l.severity.value) 
                 for l in labels if l.timestamp]
        
        if not events:
            ax.text(0.5, 0.5, 'No timestamp data available', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        # Convert to timeline
        times = [e[0] for e in events]
        event_types = [e[1] for e in events]
        severities = [e[2] for e in events]
        
        # Color by severity
        colors = ['green', 'yellow', 'orange', 'red']
        event_colors = [colors[s] for s in severities]
        
        # Plot timeline
        y_positions = np.arange(len(events))
        ax.scatter(range(len(events)), y_positions, c=event_colors, s=100, alpha=0.6)
        
        # Add event labels
        for i, (time, event, severity) in enumerate(events):
            ax.annotate(event, (i, i), xytext=(5, 5), 
                       textcoords='offset points', fontsize=8)
        
        ax.set_xlabel('Event Sequence')
        ax.set_ylabel('Event Index')
        ax.set_title('Transaction Event Timeline')
        ax.grid(True, alpha=0.3)
    
    def _generate_html_report(self, log_text: str, labels: List[EJLogLabel], 
                             result: Dict, output_dir: str):
        """Generate interactive HTML report"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>EJ Log Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; }}
                .anomaly {{ background-color: #ffebee; }}
                .normal {{ background-color: #e8f5e9; }}
                .warning {{ background-color: #fff3e0; }}
                .critical {{ background-color: #ffcdd2; }}
                .token-important {{ background-color: #ffeb3b; font-weight: bold; }}
                img {{ max-width: 100%; height: auto; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>EJ Log Analysis Report</h1>
                <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>Summary</h2>
                <p>Prediction: <strong>{result.get('class_name', 'Unknown')}</strong></p>
                <p>Confidence: <strong>{result.get('confidence', 0):.2%}</strong></p>
                <p>Anomaly Score: <strong>{result.get('anomaly_score', 0):.2f}</strong></p>
            </div>
            
            <div class="section">
                <h2>Visualizations</h2>
                <h3>SHAP Analysis</h3>
                <img src="shap_analysis.png" alt="SHAP Analysis">
                
                <h3>Attention Analysis</h3>
                <img src="attention_analysis.png" alt="Attention Analysis">
                
                <h3>Summary Dashboard</h3>
                <img src="summary_dashboard.png" alt="Summary Dashboard">
            </div>
            
            <div class="section">
                <h2>Contextual Labels</h2>
                <table>
                    <tr>
                        <th>Line</th>
                        <th>Phase</th>
                        <th>Event</th>
                        <th>Severity</th>
                        <th>Details</th>
                    </tr>
                    {self._generate_label_rows(labels)}
                </table>
            </div>
            
            <div class="section">
                <h2>Recommendations</h2>
                <ul>
                    {self._generate_recommendations(labels, result)}
                </ul>
            </div>
        </body>
        </html>
        """
        
        with open(os.path.join(output_dir, 'analysis_report.html'), 'w') as f:
            f.write(html_content)
```

## Key Benefits of This Implementation

1. **Supervisor Mode Awareness**: Distinguishes between normal operations and maintenance
2. **Recovery Tracking**: Identifies patterns in device failures and recoveries
3. **Authentication Insights**: Specific understanding of chip card failures
4. **Cash Reconciliation**: Tracks denomination-specific issues
5. **Operational Context**: Provides full transaction lifecycle understanding
6. **Visual Interpretability**: SHAP and attention visualizations show model reasoning
7. **Interactive Reports**: HTML dashboards for easy analysis sharing

## Usage Example

```python
# Initialize components
model = ContextAwareBERTModel.from_pretrained(model_path)
tokenizer = EJLogTokenizer()
labeler = EJLogLabeler()

# Create analysis dashboard
dashboard = EJLogAnalysisDashboard(model, tokenizer, labeler)

# Analyze EJ log with full visualization
ej_log_text = """[020t*629*06/18/2025*00:46*
     *TRANSACTION START*
[020t CARD INSERTED
...
DEVICE ERROR
..."""

# Generate complete analysis with visualizations
results = dashboard.create_analysis_report(
    ej_log_text,
    output_dir="./ej_analysis_results"
)

# Access individual components
print(f"Detected anomalies: {len(results['predictions']['contextual_anomalies'])}")
print(f"Operational context: {results['predictions']['operational_context']}")

# View visualizations
import webbrowser
webbrowser.open('file://./ej_analysis_results/analysis_report.html')
```

## Visualization Features

### SHAP Visualizations:
- **Waterfall plots**: Show contribution of each token to prediction
- **Force plots**: Visualize positive/negative token influences
- **Multi-class comparison**: Compare token importance across all classes
- **Token importance heatmap**: See which tokens matter most for each prediction

### Attention Visualizations:
- **Attention rollout**: Shows information flow through the model
- **Layer-wise evolution**: How attention patterns change through layers
- **Contextual alignment**: Attention mapped to transaction phases
- **Critical token focus**: Highlights attention on error indicators

### Summary Dashboard:
- **Transaction timeline**: Visual flow of events with severity
- **Anomaly progression**: How anomaly scores evolve through the log
- **Model confidence**: Certainty levels for predictions
- **Interactive HTML report**: Complete analysis in shareable format

This implementation provides complete visibility into how your model interprets EJ logs, making it easy to validate predictions and identify areas for improvement.