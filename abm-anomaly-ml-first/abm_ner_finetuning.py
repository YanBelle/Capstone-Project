#!/usr/bin/env python3
"""
ABM NER Fine-tuning Pipeline
===========================

Fine-tune BERT-based NER model specifically for ABM log patterns.
Recognizes entities like TRANSACTION_START, TIMESTAMP, CARD_NUMBER, ERROR_CODE, etc.
"""

import os
import torch
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import pandas as pd

# Hugging Face imports
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    TrainingArguments, Trainer, DataCollatorForTokenClassification
)
from datasets import Dataset
from sklearn.metrics import classification_report, confusion_matrix
from seqeval.metrics import f1_score, accuracy_score, classification_report as seq_classification_report

@dataclass
class ABMLogSample:
    """Single ABM log sample with entity annotations"""
    text: str
    entities: List[Dict[str, any]]  # [{"start": 0, "end": 5, "label": "TRANSACTION_START"}]

class ABMNERDatasetCreator:
    """Creates training dataset from ABM logs with entity annotations"""
    
    def __init__(self):
        # Define ABM-specific entity types
        self.entity_labels = [
            'O',  # Outside any entity
            'B-TRANSACTION_START',  # Beginning of transaction start marker
            'I-TRANSACTION_START',  # Inside transaction start marker
            'B-TIMESTAMP',          # Beginning of timestamp
            'I-TIMESTAMP',          # Inside timestamp
            'B-CARD_NUMBER',        # Beginning of card number (masked)
            'I-CARD_NUMBER',        # Inside card number
            'B-ERROR_CODE',         # Beginning of error code
            'I-ERROR_CODE',         # Inside error code
            'B-AMOUNT',             # Beginning of amount
            'I-AMOUNT',             # Inside amount
            'B-DEVICE_ID',          # Beginning of device identifier
            'I-DEVICE_ID',          # Inside device identifier
            'B-SESSION_BOUNDARY',   # Beginning of session boundary marker
            'I-SESSION_BOUNDARY',   # Inside session boundary marker
            'B-EVENT_TYPE',         # Beginning of event type
            'I-EVENT_TYPE',         # Inside event type
            'B-STATUS_CODE',        # Beginning of status code
            'I-STATUS_CODE',        # Inside status code
        ]
        
        self.label2id = {label: i for i, label in enumerate(self.entity_labels)}
        self.id2label = {i: label for i, label in enumerate(self.entity_labels)}
        
        # ABM-specific patterns for auto-annotation
        self.abm_patterns = {
            'TRANSACTION_START': [
                r'\*(?:TRANSACTION|CARDLESS TRANSACTION)\s+START\*',
                r'---START OF TRANSACTION---',
                r'TRANSACTION START'
            ],
            'TIMESTAMP': [
                r'\*(\d{1,3})\*(\d{2}/\d{2}/\d{4})\*(\d{2}:\d{2})\*',
                r'\d{2}:\d{2}:\d{2}',
                r'\d{2}/\d{2}/\d{4}'
            ],
            'CARD_NUMBER': [
                r'PAN\s+\d{4}\*+\d{4}',
                r'\d{4}\*+\d{4}',
                r'CARD.*\d{4}'
            ],
            'ERROR_CODE': [
                r'ESC:\s*\d{3}',
                r'VAL:\s*\d{3}',
                r'REF:\s*\d{3}',
                r'DEVICE ERROR',
                r'OPERATION ERROR'
            ],
            'AMOUNT': [
                r'JMD\d+-\d+',
                r'\$\d+\.\d{2}',
                r'AMOUNT.*\d+'
            ],
            'DEVICE_ID': [
                r'\[020t',
                r'ABM\d+',
                r'\[05p'
            ],
            'SESSION_BOUNDARY': [
                r'\*PRIMARY CARD READER ACTIVATED\*',
                r'TRANSACTION END',
                r'SESSION END'
            ],
            'EVENT_TYPE': [
                r'CARD INSERTED',
                r'PIN ENTERED',
                r'NOTES PRESENTED',
                r'NOTES TAKEN',
                r'CARD TAKEN',
                r'ATR RECEIVED'
            ],
            'STATUS_CODE': [
                r'OPERATION OK',
                r'DEVICE ERROR',
                r'COMMUNICATION TIMEOUT'
            ]
        }
    
    def create_training_data_from_files(self, input_dir: str, output_file: str) -> None:
        """Create training dataset from ABM log files"""
        samples = []
        
        for log_file in Path(input_dir).glob("*.txt"):
            print(f"Processing {log_file}")
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Split into manageable chunks (avoid token limit)
            chunks = self._split_into_chunks(content, max_chars=512)
            
            for chunk in chunks:
                entities = self._auto_annotate_chunk(chunk)
                if entities:  # Only include chunks with entities
                    samples.append(ABMLogSample(text=chunk, entities=entities))
        
        # Convert to training format
        training_data = self._convert_to_training_format(samples)
        
        # Save dataset
        with open(output_file, 'w') as f:
            json.dump(training_data, f, indent=2)
        
        print(f"Created training dataset with {len(training_data)} samples")
        print(f"Saved to: {output_file}")
    
    def _split_into_chunks(self, text: str, max_chars: int = 512) -> List[str]:
        """Split long text into smaller chunks preserving line boundaries"""
        lines = text.split('\n')
        chunks = []
        current_chunk = []
        current_length = 0
        
        for line in lines:
            line_length = len(line) + 1  # +1 for newline
            
            if current_length + line_length > max_chars and current_chunk:
                chunks.append('\n'.join(current_chunk))
                current_chunk = [line]
                current_length = line_length
            else:
                current_chunk.append(line)
                current_length += line_length
        
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return chunks
    
    def _auto_annotate_chunk(self, text: str) -> List[Dict[str, any]]:
        """Automatically annotate ABM log chunk with entity labels"""
        entities = []
        
        for entity_type, patterns in self.abm_patterns.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    entities.append({
                        'start': match.start(),
                        'end': match.end(),
                        'label': entity_type,
                        'text': match.group()
                    })
        
        # Remove overlapping entities (keep longest)
        entities = self._remove_overlapping_entities(entities)
        return entities
    
    def _remove_overlapping_entities(self, entities: List[Dict]) -> List[Dict]:
        """Remove overlapping entities, keeping the longest ones"""
        if not entities:
            return entities
        
        # Sort by start position
        entities.sort(key=lambda x: x['start'])
        
        non_overlapping = []
        for entity in entities:
            # Check if it overlaps with any existing entity
            overlaps = False
            for existing in non_overlapping:
                if (entity['start'] < existing['end'] and entity['end'] > existing['start']):
                    # Keep the longer entity
                    if (entity['end'] - entity['start']) > (existing['end'] - existing['start']):
                        non_overlapping.remove(existing)
                        break
                    else:
                        overlaps = True
                        break
            
            if not overlaps:
                non_overlapping.append(entity)
        
        return non_overlapping
    
    def _convert_to_training_format(self, samples: List[ABMLogSample]) -> List[Dict]:
        """Convert samples to HuggingFace training format"""
        training_data = []
        
        for sample in samples:
            # Tokenize text into words
            words = sample.text.split()
            
            # Create BIO tags for each word
            labels = ['O'] * len(words)
            
            # Map entities to word-level labels
            char_to_word = self._create_char_to_word_mapping(sample.text, words)
            
            for entity in sample.entities:
                start_word = char_to_word.get(entity['start'])
                end_word = char_to_word.get(entity['end'] - 1)
                
                if start_word is not None and end_word is not None:
                    # Set B- tag for first word, I- tags for subsequent words
                    labels[start_word] = f"B-{entity['label']}"
                    for i in range(start_word + 1, end_word + 1):
                        if i < len(labels):
                            labels[i] = f"I-{entity['label']}"
            
            training_data.append({
                'id': len(training_data),
                'tokens': words,
                'ner_tags': labels
            })
        
        return training_data
    
    def _create_char_to_word_mapping(self, text: str, words: List[str]) -> Dict[int, int]:
        """Create mapping from character position to word index"""
        char_to_word = {}
        char_pos = 0
        
        for word_idx, word in enumerate(words):
            # Find word in text starting from char_pos
            word_start = text.find(word, char_pos)
            if word_start != -1:
                word_end = word_start + len(word)
                for char_idx in range(word_start, word_end):
                    char_to_word[char_idx] = word_idx
                char_pos = word_end
        
        return char_to_word

class ABMNERFineTuner:
    """Fine-tune BERT model for ABM-specific NER"""
    
    def __init__(self, model_name: str = "bert-base-uncased", output_dir: str = "./abm-ner-model"):
        self.model_name = model_name
        self.output_dir = output_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Entity labels (same as dataset creator)
        self.entity_labels = [
            'O', 'B-TRANSACTION_START', 'I-TRANSACTION_START', 'B-TIMESTAMP', 'I-TIMESTAMP',
            'B-CARD_NUMBER', 'I-CARD_NUMBER', 'B-ERROR_CODE', 'I-ERROR_CODE',
            'B-AMOUNT', 'I-AMOUNT', 'B-DEVICE_ID', 'I-DEVICE_ID',
            'B-SESSION_BOUNDARY', 'I-SESSION_BOUNDARY', 'B-EVENT_TYPE', 'I-EVENT_TYPE',
            'B-STATUS_CODE', 'I-STATUS_CODE'
        ]
        
        self.label2id = {label: i for i, label in enumerate(self.entity_labels)}
        self.id2label = {i: label for i, label in enumerate(self.entity_labels)}
        
        # Initialize model
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=len(self.entity_labels),
            id2label=self.id2label,
            label2id=self.label2id
        )
    
    def load_dataset(self, dataset_file: str) -> Tuple[Dataset, Dataset]:
        """Load and split dataset"""
        with open(dataset_file, 'r') as f:
            data = json.load(f)
        
        # Convert string labels to IDs
        for sample in data:
            sample['labels'] = [self.label2id[label] for label in sample['ner_tags']]
        
        # Split train/validation
        split_idx = int(0.8 * len(data))
        train_data = data[:split_idx]
        val_data = data[split_idx:]
        
        train_dataset = Dataset.from_list(train_data)
        val_dataset = Dataset.from_list(val_data)
        
        # Tokenize datasets
        train_dataset = train_dataset.map(
            self._tokenize_and_align_labels,
            batched=True,
            remove_columns=train_dataset.column_names
        )
        
        val_dataset = val_dataset.map(
            self._tokenize_and_align_labels,
            batched=True,
            remove_columns=val_dataset.column_names
        )
        
        return train_dataset, val_dataset
    
    def _tokenize_and_align_labels(self, examples):
        """Tokenize text and align labels with subword tokens"""
        tokenized_inputs = self.tokenizer(
            examples['tokens'],
            truncation=True,
            is_split_into_words=True,
            padding=True,
            max_length=512
        )
        
        labels = []
        for i, label in enumerate(examples['labels']):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            label_ids = []
            previous_word_idx = None
            
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)  # Special token
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])  # First subword of word
                else:
                    label_ids.append(-100)  # Subsequent subwords
                previous_word_idx = word_idx
            
            labels.append(label_ids)
        
        tokenized_inputs['labels'] = labels
        return tokenized_inputs
    
    def fine_tune(self, train_dataset: Dataset, val_dataset: Dataset, 
                  epochs: int = 3, batch_size: int = 16, learning_rate: float = 2e-5):
        """Fine-tune the model"""
        
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="steps",
            logging_steps=100,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            learning_rate=learning_rate,
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1",
            greater_is_better=True,
            save_total_limit=2,
            report_to=None,  # Disable wandb
        )
        
        data_collator = DataCollatorForTokenClassification(self.tokenizer)
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self._compute_metrics,
        )
        
        print("Starting fine-tuning...")
        trainer.train()
        
        # Save final model
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        print(f"Model saved to: {self.output_dir}")
        
        # Evaluate on validation set
        eval_results = trainer.evaluate()
        print("Validation Results:")
        for key, value in eval_results.items():
            print(f"  {key}: {value:.4f}")
        
        return trainer
    
    def _compute_metrics(self, eval_pred):
        """Compute evaluation metrics"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)
        
        # Remove ignored index (special tokens)
        true_predictions = [
            [self.id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.id2label[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        
        # Calculate metrics
        f1 = f1_score(true_labels, true_predictions)
        accuracy = accuracy_score(true_labels, true_predictions)
        
        return {
            "f1": f1,
            "accuracy": accuracy,
        }

def create_sample_abm_dataset():
    """Create a sample ABM dataset for demonstration"""
    sample_logs = [
        """[020t*632*06/18/2025*04:48*
     *TRANSACTION START*
[020t CARD INSERTED
 04:48:38 ATR RECEIVED T=0
[020t 04:48:40 OPCODE = FI      
  PAN 0004263********2113
  ---START OF TRANSACTION---
[020t 04:48:55 PIN ENTERED
[020t 04:49:01 OPCODE = BBC     
 04:49:02 GENAC 1 : ARQC
 04:49:04 GENAC 2 : TC""",
        
        """*7231*1*(Iw(1*3, M-02, R-10011
A/C 
DEVICE ERROR
ESC: 000
VAL: 000
REF: 000
REJECTS:000*(1
[020t 04:49:11 NOTES STACKED
[020t 04:49:13 CARD TAKEN""",
        
        """*PRIMARY CARD READER ACTIVATED*
[020t*347*06/18/2025*16:16*
     *TRANSACTION START*
[020t CARD INSERTED
[020t 16:16:43 CARD TAKEN
TRANSACTION END""",
    ]
    
    # Save sample logs
    os.makedirs("./sample_abm_logs", exist_ok=True)
    for i, log in enumerate(sample_logs):
        with open(f"./sample_abm_logs/sample_{i}.txt", 'w') as f:
            f.write(log)
    
    print("Sample ABM logs created in ./sample_abm_logs/")

def main():
    """Main training pipeline"""
    print("🏧 ABM NER Fine-tuning Pipeline")
    print("=" * 50)
    
    # Step 1: Create sample dataset (for demo)
    print("1. Creating sample ABM dataset...")
    create_sample_abm_dataset()
    
    # Step 2: Create training dataset
    print("2. Creating training dataset...")
    dataset_creator = ABMNERDatasetCreator()
    dataset_creator.create_training_data_from_files(
        input_dir="./sample_abm_logs",
        output_file="./abm_ner_dataset.json"
    )
    
    # Step 3: Fine-tune model
    print("3. Fine-tuning BERT for ABM NER...")
    fine_tuner = ABMNERFineTuner(
        model_name="bert-base-uncased",
        output_dir="./abm-ner-model"
    )
    
    # Load dataset
    train_dataset, val_dataset = fine_tuner.load_dataset("./abm_ner_dataset.json")
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Fine-tune
    trainer = fine_tuner.fine_tune(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=3,
        batch_size=8,  # Reduce if GPU memory issues
        learning_rate=2e-5
    )
    
    print("✅ Fine-tuning completed!")
    print(f"Model saved to: ./abm-ner-model")
    
    # Step 4: Test the model
    print("4. Testing fine-tuned model...")
    test_ner_model()

def test_ner_model():
    """Test the fine-tuned NER model"""
    from transformers import pipeline
    
    try:
        # Load fine-tuned model
        ner_pipeline = pipeline(
            "ner",
            model="./abm-ner-model",
            tokenizer="./abm-ner-model",
            aggregation_strategy="simple"
        )
        
        # Test text
        test_text = """[020t*632*06/18/2025*04:48*
     *TRANSACTION START*
[020t CARD INSERTED
  PAN 0004263********2113
DEVICE ERROR
ESC: 000"""
        
        # Predict entities
        entities = ner_pipeline(test_text)
        
        print("🧪 NER Test Results:")
        print(f"Text: {test_text}")
        print("\nDetected Entities:")
        for entity in entities:
            print(f"  - {entity['entity_group']}: '{entity['word']}' (confidence: {entity['score']:.3f})")
        
    except Exception as e:
        print(f"❌ Error testing model: {e}")
        print("Make sure the model was trained successfully")

if __name__ == "__main__":
    main()
