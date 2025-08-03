import time
import random
from typing import Dict, Any, List
import re

class FastAnalyzer:
    """Super fast analyzer that returns realistic mock data for debugging"""
    
    def __init__(self, model_path: str = None):
        print("Fast analyzer initialized - using mock data for instant responses")
        
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Instant analysis with realistic mock data"""
        start_time = time.time()
        
        print(f"Fast analyzing text: {text[:50]}...")
        
        # Simulate very quick tokenization
        time.sleep(0.05)  # 50ms delay for realism
        
        # Simple tokenization
        tokens = self._tokenize_simple(text)
        
        # Generate realistic predictions based on text content
        predicted_class, probabilities = self._analyze_text_content(text)
        
        # Generate mock attention weights
        attention_weights = self._generate_attention_weights(len(tokens))
        
        # Generate token importance
        token_importance = self._generate_token_importance(tokens, text)
        
        result = {
            "text": text,
            "tokens": tokens,
            "predicted_class": predicted_class,
            "probabilities": probabilities,
            "attention_weights": attention_weights,
            "token_importance": token_importance,
            "hidden_states": {"cls_embeddings": []},
            "analysis_time": f"{time.time() - start_time:.3f}s"
        }
        
        print(f"Fast analysis completed in {time.time() - start_time:.3f}s")
        return result
    
    def _tokenize_simple(self, text: str) -> List[str]:
        """EJ log-aware tokenization that focuses on meaningful content"""
        # Add special tokens
        tokens = ["[CLS]"]
        
        # Clean up EJ log prefixes and extract meaningful content
        lines = text.split('\n')
        meaningful_words = []
        
        for line in lines:
            # Remove common EJ log prefixes like [020t, timestamps etc.
            line = re.sub(r'^\[?\d+t?\]?\s*', '', line.strip())
            line = re.sub(r'^\d{2}:\d{2}:\d{2}\s*', '', line)  # Remove timestamps
            
            if line.strip():
                # Extract meaningful words and phrases
                words = re.findall(r'\*[^*]+\*|[A-Z]{2,}|UNABLE TO PROCESS|THANK YOU|\w+|\*+|[^\w\s]', line)
                meaningful_words.extend(words)
        
        # Process meaningful words
        for word in meaningful_words:
            word = word.strip()
            if not word:
                continue
                
            # Keep important phrases intact
            if word in ["UNABLE TO PROCESS", "THANK YOU", "TRANSACTION START", "TRANSACTION END"]:
                tokens.append(word)
            elif len(word) > 8:
                # Split very long words/codes
                mid = len(word) // 2
                tokens.extend([word[:mid] + "##", "##" + word[mid:]])
            else:
                tokens.append(word)
        
        tokens.append("[SEP]")
        return tokens[:64]  # Limit for speed
    
    def _analyze_text_content(self, text: str) -> tuple:
        """Analyze EJ log content to generate realistic anomaly predictions"""
        text_lower = text.lower()
        
        # Define classes: 0=Normal, 1=Transaction Error, 2=Fraud/Security, 3=System Error
        
        # Critical failure indicators (Transaction Error - Class 1)
        critical_failures = ["unable to process", "declined", "transaction failed", 
                           "timeout", "pin retry exceeded", "card blocked"]
        
        # Fraud/Security indicators (Class 2)
        fraud_indicators = ["suspicious", "fraud", "unauthorized", "blocked card",
                          "security violation", "invalid pin", "multiple attempts"]
        
        # System error indicators (Class 3)  
        system_errors = ["system error", "communication error", "host unavailable",
                        "network timeout", "device error", "terminal error"]
        
        # Transaction outcome analysis
        has_aac = "aac" in text_lower  # Authentication failure
        has_arqc = "arqc" in text_lower  # Authorization request
        has_unable = any(indicator in text_lower for indicator in critical_failures)
        has_fraud = any(indicator in text_lower for indicator in fraud_indicators)
        has_system = any(indicator in text_lower for indicator in system_errors)
        
        # Advanced pattern analysis
        transaction_complete = "transaction end" in text_lower and "card taken" in text_lower
        has_thank_you = "thank you" in text_lower
        has_error_message = has_unable or "error" in text_lower
        
        # Classification logic
        if has_fraud:
            # Fraud/Security issue detected
            return 2, [0.05, 0.10, 0.80, 0.05]
        
        elif has_system:
            # System error detected
            return 3, [0.05, 0.15, 0.10, 0.70]
        
        elif has_unable or (has_aac and not transaction_complete):
            # Transaction error - unable to process or authentication failure
            if has_aac:
                # AAC with incomplete transaction = high confidence error
                return 1, [0.05, 0.85, 0.05, 0.05]
            else:
                # General transaction failure
                return 1, [0.10, 0.75, 0.10, 0.05]
        
        elif has_error_message and not has_thank_you:
            # Error without proper completion
            return 1, [0.15, 0.65, 0.15, 0.05]
        
        elif transaction_complete and has_thank_you and not has_error_message:
            # Normal successful transaction
            return 0, [0.85, 0.10, 0.03, 0.02]
        
        elif has_arqc and transaction_complete:
            # ARQC with completion - likely normal but flagged for review
            return 0, [0.70, 0.20, 0.05, 0.05]
        
        else:
            # Unclear or partial transaction - medium confidence normal
            return 0, [0.60, 0.25, 0.10, 0.05]
    
    def _generate_attention_weights(self, num_tokens: int) -> List[Dict]:
        """Generate realistic attention weights"""
        # Create attention matrix
        attention_matrix = []
        for i in range(num_tokens):
            row = []
            for j in range(num_tokens):
                if i == j:
                    weight = 0.3 + random.random() * 0.4  # Self attention
                elif abs(i - j) == 1:
                    weight = 0.2 + random.random() * 0.3  # Adjacent tokens
                else:
                    weight = random.random() * 0.2  # Distant tokens
                row.append(weight)
            # Normalize
            total = sum(row)
            row = [w / total for w in row]
            attention_matrix.append(row)
        
        return [{
            "layer": 5,  # Simulate layer 5
            "heads": [{
                "head": 0,
                "attention": attention_matrix
            }]
        }]
    
    def _generate_token_importance(self, tokens: List[str], text: str) -> List[float]:
        """Generate intelligent token importance scores focused on EJ log anomaly detection"""
        importance = []
        text_lower = text.lower()
        
        # Define anomaly indicators with importance weights
        high_importance_indicators = {
            # Transaction outcomes (highest importance)
            "unable": 0.95, "declined": 0.95, "failed": 0.95, "timeout": 0.95, 
            "error": 0.95, "rejected": 0.95, "blocked": 0.95,
            
            # Critical transaction codes
            "aac": 0.90,  # Application Authentication Cryptogram (rejection)
            "arqc": 0.85,  # Authorization Request Cryptogram
            "genac": 0.80,  # Generate AC command
            
            # Important opcodes and operations
            "opcode": 0.75, "bbc": 0.70, "fi": 0.65,
            
            # Transaction flow indicators
            "process": 0.70, "transaction": 0.60, "authentication": 0.75,
            
            # Financial terms
            "pin": 0.65, "card": 0.60, "pan": 0.55,
        }
        
        medium_importance_indicators = {
            # Status and timing
            "received": 0.45, "entered": 0.45, "taken": 0.40, "inserted": 0.40,
            "start": 0.35, "end": 0.35, "activated": 0.35,
            
            # Merchant/location info (medium relevance)
            "branch": 0.30, "machine": 0.30, "tran": 0.30,
        }
        
        # Technical noise (very low importance)
        technical_noise = {
            "[cls]", "[sep]", "[pad]", "[mask]", "020t", "##"
        }
        
        for token in tokens:
            token_lower = token.lower().strip()
            
            # Skip empty tokens
            if not token_lower:
                importance.append(0.1)
                continue
            
            # Technical noise gets very low importance
            if any(noise in token_lower for noise in technical_noise):
                importance.append(0.05)
                continue
            
            # Check for high importance indicators
            high_score = 0.0
            for indicator, score in high_importance_indicators.items():
                if indicator in token_lower:
                    high_score = max(high_score, score)
            
            if high_score > 0:
                importance.append(min(high_score + random.random() * 0.05, 1.0))
                continue
            
            # Check for medium importance indicators
            medium_score = 0.0
            for indicator, score in medium_importance_indicators.items():
                if indicator in token_lower:
                    medium_score = max(medium_score, score)
            
            if medium_score > 0:
                importance.append(medium_score + random.random() * 0.1)
                continue
            
            # Numbers and times - contextual importance
            if any(char.isdigit() for char in token):
                # Time stamps and amounts are medium importance
                if ":" in token or len([c for c in token if c.isdigit()]) >= 4:
                    importance.append(0.35 + random.random() * 0.15)
                else:
                    importance.append(0.25 + random.random() * 0.1)
                continue
            
            # Special characters and punctuation
            if not token.replace("*", "").replace("-", "").replace(".", "").isalnum():
                importance.append(0.15 + random.random() * 0.1)
                continue
            
            # Regular words get baseline importance
            importance.append(0.20 + random.random() * 0.15)
        
        return importance
