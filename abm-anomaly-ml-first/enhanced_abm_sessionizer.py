#!/usr/bin/env python3
"""
Fine-tuned ABM NER Integration
=============================

Integration script showing how to use the fine-tuned ABM NER model
with the intelligent sessionizer for improved sessionization.
"""

import os
import torch
import json
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

# Hugging Face imports
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

class FineTunedABMSessionizer:
    """Enhanced sessionizer using fine-tuned ABM NER model"""
    
    def __init__(self, model_path: str = "./abm-ner-model", confidence_threshold: float = 0.8):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        
        # Load fine-tuned NER model
        try:
            self.ner_pipeline = pipeline(
                "ner",
                model=model_path,
                tokenizer=model_path,
                aggregation_strategy="simple",
                device=0 if torch.cuda.is_available() else -1
            )
            self.model_available = True
            print(f"✅ Fine-tuned ABM NER model loaded from {model_path}")
        except Exception as e:
            print(f"⚠️ Could not load fine-tuned model: {e}")
            self.model_available = False
            # Fallback to generic NER
            try:
                self.ner_pipeline = pipeline(
                    "ner",
                    model="dbmdz/bert-large-cased-finetuned-conll03-english",
                    aggregation_strategy="simple",
                    device=0 if torch.cuda.is_available() else -1
                )
            except:
                self.ner_pipeline = None
        
        # ABM-specific boundary patterns (enhanced with NER)
        self.session_boundary_patterns = [
            r'\*(?:TRANSACTION|CARDLESS TRANSACTION)\s+START\*',
            r'\*PRIMARY CARD READER ACTIVATED\*',
            r'---START OF TRANSACTION---',
            r'TRANSACTION START',
            r'SESSION START'
        ]
    
    def extract_abm_entities(self, text: str) -> List[Dict]:
        """Extract ABM-specific entities using fine-tuned NER"""
        if not self.ner_pipeline:
            return []
        
        try:
            # Split text into manageable chunks (BERT token limit)
            chunks = self._split_text_for_ner(text, max_length=500)
            all_entities = []
            char_offset = 0
            
            for chunk in chunks:
                entities = self.ner_pipeline(chunk)
                
                # Adjust entity positions for chunk offset
                for entity in entities:
                    entity['start'] += char_offset
                    entity['end'] += char_offset
                    all_entities.append(entity)
                
                char_offset += len(chunk)
            
            return all_entities
            
        except Exception as e:
            print(f"Error in NER extraction: {e}")
            return []
    
    def _split_text_for_ner(self, text: str, max_length: int = 500) -> List[str]:
        """Split text into chunks suitable for NER processing"""
        if len(text) <= max_length:
            return [text]
        
        # Split by lines to preserve log structure
        lines = text.split('\n')
        chunks = []
        current_chunk = []
        current_length = 0
        
        for line in lines:
            line_length = len(line) + 1  # +1 for newline
            
            if current_length + line_length > max_length and current_chunk:
                chunks.append('\n'.join(current_chunk))
                current_chunk = [line]
                current_length = line_length
            else:
                current_chunk.append(line)
                current_length += line_length
        
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return chunks
    
    def find_session_boundaries_with_ner(self, text: str) -> List[Tuple[int, str, float]]:
        """Find session boundaries using both NER and regex patterns"""
        boundaries = []
        
        # Method 1: Use fine-tuned NER to find session boundaries
        if self.model_available:
            entities = self.extract_abm_entities(text)
            
            for entity in entities:
                if entity['entity_group'] in ['TRANSACTION_START', 'SESSION_BOUNDARY']:
                    if entity['score'] >= self.confidence_threshold:
                        boundaries.append((
                            entity['start'], 
                            f"NER_{entity['entity_group']}", 
                            entity['score']
                        ))
        
        # Method 2: Regex fallback/supplement
        for pattern in self.session_boundary_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE):
                boundaries.append((
                    match.start(), 
                    "REGEX_TRANSACTION_START", 
                    0.9  # High confidence for explicit patterns
                ))
        
        # Remove duplicates and sort by position
        boundaries = list(set(boundaries))
        boundaries.sort(key=lambda x: x[0])
        
        return boundaries
    
    def enhanced_sessionization(self, text: str, filename: str = "unknown") -> List[Dict]:
        """Perform enhanced sessionization using fine-tuned NER"""
        print(f"🔬 Processing {filename} with fine-tuned ABM NER...")
        
        # Extract all entities
        entities = self.extract_abm_entities(text)
        
        # Find session boundaries
        boundaries = self.find_session_boundaries_with_ner(text)
        
        if not boundaries:
            print("⚠️ No session boundaries found, treating as single session")
            return [self._create_session_dict(text, 0, len(text), entities, filename, "SINGLE")]
        
        # Split into sessions based on boundaries
        sessions = []
        start_pos = 0
        
        for i, (boundary_pos, boundary_type, confidence) in enumerate(boundaries):
            if i > 0:  # Create session from previous boundary to current
                session_text = text[start_pos:boundary_pos].strip()
                if session_text:
                    session_entities = self._filter_entities_for_range(entities, start_pos, boundary_pos)
                    sessions.append(self._create_session_dict(
                        session_text, start_pos, boundary_pos, session_entities, filename, boundary_type
                    ))
            start_pos = boundary_pos
        
        # Handle last session
        if start_pos < len(text):
            session_text = text[start_pos:].strip()
            if session_text:
                session_entities = self._filter_entities_for_range(entities, start_pos, len(text))
                sessions.append(self._create_session_dict(
                    session_text, start_pos, len(text), session_entities, filename, "FINAL"
                ))
        
        print(f"✅ Extracted {len(sessions)} sessions using fine-tuned NER")
        return sessions
    
    def _filter_entities_for_range(self, entities: List[Dict], start: int, end: int) -> List[Dict]:
        """Filter entities that fall within the specified range"""
        filtered = []
        for entity in entities:
            if entity['start'] >= start and entity['end'] <= end:
                # Adjust positions relative to session start
                adjusted_entity = entity.copy()
                adjusted_entity['start'] -= start
                adjusted_entity['end'] -= start
                filtered.append(adjusted_entity)
        return filtered
    
    def _create_session_dict(self, text: str, start: int, end: int, entities: List[Dict], 
                           filename: str, boundary_type: str) -> Dict:
        """Create session dictionary with enhanced NER information"""
        session_id = f"NER_{filename}_{start}_{end}_{datetime.now().strftime('%H%M%S')}"
        
        # Extract key information using entities
        timestamps = [e for e in entities if e['entity_group'] == 'TIMESTAMP']
        card_numbers = [e for e in entities if e['entity_group'] == 'CARD_NUMBER']
        error_codes = [e for e in entities if e['entity_group'] == 'ERROR_CODE']
        amounts = [e for e in entities if e['entity_group'] == 'AMOUNT']
        
        return {
            'session_id': session_id,
            'raw_text': text,
            'start_position': start,
            'end_position': end,
            'boundary_type': boundary_type,
            'sessionization_method': 'fine_tuned_ner',
            'entities': entities,
            'extracted_info': {
                'timestamps': timestamps,
                'card_numbers': card_numbers,
                'error_codes': error_codes,
                'amounts': amounts,
                'entity_count': len(entities)
            },
            'quality_score': self._calculate_session_quality(text, entities)
        }
    
    def _calculate_session_quality(self, text: str, entities: List[Dict]) -> float:
        """Calculate session quality based on extracted entities"""
        score = 0.5  # Base score
        
        # Bonus for finding important entities
        entity_types = [e['entity_group'] for e in entities]
        
        if 'TRANSACTION_START' in entity_types:
            score += 0.2
        if 'TIMESTAMP' in entity_types:
            score += 0.1
        if 'CARD_NUMBER' in entity_types:
            score += 0.1
        if any(et in entity_types for et in ['ERROR_CODE', 'STATUS_CODE']):
            score += 0.1
        
        # Penalty for very short sessions
        if len(text) < 50:
            score -= 0.2
        
        return min(1.0, max(0.0, score))

# Enhanced integration with existing intelligent sessionizer
class EnhancedIntelligentSessionizer:
    """Combines fine-tuned NER with existing intelligent sessionizer"""
    
    def __init__(self, abm_model_path: str = "./abm-ner-model", use_fine_tuned: bool = True):
        self.use_fine_tuned = use_fine_tuned
        
        if use_fine_tuned:
            self.fine_tuned_sessionizer = FineTunedABMSessionizer(abm_model_path)
        
        # Fallback to existing intelligent sessionizer
        try:
            from intelligent_sessionizer import IntelligentSessionizer
            self.fallback_sessionizer = IntelligentSessionizer(use_ner=True)
        except ImportError:
            print("⚠️ Intelligent sessionizer not available")
            self.fallback_sessionizer = None
    
    def sessionize(self, text: str, filename: str = "unknown") -> List[Dict]:
        """Enhanced sessionization with fine-tuned model"""
        
        if self.use_fine_tuned and self.fine_tuned_sessionizer.model_available:
            print("🎯 Using fine-tuned ABM NER model")
            sessions = self.fine_tuned_sessionizer.enhanced_sessionization(text, filename)
            
            # Validate results
            if len(sessions) > 0 and all(s['quality_score'] > 0.3 for s in sessions):
                return sessions
            else:
                print("⚠️ Fine-tuned results low quality, falling back...")
        
        # Fallback to existing intelligent sessionizer
        if self.fallback_sessionizer:
            print("🔄 Using fallback intelligent sessionizer")
            try:
                sessions = self.fallback_sessionizer.split_into_sessions(text, filename)
                # Convert to enhanced format
                return [self._convert_to_enhanced_format(s) for s in sessions]
            except Exception as e:
                print(f"❌ Fallback failed: {e}")
        
        # Final fallback - simple regex-based splitting
        print("🔄 Using simple regex fallback")
        return self._simple_regex_sessionization(text, filename)
    
    def _convert_to_enhanced_format(self, session) -> Dict:
        """Convert existing session format to enhanced format"""
        return {
            'session_id': session.session_id,
            'raw_text': session.raw_text,
            'start_position': 0,
            'end_position': len(session.raw_text),
            'boundary_type': 'FALLBACK',
            'sessionization_method': 'fallback_intelligent',
            'entities': [],
            'extracted_info': {
                'timestamps': [],
                'card_numbers': [],
                'error_codes': [],
                'amounts': [],
                'entity_count': 0
            },
            'quality_score': 0.5
        }
    
    def _simple_regex_sessionization(self, text: str, filename: str) -> List[Dict]:
        """Simple regex-based fallback sessionization"""
        patterns = [
            r'\*TRANSACTION START\*',
            r'\*PRIMARY CARD READER ACTIVATED\*'
        ]
        
        boundaries = []
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                boundaries.append(match.start())
        
        boundaries.sort()
        
        if not boundaries:
            return [{
                'session_id': f"SIMPLE_{filename}_0",
                'raw_text': text,
                'start_position': 0,
                'end_position': len(text),
                'boundary_type': 'SINGLE',
                'sessionization_method': 'simple_regex',
                'entities': [],
                'extracted_info': {
                    'timestamps': [],
                    'card_numbers': [],
                    'error_codes': [],
                    'amounts': [],
                    'entity_count': 0
                },
                'quality_score': 0.3
            }]
        
        sessions = []
        start_pos = 0
        
        for boundary in boundaries:
            if start_pos < boundary:
                session_text = text[start_pos:boundary].strip()
                if session_text:
                    sessions.append({
                        'session_id': f"SIMPLE_{filename}_{start_pos}",
                        'raw_text': session_text,
                        'start_position': start_pos,
                        'end_position': boundary,
                        'boundary_type': 'REGEX',
                        'sessionization_method': 'simple_regex',
                        'entities': [],
                        'extracted_info': {
                            'timestamps': [],
                            'card_numbers': [],
                            'error_codes': [],
                            'amounts': [],
                            'entity_count': 0
                        },
                        'quality_score': 0.4
                    })
            start_pos = boundary
        
        # Last session
        if start_pos < len(text):
            session_text = text[start_pos:].strip()
            if session_text:
                sessions.append({
                    'session_id': f"SIMPLE_{filename}_{start_pos}",
                    'raw_text': session_text,
                    'start_position': start_pos,
                    'end_position': len(text),
                    'boundary_type': 'FINAL',
                    'sessionization_method': 'simple_regex',
                    'entities': [],
                    'extracted_info': {
                        'timestamps': [],
                        'card_numbers': [],
                        'error_codes': [],
                        'amounts': [],
                        'entity_count': 0
                    },
                    'quality_score': 0.4
                })
        
        return sessions

def demo_fine_tuned_sessionization():
    """Demo the fine-tuned sessionization"""
    
    # Sample ABM log
    sample_log = """[020t*632*06/18/2025*04:48*
     *TRANSACTION START*
[020t CARD INSERTED
 04:48:38 ATR RECEIVED T=0
[020t 04:48:40 OPCODE = FI      
  PAN 0004263********2113
  ---START OF TRANSACTION---
[020t 04:48:55 PIN ENTERED
[020t 04:49:01 OPCODE = BBC     
 04:49:02 GENAC 1 : ARQC
 04:49:04 GENAC 2 : TC

*7231*1*(Iw(1*3, M-02, R-10011
A/C 
DEVICE ERROR
ESC: 000
VAL: 000
REF: 000
REJECTS:000*(1
[020t 04:49:11 NOTES STACKED
[020t 04:49:13 CARD TAKEN

*PRIMARY CARD READER ACTIVATED*
[020t*347*06/18/2025*16:16*
     *TRANSACTION START*
[020t CARD INSERTED
[020t 16:16:43 CARD TAKEN
TRANSACTION END"""
    
    print("🧪 Testing Fine-tuned ABM Sessionization")
    print("=" * 50)
    
    # Test enhanced sessionizer
    sessionizer = EnhancedIntelligentSessionizer(
        abm_model_path="./abm-ner-model",
        use_fine_tuned=True
    )
    
    sessions = sessionizer.sessionize(sample_log, "demo.txt")
    
    print(f"\n📊 Results:")
    print(f"Total sessions: {len(sessions)}")
    
    for i, session in enumerate(sessions):
        print(f"\n🔍 Session {i+1}:")
        print(f"  Method: {session['sessionization_method']}")
        print(f"  Quality Score: {session['quality_score']:.2f}")
        print(f"  Entities Found: {session['extracted_info']['entity_count']}")
        print(f"  Text Length: {len(session['raw_text'])} chars")
        print(f"  Boundary Type: {session['boundary_type']}")
        
        if session['entities']:
            print("  🎯 Detected Entities:")
            for entity in session['entities'][:5]:  # Show first 5
                print(f"    - {entity['entity_group']}: '{entity['word']}' ({entity['score']:.3f})")

if __name__ == "__main__":
    demo_fine_tuned_sessionization()
