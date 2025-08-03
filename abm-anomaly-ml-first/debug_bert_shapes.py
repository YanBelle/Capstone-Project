#!/usr/bin/env python3

import torch
from transformers import BertTokenizer, BertModel

def debug_bert_shapes():
    """Debug the exact shapes of BERT attention outputs"""
    
    # Initialize BERT components
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name, output_attentions=True)
    model.eval()
    
    # Test text
    text = "CARD INSERTED PIN ENTERED WITHDRAWAL COMPLETE"
    
    # Tokenize
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    
    print(f"Input IDs shape: {inputs['input_ids'].shape}")
    print(f"Attention mask shape: {inputs['attention_mask'].shape}")
    
    # Get BERT outputs
    with torch.no_grad():
        outputs = model(**inputs)
        attention_weights = outputs.attentions
        hidden_states = outputs.last_hidden_state
    
    print(f"\nNumber of attention layers: {len(attention_weights)}")
    print(f"Hidden states shape: {hidden_states.shape}")
    
    # Examine each layer's attention
    for i, layer_attention in enumerate(attention_weights):
        print(f"Layer {i} attention shape: {layer_attention.shape}")
    
    # Test stacking
    attention_stack = torch.stack(attention_weights)
    print(f"\nStacked attention shape: {attention_stack.shape}")
    
    # Test different averaging approaches
    print("\n--- Testing averaging approaches ---")
    
    # Approach 1: Average across layers and heads
    try:
        avg1 = attention_stack.mean(dim=(0, 2)).squeeze()
        print(f"Approach 1 (mean dim 0,2 + squeeze): {avg1.shape}")
    except Exception as e:
        print(f"Approach 1 failed: {e}")
    
    # Approach 2: Average across layers and heads differently
    try:
        avg2 = attention_stack.mean(dim=0).mean(dim=1).squeeze()
        print(f"Approach 2 (mean dim 0, then 1 + squeeze): {avg2.shape}")
    except Exception as e:
        print(f"Approach 2 failed: {e}")
    
    # Approach 3: Manual processing
    try:
        # Start with first layer to understand shape
        first_layer = attention_weights[0]
        print(f"First layer shape: {first_layer.shape}")
        
        # Average across heads for first layer
        first_avg = first_layer.mean(dim=1).squeeze()
        print(f"First layer averaged across heads: {first_avg.shape}")
        
        # Now average all layers
        all_layers_avg = []
        for layer_attn in attention_weights:
            layer_avg = layer_attn.mean(dim=1).squeeze()  # Average heads
            all_layers_avg.append(layer_avg)
        
        final_avg = torch.stack(all_layers_avg).mean(dim=0)
        print(f"Manual approach final shape: {final_avg.shape}")
        
    except Exception as e:
        print(f"Manual approach failed: {e}")

if __name__ == "__main__":
    debug_bert_shapes()
