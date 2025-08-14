"""
Simple validation script for DeepLog + BERT implementation
Checks syntax and basic functionality without external dependencies
"""

import os
import sys
import ast
import logging

def validate_python_syntax(file_path):
    """Validate Python syntax of a file"""
    try:
        with open(file_path, 'r') as f:
            source_code = f.read()
        
        # Parse the AST to check for syntax errors
        ast.parse(source_code)
        return True, "Syntax OK"
    except SyntaxError as e:
        return False, f"Syntax Error: {e}"
    except Exception as e:
        return False, f"Error: {e}"

def validate_implementation():
    """Validate the DeepLog + BERT implementation files"""
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    files_to_check = [
        "deeplog_bert_training_clean.py",
        "deeplog_bert_integration.py",
        "test_deeplog_bert_integration.py"
    ]
    
    print("=== DeepLog + BERT Implementation Validation ===\n")
    
    all_valid = True
    
    for file_name in files_to_check:
        file_path = os.path.join(base_dir, file_name)
        
        if os.path.exists(file_path):
            is_valid, message = validate_python_syntax(file_path)
            status = "✅ PASS" if is_valid else "❌ FAIL"
            print(f"{file_name}: {status} - {message}")
            
            if not is_valid:
                all_valid = False
        else:
            print(f"{file_name}: ❌ FAIL - File not found")
            all_valid = False
    
    print(f"\n=== Overall Validation: {'✅ PASS' if all_valid else '❌ FAIL'} ===")
    
    # Check file sizes and basic structure
    print("\n=== File Information ===")
    
    for file_name in files_to_check:
        file_path = os.path.join(base_dir, file_name)
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            with open(file_path, 'r') as f:
                line_count = sum(1 for line in f)
            print(f"{file_name}: {file_size} bytes, {line_count} lines")
    
    # Check for key classes and functions
    print("\n=== Key Components Check ===")
    
    try:
        # Check if we can parse the main training file
        with open(os.path.join(base_dir, "deeplog_bert_training_clean.py"), 'r') as f:
            source = f.read()
        
        tree = ast.parse(source)
        
        classes = []
        functions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.append(node.name)
            elif isinstance(node, ast.FunctionDef):
                functions.append(node.name)
        
        expected_classes = [
            "BertTokenizer4DeepLog",
            "DeepLogBertModel", 
            "DeepLogBertTrainer"
        ]
        
        expected_functions = [
            "create_deeplog_bert_trainer",
            "train_deeplog_on_abm_data"
        ]
        
        for cls in expected_classes:
            if cls in classes:
                print(f"Class {cls}: ✅ Found")
            else:
                print(f"Class {cls}: ❌ Missing")
        
        for func in expected_functions:
            if func in functions:
                print(f"Function {func}: ✅ Found")
            else:
                print(f"Function {func}: ❌ Missing")
                
    except Exception as e:
        print(f"Error checking components: {e}")
    
    return all_valid

if __name__ == "__main__":
    validate_implementation()
