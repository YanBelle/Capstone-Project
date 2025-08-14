import ast
import os

# Check syntax of our main files
files = [
    "deeplog_bert_training_clean.py",
    "deeplog_bert_integration.py", 
    "test_deeplog_bert_integration.py"
]

print("DeepLog + BERT Implementation Validation")
print("=" * 50)

for filename in files:
    try:
        with open(filename, 'r') as f:
            source = f.read()
        ast.parse(source)
        print(f"✅ {filename}: Syntax OK")
    except SyntaxError as e:
        print(f"❌ {filename}: Syntax Error - {e}")
    except FileNotFoundError:
        print(f"❌ {filename}: File not found")
    except Exception as e:
        print(f"❌ {filename}: Error - {e}")

print("\nValidation complete!")
