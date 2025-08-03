#!/usr/bin/env python3

# Quick syntax check
try:
    import services.anomaly_detector.ej_contextual_labeler
    print("✅ No syntax errors found")
except SyntaxError as e:
    print(f"❌ Syntax error: {e}")
    print(f"Line {e.lineno}: {e.text}")
except Exception as e:
    print(f"Other error: {e}")
