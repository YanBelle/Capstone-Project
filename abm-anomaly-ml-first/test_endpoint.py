#!/usr/bin/env python3

# Test script to debug the BERT DeepLog API endpoint issue

import sys
sys.path.append('/home/yc/development/Capstone-Project/abm-anomaly-ml-first/services/api')

try:
    print("1. Importing bert_deeplog_api...")
    import bert_deeplog_api
    print(f"   SUCCESS: Module imported")
    
    print("2. Checking router...")
    router = bert_deeplog_api.router
    print(f"   Router type: {type(router)}")
    
    print("3. Listing all routes...")
    for route in router.routes:
        print(f"   Route: {route.methods} {route.path}")
    
    print("4. Checking for load_ej_sessions function...")
    if hasattr(bert_deeplog_api, 'load_ej_sessions'):
        print("   SUCCESS: load_ej_sessions function found")
    else:
        print("   ERROR: load_ej_sessions function NOT found")
        print("   Available functions:", [name for name in dir(bert_deeplog_api) if not name.startswith('_')])
        
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
