#!/usr/bin/env python3

"""
SVM Debug System Deployment Verification Script
"""

import os
import json
import sys
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a file exists and return status"""
    if os.path.exists(filepath):
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description}: {filepath} (NOT FOUND)")
        return False

def check_directory_structure():
    """Check if all required directories exist"""
    print("\n🔍 Checking Directory Structure...")
    
    required_dirs = [
        "services/api",
        "services/dashboard/src", 
        "services/anomaly-detector",
        "debug_output",
        "static/debug"
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✅ Directory: {dir_path}")
        else:
            print(f"❌ Directory: {dir_path} (NOT FOUND)")
            all_exist = False
    
    return all_exist

def check_svm_debug_files():
    """Check if all SVM debug files are in place"""
    print("\n🔍 Checking SVM Debug Files...")
    
    files_to_check = [
        ("services/api/svm_debug_api.py", "SVM Debug API"),
        ("services/dashboard/src/SVMDebugDashboard.js", "React SVM Debug Component"),
        ("services/anomaly-detector/svm_visualizer.py", "SVM Visualizer"),
        ("debug_svm_cli.py", "CLI Debug Tool"),
        ("example_sessions.json", "Example Test Sessions"),
        ("svm_requirements.txt", "SVM Requirements"),
        ("deploy_svm_debug_system.sh", "Deployment Script")
    ]
    
    all_exist = True
    for filepath, description in files_to_check:
        if not check_file_exists(filepath, description):
            all_exist = False
    
    return all_exist

def check_dashboard_integration():
    """Check if Dashboard.js has been properly updated"""
    print("\n🔍 Checking Dashboard Integration...")
    
    dashboard_file = "services/dashboard/src/Dashboard.js"
    if not os.path.exists(dashboard_file):
        print(f"❌ Dashboard file not found: {dashboard_file}")
        return False
    
    with open(dashboard_file, 'r') as f:
        content = f.read()
    
    checks = [
        ("SVMDebugDashboard import", "import SVMDebugDashboard"),
        ("svm-debug in navigation", "'svm-debug'"),
        ("SVM Debug tab label", "'SVM Debug'"),
        ("SVM debug component", "<SVMDebugDashboard />")
    ]
    
    all_integrated = True
    for check_name, search_string in checks:
        if search_string in content:
            print(f"✅ {check_name}: Found")
        else:
            print(f"❌ {check_name}: NOT FOUND")
            all_integrated = False
    
    return all_integrated

def check_api_integration():
    """Check if main API has SVM debug routes"""
    print("\n🔍 Checking API Integration...")
    
    api_file = "services/api/main.py"
    if not os.path.exists(api_file):
        print(f"❌ API file not found: {api_file}")
        return False
    
    with open(api_file, 'r') as f:
        content = f.read()
    
    checks = [
        ("SVM debug import", "from svm_debug_api import router"),
        ("SVM debug router", "app.include_router(svm_debug_router"),
        ("SVM debug prefix", 'prefix="/api/v1"'),
        ("SVM debug tags", 'tags=["svm-debug"]')
    ]
    
    all_integrated = True
    for check_name, search_string in checks:
        if search_string in content:
            print(f"✅ {check_name}: Found")
        else:
            print(f"❌ {check_name}: NOT FOUND")
            all_integrated = False
    
    return all_integrated

def check_dependencies():
    """Check if required Python packages can be imported"""
    print("\n🔍 Checking Python Dependencies...")
    
    required_packages = [
        "plotly",
        "matplotlib", 
        "seaborn",
        "sklearn",
        "numpy",
        "pandas",
        "fastapi"
    ]
    
    all_available = True
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}: Available")
        except ImportError:
            print(f"❌ {package}: NOT AVAILABLE")
            all_available = False
    
    return all_available

def create_test_session():
    """Create test data for SVM debugging"""
    print("\n🔧 Creating Test Session Data...")
    
    test_data = {
        "sessions": [
            {
                "session_id": "test_svm_debug_1",
                "raw_text": "Normal ATM transaction: user inserted card, entered PIN, checked balance, withdrew $100, transaction completed successfully",
                "expected_anomaly": False,
                "features": {
                    "transaction_amount": 100,
                    "transaction_type": "withdrawal",
                    "time_of_day": "14:30",
                    "day_of_week": "Tuesday"
                }
            },
            {
                "session_id": "test_svm_debug_2", 
                "raw_text": "Unusual behavior detected: multiple failed PIN attempts, card skimmer device detected, suspicious activity around ATM terminal",
                "expected_anomaly": True,
                "features": {
                    "failed_attempts": 5,
                    "device_tampered": True,
                    "suspicious_activity": True,
                    "time_of_day": "02:15"
                }
            }
        ]
    }
    
    if not os.path.exists("example_sessions.json"):
        with open("example_sessions.json", 'w') as f:
            json.dump(test_data, f, indent=2)
        print("✅ Created example_sessions.json")
    else:
        print("✅ example_sessions.json already exists")

def generate_deployment_summary():
    """Generate deployment summary and next steps"""
    print("\n" + "="*60)
    print("🎉 SVM DEBUG SYSTEM DEPLOYMENT SUMMARY")
    print("="*60)
    
    # Check all components
    dir_ok = check_directory_structure()
    files_ok = check_svm_debug_files()
    dashboard_ok = check_dashboard_integration()
    api_ok = check_api_integration()
    deps_ok = check_dependencies()
    
    create_test_session()
    
    print(f"\n📊 DEPLOYMENT STATUS:")
    print(f"   Directory Structure: {'✅ OK' if dir_ok else '❌ ISSUES'}")
    print(f"   SVM Debug Files: {'✅ OK' if files_ok else '❌ MISSING'}")
    print(f"   Dashboard Integration: {'✅ OK' if dashboard_ok else '❌ ISSUES'}")
    print(f"   API Integration: {'✅ OK' if api_ok else '❌ ISSUES'}")
    print(f"   Python Dependencies: {'✅ OK' if deps_ok else '❌ MISSING'}")
    
    overall_status = all([dir_ok, files_ok, dashboard_ok, api_ok, deps_ok])
    
    if overall_status:
        print(f"\n🎊 OVERALL STATUS: ✅ DEPLOYMENT SUCCESSFUL!")
        print("\n🚀 NEXT STEPS:")
        print("   1. Start your Docker containers:")
        print("      docker-compose up -d")
        print("   2. Access the dashboard:")
        print("      http://localhost:3000")
        print("   3. Click on the 'SVM Debug' tab")
        print("   4. Test SVM debugging with CLI:")
        print("      python debug_svm_cli.py --session-file example_sessions.json")
        print("   5. Check API endpoints:")
        print("      http://localhost:8000/docs")
        
        print("\n📋 AVAILABLE INTERFACES:")
        print("   🌐 Web Dashboard: http://localhost:3000 (SVM Debug tab)")
        print("   🔧 REST API: http://localhost:8000/api/v1/svm-debug/")
        print("   💻 CLI Tool: python debug_svm_cli.py --help")
        print("   📖 API Docs: http://localhost:8000/docs")
        
    else:
        print(f"\n⚠️  OVERALL STATUS: ❌ DEPLOYMENT INCOMPLETE")
        print("\n🔧 REQUIRED ACTIONS:")
        if not files_ok:
            print("   - Run: ./implement_svm_debug.sh")
        if not dashboard_ok:
            print("   - Manually update Dashboard.js with SVM debug integration")
        if not api_ok:
            print("   - Manually update main.py with SVM debug routes")
        if not deps_ok:
            print("   - Install missing Python packages: pip install -r svm_requirements.txt")
    
    print("\n💡 TROUBLESHOOTING:")
    print("   - If containers won't start: docker-compose down && docker-compose up --build")
    print("   - If API not responding: Check logs with docker-compose logs api")
    print("   - If SVM debug not working: Ensure models are trained first")
    print("   - For detailed logs: Check docker-compose logs")

if __name__ == "__main__":
    generate_deployment_summary()
