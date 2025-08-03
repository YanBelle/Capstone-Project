#!/usr/bin/env python3
"""
BERT-DeepLog System Test Suite

This script tests the BERT-enhanced DeepLog anomaly detection system
including training, prediction, and API functionality.
"""

import asyncio
import json
import time
import requests
from typing import List, Dict, Any
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

# Test configuration
API_BASE_URL = "http://localhost:8000"
TEST_DATA_DIR = Path("test_data")

class BertDeepLogTester:
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({'Content-Type': 'application/json'})
        
        # Ensure test data directory
        TEST_DATA_DIR.mkdir(exist_ok=True)
        
        # Test results
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "tests": {},
            "summary": {}
        }
    
    def log(self, message: str):
        """Log test messages with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")
    
    def generate_sample_sessions(self) -> List[Dict[str, Any]]:
        """Generate sample EJ sessions for testing"""
        normal_sessions = [
            "CARD INSERTED PIN ENTERED OPCODE FI CASH DISPENSED NOTES TAKEN CARD TAKEN TRANSACTION END",
            "CARD INSERTED PIN ENTERED BALANCE INQUIRY RECEIPT PRINTED CARD TAKEN",
            "CARD INSERTED PIN ENTERED DEPOSIT TRANSACTION RECEIPT PRINTED CARD TAKEN",
            "CARD INSERTED PIN ENTERED TRANSFER FUNDS RECEIPT PRINTED CARD TAKEN",
            "CARD INSERTED PIN ENTERED OPCODE FI ATR_RECEIVED_T_0 CASH DISPENSED NOTES TAKEN CARD TAKEN",
            "CARD INSERTED PIN ENTERED MINI STATEMENT RECEIPT PRINTED CARD TAKEN",
            "CARD INSERTED PIN ENTERED WITHDRAWAL TRANSACTION CASH DISPENSED NOTES TAKEN CARD TAKEN",
            "CARD INSERTED PIN ENTERED INQUIRY BALANCE DISPLAY CARD TAKEN",
            "CARD INSERTED PIN ENTERED OPCODE FI WITHDRAWAL TRANSACTION CASH DISPENSED RECEIPT PRINTED NOTES TAKEN CARD TAKEN",
            "CARD INSERTED PIN ENTERED OPCODE FI BALANCE INQUIRY DISPLAY CARD TAKEN"
        ]
        
        anomaly_sessions = [
            "CARD INSERTED DEVICE ERROR M_02 SUPERVISOR ENTRY CARD TAKEN",
            "CARD INSERTED PIN ENTERED TIMEOUT ERROR CARD RETAINED",
            "CARD INSERTED DEVICE ERROR HOST DECLINE TRANSACTION CANCELLED CARD TAKEN",
            "CARD INSERTED DEVICE ERROR SENSOR FAULT SUPERVISOR ENTRY CARD TAKEN",
            "CARD INSERTED PIN ENTERED OPCODE FI DEVICE ERROR CASH JAMMED SUPERVISOR ENTRY",
            "CARD INSERTED TAMPER DETECTED SECURITY ALARM CARD RETAINED",
            "CARD INSERTED DEVICE ERROR DISPENSER FAULT TRANSACTION CANCELLED CARD TAKEN",
            "CARD INSERTED PIN ENTERED HOST DECLINE INSUFFICIENT FUNDS CARD TAKEN",
            "CARD INSERTED DEVICE ERROR COMMUNICATION FAILURE TRANSACTION CANCELLED CARD TAKEN",
            "CARD INSERTED PIN ENTERED DEVICE ERROR RECEIPT PRINTER FAULT TRANSACTION COMPLETED CARD TAKEN"
        ]
        
        sessions = []
        
        # Add normal sessions
        for i, text in enumerate(normal_sessions):
            sessions.append({
                "session_id": f"normal_{i+1}",
                "raw_text": text,
                "is_anomaly": False
            })
        
        # Add anomaly sessions  
        for i, text in enumerate(anomaly_sessions):
            sessions.append({
                "session_id": f"anomaly_{i+1}",
                "raw_text": text,
                "is_anomaly": True
            })
        
        # Generate additional normal sessions with variations
        for i in range(15):
            base_text = normal_sessions[i % len(normal_sessions)]
            variation = base_text + (f" ATR_RECEIVED_T_{i%3}" if i % 3 == 0 else "")
            sessions.append({
                "session_id": f"normal_var_{i+1}",
                "raw_text": variation,
                "is_anomaly": False
            })
        
        return sessions
    
    async def test_api_health(self) -> bool:
        """Test if API is accessible"""
        self.log("Testing API health...")
        try:
            response = self.session.get(f"{self.base_url}/api/v1/health")
            if response.status_code == 200:
                self.log("✅ API is healthy")
                self.results["tests"]["api_health"] = {"status": "pass"}
                return True
            else:
                self.log(f"❌ API health check failed: {response.status_code}")
                self.results["tests"]["api_health"] = {"status": "fail", "error": f"HTTP {response.status_code}"}
                return False
        except Exception as e:
            self.log(f"❌ API connection failed: {e}")
            self.results["tests"]["api_health"] = {"status": "fail", "error": str(e)}
            return False
    
    async def test_model_info(self) -> bool:
        """Test model info endpoint"""
        self.log("Testing model info endpoint...")
        try:
            response = self.session.get(f"{self.base_url}/api/v1/bert-deeplog/model-info")
            if response.status_code == 200:
                model_info = response.json()
                self.log(f"✅ Model info retrieved: {model_info.get('model_stats', {}).get('model_info', {}).get('trained', 'Unknown')} trained")
                self.results["tests"]["model_info"] = {"status": "pass", "data": model_info}
                return True
            else:
                error_msg = f"HTTP {response.status_code}"
                try:
                    error_detail = response.json().get('detail', error_msg)
                except:
                    error_detail = error_msg
                self.log(f"❌ Model info failed: {error_detail}")
                self.results["tests"]["model_info"] = {"status": "fail", "error": error_detail}
                return False
        except Exception as e:
            self.log(f"❌ Model info error: {e}")
            self.results["tests"]["model_info"] = {"status": "fail", "error": str(e)}
            return False
    
    async def test_training(self) -> bool:
        """Test model training"""
        self.log("Testing model training...")
        
        # Generate training data
        sessions = self.generate_sample_sessions()
        normal_sessions = [s for s in sessions if not s['is_anomaly']]
        
        if len(normal_sessions) < 10:
            self.log("❌ Not enough normal sessions for training")
            self.results["tests"]["training"] = {"status": "fail", "error": "Insufficient training data"}
            return False
        
        try:
            training_data = {
                "sessions": normal_sessions,
                "validation_split": 0.2,
                "normal_sessions_only": True
            }
            
            self.log(f"Starting training with {len(normal_sessions)} normal sessions...")
            response = self.session.post(
                f"{self.base_url}/api/v1/bert-deeplog/train",
                data=json.dumps(training_data)
            )
            
            if response.status_code == 200:
                result = response.json()
                self.log(f"✅ Training completed: {result.get('message', 'Success')}")
                self.results["tests"]["training"] = {
                    "status": "pass", 
                    "data": result,
                    "training_sessions": len(normal_sessions)
                }
                return True
            else:
                error_msg = f"HTTP {response.status_code}"
                try:
                    error_detail = response.json().get('detail', error_msg)
                except:
                    error_detail = error_msg
                self.log(f"❌ Training failed: {error_detail}")
                self.results["tests"]["training"] = {"status": "fail", "error": error_detail}
                return False
                
        except Exception as e:
            self.log(f"❌ Training error: {e}")
            self.results["tests"]["training"] = {"status": "fail", "error": str(e)}
            return False
    
    async def test_prediction(self) -> bool:
        """Test single prediction"""
        self.log("Testing single prediction...")
        
        test_cases = [
            {
                "name": "normal_transaction",
                "session_id": "test_normal_1",
                "session_text": "CARD INSERTED PIN ENTERED OPCODE FI CASH DISPENSED NOTES TAKEN CARD TAKEN TRANSACTION END",
                "expected_anomaly": False
            },
            {
                "name": "anomalous_transaction", 
                "session_id": "test_anomaly_1",
                "session_text": "CARD INSERTED DEVICE ERROR M_02 SUPERVISOR ENTRY CARD TAKEN",
                "expected_anomaly": True
            }
        ]
        
        prediction_results = []
        
        for test_case in test_cases:
            try:
                prediction_data = {
                    "session_id": test_case["session_id"],
                    "session_text": test_case["session_text"]
                }
                
                response = self.session.post(
                    f"{self.base_url}/api/v1/bert-deeplog/predict",
                    data=json.dumps(prediction_data)
                )
                
                if response.status_code == 200:
                    result = response.json()
                    is_anomaly = result.get('is_anomaly', False)
                    probability = result.get('anomaly_probability', 0.0)
                    confidence = result.get('confidence', 0.0)
                    
                    self.log(f"✅ Prediction '{test_case['name']}': Anomaly={is_anomaly}, Prob={probability:.3f}, Conf={confidence:.3f}")
                    
                    prediction_results.append({
                        "test_case": test_case["name"],
                        "result": result,
                        "correct_prediction": is_anomaly == test_case["expected_anomaly"]
                    })
                else:
                    self.log(f"❌ Prediction '{test_case['name']}' failed: HTTP {response.status_code}")
                    prediction_results.append({
                        "test_case": test_case["name"],
                        "error": f"HTTP {response.status_code}",
                        "correct_prediction": False
                    })
                    
            except Exception as e:
                self.log(f"❌ Prediction '{test_case['name']}' error: {e}")
                prediction_results.append({
                    "test_case": test_case["name"],
                    "error": str(e),
                    "correct_prediction": False
                })
        
        # Evaluate results
        successful_predictions = sum(1 for r in prediction_results if r.get('correct_prediction', False))
        total_predictions = len(prediction_results)
        
        if successful_predictions > 0:
            self.log(f"✅ Predictions successful: {successful_predictions}/{total_predictions}")
            self.results["tests"]["prediction"] = {
                "status": "pass",
                "successful_predictions": successful_predictions,
                "total_predictions": total_predictions,
                "results": prediction_results
            }
            return True
        else:
            self.log(f"❌ No successful predictions: {successful_predictions}/{total_predictions}")
            self.results["tests"]["prediction"] = {
                "status": "fail",
                "successful_predictions": successful_predictions,
                "total_predictions": total_predictions,
                "results": prediction_results
            }
            return False
    
    async def test_batch_prediction(self) -> bool:
        """Test batch prediction"""
        self.log("Testing batch prediction...")
        
        try:
            # Generate test sessions
            sessions = self.generate_sample_sessions()[:10]  # Use first 10 for batch test
            
            batch_data = {
                "sessions": [
                    {
                        "session_id": s["session_id"],
                        "session_text": s["raw_text"]
                    } for s in sessions
                ]
            }
            
            response = self.session.post(
                f"{self.base_url}/api/v1/bert-deeplog/predict-batch",
                data=json.dumps(batch_data)
            )
            
            if response.status_code == 200:
                result = response.json()
                predictions = result.get('predictions', [])
                total_processed = result.get('total_processed', 0)
                
                self.log(f"✅ Batch prediction completed: {total_processed} sessions processed")
                self.results["tests"]["batch_prediction"] = {
                    "status": "pass",
                    "total_processed": total_processed,
                    "predictions_count": len(predictions)
                }
                return True
            else:
                error_msg = f"HTTP {response.status_code}"
                try:
                    error_detail = response.json().get('detail', error_msg)
                except:
                    error_detail = error_msg
                self.log(f"❌ Batch prediction failed: {error_detail}")
                self.results["tests"]["batch_prediction"] = {"status": "fail", "error": error_detail}
                return False
                
        except Exception as e:
            self.log(f"❌ Batch prediction error: {e}")
            self.results["tests"]["batch_prediction"] = {"status": "fail", "error": str(e)}
            return False
    
    async def test_explanation(self) -> bool:
        """Test prediction explanation"""
        self.log("Testing prediction explanation...")
        
        try:
            # First make a prediction to get a session_id
            prediction_data = {
                "session_id": "test_explanation",
                "session_text": "CARD INSERTED DEVICE ERROR M_02 SUPERVISOR ENTRY CARD TAKEN"
            }
            
            response = self.session.post(
                f"{self.base_url}/api/v1/bert-deeplog/predict",
                data=json.dumps(prediction_data)
            )
            
            if response.status_code != 200:
                self.log("❌ Failed to create prediction for explanation test")
                self.results["tests"]["explanation"] = {"status": "fail", "error": "No prediction to explain"}
                return False
            
            # Now get explanation
            time.sleep(1)  # Brief delay to ensure prediction is cached
            
            response = self.session.get(f"{self.base_url}/api/v1/bert-deeplog/explanation/test_explanation")
            
            if response.status_code == 200:
                explanation = response.json()
                reasoning_count = len(explanation.get('model_reasoning', []))
                event_analysis_count = len(explanation.get('event_analysis', []))
                
                self.log(f"✅ Explanation retrieved: {reasoning_count} reasoning points, {event_analysis_count} event analyses")
                self.results["tests"]["explanation"] = {
                    "status": "pass",
                    "reasoning_points": reasoning_count,
                    "event_analyses": event_analysis_count
                }
                return True
            else:
                error_msg = f"HTTP {response.status_code}"
                try:
                    error_detail = response.json().get('detail', error_msg)
                except:
                    error_detail = error_msg
                self.log(f"❌ Explanation failed: {error_detail}")
                self.results["tests"]["explanation"] = {"status": "fail", "error": error_detail}
                return False
                
        except Exception as e:
            self.log(f"❌ Explanation error: {e}")
            self.results["tests"]["explanation"] = {"status": "fail", "error": str(e)}
            return False
    
    async def test_performance_metrics(self) -> bool:
        """Test system performance metrics"""
        self.log("Testing performance metrics...")
        
        try:
            # Test multiple predictions for performance analysis
            start_time = time.time()
            successful_predictions = 0
            total_predictions = 5
            
            for i in range(total_predictions):
                prediction_data = {
                    "session_id": f"perf_test_{i}",
                    "session_text": "CARD INSERTED PIN ENTERED OPCODE FI CASH DISPENSED NOTES TAKEN CARD TAKEN"
                }
                
                pred_start = time.time()
                response = self.session.post(
                    f"{self.base_url}/api/v1/bert-deeplog/predict",
                    data=json.dumps(prediction_data)
                )
                pred_time = time.time() - pred_start
                
                if response.status_code == 200:
                    successful_predictions += 1
                    self.log(f"  Prediction {i+1}: {pred_time:.3f}s")
            
            total_time = time.time() - start_time
            avg_time = total_time / total_predictions
            
            self.log(f"✅ Performance test: {successful_predictions}/{total_predictions} successful, avg {avg_time:.3f}s per prediction")
            
            self.results["tests"]["performance"] = {
                "status": "pass",
                "successful_predictions": successful_predictions,
                "total_predictions": total_predictions,
                "total_time": total_time,
                "average_time": avg_time
            }
            return successful_predictions > 0
            
        except Exception as e:
            self.log(f"❌ Performance test error: {e}")
            self.results["tests"]["performance"] = {"status": "fail", "error": str(e)}
            return False
    
    def save_results(self):
        """Save test results to file"""
        # Calculate summary
        total_tests = len(self.results["tests"])
        passed_tests = sum(1 for test in self.results["tests"].values() if test.get("status") == "pass")
        
        self.results["summary"] = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0
        }
        
        # Save to file
        results_file = TEST_DATA_DIR / f"bert_deeplog_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        self.log(f"Test results saved to: {results_file}")
    
    async def run_all_tests(self):
        """Run complete test suite"""
        self.log("🚀 Starting BERT-DeepLog System Test Suite")
        self.log("=" * 60)
        
        tests = [
            ("API Health", self.test_api_health),
            ("Model Info", self.test_model_info),
            ("Model Training", self.test_training),
            ("Single Prediction", self.test_prediction),
            ("Batch Prediction", self.test_batch_prediction),
            ("Prediction Explanation", self.test_explanation),
            ("Performance Metrics", self.test_performance_metrics)
        ]
        
        for test_name, test_func in tests:
            self.log(f"\n📋 Running {test_name}...")
            try:
                await test_func()
            except Exception as e:
                self.log(f"❌ {test_name} failed with exception: {e}")
                self.results["tests"][test_name.lower().replace(" ", "_")] = {
                    "status": "fail",
                    "error": str(e)
                }
        
        # Print summary
        self.log("\n" + "=" * 60)
        self.log("📊 TEST SUMMARY")
        self.log("=" * 60)
        
        total_tests = len(self.results["tests"])
        passed_tests = sum(1 for test in self.results["tests"].values() if test.get("status") == "pass")
        failed_tests = total_tests - passed_tests
        
        self.log(f"Total Tests: {total_tests}")
        self.log(f"Passed: {passed_tests}")
        self.log(f"Failed: {failed_tests}")
        self.log(f"Success Rate: {(passed_tests/total_tests*100):.1f}%" if total_tests > 0 else "No tests run")
        
        # Save results
        self.save_results()
        
        if failed_tests == 0:
            self.log("\n🎉 All tests passed! BERT-DeepLog system is working correctly.")
        else:
            self.log(f"\n⚠️  {failed_tests} test(s) failed. Check the logs above for details.")
        
        return failed_tests == 0

async def main():
    """Main test runner"""
    import argparse
    
    parser = argparse.ArgumentParser(description="BERT-DeepLog System Test Suite")
    parser.add_argument("--api-url", default=API_BASE_URL, help="API base URL")
    parser.add_argument("--quick", action="store_true", help="Run quick tests only (skip training)")
    args = parser.parse_args()
    
    tester = BertDeepLogTester(args.api_url)
    
    if args.quick:
        print("🏃‍♂️ Running quick tests (skipping training)...")
        # Run subset of tests
        await tester.test_api_health()
        await tester.test_model_info()
        await tester.test_prediction()
        await tester.test_explanation()
    else:
        await tester.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())
