#!/usr/bin/env python3
"""
Testing Script untuk SIPD Nebius Chatbot

Script ini melakukan testing komprehensif untuk:
1. Koneksi ke Nebius AI
2. Fungsionalitas chatbot
3. Performance testing
4. API endpoint testing

Usage:
    python test_nebius_chatbot.py
    python test_nebius_chatbot.py --quick
    python test_nebius_chatbot.py --performance
"""

import asyncio
import aiohttp
import json
import time
import argparse
import sys
from typing import Dict, List, Any
from datetime import datetime
import os
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

try:
    from nebius_chatbot_config import NebiusChatbotConfig
    from nebius_client import NebiusClient
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Pastikan file nebius_chatbot_config.py dan nebius_client.py ada")
    sys.exit(1)

class NebiusChatbotTester:
    """Comprehensive tester untuk Nebius Chatbot"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.config = NebiusChatbotConfig()
        self.test_results = []
        self.session_id = f"test_session_{int(time.time())}"
        
    async def run_all_tests(self, quick: bool = False, performance: bool = False):
        """Jalankan semua test"""
        print("🚀 Starting SIPD Nebius Chatbot Tests...\n")
        
        # Basic tests
        await self.test_environment()
        await self.test_nebius_connection()
        
        if not quick:
            await self.test_chatbot_server()
            await self.test_chat_functionality()
            await self.test_api_endpoints()
            
        if performance:
            await self.test_performance()
            
        self.print_summary()
        
    async def test_environment(self):
        """Test environment setup"""
        print("🔧 Testing Environment Setup...")
        
        # Check Python version
        python_version = sys.version_info
        if python_version >= (3, 8):
            self.log_success("Python version", f"{python_version.major}.{python_version.minor}")
        else:
            self.log_error("Python version", "Requires Python 3.8+")
            
        # Check required environment variables
        required_vars = ['NEBIUS_API_KEY']
        for var in required_vars:
            if os.getenv(var):
                self.log_success(f"Environment variable {var}", "✓ Set")
            else:
                self.log_error(f"Environment variable {var}", "❌ Not set")
                
        # Check required files
        required_files = [
            'nebius_chatbot.py',
            'nebius_chatbot_config.py', 
            'nebius_client.py',
            'run_nebius_chatbot.py'
        ]
        
        for file in required_files:
            if Path(file).exists():
                self.log_success(f"Required file {file}", "✓ Exists")
            else:
                self.log_error(f"Required file {file}", "❌ Missing")
                
        print()
        
    async def test_nebius_connection(self):
        """Test koneksi ke Nebius AI"""
        print("🔗 Testing Nebius AI Connection...")
        
        try:
            client = NebiusClient()
            
            # Test simple completion
            start_time = time.time()
            response = await client.generate_response(
                "Test connection",
                max_tokens=10
            )
            response_time = time.time() - start_time
            
            if response and len(response.strip()) > 0:
                self.log_success("Nebius API connection", f"✓ Connected ({response_time:.2f}s)")
                self.log_success("Response generation", f"✓ Working: '{response[:50]}...'")
            else:
                self.log_error("Nebius API connection", "❌ Empty response")
                
        except Exception as e:
            self.log_error("Nebius API connection", f"❌ Failed: {str(e)}")
            
        print()
        
    async def test_chatbot_server(self):
        """Test apakah chatbot server berjalan"""
        print("🖥️ Testing Chatbot Server...")
        
        try:
            async with aiohttp.ClientSession() as session:
                # Test health endpoint
                async with session.get(f"{self.base_url}/health") as response:
                    if response.status == 200:
                        health_data = await response.json()
                        self.log_success("Server health", f"✓ Healthy: {health_data.get('status')}")
                    else:
                        self.log_error("Server health", f"❌ Status: {response.status}")
                        
        except aiohttp.ClientConnectorError:
            self.log_error("Server connection", "❌ Server not running")
            print("💡 Hint: Jalankan 'python run_nebius_chatbot.py' terlebih dahulu")
        except Exception as e:
            self.log_error("Server connection", f"❌ Error: {str(e)}")
            
        print()
        
    async def test_chat_functionality(self):
        """Test fungsionalitas chat"""
        print("💬 Testing Chat Functionality...")
        
        test_messages = [
            {
                "message": "Halo, saya butuh bantuan dengan SIPD",
                "expected_intent": "greeting",
                "description": "Basic greeting"
            },
            {
                "message": "Saya tidak bisa login ke sistem SIPD",
                "expected_intent": "login_issue", 
                "description": "Login problem"
            },
            {
                "message": "Bagaimana cara membuat DPA?",
                "expected_intent": "dpa_issue",
                "description": "DPA question"
            },
            {
                "message": "Sistem error terus, saya frustasi!",
                "expected_sentiment": "negative",
                "description": "Negative sentiment"
            }
        ]
        
        try:
            async with aiohttp.ClientSession() as session:
                for test_case in test_messages:
                    await self.test_single_message(session, test_case)
                    await asyncio.sleep(0.5)  # Rate limiting
                    
        except Exception as e:
            self.log_error("Chat functionality", f"❌ Error: {str(e)}")
            
        print()
        
    async def test_single_message(self, session: aiohttp.ClientSession, test_case: Dict):
        """Test single chat message"""
        payload = {
            "message": test_case["message"],
            "session_id": self.session_id
        }
        
        try:
            start_time = time.time()
            async with session.post(
                f"{self.base_url}/chat",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                response_time = time.time() - start_time
                
                if response.status == 200:
                    data = await response.json()
                    
                    # Check response structure
                    required_fields = ['response', 'session_id', 'intent', 'sentiment']
                    missing_fields = [f for f in required_fields if f not in data]
                    
                    if not missing_fields:
                        self.log_success(
                            f"Chat: {test_case['description']}",
                            f"✓ Response ({response_time:.2f}s): {data['response'][:50]}..."
                        )
                        
                        # Check intent if specified
                        if 'expected_intent' in test_case:
                            if test_case['expected_intent'] in data.get('intent', ''):
                                self.log_success(
                                    f"Intent detection: {test_case['description']}",
                                    f"✓ Detected: {data['intent']}"
                                )
                            else:
                                self.log_warning(
                                    f"Intent detection: {test_case['description']}",
                                    f"⚠️ Expected: {test_case['expected_intent']}, Got: {data['intent']}"
                                )
                                
                        # Check sentiment if specified
                        if 'expected_sentiment' in test_case:
                            if test_case['expected_sentiment'] in data.get('sentiment', ''):
                                self.log_success(
                                    f"Sentiment analysis: {test_case['description']}",
                                    f"✓ Detected: {data['sentiment']}"
                                )
                            else:
                                self.log_warning(
                                    f"Sentiment analysis: {test_case['description']}",
                                    f"⚠️ Expected: {test_case['expected_sentiment']}, Got: {data['sentiment']}"
                                )
                    else:
                        self.log_error(
                            f"Chat: {test_case['description']}",
                            f"❌ Missing fields: {missing_fields}"
                        )
                else:
                    self.log_error(
                        f"Chat: {test_case['description']}",
                        f"❌ HTTP {response.status}"
                    )
                    
        except Exception as e:
            self.log_error(
                f"Chat: {test_case['description']}",
                f"❌ Error: {str(e)}"
            )
            
    async def test_api_endpoints(self):
        """Test semua API endpoints"""
        print("🔌 Testing API Endpoints...")
        
        endpoints = [
            {
                "method": "GET",
                "path": "/health",
                "description": "Health check"
            },
            {
                "method": "GET",
                "path": "/stats",
                "description": "Statistics"
            },
            {
                "method": "GET",
                "path": f"/chat/history/{self.session_id}",
                "description": "Chat history"
            },
            {
                "method": "GET",
                "path": "/docs",
                "description": "API documentation"
            }
        ]
        
        try:
            async with aiohttp.ClientSession() as session:
                for endpoint in endpoints:
                    await self.test_endpoint(session, endpoint)
                    
        except Exception as e:
            self.log_error("API endpoints", f"❌ Error: {str(e)}")
            
        print()
        
    async def test_endpoint(self, session: aiohttp.ClientSession, endpoint: Dict):
        """Test single API endpoint"""
        try:
            url = f"{self.base_url}{endpoint['path']}"
            
            if endpoint['method'] == 'GET':
                async with session.get(url) as response:
                    if response.status == 200:
                        self.log_success(
                            f"Endpoint {endpoint['path']}",
                            f"✓ {endpoint['description']}"
                        )
                    else:
                        self.log_error(
                            f"Endpoint {endpoint['path']}",
                            f"❌ HTTP {response.status}"
                        )
                        
        except Exception as e:
            self.log_error(
                f"Endpoint {endpoint['path']}",
                f"❌ Error: {str(e)}"
            )
            
    async def test_performance(self):
        """Test performance dengan concurrent requests"""
        print("⚡ Testing Performance...")
        
        concurrent_requests = 5
        test_message = "Test performance message"
        
        try:
            async with aiohttp.ClientSession() as session:
                # Concurrent chat requests
                start_time = time.time()
                
                tasks = []
                for i in range(concurrent_requests):
                    task = self.send_chat_request(
                        session, 
                        test_message, 
                        f"perf_session_{i}"
                    )
                    tasks.append(task)
                    
                results = await asyncio.gather(*tasks, return_exceptions=True)
                total_time = time.time() - start_time
                
                successful_requests = sum(1 for r in results if not isinstance(r, Exception))
                avg_time = total_time / concurrent_requests
                
                self.log_success(
                    "Concurrent requests",
                    f"✓ {successful_requests}/{concurrent_requests} successful"
                )
                self.log_success(
                    "Average response time",
                    f"✓ {avg_time:.2f}s per request"
                )
                
                if avg_time < 5.0:
                    self.log_success("Performance", "✓ Good performance")
                elif avg_time < 10.0:
                    self.log_warning("Performance", "⚠️ Acceptable performance")
                else:
                    self.log_error("Performance", "❌ Slow performance")
                    
        except Exception as e:
            self.log_error("Performance test", f"❌ Error: {str(e)}")
            
        print()
        
    async def send_chat_request(self, session: aiohttp.ClientSession, message: str, session_id: str):
        """Send single chat request"""
        payload = {
            "message": message,
            "session_id": session_id
        }
        
        async with session.post(
            f"{self.base_url}/chat",
            json=payload,
            headers={"Content-Type": "application/json"}
        ) as response:
            return await response.json()
            
    def log_success(self, test_name: str, message: str):
        """Log successful test"""
        print(f"  ✅ {test_name}: {message}")
        self.test_results.append({
            "test": test_name,
            "status": "success",
            "message": message,
            "timestamp": datetime.now().isoformat()
        })
        
    def log_error(self, test_name: str, message: str):
        """Log failed test"""
        print(f"  ❌ {test_name}: {message}")
        self.test_results.append({
            "test": test_name,
            "status": "error",
            "message": message,
            "timestamp": datetime.now().isoformat()
        })
        
    def log_warning(self, test_name: str, message: str):
        """Log warning test"""
        print(f"  ⚠️ {test_name}: {message}")
        self.test_results.append({
            "test": test_name,
            "status": "warning",
            "message": message,
            "timestamp": datetime.now().isoformat()
        })
        
    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*60)
        print("📊 TEST SUMMARY")
        print("="*60)
        
        total_tests = len(self.test_results)
        successful_tests = sum(1 for r in self.test_results if r['status'] == 'success')
        warning_tests = sum(1 for r in self.test_results if r['status'] == 'warning')
        failed_tests = sum(1 for r in self.test_results if r['status'] == 'error')
        
        print(f"Total Tests: {total_tests}")
        print(f"✅ Successful: {successful_tests}")
        print(f"⚠️ Warnings: {warning_tests}")
        print(f"❌ Failed: {failed_tests}")
        
        success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
        print(f"\n🎯 Success Rate: {success_rate:.1f}%")
        
        if failed_tests == 0:
            print("\n🎉 All tests passed! Chatbot siap digunakan.")
        elif failed_tests <= 2:
            print("\n⚠️ Beberapa test gagal, tapi chatbot masih bisa digunakan.")
        else:
            print("\n❌ Banyak test gagal, periksa konfigurasi dan koneksi.")
            
        # Save results to file
        self.save_results()
        
    def save_results(self):
        """Save test results to file"""
        try:
            results_file = f"test_results_{int(time.time())}.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "timestamp": datetime.now().isoformat(),
                    "summary": {
                        "total": len(self.test_results),
                        "successful": sum(1 for r in self.test_results if r['status'] == 'success'),
                        "warnings": sum(1 for r in self.test_results if r['status'] == 'warning'),
                        "failed": sum(1 for r in self.test_results if r['status'] == 'error')
                    },
                    "results": self.test_results
                }, f, indent=2, ensure_ascii=False)
                
            print(f"\n💾 Test results saved to: {results_file}")
            
        except Exception as e:
            print(f"\n⚠️ Could not save results: {e}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Test SIPD Nebius Chatbot")
    parser.add_argument("--quick", action="store_true", help="Run quick tests only")
    parser.add_argument("--performance", action="store_true", help="Include performance tests")
    parser.add_argument("--url", default="http://localhost:8000", help="Chatbot server URL")
    
    args = parser.parse_args()
    
    print("🤖 SIPD Nebius Chatbot Tester")
    print("=" * 40)
    print(f"Target URL: {args.url}")
    print(f"Quick mode: {args.quick}")
    print(f"Performance tests: {args.performance}")
    print()
    
    tester = NebiusChatbotTester(args.url)
    
    try:
        asyncio.run(tester.run_all_tests(
            quick=args.quick,
            performance=args.performance
        ))
    except KeyboardInterrupt:
        print("\n⚠️ Testing interrupted by user")
    except Exception as e:
        print(f"\n❌ Testing failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()