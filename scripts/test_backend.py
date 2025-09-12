#!/usr/bin/env python3
"""
Neural Options Oracle++ Backend Test Script

This script tests all major backend functionality including:
- Database connection
- API endpoints
- Session management
- Trading operations
- Educational content
"""
import asyncio
import sys
import requests
import json
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.database import db_manager
from config.logging import get_core_logger

logger = get_core_logger()

# API Base URL
API_BASE_URL = "http://localhost:8080"

class BackendTester:
    """Comprehensive backend testing"""
    
    def __init__(self):
        self.session_token = None
        self.test_results = {}
        self.api_available = False
    
    def test_api_connection(self) -> bool:
        """Test if API server is running"""
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=5)
            if response.status_code == 200:
                logger.info("✅ API server is running")
                self.api_available = True
                return True
            else:
                logger.error(f"❌ API server returned status {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            logger.error("❌ API server is not running")
            logger.info("💡 Start the server with: python main.py")
            return False
        except Exception as e:
            logger.error(f"❌ API connection test failed: {e}")
            return False
    
    async def test_database_connection(self) -> bool:
        """Test direct database connection"""
        try:
            health = await db_manager.health_check()
            if health["status"] == "healthy":
                logger.info("✅ Database connection successful")
                return True
            else:
                logger.error(f"❌ Database unhealthy: {health}")
                return False
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            return False
    
    def test_session_management(self) -> bool:
        """Test session creation and management"""
        try:
            # Create session
            response = requests.post(f"{API_BASE_URL}/api/v1/session/create", 
                                   json={"risk_profile": "moderate"})
            
            if response.status_code == 200:
                data = response.json()
                self.session_token = data["session_token"]
                logger.info(f"✅ Session created: {self.session_token[:8]}...")
                
                # Test session info
                response = requests.get(f"{API_BASE_URL}/api/v1/session/info",
                                      headers={"X-Session-Token": self.session_token})
                
                if response.status_code == 200:
                    logger.info("✅ Session info retrieval successful")
                    return True
                else:
                    logger.error(f"❌ Session info failed: {response.status_code}")
                    return False
            else:
                logger.error(f"❌ Session creation failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Session management test failed: {e}")
            return False
    
    def test_analysis_api(self) -> bool:
        """Test stock analysis endpoints"""
        try:
            headers = {}
            if self.session_token:
                headers["X-Session-Token"] = self.session_token
            
            # Test quick analysis
            response = requests.get(f"{API_BASE_URL}/api/v1/analysis/quick/AAPL", 
                                  headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"✅ Quick analysis successful: {data['direction']} signal for AAPL")
                
                # Test full analysis
                analysis_request = {
                    "symbol": "AAPL",
                    "risk_profile": "moderate",
                    "analysis_type": "full"
                }
                
                response = requests.post(f"{API_BASE_URL}/api/v1/analysis/analyze/AAPL",
                                       json=analysis_request,
                                       headers=headers)
                
                if response.status_code == 200:
                    data = response.json()
                    logger.info(f"✅ Full analysis successful: {data['confidence']:.2f} confidence")
                    return True
                else:
                    logger.error(f"❌ Full analysis failed: {response.status_code}")
                    return False
            else:
                logger.error(f"❌ Quick analysis failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Analysis API test failed: {e}")
            return False
    
    def test_trading_api(self) -> bool:
        """Test trading endpoints"""
        try:
            headers = {}
            if self.session_token:
                headers["X-Session-Token"] = self.session_token
            
            # Test paper trade execution
            trade_request = {
                "symbol": "AAPL",
                "action": "buy",
                "quantity": 1,
                "order_type": "market",
                "option_details": {
                    "type": "call",
                    "strike": 155.0,
                    "expiration": "2024-03-15"
                }
            }
            
            response = requests.post(f"{API_BASE_URL}/api/v1/trading/execute",
                                   json=trade_request,
                                   headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"✅ Paper trade executed: ${data['total_value']:.2f}")
                
                # Test positions retrieval
                response = requests.get(f"{API_BASE_URL}/api/v1/trading/positions",
                                      headers=headers)
                
                if response.status_code == 200:
                    positions = response.json()
                    logger.info(f"✅ Positions retrieved: {len(positions)} positions")
                    return True
                else:
                    logger.error(f"❌ Positions retrieval failed: {response.status_code}")
                    return False
            else:
                logger.error(f"❌ Paper trade failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Trading API test failed: {e}")
            return False
    
    def test_portfolio_api(self) -> bool:
        """Test portfolio endpoints"""
        try:
            headers = {}
            if self.session_token:
                headers["X-Session-Token"] = self.session_token
            
            # Test portfolio summary
            response = requests.get(f"{API_BASE_URL}/api/v1/portfolio/summary",
                                  headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"✅ Portfolio summary: ${data['total_value']:.2f} total value")
                
                # Test portfolio Greeks
                response = requests.get(f"{API_BASE_URL}/api/v1/portfolio/greeks",
                                      headers=headers)
                
                if response.status_code == 200:
                    greeks = response.json()
                    logger.info(f"✅ Portfolio Greeks: {greeks['total_delta']:.2f} delta")
                    return True
                else:
                    logger.error(f"❌ Portfolio Greeks failed: {response.status_code}")
                    return False
            else:
                logger.error(f"❌ Portfolio summary failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Portfolio API test failed: {e}")
            return False
    
    def test_education_api(self) -> bool:
        """Test education endpoints"""
        try:
            headers = {}
            if self.session_token:
                headers["X-Session-Token"] = self.session_token
            
            # Test educational content
            response = requests.get(f"{API_BASE_URL}/api/v1/education/content",
                                  headers=headers)
            
            if response.status_code == 200:
                content = response.json()
                logger.info(f"✅ Educational content: {len(content)} items")
                
                # Test quiz generation
                quiz_request = {
                    "topic": "options_basics",
                    "difficulty": "beginner",
                    "question_count": 3
                }
                
                response = requests.post(f"{API_BASE_URL}/api/v1/education/quiz",
                                       json=quiz_request,
                                       headers=headers)
                
                if response.status_code == 200:
                    quiz = response.json()
                    logger.info(f"✅ Quiz generated: {len(quiz['questions'])} questions")
                    return True
                else:
                    logger.error(f"❌ Quiz generation failed: {response.status_code}")
                    return False
            else:
                logger.error(f"❌ Educational content failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Education API test failed: {e}")
            return False
    
    def test_system_api(self) -> bool:
        """Test system endpoints"""
        try:
            headers = {}
            if self.session_token:
                headers["X-Session-Token"] = self.session_token
            
            # Test system status
            response = requests.get(f"{API_BASE_URL}/api/v1/system/status",
                                  headers=headers)
            
            if response.status_code == 200:
                status = response.json()
                logger.info(f"✅ System status: {status['system']['status']}")
                
                # Test system analytics
                response = requests.get(f"{API_BASE_URL}/api/v1/system/analytics",
                                      headers=headers)
                
                if response.status_code == 200:
                    analytics = response.json()
                    logger.info("✅ System analytics retrieved")
                    return True
                else:
                    logger.error(f"❌ System analytics failed: {response.status_code}")
                    return False
            else:
                logger.error(f"❌ System status failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ System API test failed: {e}")
            return False
    
    async def run_all_tests(self):
        """Run comprehensive backend tests"""
        
        logger.info("🧪 Starting Neural Options Oracle++ Backend Tests")
        logger.info("=" * 60)
        
        tests = [
            ("Database Connection", self.test_database_connection()),
            ("API Connection", self.test_api_connection()),
        ]
        
        # Run async tests first
        for test_name, test_coro in tests[:1]:  # Database test
            logger.info(f"Running: {test_name}")
            result = await test_coro
            self.test_results[test_name] = result
        
        # Run sync API tests if API is available
        if self.api_available:
            api_tests = [
                ("API Connection", self.test_api_connection),
                ("Session Management", self.test_session_management),
                ("Analysis API", self.test_analysis_api),
                ("Trading API", self.test_trading_api),
                ("Portfolio API", self.test_portfolio_api),
                ("Education API", self.test_education_api),
                ("System API", self.test_system_api),
            ]
            
            for test_name, test_func in api_tests[1:]:  # Skip API connection, already done
                logger.info(f"Running: {test_name}")
                result = test_func()
                self.test_results[test_name] = result
        
        # Summary
        self.print_test_summary()
    
    def print_test_summary(self):
        """Print comprehensive test results"""
        
        logger.info("=" * 60)
        logger.info("🎯 Test Results Summary")
        logger.info("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result)
        failed_tests = total_tests - passed_tests
        
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"{status}: {test_name}")
        
        logger.info("=" * 60)
        logger.info(f"📊 Overall Results: {passed_tests}/{total_tests} tests passed")
        
        if failed_tests == 0:
            logger.info("🎉 All tests passed! Backend is ready for production.")
        elif passed_tests >= total_tests * 0.8:
            logger.info("⚠️ Most tests passed. Minor issues detected.")
        else:
            logger.info("❌ Multiple test failures. Check configuration and dependencies.")
        
        logger.info("\n📋 Next Steps:")
        if not self.api_available:
            logger.info("1. Start the API server: python main.py")
            logger.info("2. Re-run tests: python scripts/test_backend.py")
        else:
            logger.info("1. Backend is functional!")
            logger.info("2. Start implementing AI agents")
            logger.info("3. Access API docs: http://localhost:8080/docs")


async def main():
    """Main test function"""
    tester = BackendTester()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())