"""
ML Models Test - Essential Components Only
Tests our 4 core ML models with real agent integration
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

async def test_core_ml_models():
    """Test all 4 core ML models"""
    
    print("🧪 CORE ML MODELS TEST")
    print("=" * 40)
    
    results = []
    
    # 1. OpenAI Sentiment
    try:
        from src.ml.openai_sentiment_model import openai_sentiment
        print("🧠 Testing OpenAI Sentiment...")
        
        result = await openai_sentiment.analyze_text("AAPL bullish momentum")
        print(f"   Score: {result.score:.2f}, Category: {result.category}")
        print("✅ OpenAI Sentiment: WORKING")
        results.append(True)
    except Exception as e:
        print(f"❌ OpenAI Sentiment: {e}")
        results.append(False)
    
    # 2. LightGBM Flow 
    try:
        from src.ml.lightgbm_flow_model import lightgbm_flow_predictor
        print("\n⚡ Testing LightGBM Flow...")
        
        flow_data = {'put_call_ratio': 1.2, 'call_volume': 15000, 'put_volume': 18000}
        result = await lightgbm_flow_predictor.predict_flow(flow_data)
        print(f"   Sentiment: {result.flow_sentiment}, Risk: {result.risk_level}")
        print("✅ LightGBM Flow: WORKING")
        results.append(True)
    except Exception as e:
        print(f"❌ LightGBM Flow: {e}")
        results.append(False)
    
    # 3. Ensemble Model
    try:
        from src.ml.ensemble_model import ensemble_model
        print("\n🎯 Testing Ensemble Model...")
        
        market_data = {'quote': {'price': 150.0}, 'technical': {'rsi': 65.0}}
        result = await ensemble_model.generate_signal('TEST', market_data)
        print(f"   Direction: {result.direction}, Score: {result.final_score:.2f}")
        print("✅ Ensemble Model: WORKING") 
        results.append(True)
    except Exception as e:
        print(f"❌ Ensemble Model: {e}")
        results.append(False)
    
    # 4. RL Trading Agent
    try:
        from src.ml.rl_trading_agent import RealRLTradingAgent
        print("\n🤖 Testing RL Trading Agent...")
        
        agent = RealRLTradingAgent()
        metrics = agent.get_real_performance_metrics()
        print(f"   Agent Type: {metrics['agent_type']}")
        print("✅ RL Agent: WORKING")
        results.append(True)
    except Exception as e:
        print(f"❌ RL Agent: {e}")
        results.append(False)
    
    # Results
    passed = sum(results)
    print(f"\n📊 {passed}/4 models working ({passed/4*100:.0f}%)")
    print("🎉 ML PIPELINE: READY!" if passed == 4 else "⚠️  Some issues need attention")

if __name__ == "__main__":
    asyncio.run(test_core_ml_models())