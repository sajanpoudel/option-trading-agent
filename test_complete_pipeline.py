#!/usr/bin/env python3
"""
Complete ML Pipeline Test with Real Agent Integration
Tests the full pipeline: Agents -> ML Models -> Decision Engine -> Signal Generation
"""
import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.orchestrator import OptionsOracleOrchestrator
from src.core.decision_engine import DecisionEngine
from src.ml.ensemble_model import EnsembleDecisionModel
from datetime import datetime

async def test_complete_pipeline():
    """Test complete pipeline with real agent integration"""
    
    print("🚀 COMPLETE ML PIPELINE TEST")
    print("=" * 50)
    
    # Initialize components
    print("🔧 Initializing components...")
    orchestrator = OptionsOracleOrchestrator()
    decision_engine = DecisionEngine()
    ensemble_model = EnsembleDecisionModel()
    
    # Test symbol
    test_symbol = "AAPL"
    user_risk_profile = {
        'risk_level': 'moderate',
        'experience': 'intermediate',
        'max_position_size': 0.05
    }
    
    print(f"📊 Testing with symbol: {test_symbol}")
    print(f"👤 Risk profile: {user_risk_profile['risk_level']}")
    print()
    
    try:
        # Step 1: Agent Analysis
        print("🤖 Step 1: Running AI Agent Analysis...")
        agent_results = await orchestrator.analyze_stock(test_symbol, user_risk_profile)
        print(f"   ✅ Technical Analysis: {agent_results.get('technical', {}).get('scenario', 'N/A')}")
        print(f"   ✅ Sentiment Analysis: {agent_results.get('sentiment', {}).get('overall_sentiment', 'N/A')}")
        print(f"   ✅ Options Flow: {agent_results.get('flow', {}).get('flow_direction', 'N/A')}")
        print(f"   ✅ Historical Pattern: {agent_results.get('history', {}).get('pattern_type', 'N/A')}")
        print()
        
        # Step 2: ML Model Processing
        print("🧠 Step 2: ML Model Processing...")
        
        # Prepare market data for ensemble
        market_data = {
            'symbol': test_symbol,
            'price': 150.0,  # Example price
            'volume': 50000000,
            'timestamp': datetime.now()
        }
        
        # Generate ensemble signal
        ensemble_signal = await ensemble_model.generate_signal(
            test_symbol, 
            market_data,
            sentiment_data=agent_results.get('sentiment', {}),
            options_data=agent_results.get('flow', {})
        )
        
        print(f"   ✅ Ensemble Direction: {ensemble_signal.direction}")
        print(f"   ✅ Ensemble Score: {ensemble_signal.final_score:.3f}")
        print(f"   ✅ Confidence: {ensemble_signal.confidence:.3f}")
        print()
        
        # Step 3: Decision Engine Processing
        print("🎯 Step 3: Decision Engine Processing...")
        final_decision = await decision_engine.process_stock(test_symbol, user_risk_profile)
        
        print(f"   ✅ Final Signal: {final_decision['signal']['direction']}")
        print(f"   ✅ Strategy: {final_decision['signal']['options_strategy']}")
        print(f"   ✅ Scenario: {final_decision['scenario']}")
        print(f"   ✅ Overall Confidence: {final_decision['confidence']:.3f}")
        print()
        
        # Step 4: Strike Recommendations
        print("⚡ Step 4: Risk-Based Strike Selection...")
        strikes = final_decision.get('strike_recommendations', [])
        print(f"   ✅ Generated {len(strikes)} strike recommendations")
        
        if strikes:
            top_strike = strikes[0]
            print(f"   🎯 Top Recommendation:")
            print(f"      Strike: ${top_strike.get('strike', 'N/A')}")
            print(f"      Type: {top_strike.get('option_type', 'N/A')}")
            print(f"      Risk Score: {top_strike.get('risk_score', 'N/A')}")
        print()
        
        # Summary
        print("📈 PIPELINE TEST RESULTS")
        print("=" * 30)
        print(f"Symbol: {test_symbol}")
        print(f"Agent Analysis: ✅ Complete")
        print(f"ML Processing: ✅ Complete")  
        print(f"Decision Engine: ✅ Complete")
        print(f"Strike Selection: ✅ Complete")
        print(f"Final Signal: {final_decision['signal']['direction']}")
        print(f"Pipeline Status: 🎉 FULLY OPERATIONAL")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_complete_pipeline())
    sys.exit(0 if success else 1)