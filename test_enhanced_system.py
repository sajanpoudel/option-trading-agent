"""
Test script for enhanced professional stock indicators system
"""
import asyncio
from src.data.market_data_manager import market_data_manager

async def test_enhanced_indicators():
    print('🧪 Testing Enhanced Professional Stock Indicators System')
    print('=' * 60)
    
    # Test with AAPL
    symbol = 'AAPL'
    print(f'📊 Testing comprehensive analysis for {symbol}...')
    
    try:
        # Get comprehensive market data
        result = await market_data_manager.get_comprehensive_data(symbol)
        
        # Display technical indicators
        tech_indicators = result.get('technical_indicators', {})
        
        print(f'\n📈 Professional Technical Indicators for {symbol}:')
        print('-' * 50)
        
        # Current price and basic metrics
        current_price = tech_indicators.get('current_price', 0)
        change_percent = tech_indicators.get('change_percent', 0)
        print(f'Current Price: ${current_price:.2f}')
        print(f'Change: {change_percent:.2f}%')
        
        # Moving averages
        print(f'\n🔄 Moving Averages:')
        print(f'  SMA(20): ${tech_indicators.get("sma_20", 0):.2f}')
        print(f'  EMA(20): ${tech_indicators.get("ema_20", 0):.2f}')
        print(f'  WMA(20): ${tech_indicators.get("wma_20", 0):.2f}')
        
        # Oscillators
        print(f'\n📊 Oscillators:')
        print(f'  RSI(14): {tech_indicators.get("rsi", 0):.2f}')
        print(f'  Stochastic: {tech_indicators.get("stochastic_k", 0):.2f}')
        print(f'  Williams %R: {tech_indicators.get("williams_r", 0):.2f}')
        
        # MACD
        print(f'\n⚡ MACD:')
        print(f'  MACD Line: {tech_indicators.get("macd", 0):.4f}')
        print(f'  Signal Line: {tech_indicators.get("macd_signal", 0):.4f}')
        print(f'  Histogram: {tech_indicators.get("macd_histogram", 0):.4f}')
        
        # Bollinger Bands
        print(f'\n🎯 Bollinger Bands:')
        print(f'  Upper: ${tech_indicators.get("bb_upper", 0):.2f}')
        print(f'  Middle: ${tech_indicators.get("bb_middle", 0):.2f}')
        print(f'  Lower: ${tech_indicators.get("bb_lower", 0):.2f}')
        print(f'  %B: {tech_indicators.get("bb_percent", 0):.2f}')
        
        # Advanced indicators
        print(f'\n🔥 Advanced Indicators:')
        print(f'  ADX: {tech_indicators.get("adx", 0):.2f}')
        print(f'  Supertrend: ${tech_indicators.get("supertrend", 0):.2f}')
        print(f'  VWAP: ${tech_indicators.get("vwap", 0):.2f}')
        print(f'  Volatility: {tech_indicators.get("volatility", 0):.2f}%')
        
        # Support/Resistance
        print(f'\n🎯 Support/Resistance:')
        print(f'  Resistance: ${tech_indicators.get("resistance", 0):.2f}')
        print(f'  Support: ${tech_indicators.get("support", 0):.2f}')
        
        # Data source and quality
        data_source = tech_indicators.get('source', 'unknown')
        data_points = tech_indicators.get('data_points', 0)
        print(f'\n📈 Data Quality:')
        print(f'  Source: {data_source}')
        print(f'  Data Points: {data_points}')
        
        print(f'\n✅ Professional indicators integration successful!')
        
        # Test AI agent with enhanced data
        print(f'\n🤖 Testing Technical Analysis Agent with enhanced data...')
        from agents.technical_agent import TechnicalAnalysisAgent
        from openai import OpenAI
        from config.settings import settings
        
        client = OpenAI(api_key=settings.openai_api_key)
        tech_agent = TechnicalAnalysisAgent(client)
        
        agent_result = await tech_agent.analyze(symbol)
        
        print(f'🧠 Agent Analysis Results:')
        print(f'  Technical Score: {agent_result.get("technical_score", 0):.2f}')
        print(f'  Confidence: {agent_result.get("confidence", 0):.2f}')
        print(f'  Market Scenario: {agent_result.get("market_scenario", "unknown")}')
        print(f'  Key Signals: {len(agent_result.get("key_signals", []))} signals detected')
        
        return True
        
    except Exception as e:
        print(f'❌ Error testing enhanced system: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    asyncio.run(test_enhanced_indicators())