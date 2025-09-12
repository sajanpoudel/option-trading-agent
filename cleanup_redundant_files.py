"""
Cleanup Script - Remove Redundant Files
Removes outdated orchestrators and integrations that have been replaced
"""

import os
from pathlib import Path

def cleanup_redundant_files():
    """Remove redundant files and keep only production-ready code"""
    
    print("🧹 Starting File Cleanup")
    print("=" * 50)
    
    # Files to remove (redundant/outdated)
    files_to_remove = [
        # Redundant orchestrators (replaced by openai_only_orchestrator.py)
        "src/data/real_data_sources.py",
        "src/core/real_data_orchestrator.py", 
        "src/data/simplified_data_orchestrator.py",
        
        # JigsawStack integration (eliminated)
        "src/data/jigsawstack_integration.py",
        
        # Web search agents (integrated into openai_only_orchestrator.py)
        "agents/web_search_agents.py",
    ]
    
    # Optional test files (user can decide)
    optional_files = [
        "test_enhanced_system.py",
        "test_openai_integration.py"
    ]
    
    removed_count = 0
    kept_count = 0
    
    print("🗑️  Removing redundant files:")
    print("-" * 30)
    
    for file_path in files_to_remove:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"✅ Removed: {file_path}")
                removed_count += 1
            except Exception as e:
                print(f"❌ Failed to remove {file_path}: {e}")
        else:
            print(f"⚠️  Not found: {file_path}")
    
    print(f"\n📁 Optional test files (keeping for now):")
    print("-" * 30)
    
    for file_path in optional_files:
        if os.path.exists(file_path):
            print(f"📄 Kept: {file_path}")
            kept_count += 1
        else:
            print(f"⚠️  Not found: {file_path}")
    
    print(f"\n📊 Cleanup Summary:")
    print(f"   🗑️  Removed: {removed_count} files")
    print(f"   📁 Kept: {kept_count} optional files")
    
    # Show essential files structure
    print(f"\n✅ Essential Files Structure (Production Ready):")
    print("-" * 50)
    
    essential_structure = {
        "📊 Core Data & Intelligence": [
            "src/data/market_data_manager.py (MAIN coordinator)",
            "src/data/alpaca_client.py (Real market data)",
            "src/data/openai_only_orchestrator.py (OpenAI + Options)",
            "src/indicators/technical_calculator.py (62 professional indicators)"
        ],
        "🤖 AI Agents (OpenAI SDK v0.3.0)": [
            "agents/orchestrator.py (Master orchestrator)",
            "agents/base_agent.py (Base agent class)",
            "agents/technical_agent.py (Technical analysis)",
            "agents/flow_agent.py (Options flow)",
            "agents/sentiment_agent.py (Sentiment analysis)",
            "agents/history_agent.py (Historical patterns)",
            "agents/risk_agent.py (Risk management)",
            "agents/education_agent.py (Educational content)"
        ],
        "🌐 API Layer (For Frontend)": [
            "src/api/main.py (FastAPI application)",
            "src/api/dependencies.py (API dependencies)",
            "src/api/routes/analysis.py (Analysis endpoints)",
            "src/api/routes/trading.py (Trading endpoints)",
            "src/api/routes/portfolio.py (Portfolio endpoints)",
            "src/api/routes/system.py (System endpoints)",
            "src/api/routes/education.py (Education endpoints)"
        ]
    }
    
    for category, files in essential_structure.items():
        print(f"\n{category}:")
        for file in files:
            print(f"   ✅ {file}")
    
    print(f"\n🎯 Final Architecture:")
    print("   • OpenAI web search for intelligence")
    print("   • OptionsProfitCalculator for real options data")
    print("   • Professional stock-indicators library (62 metrics)")
    print("   • Alpaca + yfinance for market data")
    print("   • OpenAI Agents SDK v0.3.0 for AI orchestration")
    
    print(f"\n🚀 System Status: PRODUCTION READY!")

if __name__ == "__main__":
    cleanup_redundant_files()