# Neural Options Oracle++ - Database Schema

## Database Architecture Overview

The Neural Options Oracle++ system uses **Supabase** as the primary database solution, which provides:

- **PostgreSQL Database**: Full-featured relational database with JSONB support
- **Real-time Subscriptions**: WebSocket-based live updates
- **Authentication**: Built-in user management and JWT tokens
- **Row Level Security**: Fine-grained access control
- **Auto-generated APIs**: REST and GraphQL endpoints
- **Edge Functions**: Server-side logic execution

## Supabase Database Schema

### Core Tables

#### Users Table (Built-in with Supabase Auth)
```sql
-- Supabase provides auth.users table automatically
-- We extend it with user profiles

CREATE TABLE user_profiles (
    id UUID REFERENCES auth.users(id) ON DELETE CASCADE PRIMARY KEY,
    risk_profile TEXT CHECK (risk_profile IN ('conservative', 'moderate', 'aggressive')) DEFAULT 'moderate',
    experience_level TEXT CHECK (experience_level IN ('beginner', 'intermediate', 'advanced')) DEFAULT 'beginner',
    account_balance DECIMAL(15,2) DEFAULT 100000.00, -- Paper trading balance
    settings JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_login_at TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT TRUE
);

-- Enable RLS
ALTER TABLE user_profiles ENABLE ROW LEVEL SECURITY;

-- RLS Policies
CREATE POLICY "Users can view own profile" ON user_profiles
    FOR SELECT USING (auth.uid() = id);

CREATE POLICY "Users can update own profile" ON user_profiles
    FOR UPDATE USING (auth.uid() = id);

-- Indexes
CREATE INDEX idx_user_profiles_risk_profile ON user_profiles(risk_profile);
CREATE INDEX idx_user_profiles_created_at ON user_profiles(created_at);
```

#### Stocks Table
```sql
CREATE TABLE stocks (
    symbol VARCHAR(10) PRIMARY KEY,
    company_name VARCHAR(255) NOT NULL,
    sector VARCHAR(100),
    industry VARCHAR(100),
    market_cap BIGINT,
    exchange VARCHAR(50),
    currency VARCHAR(3) DEFAULT 'USD',
    is_active BOOLEAN DEFAULT TRUE,
    options_available BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Additional metadata
    metadata JSONB DEFAULT '{}'
);

-- Indexes
CREATE INDEX idx_stocks_sector ON stocks(sector);
CREATE INDEX idx_stocks_market_cap ON stocks(market_cap);
CREATE INDEX idx_stocks_options_available ON stocks(options_available) WHERE options_available = TRUE;
```

#### Trading Signals Table
```sql
CREATE TABLE trading_signals (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    symbol VARCHAR(10) NOT NULL REFERENCES stocks(symbol),
    analysis_id UUID, -- Reference to detailed analysis
    
    -- Signal data
    signal_type signal_type_enum NOT NULL,
    direction direction_enum NOT NULL,
    strength strength_enum NOT NULL,
    confidence_score DECIMAL(3,2) CHECK (confidence_score >= 0 AND confidence_score <= 1),
    
    -- Market scenario and weights
    market_scenario market_scenario_enum,
    agent_weights JSONB, -- Dynamic weights for each agent
    
    -- Agent analysis results
    technical_analysis JSONB,
    sentiment_analysis JSONB,
    flow_analysis JSONB,
    historical_analysis JSONB,
    
    -- Strike recommendations
    strike_recommendations JSONB,
    
    -- Educational content
    educational_content JSONB,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE,
    
    CONSTRAINT signals_confidence_range CHECK (confidence_score BETWEEN 0.0 AND 1.0)
);

-- Enums for trading signals
CREATE TYPE signal_type_enum AS ENUM ('technical', 'fundamental', 'hybrid');
CREATE TYPE direction_enum AS ENUM ('BUY', 'SELL', 'HOLD', 'STRONG_BUY', 'STRONG_SELL');
CREATE TYPE strength_enum AS ENUM ('weak', 'moderate', 'strong');
CREATE TYPE market_scenario_enum AS ENUM ('strong_uptrend', 'strong_downtrend', 'range_bound', 'breakout', 'potential_reversal', 'high_volatility', 'low_volatility');

-- Indexes
CREATE INDEX idx_signals_user_id ON trading_signals(user_id);
CREATE INDEX idx_signals_symbol ON trading_signals(symbol);
CREATE INDEX idx_signals_created_at ON trading_signals(created_at);
CREATE INDEX idx_signals_direction ON trading_signals(direction);
CREATE INDEX idx_signals_market_scenario ON trading_signals(market_scenario);
```

#### Positions Table
```sql
CREATE TABLE positions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    signal_id UUID REFERENCES trading_signals(id),
    
    -- Position details
    symbol VARCHAR(10) NOT NULL REFERENCES stocks(symbol),
    option_symbol VARCHAR(50), -- Full option symbol (e.g., AAPL240119C00155000)
    position_type position_type_enum NOT NULL,
    option_type option_type_enum,
    
    -- Strike and expiration
    strike_price DECIMAL(10,2),
    expiration_date DATE,
    
    -- Quantity and pricing
    quantity INTEGER NOT NULL,
    entry_price DECIMAL(10,4) NOT NULL,
    current_price DECIMAL(10,4),
    
    -- P&L tracking
    unrealized_pnl DECIMAL(15,2) DEFAULT 0,
    unrealized_pnl_percent DECIMAL(5,2) DEFAULT 0,
    realized_pnl DECIMAL(15,2) DEFAULT 0,
    
    -- Greeks (for options)
    delta DECIMAL(8,6),
    gamma DECIMAL(8,6),
    theta DECIMAL(8,6),
    vega DECIMAL(8,6),
    rho DECIMAL(8,6),
    
    -- Position management
    status position_status_enum DEFAULT 'open',
    entry_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    close_date TIMESTAMP WITH TIME ZONE,
    close_price DECIMAL(10,4),
    
    -- Risk management
    stop_loss_price DECIMAL(10,4),
    take_profit_price DECIMAL(10,4),
    
    -- Order tracking
    entry_order_id VARCHAR(100),
    close_order_id VARCHAR(100),
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Enums for positions
CREATE TYPE position_type_enum AS ENUM ('stock', 'option');
CREATE TYPE option_type_enum AS ENUM ('call', 'put');
CREATE TYPE position_status_enum AS ENUM ('open', 'closed', 'expired');

-- Indexes
CREATE INDEX idx_positions_user_id ON positions(user_id);
CREATE INDEX idx_positions_symbol ON positions(symbol);
CREATE INDEX idx_positions_status ON positions(status);
CREATE INDEX idx_positions_expiration ON positions(expiration_date) WHERE option_type IS NOT NULL;
CREATE INDEX idx_positions_entry_date ON positions(entry_date);
```

#### Orders Table
```sql
CREATE TABLE orders (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    position_id UUID REFERENCES positions(id),
    
    -- Order details
    symbol VARCHAR(10) NOT NULL,
    option_symbol VARCHAR(50),
    side order_side_enum NOT NULL,
    order_type order_type_enum NOT NULL,
    quantity INTEGER NOT NULL,
    
    -- Pricing
    limit_price DECIMAL(10,4),
    stop_price DECIMAL(10,4),
    filled_price DECIMAL(10,4),
    filled_quantity INTEGER DEFAULT 0,
    
    -- Order status and timing
    status order_status_enum DEFAULT 'pending',
    time_in_force time_in_force_enum DEFAULT 'day',
    
    -- External order tracking
    broker_order_id VARCHAR(100),
    broker VARCHAR(50) DEFAULT 'alpaca',
    
    -- Order lifecycle
    submitted_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    filled_at TIMESTAMP WITH TIME ZONE,
    cancelled_at TIMESTAMP WITH TIME ZONE,
    
    -- Commission and fees
    commission DECIMAL(10,4) DEFAULT 0,
    fees DECIMAL(10,4) DEFAULT 0,
    
    -- Order metadata
    metadata JSONB DEFAULT '{}'
);

-- Enums for orders
CREATE TYPE order_side_enum AS ENUM ('buy', 'sell');
CREATE TYPE order_type_enum AS ENUM ('market', 'limit', 'stop', 'stop_limit');
CREATE TYPE order_status_enum AS ENUM ('pending', 'submitted', 'filled', 'partially_filled', 'cancelled', 'rejected');
CREATE TYPE time_in_force_enum AS ENUM ('day', 'gtc', 'ioc', 'fok');

-- Indexes
CREATE INDEX idx_orders_user_id ON orders(user_id);
CREATE INDEX idx_orders_status ON orders(status);
CREATE INDEX idx_orders_submitted_at ON orders(submitted_at);
CREATE INDEX idx_orders_broker_order_id ON orders(broker_order_id);
```

#### Educational Progress Table
```sql
CREATE TABLE educational_progress (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    
    -- Lesson tracking
    lesson_id VARCHAR(100) NOT NULL,
    lesson_topic VARCHAR(100) NOT NULL,
    lesson_difficulty difficulty_enum NOT NULL,
    
    -- Progress metrics
    completion_status completion_status_enum DEFAULT 'not_started',
    completion_percentage INTEGER DEFAULT 0 CHECK (completion_percentage BETWEEN 0 AND 100),
    time_spent_minutes INTEGER DEFAULT 0,
    
    -- Quiz results
    quiz_attempts INTEGER DEFAULT 0,
    best_quiz_score DECIMAL(3,2),
    latest_quiz_score DECIMAL(3,2),
    
    -- Learning analytics
    concepts_mastered JSONB DEFAULT '[]',
    areas_for_improvement JSONB DEFAULT '[]',
    
    -- Timestamps
    first_accessed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_accessed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    
    CONSTRAINT unique_user_lesson UNIQUE(user_id, lesson_id)
);

-- Enums for education
CREATE TYPE difficulty_enum AS ENUM ('beginner', 'intermediate', 'advanced');
CREATE TYPE completion_status_enum AS ENUM ('not_started', 'in_progress', 'completed', 'mastered');

-- Indexes
CREATE INDEX idx_edu_progress_user_id ON educational_progress(user_id);
CREATE INDEX idx_edu_progress_topic ON educational_progress(lesson_topic);
CREATE INDEX idx_edu_progress_completion ON educational_progress(completion_status);
```

#### Learning Analytics Table
```sql
CREATE TABLE learning_analytics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    
    -- Overall learning metrics
    total_lessons_completed INTEGER DEFAULT 0,
    total_study_time_minutes INTEGER DEFAULT 0,
    current_streak_days INTEGER DEFAULT 0,
    longest_streak_days INTEGER DEFAULT 0,
    
    -- Topic mastery levels
    topic_mastery JSONB DEFAULT '{}', -- {"options_basics": 0.85, "technical_analysis": 0.72}
    
    -- Learning preferences
    preferred_learning_style learning_style_enum,
    optimal_session_length INTEGER DEFAULT 30, -- minutes
    
    -- Performance trends
    weekly_progress JSONB DEFAULT '[]',
    quiz_performance_trend JSONB DEFAULT '[]',
    
    -- Adaptive learning data
    knowledge_gaps JSONB DEFAULT '[]',
    recommended_topics JSONB DEFAULT '[]',
    
    last_updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Enum for learning style
CREATE TYPE learning_style_enum AS ENUM ('visual', 'auditory', 'kinesthetic', 'mixed');

-- Indexes
CREATE INDEX idx_learning_analytics_user_id ON learning_analytics(user_id);
```

### Relationship Tables

#### User Sessions Table
```sql
CREATE TABLE user_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_hash VARCHAR(255) NOT NULL,
    refresh_token_hash VARCHAR(255),
    
    -- Session metadata
    ip_address INET,
    user_agent TEXT,
    device_info JSONB,
    
    -- Session lifecycle
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    last_accessed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    is_active BOOLEAN DEFAULT TRUE
);

-- Indexes
CREATE INDEX idx_sessions_user_id ON user_sessions(user_id);
CREATE INDEX idx_sessions_token_hash ON user_sessions(token_hash);
CREATE INDEX idx_sessions_expires_at ON user_sessions(expires_at);
```

## Real-time Subscriptions with Supabase

Supabase provides real-time functionality through PostgreSQL's built-in LISTEN/NOTIFY mechanism:

### Real-time Channels
```javascript
// Position updates
const positionChannel = supabase
  .channel('positions')
  .on(
    'postgres_changes',
    {
      event: '*',
      schema: 'public',
      table: 'positions',
      filter: `user_id=eq.${userId}`
    },
    payload => {
      console.log('Position update:', payload)
      updatePositionInUI(payload)
    }
  )
  .subscribe()

// Trading signal updates
const signalChannel = supabase
  .channel('trading_signals')
  .on(
    'postgres_changes',
    {
      event: 'INSERT',
      schema: 'public',
      table: 'trading_signals',
      filter: `user_id=eq.${userId}`
    },
    payload => {
      displayNewSignal(payload.new)
    }
  )
  .subscribe()
```

### Caching Strategy
```sql
-- Market data cache table
CREATE TABLE market_data_cache (
    symbol TEXT PRIMARY KEY,
    price_data JSONB,
    technical_indicators JSONB,
    options_chain JSONB,
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Cache policies
CREATE POLICY "Market data readable by authenticated users" ON market_data_cache
    FOR SELECT TO authenticated USING (true);
```

## Time Series Data with PostgreSQL

Using PostgreSQL with proper indexing for time series data:

### Historical Performance Tracking
```sql
-- Portfolio snapshots (daily)
CREATE TABLE portfolio_snapshots (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    snapshot_date DATE NOT NULL,
    total_value DECIMAL(15,2),
    total_pnl DECIMAL(15,2),
    total_pnl_percent DECIMAL(5,2),
    portfolio_greeks JSONB,
    position_count INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    UNIQUE(user_id, snapshot_date)
);

-- Market data history
CREATE TABLE market_data_history (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    symbol TEXT NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    price_data JSONB,
    technical_indicators JSONB,
    volume BIGINT,
    
    INDEX (symbol, timestamp DESC)
);

-- Agent performance tracking
CREATE TABLE agent_performance_logs (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    agent_type TEXT NOT NULL,
    symbol TEXT,
    accuracy DECIMAL(3,2),
    confidence DECIMAL(3,2),
    response_time_ms INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    INDEX (agent_type, created_at DESC)
);
```

## Database Initialization Scripts

### PostgreSQL Initialization
```sql
-- init/001_create_extensions.sql
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- init/002_create_enums.sql
-- (Enum definitions from above)

-- init/003_create_tables.sql
-- (Table definitions from above)

-- init/004_create_indexes.sql
-- Performance indexes
CREATE INDEX CONCURRENTLY idx_positions_user_symbol ON positions(user_id, symbol);
CREATE INDEX CONCURRENTLY idx_signals_user_created ON trading_signals(user_id, created_at DESC);
CREATE INDEX CONCURRENTLY idx_orders_user_status ON orders(user_id, status);

-- init/005_create_functions.sql
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Add triggers for updated_at
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_positions_updated_at BEFORE UPDATE ON positions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- init/006_create_views.sql
CREATE VIEW system_portfolio_summary AS
SELECT 
    COUNT(p.id) as total_positions,
    SUM(p.quantity * p.current_price * 100) as total_value,
    SUM(p.unrealized_pnl) as total_unrealized_pnl,
    AVG(p.unrealized_pnl_percent) as avg_return_percent,
    SUM(p.delta * p.quantity) as portfolio_delta,
    SUM(p.gamma * p.quantity) as portfolio_gamma,
    SUM(p.theta * p.quantity) as portfolio_theta,
    SUM(p.vega * p.quantity) as portfolio_vega,
    NOW() as calculated_at
FROM positions p
WHERE p.status = 'open';

-- System performance analytics view
CREATE VIEW system_trading_performance AS
SELECT 
    COUNT(*) as total_trades,
    COUNT(CASE WHEN realized_pnl > 0 THEN 1 END) as winning_trades,
    COUNT(CASE WHEN realized_pnl < 0 THEN 1 END) as losing_trades,
    ROUND(
        COUNT(CASE WHEN realized_pnl > 0 THEN 1 END)::DECIMAL / COUNT(*)::DECIMAL * 100, 
        2
    ) as win_rate,
    SUM(realized_pnl) as total_realized_pnl,
    AVG(realized_pnl) as avg_trade_pnl,
    MAX(realized_pnl) as best_trade,
    MIN(realized_pnl) as worst_trade,
    NOW() as calculated_at
FROM positions
WHERE status = 'closed';
```

### Database Migration Strategy
```python
# migrations/migration_manager.py
class MigrationManager:
    def __init__(self, db_connection):
        self.db = db_connection
        
    def run_migrations(self):
        """Run all pending migrations"""
        current_version = self.get_current_version()
        migration_files = self.get_migration_files()
        
        for migration in migration_files:
            if migration.version > current_version:
                self.run_migration(migration)
                self.update_version(migration.version)
                
    def create_migration(self, name: str):
        """Create a new migration file"""
        version = self.get_next_version()
        filename = f"V{version:04d}__{name}.sql"
        # Create migration file template
```

## Python Backend Integration

### Supabase Client Setup
```python
# config/database.py
import os
from supabase import create_client, Client
from typing import Optional, Dict, Any, List
import asyncio
from datetime import datetime, timedelta
import uuid

class SupabaseManager:
    """Supabase database manager for Neural Options Oracle++"""
    
    def __init__(self):
        self.url: str = os.environ.get("SUPABASE_URL")
        self.key: str = os.environ.get("SUPABASE_ANON_KEY")
        self.service_key: str = os.environ.get("SUPABASE_SERVICE_KEY")
        
        # Use service key for backend operations (bypasses RLS)
        self.client: Client = create_client(self.url, self.service_key)
        
    async def create_browser_session(self, 
                                   ip_address: str = None,
                                   user_agent: str = None,
                                   device_info: Dict = None) -> str:
        """Create a new browser session"""
        
        session_token = str(uuid.uuid4())
        
        session_data = {
            "session_token": session_token,
            "ip_address": ip_address,
            "user_agent": user_agent,
            "device_info": device_info or {},
            "expires_at": (datetime.now() + timedelta(hours=24)).isoformat()
        }
        
        result = self.client.table("browser_sessions").insert(session_data).execute()
        return session_token
    
    async def get_session(self, session_token: str) -> Optional[Dict]:
        """Get browser session data"""
        
        result = self.client.table("browser_sessions")\
            .select("*")\
            .eq("session_token", session_token)\
            .eq("is_active", True)\
            .gt("expires_at", datetime.now().isoformat())\
            .single()\
            .execute()
            
        return result.data if result.data else None
    
    async def save_trading_signal(self, signal_data: Dict) -> str:
        """Save trading signal to database"""
        
        signal_record = {
            "symbol": signal_data["symbol"],
            "signal_type": signal_data.get("signal_type", "hybrid"),
            "direction": signal_data["direction"],
            "strength": signal_data["strength"],
            "confidence_score": signal_data["confidence_score"],
            "market_scenario": signal_data.get("market_scenario"),
            "agent_weights": signal_data.get("agent_weights", {}),
            "technical_analysis": signal_data.get("technical_analysis", {}),
            "sentiment_analysis": signal_data.get("sentiment_analysis", {}),
            "flow_analysis": signal_data.get("flow_analysis", {}),
            "historical_analysis": signal_data.get("historical_analysis", {}),
            "strike_recommendations": signal_data.get("strike_recommendations", []),
            "educational_content": signal_data.get("educational_content", {}),
            "expires_at": (datetime.now() + timedelta(hours=24)).isoformat()
        }
        
        result = self.client.table("trading_signals").insert(signal_record).execute()
        return result.data[0]["id"] if result.data else None
    
    async def create_position(self, position_data: Dict) -> str:
        """Create new trading position"""
        
        position_record = {
            "signal_id": position_data.get("signal_id"),
            "symbol": position_data["symbol"],
            "option_symbol": position_data.get("option_symbol"),
            "position_type": position_data["position_type"],
            "option_type": position_data.get("option_type"),
            "strike_price": position_data.get("strike_price"),
            "expiration_date": position_data.get("expiration_date"),
            "quantity": position_data["quantity"],
            "entry_price": position_data["entry_price"],
            "current_price": position_data.get("current_price", position_data["entry_price"]),
            "delta": position_data.get("delta"),
            "gamma": position_data.get("gamma"),
            "theta": position_data.get("theta"),
            "vega": position_data.get("vega"),
            "rho": position_data.get("rho"),
            "entry_order_id": position_data.get("entry_order_id")
        }
        
        result = self.client.table("positions").insert(position_record).execute()
        return result.data[0]["id"] if result.data else None
    
    async def update_position_pnl(self, position_id: str, 
                                current_price: float) -> Dict:
        """Update position P&L in real-time"""
        
        # Get current position
        position_result = self.client.table("positions")\
            .select("*")\
            .eq("id", position_id)\
            .single()\
            .execute()
            
        if not position_result.data:
            return None
            
        position = position_result.data
        entry_price = float(position["entry_price"])
        quantity = int(position["quantity"])
        
        # Calculate P&L
        if position["position_type"] == "option":
            # Options P&L (per contract = 100 shares)
            unrealized_pnl = (current_price - entry_price) * quantity * 100
        else:
            # Stock P&L
            unrealized_pnl = (current_price - entry_price) * quantity
            
        unrealized_pnl_percent = ((current_price - entry_price) / entry_price) * 100
        
        # Update position
        update_data = {
            "current_price": current_price,
            "unrealized_pnl": unrealized_pnl,
            "unrealized_pnl_percent": unrealized_pnl_percent,
            "updated_at": datetime.now().isoformat()
        }
        
        result = self.client.table("positions")\
            .update(update_data)\
            .eq("id", position_id)\
            .execute()
            
        return result.data[0] if result.data else None
    
    async def get_portfolio_summary(self) -> Dict:
        """Get real-time portfolio summary"""
        
        result = self.client.rpc("system_portfolio_summary").execute()
        return result.data if result.data else {}
    
    async def save_educational_content(self, content_data: Dict) -> str:
        """Save educational content"""
        
        content_record = {
            "content_id": content_data["content_id"],
            "title": content_data["title"],
            "topic": content_data["topic"],
            "difficulty": content_data["difficulty"],
            "content_type": content_data["content_type"],
            "content": content_data["content"],
            "prerequisites": content_data.get("prerequisites", []),
            "learning_objectives": content_data.get("learning_objectives", []),
            "estimated_duration_minutes": content_data.get("estimated_duration_minutes", 15),
            "tags": content_data.get("tags", [])
        }
        
        result = self.client.table("educational_content").insert(content_record).execute()
        return result.data[0]["id"] if result.data else None
    
    async def get_system_analytics(self) -> Dict:
        """Get system-wide analytics"""
        
        # Get trading performance
        trading_perf = self.client.rpc("system_trading_performance").execute()
        
        # Get portfolio summary  
        portfolio_summary = self.client.rpc("system_portfolio_summary").execute()
        
        # Get active sessions count
        sessions_result = self.client.table("browser_sessions")\
            .select("count", count="exact")\
            .eq("is_active", True)\
            .gt("expires_at", datetime.now().isoformat())\
            .execute()
            
        return {
            "trading_performance": trading_perf.data if trading_perf.data else {},
            "portfolio_summary": portfolio_summary.data if portfolio_summary.data else {},
            "active_sessions": sessions_result.count or 0,
            "timestamp": datetime.now().isoformat()
        }

# Usage example in FastAPI
from fastapi import FastAPI, WebSocket
import json

app = FastAPI()
db_manager = SupabaseManager()

@app.post("/api/analyze/{symbol}")
async def analyze_symbol(symbol: str, session_token: str = None):
    """Analyze symbol and generate trading signal"""
    
    # Validate session if provided
    if session_token:
        session = await db_manager.get_session(session_token)
        if not session:
            return {"error": "Invalid session"}
    
    # Run AI analysis (your existing agent orchestration)
    signal_data = await run_ai_analysis(symbol)
    
    # Save to database
    signal_id = await db_manager.save_trading_signal(signal_data)
    
    return {
        "signal_id": signal_id,
        "signal_data": signal_data
    }

@app.websocket("/ws/positions")
async def websocket_positions_endpoint(websocket: WebSocket):
    """Real-time position updates via WebSocket"""
    
    await websocket.accept()
    
    # Subscribe to Supabase real-time changes
    def handle_position_change(payload):
        asyncio.create_task(websocket.send_text(json.dumps(payload)))
    
    # Note: Supabase real-time would be configured on the frontend
    # This is just an example of how to handle updates
    
    try:
        while True:
            data = await websocket.receive_text()
            # Handle client messages
    except:
        # Handle disconnect
        pass
```

### Environment Configuration
```bash
# .env file
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your_anon_key_here
SUPABASE_SERVICE_KEY=your_service_key_here

# API Keys for AI services
OPENAI_API_KEY=your_openai_key
GEMINI_API_KEY=your_gemini_key  
JIGSAWSTACK_API_KEY=your_jigsawstack_key

# Alpaca API (for paper trading)
ALPACA_API_KEY=your_alpaca_key
ALPACA_SECRET_KEY=your_alpaca_secret
```

This comprehensive database schema provides the foundation for the Neural Options Oracle++ system with Supabase integration, removing authentication complexity while maintaining real-time capabilities and educational analytics. The system operates with browser sessions instead of user accounts, making it perfect for a demo/educational environment.