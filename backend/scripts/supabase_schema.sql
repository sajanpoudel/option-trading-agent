-- Neural Options Oracle++ Database Schema
-- Complete SQL script for Supabase setup
-- Run this in Supabase SQL Editor if the Python script has issues

-- ==============================================
-- 1. CREATE EXTENSIONS
-- ==============================================

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ==============================================
-- 2. CREATE ENUMS
-- ==============================================

-- Trading signal enums
DO $$ BEGIN
    CREATE TYPE signal_type_enum AS ENUM ('technical', 'fundamental', 'hybrid');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE direction_enum AS ENUM ('BUY', 'SELL', 'HOLD', 'STRONG_BUY', 'STRONG_SELL');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE strength_enum AS ENUM ('weak', 'moderate', 'strong');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE market_scenario_enum AS ENUM ('strong_uptrend', 'strong_downtrend', 'range_bound', 'breakout', 'potential_reversal', 'high_volatility', 'low_volatility');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- Position enums
DO $$ BEGIN
    CREATE TYPE position_type_enum AS ENUM ('stock', 'option');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE option_type_enum AS ENUM ('call', 'put');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE position_status_enum AS ENUM ('open', 'closed', 'expired');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- Order enums
DO $$ BEGIN
    CREATE TYPE order_side_enum AS ENUM ('buy', 'sell');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE order_type_enum AS ENUM ('market', 'limit', 'stop', 'stop_limit');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE order_status_enum AS ENUM ('pending', 'submitted', 'filled', 'partially_filled', 'cancelled', 'rejected');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE time_in_force_enum AS ENUM ('day', 'gtc', 'ioc', 'fok');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- Education enums
DO $$ BEGIN
    CREATE TYPE difficulty_enum AS ENUM ('beginner', 'intermediate', 'advanced');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE completion_status_enum AS ENUM ('not_started', 'in_progress', 'completed', 'mastered');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE content_type_enum AS ENUM ('lesson', 'quiz', 'interactive', 'video', 'simulation');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE learning_style_enum AS ENUM ('visual', 'auditory', 'kinesthetic', 'mixed');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- ==============================================
-- 3. CREATE TABLES
-- ==============================================

-- System Configuration Table
CREATE TABLE IF NOT EXISTS system_config (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    config_key VARCHAR(100) UNIQUE NOT NULL,
    config_value JSONB NOT NULL,
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Browser Sessions Table (no user authentication)
CREATE TABLE IF NOT EXISTS browser_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_token VARCHAR(255) UNIQUE NOT NULL,
    
    -- Session metadata (no user auth)
    ip_address INET,
    user_agent TEXT,
    device_info JSONB DEFAULT '{}',
    
    -- Session preferences
    risk_profile TEXT CHECK (risk_profile IN ('conservative', 'moderate', 'aggressive')) DEFAULT 'moderate',
    preferences JSONB DEFAULT '{}',
    
    -- Session lifecycle
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE DEFAULT (NOW() + INTERVAL '24 hours'),
    last_accessed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    is_active BOOLEAN DEFAULT TRUE
);

-- Stocks Table
CREATE TABLE IF NOT EXISTS stocks (
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

-- Trading Signals Table
CREATE TABLE IF NOT EXISTS trading_signals (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol VARCHAR(10) NOT NULL REFERENCES stocks(symbol),
    analysis_id UUID, -- Reference to detailed analysis
    
    -- Signal data
    signal_type signal_type_enum NOT NULL,
    direction direction_enum NOT NULL,
    strength strength_enum NOT NULL,
    confidence_score DECIMAL(3,2) CHECK (confidence_score >= 0 AND confidence_score <= 1),
    
    -- Market scenario and weights
    market_scenario market_scenario_enum,
    agent_weights JSONB DEFAULT '{}', -- Dynamic weights for each agent
    
    -- Agent analysis results
    technical_analysis JSONB DEFAULT '{}',
    sentiment_analysis JSONB DEFAULT '{}',
    flow_analysis JSONB DEFAULT '{}',
    historical_analysis JSONB DEFAULT '{}',
    
    -- Strike recommendations
    strike_recommendations JSONB DEFAULT '[]',
    
    -- Educational content
    educational_content JSONB DEFAULT '{}',
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE,
    
    CONSTRAINT signals_confidence_range CHECK (confidence_score BETWEEN 0.0 AND 1.0)
);

-- Positions Table
CREATE TABLE IF NOT EXISTS positions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
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

-- Orders Table
CREATE TABLE IF NOT EXISTS orders (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
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

-- Educational Content Table
CREATE TABLE IF NOT EXISTS educational_content (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Content details
    content_id VARCHAR(100) UNIQUE NOT NULL,
    title VARCHAR(255) NOT NULL,
    topic VARCHAR(100) NOT NULL,
    difficulty difficulty_enum NOT NULL,
    content_type content_type_enum NOT NULL,
    
    -- Content data
    content JSONB NOT NULL, -- Lesson content, quiz questions, etc.
    prerequisites JSONB DEFAULT '[]', -- Required prior lessons
    learning_objectives JSONB DEFAULT '[]',
    
    -- Metadata
    estimated_duration_minutes INTEGER DEFAULT 15,
    tags JSONB DEFAULT '[]',
    
    -- Status
    is_active BOOLEAN DEFAULT TRUE,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Trading Analytics Table (session-based, no user auth)
CREATE TABLE IF NOT EXISTS trading_analytics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Session tracking (no user auth)
    session_id UUID UNIQUE NOT NULL,
    
    -- Trading metrics
    total_signals_generated INTEGER DEFAULT 0,
    total_positions_tracked INTEGER DEFAULT 0,
    successful_predictions INTEGER DEFAULT 0,
    
    -- Performance analytics
    win_rate DECIMAL(5,2) DEFAULT 0,
    total_pnl DECIMAL(15,2) DEFAULT 0,
    best_trade DECIMAL(15,2) DEFAULT 0,
    worst_trade DECIMAL(15,2) DEFAULT 0,
    
    -- Strategy effectiveness
    strategy_performance JSONB DEFAULT '{}',
    agent_accuracy JSONB DEFAULT '{}',
    
    -- System metrics
    analysis_response_times JSONB DEFAULT '[]',
    error_count INTEGER DEFAULT 0,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Market Data Cache Table
CREATE TABLE IF NOT EXISTS market_data_cache (
    symbol TEXT PRIMARY KEY,
    price_data JSONB,
    technical_indicators JSONB,
    options_chain JSONB,
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- System Performance Snapshots
CREATE TABLE IF NOT EXISTS system_snapshots (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    snapshot_date DATE NOT NULL UNIQUE,
    
    -- System metrics
    total_signals_generated INTEGER DEFAULT 0,
    total_positions_tracked INTEGER DEFAULT 0,
    active_sessions INTEGER DEFAULT 0,
    
    -- Performance metrics
    avg_analysis_time_ms INTEGER,
    success_rate DECIMAL(5,2),
    error_rate DECIMAL(5,2),
    
    -- Market data
    symbols_analyzed JSONB DEFAULT '[]',
    market_conditions JSONB DEFAULT '{}',
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ==============================================
-- 4. CREATE INDEXES
-- ==============================================

-- Browser Sessions Indexes
CREATE INDEX IF NOT EXISTS idx_browser_sessions_token ON browser_sessions(session_token);
CREATE INDEX IF NOT EXISTS idx_browser_sessions_expires ON browser_sessions(expires_at);
CREATE INDEX IF NOT EXISTS idx_browser_sessions_active ON browser_sessions(is_active) WHERE is_active = TRUE;

-- Stocks Indexes
CREATE INDEX IF NOT EXISTS idx_stocks_sector ON stocks(sector);
CREATE INDEX IF NOT EXISTS idx_stocks_market_cap ON stocks(market_cap);
CREATE INDEX IF NOT EXISTS idx_stocks_options_available ON stocks(options_available) WHERE options_available = TRUE;

-- Trading Signals Indexes
CREATE INDEX IF NOT EXISTS idx_signals_symbol ON trading_signals(symbol);
CREATE INDEX IF NOT EXISTS idx_signals_created_at ON trading_signals(created_at);
CREATE INDEX IF NOT EXISTS idx_signals_direction ON trading_signals(direction);
CREATE INDEX IF NOT EXISTS idx_signals_market_scenario ON trading_signals(market_scenario);

-- Positions Indexes
CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol);
CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status);
CREATE INDEX IF NOT EXISTS idx_positions_expiration ON positions(expiration_date) WHERE option_type IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_positions_entry_date ON positions(entry_date);

-- Orders Indexes
CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);
CREATE INDEX IF NOT EXISTS idx_orders_submitted_at ON orders(submitted_at);
CREATE INDEX IF NOT EXISTS idx_orders_broker_order_id ON orders(broker_order_id);

-- Educational Content Indexes
CREATE INDEX IF NOT EXISTS idx_edu_content_topic ON educational_content(topic);
CREATE INDEX IF NOT EXISTS idx_edu_content_difficulty ON educational_content(difficulty);
CREATE INDEX IF NOT EXISTS idx_edu_content_type ON educational_content(content_type);
CREATE INDEX IF NOT EXISTS idx_edu_content_active ON educational_content(is_active) WHERE is_active = TRUE;

-- Trading Analytics Indexes
CREATE INDEX IF NOT EXISTS idx_trading_analytics_session ON trading_analytics(session_id);
CREATE INDEX IF NOT EXISTS idx_trading_analytics_created ON trading_analytics(created_at);

-- Performance Indexes
CREATE INDEX IF NOT EXISTS idx_positions_symbol_status ON positions(symbol, status);
CREATE INDEX IF NOT EXISTS idx_signals_symbol_created ON trading_signals(symbol, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_orders_status_created ON orders(status, submitted_at DESC);

-- ==============================================
-- 5. CREATE FUNCTIONS AND TRIGGERS
-- ==============================================

-- Updated At Trigger Function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Add Updated At Triggers
DROP TRIGGER IF EXISTS update_system_config_updated_at ON system_config;
CREATE TRIGGER update_system_config_updated_at 
    BEFORE UPDATE ON system_config
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_positions_updated_at ON positions;
CREATE TRIGGER update_positions_updated_at 
    BEFORE UPDATE ON positions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_browser_sessions_updated_at ON browser_sessions;
CREATE TRIGGER update_browser_sessions_updated_at 
    BEFORE UPDATE ON browser_sessions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_stocks_updated_at ON stocks;
CREATE TRIGGER update_stocks_updated_at 
    BEFORE UPDATE ON stocks
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_educational_content_updated_at ON educational_content;
CREATE TRIGGER update_educational_content_updated_at 
    BEFORE UPDATE ON educational_content
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ==============================================
-- 6. CREATE VIEWS
-- ==============================================

-- System Portfolio Summary View
CREATE OR REPLACE VIEW system_portfolio_summary AS
SELECT 
    COUNT(p.id) as total_positions,
    COALESCE(SUM(p.quantity * p.current_price * 
        CASE WHEN p.position_type = 'option' THEN 100 ELSE 1 END), 0) as total_value,
    COALESCE(SUM(p.unrealized_pnl), 0) as total_unrealized_pnl,
    COALESCE(AVG(p.unrealized_pnl_percent), 0) as avg_return_percent,
    COALESCE(SUM(p.delta * p.quantity), 0) as portfolio_delta,
    COALESCE(SUM(p.gamma * p.quantity), 0) as portfolio_gamma,
    COALESCE(SUM(p.theta * p.quantity), 0) as portfolio_theta,
    COALESCE(SUM(p.vega * p.quantity), 0) as portfolio_vega,
    NOW() as calculated_at
FROM positions p
WHERE p.status = 'open';

-- System Trading Performance View
CREATE OR REPLACE VIEW system_trading_performance AS
SELECT 
    COUNT(*) as total_trades,
    COUNT(CASE WHEN realized_pnl > 0 THEN 1 END) as winning_trades,
    COUNT(CASE WHEN realized_pnl < 0 THEN 1 END) as losing_trades,
    CASE 
        WHEN COUNT(*) > 0 THEN
            ROUND(COUNT(CASE WHEN realized_pnl > 0 THEN 1 END)::DECIMAL / COUNT(*)::DECIMAL * 100, 2)
        ELSE 0
    END as win_rate,
    COALESCE(SUM(realized_pnl), 0) as total_realized_pnl,
    COALESCE(AVG(realized_pnl), 0) as avg_trade_pnl,
    COALESCE(MAX(realized_pnl), 0) as best_trade,
    COALESCE(MIN(realized_pnl), 0) as worst_trade,
    NOW() as calculated_at
FROM positions
WHERE status = 'closed';

-- ==============================================
-- 7. INSERT DEFAULT DATA
-- ==============================================

-- Insert Default System Configuration
INSERT INTO system_config (config_key, config_value, description) VALUES
('default_risk_profile', '"moderate"', 'Default risk profile for analysis'),
('paper_trading_balance', '100000.00', 'Starting paper trading balance'),
('max_positions', '10', 'Maximum concurrent positions'),
('analysis_timeout', '30', 'Analysis timeout in seconds')
ON CONFLICT (config_key) DO NOTHING;

-- Insert Sample Stocks
INSERT INTO stocks (symbol, company_name, sector, industry, market_cap, exchange, options_available) VALUES
('AAPL', 'Apple Inc.', 'Technology', 'Consumer Electronics', 3000000000000, 'NASDAQ', TRUE),
('GOOGL', 'Alphabet Inc.', 'Technology', 'Internet Software & Services', 2000000000000, 'NASDAQ', TRUE),
('MSFT', 'Microsoft Corporation', 'Technology', 'Software', 2800000000000, 'NASDAQ', TRUE),
('TSLA', 'Tesla Inc.', 'Consumer Cyclical', 'Auto Manufacturers', 800000000000, 'NASDAQ', TRUE),
('AMZN', 'Amazon.com Inc.', 'Consumer Cyclical', 'Internet Retail', 1500000000000, 'NASDAQ', TRUE),
('NVDA', 'NVIDIA Corporation', 'Technology', 'Semiconductors', 1700000000000, 'NASDAQ', TRUE),
('META', 'Meta Platforms Inc.', 'Technology', 'Internet Software & Services', 900000000000, 'NASDAQ', TRUE),
('NFLX', 'Netflix Inc.', 'Communication Services', 'Entertainment', 200000000000, 'NASDAQ', TRUE),
('JPM', 'JPMorgan Chase & Co.', 'Financial Services', 'Banks', 500000000000, 'NYSE', TRUE),
('JNJ', 'Johnson & Johnson', 'Healthcare', 'Drug Manufacturers', 400000000000, 'NYSE', TRUE)
ON CONFLICT (symbol) DO UPDATE SET
    company_name = EXCLUDED.company_name,
    sector = EXCLUDED.sector,
    industry = EXCLUDED.industry,
    market_cap = EXCLUDED.market_cap,
    updated_at = NOW();

-- Insert Sample Educational Content
INSERT INTO educational_content (content_id, title, topic, difficulty, content_type, content, learning_objectives, tags) VALUES
('options_basics_001', 'Introduction to Options Trading', 'options_basics', 'beginner', 'lesson', 
 '{"introduction": "Options are financial contracts that give you the right, but not the obligation, to buy or sell a stock at a specific price.", "key_concepts": ["Call options give you the right to buy", "Put options give you the right to sell", "Strike price is the contract price", "Expiration date is when the contract expires"], "examples": [{"scenario": "Buying a call option on AAPL", "explanation": "If you think AAPL will go up, you can buy a call option."}]}',
 '["Understand basic options terminology", "Distinguish between calls and puts", "Identify key option components"]',
 '["options", "basics", "beginner"]'),

('technical_analysis_001', 'Reading Stock Charts', 'technical_analysis', 'beginner', 'lesson',
 '{"introduction": "Technical analysis involves studying price charts to predict future price movements.", "key_concepts": ["Support and resistance levels", "Moving averages", "Volume indicators", "Chart patterns"], "practical_tips": ["Look for trends in price movement", "Use multiple timeframes", "Consider volume confirmation"]}',
 '["Read basic stock charts", "Identify trends and patterns", "Understand key technical indicators"]',
 '["technical_analysis", "charts", "beginner"]'),

('options_greeks_001', 'Understanding Options Greeks', 'options_advanced', 'intermediate', 'lesson',
 '{"introduction": "Options Greeks measure how option prices change relative to various factors.", "greeks": {"delta": "Measures price sensitivity to stock price changes", "gamma": "Measures how delta changes", "theta": "Measures time decay", "vega": "Measures volatility sensitivity", "rho": "Measures interest rate sensitivity"}, "practical_examples": [{"scenario": "High delta call option", "explanation": "A call with 0.7 delta will gain $0.70 for every $1 stock increase"}]}',
 '["Understand all five Greeks", "Apply Greeks to trading decisions", "Calculate risk exposure"]',
 '["greeks", "options", "intermediate"]'),

('risk_management_001', 'Portfolio Risk Management', 'risk_management', 'intermediate', 'lesson',
 '{"introduction": "Risk management is crucial for long-term trading success.", "key_concepts": ["Position sizing", "Stop losses", "Diversification", "Risk-reward ratios"], "practical_examples": [{"scenario": "2% rule", "explanation": "Never risk more than 2% of your account on a single trade"}]}',
 '["Calculate appropriate position sizes", "Set effective stop losses", "Understand risk-reward ratios"]',
 '["risk", "management", "intermediate"]')

ON CONFLICT (content_id) DO NOTHING;

-- ==============================================
-- COMPLETION MESSAGE
-- ==============================================

-- This will show in the Results tab if run successfully
SELECT 
    '🎉 Neural Options Oracle++ Database Schema Created Successfully!' as message,
    'Tables: ' || count(*) as tables_created
FROM information_schema.tables 
WHERE table_schema = 'public' 
AND table_name IN (
    'system_config', 'browser_sessions', 'stocks', 'trading_signals', 
    'positions', 'orders', 'educational_content', 'trading_analytics',
    'market_data_cache', 'system_snapshots'
);