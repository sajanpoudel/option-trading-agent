-- Neural Options Oracle++ Essential Tables
-- Copy and paste this into Supabase SQL Editor

-- Create system_config table
CREATE TABLE IF NOT EXISTS system_config (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    config_key VARCHAR(255) UNIQUE NOT NULL,
    config_value JSONB,
    description TEXT,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create browser_sessions table
CREATE TABLE IF NOT EXISTS browser_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_token VARCHAR(255) UNIQUE NOT NULL,
    ip_address INET,
    user_agent TEXT,
    device_info JSONB DEFAULT '{}',
    risk_profile VARCHAR(50) DEFAULT 'moderate',
    preferences JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT true,
    expires_at TIMESTAMP WITH TIME ZONE,
    last_accessed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create stocks table
CREATE TABLE IF NOT EXISTS stocks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol VARCHAR(10) UNIQUE NOT NULL,
    company_name VARCHAR(255),
    sector VARCHAR(100),
    market_cap BIGINT,
    is_active BOOLEAN DEFAULT true,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create trading_signals table
CREATE TABLE IF NOT EXISTS trading_signals (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol VARCHAR(10) NOT NULL,
    signal_type VARCHAR(50) DEFAULT 'hybrid',
    direction VARCHAR(20) NOT NULL,
    strength VARCHAR(20),
    confidence_score DECIMAL(4,3),
    market_scenario VARCHAR(50),
    agent_weights JSONB DEFAULT '{}',
    technical_analysis JSONB DEFAULT '{}',
    sentiment_analysis JSONB DEFAULT '{}',
    flow_analysis JSONB DEFAULT '{}',
    historical_analysis JSONB DEFAULT '{}',
    strike_recommendations JSONB DEFAULT '[]',
    educational_content JSONB DEFAULT '{}',
    expires_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create positions table
CREATE TABLE IF NOT EXISTS positions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    signal_id UUID REFERENCES trading_signals(id),
    session_token VARCHAR(255),
    symbol VARCHAR(10) NOT NULL,
    option_symbol VARCHAR(50),
    position_type VARCHAR(20) DEFAULT 'option',
    option_type VARCHAR(10),
    strike_price DECIMAL(10,2),
    expiration_date DATE,
    quantity INTEGER NOT NULL,
    entry_price DECIMAL(10,2) NOT NULL,
    current_price DECIMAL(10,2),
    unrealized_pnl DECIMAL(15,2) DEFAULT 0,
    unrealized_pnl_percent DECIMAL(8,4) DEFAULT 0,
    realized_pnl DECIMAL(15,2) DEFAULT 0,
    delta DECIMAL(8,6),
    gamma DECIMAL(8,6),
    theta DECIMAL(8,6),
    vega DECIMAL(8,6),
    rho DECIMAL(8,6),
    status VARCHAR(20) DEFAULT 'open',
    entry_order_id VARCHAR(255),
    exit_order_id VARCHAR(255),
    entry_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    exit_date TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create orders table
CREATE TABLE IF NOT EXISTS orders (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    position_id UUID REFERENCES positions(id),
    session_token VARCHAR(255),
    symbol VARCHAR(10) NOT NULL,
    option_symbol VARCHAR(50),
    side VARCHAR(10) NOT NULL,
    order_type VARCHAR(20) DEFAULT 'market',
    quantity INTEGER NOT NULL,
    price DECIMAL(10,2),
    stop_price DECIMAL(10,2),
    time_in_force VARCHAR(10) DEFAULT 'day',
    status VARCHAR(20) DEFAULT 'pending',
    filled_quantity INTEGER DEFAULT 0,
    average_fill_price DECIMAL(10,2),
    external_order_id VARCHAR(255),
    submitted_at TIMESTAMP WITH TIME ZONE,
    filled_at TIMESTAMP WITH TIME ZONE,
    cancelled_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create educational_content table
CREATE TABLE IF NOT EXISTS educational_content (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content_id VARCHAR(255) UNIQUE NOT NULL,
    title VARCHAR(255) NOT NULL,
    topic VARCHAR(100),
    difficulty VARCHAR(20) DEFAULT 'beginner',
    content_type VARCHAR(20) DEFAULT 'lesson',
    content JSONB NOT NULL,
    prerequisites TEXT[],
    learning_objectives TEXT[],
    estimated_duration_minutes INTEGER DEFAULT 15,
    tags TEXT[],
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_browser_sessions_token ON browser_sessions(session_token);
CREATE INDEX IF NOT EXISTS idx_browser_sessions_active ON browser_sessions(is_active, expires_at);
CREATE INDEX IF NOT EXISTS idx_stocks_symbol ON stocks(symbol);
CREATE INDEX IF NOT EXISTS idx_trading_signals_symbol ON trading_signals(symbol);
CREATE INDEX IF NOT EXISTS idx_trading_signals_created ON trading_signals(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol);
CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status);
CREATE INDEX IF NOT EXISTS idx_orders_symbol ON orders(symbol);
CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);
CREATE INDEX IF NOT EXISTS idx_educational_content_topic ON educational_content(topic);

-- Insert initial system configuration
INSERT INTO system_config (config_key, config_value, description) VALUES 
('app_version', '"1.0.0"', 'Neural Options Oracle++ Version'),
('default_risk_profile', '"moderate"', 'Default risk profile for new sessions'),
('session_timeout_hours', '24', 'Default session timeout in hours'),
('max_positions_per_session', '10', 'Maximum positions per session')
ON CONFLICT (config_key) DO UPDATE SET 
    config_value = EXCLUDED.config_value,
    updated_at = NOW();

-- Insert sample educational content
INSERT INTO educational_content (content_id, title, topic, difficulty, content_type, content, learning_objectives) VALUES 
(
    'options_basics_101',
    'Introduction to Options Trading',
    'options_basics',
    'beginner',
    'lesson',
    '{"sections": [{"title": "What are Options?", "content": "Options are financial contracts that give you the right, but not the obligation, to buy or sell a stock at a specific price within a certain time period."}]}',
    ARRAY['Understand what options are', 'Learn the difference between calls and puts', 'Understand expiration dates and strike prices']
)
ON CONFLICT (content_id) DO NOTHING;

COMMIT;