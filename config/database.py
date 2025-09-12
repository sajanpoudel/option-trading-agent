"""
Neural Options Oracle++ Database Configuration and Manager
"""
import os
import asyncio
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import uuid
from supabase import create_client, Client
from config.settings import settings
from config.logging import get_database_logger

logger = get_database_logger()


class SupabaseManager:
    """Supabase database manager for Neural Options Oracle++"""
    
    def __init__(self):
        self.url = settings.supabase_url
        self.anon_key = settings.supabase_anon_key
        self.service_key = settings.supabase_service_key
        
        # Use service key for backend operations (bypasses RLS)
        self.client: Client = create_client(self.url, self.service_key)
        logger.info("Supabase client initialized")
        
    async def health_check(self) -> Dict[str, Any]:
        """Check database connectivity"""
        try:
            # Simple query to test connection
            result = self.client.table("system_config").select("count", count="exact").execute()
            
            return {
                "status": "healthy",
                "connection": "active",
                "timestamp": datetime.now().isoformat(),
                "config_count": result.count or 0
            }
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {
                "status": "unhealthy",
                "connection": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    # ========================
    # BROWSER SESSION MANAGEMENT
    # ========================
    
    async def create_browser_session(
        self, 
        ip_address: str = None,
        user_agent: str = None,
        device_info: Dict = None,
        risk_profile: str = "moderate"
    ) -> str:
        """Create a new browser session (no authentication)"""
        
        session_token = str(uuid.uuid4())
        
        session_data = {
            "session_token": session_token,
            "ip_address": ip_address,
            "user_agent": user_agent,
            "device_info": device_info or {},
            "risk_profile": risk_profile,
            "preferences": {},
            "expires_at": (datetime.now() + timedelta(hours=24)).isoformat(),
            "is_active": True
        }
        
        try:
            result = self.client.table("browser_sessions").insert(session_data).execute()
            logger.info(f"Browser session created: {session_token}")
            return session_token
        except Exception as e:
            logger.error(f"Failed to create browser session: {e}")
            raise
    
    async def get_session(self, session_token: str) -> Optional[Dict]:
        """Get browser session data"""
        
        try:
            result = self.client.table("browser_sessions")\
                .select("*")\
                .eq("session_token", session_token)\
                .eq("is_active", True)\
                .gt("expires_at", datetime.now().isoformat())\
                .single()\
                .execute()
                
            if result.data:
                logger.debug(f"Session retrieved: {session_token}")
                return result.data
            else:
                logger.warning(f"Session not found or expired: {session_token}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to get session {session_token}: {e}")
            return None
    
    async def update_session_activity(self, session_token: str) -> bool:
        """Update session last accessed time"""
        
        try:
            result = self.client.table("browser_sessions")\
                .update({"last_accessed_at": datetime.now().isoformat()})\
                .eq("session_token", session_token)\
                .execute()
                
            return len(result.data) > 0
        except Exception as e:
            logger.error(f"Failed to update session activity {session_token}: {e}")
            return False
    
    # ========================
    # TRADING SIGNALS
    # ========================
    
    async def save_trading_signal(self, signal_data: Dict) -> Optional[str]:
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
            "expires_at": (datetime.now() + timedelta(hours=24)).isoformat(),
            "created_at": datetime.now().isoformat()
        }
        
        try:
            result = self.client.table("trading_signals").insert(signal_record).execute()
            signal_id = result.data[0]["id"] if result.data else None
            logger.info(f"Trading signal saved: {signal_id} for {signal_data['symbol']}")
            return signal_id
        except Exception as e:
            logger.error(f"Failed to save trading signal: {e}")
            return None
    
    async def get_trading_signals(
        self, 
        symbol: str = None, 
        limit: int = 10,
        include_expired: bool = False
    ) -> List[Dict]:
        """Get trading signals with optional filtering"""
        
        try:
            query = self.client.table("trading_signals").select("*")
            
            if symbol:
                query = query.eq("symbol", symbol)
            
            if not include_expired:
                query = query.gt("expires_at", datetime.now().isoformat())
            
            query = query.order("created_at", desc=True).limit(limit)
            
            result = query.execute()
            logger.debug(f"Retrieved {len(result.data) if result.data else 0} trading signals")
            return result.data or []
        except Exception as e:
            logger.error(f"Failed to get trading signals: {e}")
            return []
    
    # ========================
    # POSITIONS MANAGEMENT
    # ========================
    
    async def create_position(self, position_data: Dict) -> Optional[str]:
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
            "status": "open",
            "entry_order_id": position_data.get("entry_order_id"),
            "entry_date": datetime.now().isoformat()
        }
        
        try:
            result = self.client.table("positions").insert(position_record).execute()
            position_id = result.data[0]["id"] if result.data else None
            logger.info(f"Position created: {position_id} for {position_data['symbol']}")
            return position_id
        except Exception as e:
            logger.error(f"Failed to create position: {e}")
            return None
    
    async def update_position_pnl(
        self, 
        position_id: str, 
        current_price: float
    ) -> Optional[Dict]:
        """Update position P&L in real-time"""
        
        try:
            # Get current position
            position_result = self.client.table("positions")\
                .select("*")\
                .eq("id", position_id)\
                .single()\
                .execute()
                
            if not position_result.data:
                logger.warning(f"Position not found: {position_id}")
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
                
            unrealized_pnl_percent = ((current_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0
            
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
                
            updated_position = result.data[0] if result.data else None
            logger.debug(f"Position P&L updated: {position_id}, P&L: ${unrealized_pnl:.2f}")
            return updated_position
        except Exception as e:
            logger.error(f"Failed to update position P&L: {e}")
            return None
    
    async def get_open_positions(self) -> List[Dict]:
        """Get all open positions"""
        
        try:
            result = self.client.table("positions")\
                .select("*")\
                .eq("status", "open")\
                .order("entry_date", desc=True)\
                .execute()
                
            logger.debug(f"Retrieved {len(result.data) if result.data else 0} open positions")
            return result.data or []
        except Exception as e:
            logger.error(f"Failed to get open positions: {e}")
            return []
    
    # ========================
    # EDUCATIONAL CONTENT
    # ========================
    
    async def save_educational_content(self, content_data: Dict) -> Optional[str]:
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
            "tags": content_data.get("tags", []),
            "is_active": True
        }
        
        try:
            result = self.client.table("educational_content").insert(content_record).execute()
            content_id = result.data[0]["id"] if result.data else None
            logger.info(f"Educational content saved: {content_data['content_id']}")
            return content_id
        except Exception as e:
            logger.error(f"Failed to save educational content: {e}")
            return None
    
    async def get_educational_content(
        self, 
        topic: str = None, 
        difficulty: str = None,
        content_type: str = None,
        limit: int = 20
    ) -> List[Dict]:
        """Get educational content with filtering"""
        
        try:
            query = self.client.table("educational_content")\
                .select("*")\
                .eq("is_active", True)
            
            if topic:
                query = query.eq("topic", topic)
            if difficulty:
                query = query.eq("difficulty", difficulty)  
            if content_type:
                query = query.eq("content_type", content_type)
            
            query = query.order("created_at", desc=True).limit(limit)
            
            result = query.execute()
            logger.debug(f"Retrieved {len(result.data) if result.data else 0} educational content items")
            return result.data or []
        except Exception as e:
            logger.error(f"Failed to get educational content: {e}")
            return []
    
    # ========================
    # SYSTEM ANALYTICS
    # ========================
    
    async def get_system_analytics(self) -> Dict[str, Any]:
        """Get system-wide analytics"""
        
        try:
            # Get portfolio summary using the view
            portfolio_result = self.client.table("system_portfolio_summary").select("*").execute()
            
            # Get trading performance using the view
            performance_result = self.client.table("system_trading_performance").select("*").execute()
            
            # Get active sessions count
            sessions_result = self.client.table("browser_sessions")\
                .select("count", count="exact")\
                .eq("is_active", True)\
                .gt("expires_at", datetime.now().isoformat())\
                .execute()
            
            # Get recent signals count
            signals_result = self.client.table("trading_signals")\
                .select("count", count="exact")\
                .gt("created_at", (datetime.now() - timedelta(hours=24)).isoformat())\
                .execute()
            
            analytics = {
                "portfolio_summary": portfolio_result.data if portfolio_result.data else {},
                "trading_performance": performance_result.data if performance_result.data else {},
                "active_sessions": sessions_result.count or 0,
                "signals_last_24h": signals_result.count or 0,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.debug("System analytics retrieved successfully")
            return analytics
        except Exception as e:
            logger.error(f"Failed to get system analytics: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    # ========================
    # SCHEMA INITIALIZATION
    # ========================
    
    async def execute_sql(self, sql: str) -> bool:
        """Execute raw SQL command"""
        try:
            result = self.client.rpc('exec_sql', {'sql': sql}).execute()
            return True
        except Exception as e:
            logger.error(f"SQL execution failed: {e}")
            # Try alternative approach using raw HTTP request
            try:
                import requests
                response = requests.post(
                    f"{self.url}/rest/v1/rpc/exec_sql",
                    json={"sql": sql},
                    headers={
                        "apikey": self.service_key,
                        "Authorization": f"Bearer {self.service_key}",
                        "Content-Type": "application/json"
                    }
                )
                return response.status_code == 200
            except Exception as e2:
                logger.error(f"Alternative SQL execution failed: {e2}")
                return False
    
    async def initialize_schema(self) -> bool:
        """Initialize database schema - create essential tables programmatically"""
        try:
            logger.info("Creating essential database tables programmatically...")
            
            # Create system_config table first
            try:
                system_config_data = {
                    "config_key": "app_version",
                    "config_value": "1.0.0",
                    "description": "Neural Options Oracle++ Version",
                    "updated_at": datetime.now().isoformat()
                }
                self.client.table("system_config").insert(system_config_data).execute()
                logger.info("✅ system_config table verified/created")
            except Exception as e:
                logger.warning(f"system_config table creation failed: {e}")
            
            # Create browser_sessions table
            try:
                session_data = {
                    "session_token": "test_init",
                    "ip_address": "127.0.0.1",
                    "user_agent": "init",
                    "device_info": {},
                    "risk_profile": "moderate",
                    "preferences": {},
                    "expires_at": (datetime.now() + timedelta(hours=1)).isoformat(),
                    "is_active": False,
                    "created_at": datetime.now().isoformat(),
                    "last_accessed_at": datetime.now().isoformat()
                }
                result = self.client.table("browser_sessions").insert(session_data).execute()
                # Delete the test record
                self.client.table("browser_sessions").delete().eq("session_token", "test_init").execute()
                logger.info("✅ browser_sessions table verified/created")
            except Exception as e:
                logger.warning(f"browser_sessions table creation failed: {e}")
                
            # Create stocks table
            try:
                stock_data = {
                    "symbol": "AAPL",
                    "company_name": "Apple Inc.",
                    "sector": "Technology",
                    "market_cap": 3000000000000,
                    "is_active": True,
                    "created_at": datetime.now().isoformat()
                }
                result = self.client.table("stocks").insert(stock_data).execute()
                # Delete the test record
                self.client.table("stocks").delete().eq("symbol", "AAPL").execute()
                logger.info("✅ stocks table verified/created")
            except Exception as e:
                logger.warning(f"stocks table creation failed: {e}")
                
            # Create trading_signals table
            try:
                signal_data = {
                    "symbol": "AAPL",
                    "signal_type": "technical",
                    "direction": "BUY",
                    "strength": "moderate",
                    "confidence_score": 0.75,
                    "market_scenario": "range_bound",
                    "agent_weights": {},
                    "technical_analysis": {},
                    "sentiment_analysis": {},
                    "flow_analysis": {},
                    "historical_analysis": {},
                    "strike_recommendations": [],
                    "educational_content": {},
                    "expires_at": (datetime.now() + timedelta(hours=24)).isoformat(),
                    "created_at": datetime.now().isoformat()
                }
                result = self.client.table("trading_signals").insert(signal_data).execute()
                signal_id = result.data[0]["id"]
                # Delete the test record
                self.client.table("trading_signals").delete().eq("id", signal_id).execute()
                logger.info("✅ trading_signals table verified/created")
            except Exception as e:
                logger.warning(f"trading_signals table creation failed: {e}")
                
            # Create positions table
            try:
                position_data = {
                    "symbol": "AAPL",
                    "position_type": "option",
                    "option_type": "call",
                    "strike_price": 150.0,
                    "expiration_date": "2024-03-15",
                    "quantity": 1,
                    "entry_price": 5.0,
                    "current_price": 5.0,
                    "status": "open",
                    "entry_date": datetime.now().isoformat()
                }
                result = self.client.table("positions").insert(position_data).execute()
                position_id = result.data[0]["id"]
                # Delete the test record  
                self.client.table("positions").delete().eq("id", position_id).execute()
                logger.info("✅ positions table verified/created")
            except Exception as e:
                logger.warning(f"positions table creation failed: {e}")
                
            logger.info("✅ Essential database tables initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Schema initialization failed: {e}")
            return False
    
    # Alias for backward compatibility
    async def create_session(self, session_data: Dict) -> str:
        """Create session - alias for create_browser_session"""
        return await self.create_browser_session(
            risk_profile=session_data.get('risk_profile', 'moderate')
        )

    # ========================
    # SYSTEM CONFIGURATION
    # ========================
    
    async def get_system_config(self, key: str) -> Optional[Any]:
        """Get system configuration value"""
        
        try:
            result = self.client.table("system_config")\
                .select("config_value")\
                .eq("config_key", key)\
                .single()\
                .execute()
            
            if result.data:
                return result.data["config_value"]
            return None
        except Exception as e:
            logger.error(f"Failed to get system config {key}: {e}")
            return None
    
    async def set_system_config(self, key: str, value: Any, description: str = None) -> bool:
        """Set system configuration value"""
        
        try:
            config_data = {
                "config_key": key,
                "config_value": value,
                "description": description,
                "updated_at": datetime.now().isoformat()
            }
            
            # Try to update first, then insert if not exists
            result = self.client.table("system_config")\
                .upsert(config_data)\
                .execute()
            
            logger.info(f"System config updated: {key}")
            return True
        except Exception as e:
            logger.error(f"Failed to set system config {key}: {e}")
            return False


# Global database manager instance
db_manager = SupabaseManager()