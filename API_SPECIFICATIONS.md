# Neural Options Oracle++ - API Specifications

## API Overview

The Neural Options Oracle++ exposes a comprehensive RESTful API with WebSocket support for real-time updates. All APIs follow OpenAPI 3.0 specification and include authentication, rate limiting, and comprehensive error handling.

**Base URL**: `https://api.neural-oracle.com`  
**API Version**: `v1`  
**Authentication**: JWT Bearer Token  
**Rate Limiting**: 100 requests/minute per user  

## Authentication Endpoints

### POST /api/v1/auth/login
**Description**: User login with credentials

**Request Body**:
```json
{
  "email": "user@example.com",
  "password": "secure_password",
  "remember_me": false
}
```

**Response**:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600,
  "user": {
    "id": "user_123",
    "email": "user@example.com",
    "risk_profile": "moderate",
    "created_at": "2024-01-01T00:00:00Z"
  }
}
```

### POST /api/v1/auth/register
**Description**: User registration

**Request Body**:
```json
{
  "email": "user@example.com",
  "password": "secure_password",
  "risk_profile": "conservative", // "conservative", "moderate", "aggressive"
  "experience_level": "beginner"  // "beginner", "intermediate", "advanced"
}
```

### GET /api/v1/auth/profile
**Description**: Get current user profile  
**Authorization**: Required

**Response**:
```json
{
  "user": {
    "id": "user_123",
    "email": "user@example.com",
    "risk_profile": "moderate",
    "experience_level": "intermediate",
    "total_trades": 42,
    "success_rate": 0.67,
    "created_at": "2024-01-01T00:00:00Z"
  }
}
```

---

## Stock Analysis Endpoints

### POST /api/v1/analysis/stock/{symbol}
**Description**: Request comprehensive AI analysis for a stock  
**Authorization**: Required

**Path Parameters**:
- `symbol` (string): Stock symbol (e.g., "AAPL")

**Request Body**:
```json
{
  "timeframe": "1D",           // "5M", "15M", "1H", "1D"
  "analysis_depth": "full",    // "quick", "standard", "full"
  "include_education": true,
  "user_context": {
    "risk_tolerance": "moderate",
    "investment_horizon": "short_term",
    "learning_focus": ["technical_analysis", "options_basics"]
  }
}
```

**Response**:
```json
{
  "symbol": "AAPL",
  "analysis_id": "analysis_456",
  "timestamp": "2024-01-01T10:00:00Z",
  "market_scenario": "strong_uptrend",
  "agent_results": {
    "technical": {
      "scenario": "strong_uptrend",
      "indicators": {
        "ma": {"signal": 0.75, "ma_20": 150.25, "ma_50": 148.50},
        "rsi": {"value": 65.2, "signal": 0.3},
        "bb": {"upper": 152.0, "lower": 148.0, "signal": 0.6},
        "macd": {"value": 1.25, "signal": 0.8},
        "vwap": {"value": 149.75, "signal": 0.7}
      },
      "weights": {
        "ma": 0.30, "rsi": 0.15, "bb": 0.10, 
        "macd": 0.25, "vwap": 0.20
      },
      "weighted_score": 0.68,
      "confidence": 0.85
    },
    "sentiment": {
      "social_sentiment": 0.72,
      "news_sentiment": 0.68,
      "aggregate_sentiment": 0.70,
      "confidence": 0.78,
      "sources": {
        "stocktwits": {"bullish": 65, "bearish": 35},
        "reddit": {"sentiment_score": 0.74},
        "news": {"positive": 8, "negative": 2, "neutral": 5}
      }
    },
    "flow": {
      "put_call_ratio": 0.45,
      "unusual_volume": true,
      "gamma_exposure": 1250000,
      "ml_prediction": 0.73,
      "unusual_activity": [
        {
          "strike": 155.0,
          "expiration": "2024-01-19",
          "type": "call",
          "volume": 5000,
          "open_interest": 1200,
          "unusual_score": 0.92
        }
      ]
    },
    "history": {
      "pattern": "ascending_triangle",
      "pattern_confidence": 0.82,
      "similar_patterns": 15,
      "success_rate": 0.73,
      "pattern_score": 0.75
    }
  },
  "decision": {
    "overall_score": 0.71,
    "adjusted_weights": {
      "technical": 0.60,
      "sentiment": 0.10,
      "flow": 0.10,
      "history": 0.20
    },
    "signal": {
      "direction": "BUY",
      "strength": "STRONG",
      "confidence": 0.85,
      "reasoning": "Strong uptrend confirmed by technical indicators with positive sentiment and unusual call activity"
    }
  },
  "strike_recommendations": [
    {
      "rank": 1,
      "strike": 155.0,
      "expiration": "2024-01-19",
      "option_type": "call",
      "delta": 0.65,
      "probability_of_profit": 0.68,
      "max_profit": 500,
      "max_loss": 250,
      "risk_reward_ratio": 2.0,
      "cost": 250,
      "reasoning": "High delta call with good risk-reward for moderate risk profile"
    }
  ],
  "educational_content": {
    "key_concepts": ["technical_analysis", "options_delta", "risk_management"],
    "explanation": "This analysis shows a strong uptrend pattern...",
    "recommended_learning": [
      {
        "topic": "Understanding Delta",
        "importance": "high",
        "estimated_time": "10 minutes"
      }
    ]
  }
}
```

### GET /api/v1/analysis/history/{user_id}
**Description**: Get user's analysis history  
**Authorization**: Required

**Query Parameters**:
- `limit` (int, optional): Number of results (default: 50, max: 100)
- `offset` (int, optional): Pagination offset (default: 0)
- `symbol` (string, optional): Filter by symbol

**Response**:
```json
{
  "analyses": [
    {
      "analysis_id": "analysis_456",
      "symbol": "AAPL",
      "timestamp": "2024-01-01T10:00:00Z",
      "signal": "BUY",
      "confidence": 0.85,
      "outcome": "pending" // "pending", "profitable", "loss", "breakeven"
    }
  ],
  "total_count": 127,
  "has_more": true
}
```

---

## Trading Endpoints

### GET /api/v1/trading/positions
**Description**: Get all active positions  
**Authorization**: Required

**Response**:
```json
{
  "positions": [
    {
      "position_id": "pos_789",
      "symbol": "AAPL",
      "option_symbol": "AAPL240119C00155000",
      "option_type": "call",
      "strike": 155.0,
      "expiration": "2024-01-19",
      "quantity": 2,
      "entry_price": 2.50,
      "current_price": 3.20,
      "unrealized_pnl": 140.0,
      "unrealized_pnl_percent": 28.0,
      "greeks": {
        "delta": 0.65,
        "gamma": 0.05,
        "theta": -0.03,
        "vega": 0.12
      },
      "entry_date": "2024-01-01T10:30:00Z",
      "days_to_expiration": 18
    }
  ],
  "portfolio_summary": {
    "total_positions": 3,
    "total_value": 1250.0,
    "total_pnl": 180.0,
    "total_pnl_percent": 16.8,
    "portfolio_greeks": {
      "delta": 1.23,
      "gamma": 0.15,
      "theta": -0.08,
      "vega": 0.35
    }
  }
}
```

### POST /api/v1/trading/execute
**Description**: Execute trading recommendation  
**Authorization**: Required

**Request Body**:
```json
{
  "symbol": "AAPL",
  "recommendation_id": "rec_123",
  "action": "buy",              // "buy", "sell"
  "option_type": "call",        // "call", "put"
  "strike": 155.0,
  "expiration": "2024-01-19",
  "quantity": 2,
  "order_type": "market",       // "market", "limit"
  "limit_price": 2.60,          // required for limit orders
  "educational_mode": true      // adds educational explanations
}
```

**Response**:
```json
{
  "execution_id": "exec_321",
  "order_id": "order_654",
  "status": "filled",           // "pending", "filled", "cancelled", "rejected"
  "symbol": "AAPL",
  "option_symbol": "AAPL240119C00155000",
  "quantity": 2,
  "fill_price": 2.55,
  "total_cost": 510.0,
  "commission": 2.0,
  "timestamp": "2024-01-01T10:35:00Z",
  "position": {
    "position_id": "pos_789",
    "entry_price": 2.55,
    "greeks": {
      "delta": 0.65,
      "gamma": 0.05,
      "theta": -0.03,
      "vega": 0.12
    }
  },
  "educational_explanation": {
    "strategy_explanation": "You bought 2 call options at $155 strike...",
    "risk_analysis": "Maximum loss is $510 if AAPL stays below $155...",
    "profit_scenarios": "Break-even at $157.55, max profit unlimited above..."
  }
}
```

### DELETE /api/v1/trading/positions/{position_id}
**Description**: Close a position  
**Authorization**: Required

**Path Parameters**:
- `position_id` (string): Position ID to close

**Response**:
```json
{
  "position_id": "pos_789",
  "close_order_id": "order_987",
  "close_price": 3.20,
  "quantity_closed": 2,
  "realized_pnl": 130.0,
  "realized_pnl_percent": 25.5,
  "close_timestamp": "2024-01-05T14:20:00Z",
  "educational_summary": {
    "trade_outcome": "profitable",
    "key_lessons": [
      "Technical analysis correctly identified uptrend",
      "Position sizing appropriate for risk tolerance"
    ],
    "areas_for_improvement": ["Consider taking partial profits earlier"]
  }
}
```

---

## Educational Endpoints

### GET /api/v1/education/lessons
**Description**: Get personalized educational content  
**Authorization**: Required

**Query Parameters**:
- `topic` (string, optional): Filter by topic
- `difficulty` (string, optional): "beginner", "intermediate", "advanced"
- `based_on_trades` (boolean, optional): Generate based on recent trades

**Response**:
```json
{
  "lessons": [
    {
      "lesson_id": "lesson_101",
      "title": "Understanding Options Delta",
      "topic": "options_greeks",
      "difficulty": "beginner",
      "estimated_time_minutes": 15,
      "content": {
        "introduction": "Delta measures how much an option's price changes...",
        "key_points": [
          "Delta ranges from 0 to 1 for calls, 0 to -1 for puts",
          "Higher delta = more sensitive to stock price changes"
        ],
        "examples": [
          {
            "scenario": "AAPL at $150, call delta 0.50",
            "explanation": "If AAPL rises $1, call price rises ~$0.50"
          }
        ],
        "interactive_elements": [
          {
            "type": "calculator",
            "description": "Delta calculator tool",
            "config": {"stock_price": 150, "strike": 155}
          }
        ]
      },
      "quiz": {
        "questions": [
          {
            "question": "What is the delta of an at-the-money call option typically?",
            "options": ["0.25", "0.50", "0.75", "1.00"],
            "correct_answer": 1,
            "explanation": "ATM calls typically have delta around 0.50"
          }
        ]
      }
    }
  ],
  "personalized_recommendations": [
    {
      "lesson_id": "lesson_102",
      "reason": "Based on your recent AAPL trade",
      "priority": "high"
    }
  ]
}
```

### POST /api/v1/education/quiz/{lesson_id}/submit
**Description**: Submit quiz answers  
**Authorization**: Required

**Request Body**:
```json
{
  "answers": [
    {"question_id": "q1", "selected_answer": 1},
    {"question_id": "q2", "selected_answer": 0}
  ]
}
```

**Response**:
```json
{
  "quiz_result": {
    "score": 0.75,
    "passed": true,
    "total_questions": 4,
    "correct_answers": 3,
    "detailed_results": [
      {
        "question_id": "q1",
        "correct": true,
        "explanation": "Correct! ATM calls typically have delta around 0.50"
      }
    ],
    "recommended_next_steps": [
      {
        "action": "review",
        "lesson_id": "lesson_103",
        "reason": "Strengthen understanding of gamma"
      }
    ]
  },
  "progress_update": {
    "lesson_completion": 100,
    "topic_mastery": {
      "options_greeks": 0.65,
      "technical_analysis": 0.80
    }
  }
}
```

### GET /api/v1/education/progress
**Description**: Get learning progress  
**Authorization**: Required

**Response**:
```json
{
  "overall_progress": {
    "lessons_completed": 23,
    "total_lessons": 45,
    "completion_percentage": 51.1,
    "streak_days": 7,
    "total_study_time_minutes": 420
  },
  "topic_progress": [
    {
      "topic": "options_basics",
      "mastery_level": 0.85,
      "lessons_completed": 8,
      "total_lessons": 10,
      "recent_activity": "2024-01-01T09:00:00Z"
    },
    {
      "topic": "technical_analysis",
      "mastery_level": 0.72,
      "lessons_completed": 12,
      "total_lessons": 15,
      "recent_activity": "2024-01-01T14:30:00Z"
    }
  ],
  "achievements": [
    {
      "achievement_id": "first_profitable_trade",
      "title": "First Profit",
      "description": "Completed your first profitable trade",
      "unlocked_at": "2024-01-01T15:00:00Z",
      "badge_icon": "trophy"
    }
  ],
  "next_recommendations": [
    {
      "lesson_id": "lesson_201",
      "title": "Advanced Technical Patterns",
      "priority": "high",
      "reason": "Based on your strong technical analysis performance"
    }
  ]
}
```

---

## WebSocket API

### Connection Endpoint
**URL**: `wss://api.neural-oracle.com/ws`  
**Authentication**: JWT token in query parameter: `?token=<jwt_token>`

### Message Types

#### Subscribe to Real-time Data
```json
{
  "type": "subscribe",
  "channels": ["market_data", "positions", "analysis"],
  "symbols": ["AAPL", "MSFT"]
}
```

#### Market Data Updates
```json
{
  "type": "market_data",
  "symbol": "AAPL",
  "data": {
    "price": 150.25,
    "change": 1.50,
    "change_percent": 1.01,
    "volume": 1250000,
    "timestamp": "2024-01-01T10:00:00Z"
  }
}
```

#### Position Updates
```json
{
  "type": "position_update",
  "position_id": "pos_789",
  "data": {
    "current_price": 3.20,
    "unrealized_pnl": 140.0,
    "greeks": {
      "delta": 0.65,
      "gamma": 0.05,
      "theta": -0.03,
      "vega": 0.12
    },
    "timestamp": "2024-01-01T10:00:00Z"
  }
}
```

#### Analysis Completion
```json
{
  "type": "analysis_complete",
  "analysis_id": "analysis_456",
  "symbol": "AAPL",
  "signal": {
    "direction": "BUY",
    "confidence": 0.85,
    "reasoning": "Strong technical setup with bullish sentiment"
  }
}
```

---

## Error Responses

All endpoints follow consistent error response format:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid request parameters",
    "details": [
      {
        "field": "symbol",
        "message": "Symbol is required"
      }
    ],
    "request_id": "req_123456",
    "timestamp": "2024-01-01T10:00:00Z"
  }
}
```

### Error Codes
- `AUTHENTICATION_ERROR` (401): Invalid or expired token
- `AUTHORIZATION_ERROR` (403): Insufficient permissions
- `VALIDATION_ERROR` (400): Invalid request parameters
- `NOT_FOUND` (404): Resource not found
- `RATE_LIMIT_EXCEEDED` (429): Rate limit exceeded
- `INTERNAL_ERROR` (500): Server error
- `SERVICE_UNAVAILABLE` (503): External service unavailable

---

## Rate Limiting

**Default Limits**:
- 100 requests per minute per user
- 1000 requests per hour per user
- 5 concurrent WebSocket connections per user

**Headers**:
```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1704067200
```

---

## API Versioning

The API uses URL path versioning (`/api/v1/`). When breaking changes are introduced, a new version will be released. Previous versions will be supported for at least 12 months.

**Version History**:
- `v1.0` - Initial release (2024-01-01)

This comprehensive API specification ensures clear communication between the frontend and backend services while providing educational value and comprehensive trading functionality.