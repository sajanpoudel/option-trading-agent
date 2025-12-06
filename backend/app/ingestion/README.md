# Lambda Architecture - Data Ingestion Layer

## Overview

This ingestion layer implements the **Lambda Architecture** for real-time and batch processing of financial market data. It combines:

- **Speed Layer**: Apache Kafka for real-time event streaming
- **Batch Layer**: Dask distributed computing for large-scale historical analysis
- **Serving Layer**: Unified interface merging real-time and batch views

```
┌─────────────────────────────────────────────────────────┐
│              SPEED LAYER (Kafka)                        │
│  ┌─────────┐  ┌─────────┐  ┌──────────────┐            │
│  │ market- │  │ options-│  │  sentiment-  │            │
│  │  ticks  │  │  flow   │  │    events    │            │
│  └────┬────┘  └────┬────┘  └──────┬───────┘            │
└───────┼────────────┼──────────────┼────────────────────┘
        ▼            ▼              ▼
┌─────────────────────────────────────────────────────────┐
│           SERVING LAYER (Unified View)                  │
│      ┌──────────────────────────────────┐               │
│      │     IngestionManager              │               │
│      │  - Merges real-time + batch      │               │
│      │  - Exposes unified query API     │               │
│      └──────────────────────────────────┘               │
└─────────────────────────────────────────────────────────┘
        ▲            ▲              ▲
┌───────┼────────────┼──────────────┼────────────────────┐
│  ┌────┴────┐  ┌────┴────┐  ┌──────┴───────┐            │
│  │Historical│  │  Flow   │  │  Sentiment   │            │
│  │ Features │  │Aggregates│  │  Aggregates  │            │
│  └─────────┘  └─────────┘  └──────────────┘            │
│              BATCH LAYER (Dask)                         │
└─────────────────────────────────────────────────────────┘
```

## Features

### Speed Layer (Real-time)
- **Kafka Producer**: Publishes market events with compression and batching
- **Kafka Consumer**: Processes events with configurable message handlers
- **Real-time Streams**:
  - Market ticks (price/volume updates)
  - Options flow (unusual activity detection)
  - Sentiment events (news/social media analysis)

### Batch Layer (Historical)
- **Dask Cluster**: Distributed computing for large datasets
- **Batch Tasks**:
  - Historical feature calculation
  - Options flow aggregation
  - Sentiment trend analysis
- **Parallel Processing**: Process multiple symbols concurrently

### Serving Layer
- **Unified Manager**: Single interface to query both layers
- **Graceful Degradation**: Works without Kafka/Dask
- **Health Monitoring**: Status endpoints for all components

## Architecture

### Directory Structure
```
backend/app/ingestion/
├── __init__.py
├── config.py              # Configuration (env-driven)
├── manager.py             # IngestionManager - unified interface
├── kafka/
│   ├── __init__.py
│   ├── producer.py        # Kafka event publishing
│   ├── consumer.py        # Kafka event processing
│   └── topics.py          # Topic definitions and schemas
├── dask/
│   ├── __init__.py
│   ├── cluster.py         # Dask cluster management
│   └── tasks.py           # Batch processing jobs
└── streams/
    ├── __init__.py
    ├── market_stream.py   # Real-time market data
    ├── options_stream.py  # Real-time options flow
    └── sentiment_stream.py# Real-time sentiment
```

### Key Components

#### 1. Kafka Topics (Speed Layer)
| Topic | Purpose | Retention |
|-------|---------|-----------|
| `market-ticks` | Real-time price/volume updates | 1 hour |
| `options-flow` | Unusual options activity | 24 hours |
| `sentiment-events` | News/social sentiment | 24 hours |
| `technical-signals` | Technical indicator events | 2 hours |

#### 2. Data Streams
- **MarketDataStream**: Connects to Alpaca WebSocket for live quotes
- **OptionsFlowStream**: Detects unusual options volume/OI
- **SentimentStream**: Monitors news APIs and social media

#### 3. Batch Jobs (Dask)
- `calculate_historical_features()` - Technical indicators over time
- `aggregate_options_flow()` - Flow metrics by time window
- `compute_sentiment_trends()` - Multi-day sentiment analysis

## Usage

### Basic Setup (Without Ingestion)
```bash
# Run without Kafka/Dask - gracefully degrades
docker-compose up
```

### With Ingestion Layer
```bash
# Start with Kafka + Dask
docker-compose --profile ingestion up

# Or enable via environment variables
KAFKA_ENABLED=true DASK_ENABLED=true docker-compose up
```

### Environment Variables
```bash
# Kafka Configuration
KAFKA_ENABLED=true
KAFKA_BOOTSTRAP_SERVERS=kafka:9092
KAFKA_CONSUMER_GROUP=neural-oracle

# Dask Configuration
DASK_ENABLED=true
DASK_SCHEDULER_ADDRESS=tcp://dask-scheduler:8786
DASK_N_WORKERS=2
```

### API Endpoints

#### Get Ingestion Status
```bash
GET /api/v1/system/ingestion/status

Response:
{
  "ingestion_layer": {
    "is_running": true,
    "uptime_seconds": 3600,
    "kafka": {
      "enabled": true,
      "producer": {
        "is_running": true,
        "total_messages": 15420
      },
      "consumer": {
        "is_running": true,
        "message_counts": {
          "market-ticks": 12000,
          "options-flow": 2000,
          "sentiment-events": 1420
        }
      }
    },
    "dask": {
      "enabled": true,
      "cluster": {
        "connected": true,
        "workers": 2,
        "cores": 8,
        "jobs_submitted": 45
      }
    },
    "streams": {
      "market": {
        "is_streaming": true,
        "subscribed_symbols": ["SPY", "QQQ", "AAPL", "TSLA", "NVDA"]
      },
      "options": {
        "is_streaming": true,
        "total_flows_detected": 234
      },
      "sentiment": {
        "is_streaming": true,
        "total_events": 512
      }
    }
  }
}
```

#### Health Check
```bash
GET /api/v1/system/ingestion/health

Response:
{
  "healthy": true,
  "components": {
    "kafka_producer": true,
    "kafka_consumer": true,
    "dask_cluster": true,
    "market_stream": true,
    "options_stream": true,
    "sentiment_stream": true
  },
  "timestamp": "2025-12-05T12:00:00"
}
```

### Programmatic Usage

#### Subscribe to Real-time Data
```python
from backend.app.ingestion import ingestion_manager

# Subscribe a symbol to all streams
await ingestion_manager.subscribe_symbol("AAPL")

# Check if real-time data available
is_streaming = ingestion_manager.is_streaming("AAPL")

# Get latest tick (from Kafka buffer)
tick = ingestion_manager.get_latest_tick("AAPL")
```

#### Run Batch Job
```python
# Process multiple symbols in parallel on Dask cluster
results = await ingestion_manager.run_batch_job(
    job_type='features',
    symbols=['SPY', 'QQQ', 'AAPL', 'TSLA']
)
```

## Infrastructure

### Docker Services

#### Kafka + Zookeeper
```yaml
zookeeper:
  image: confluentinc/cp-zookeeper:7.5.0
  ports: ["2181:2181"]

kafka:
  image: confluentinc/cp-kafka:7.5.0
  ports: ["9092:9092"]
  depends_on: [zookeeper]
```

#### Dask Cluster
```yaml
dask-scheduler:
  image: ghcr.io/dask/dask:latest
  ports:
    - "8786:8786"  # Scheduler
    - "8787:8787"  # Dashboard

dask-worker:
  image: ghcr.io/dask/dask:latest
  replicas: 2
  memory: 2GB per worker
```

### Monitoring

#### Dask Dashboard
```
http://localhost:8787
```
- View cluster status
- Monitor task execution
- Track resource usage

#### Application Metrics
- Kafka message throughput
- Stream subscription counts
- Batch job statistics
- Component health status

## Performance

### Throughput
- **Kafka**: 10,000+ messages/sec
- **Dask**: Scales horizontally with workers
- **Streams**: 1 update/sec per symbol (configurable)

### Latency
- **Speed Layer**: <100ms event propagation
- **Batch Layer**: Minutes to hours (depending on job)
- **Serving Layer**: Sub-second query response

### Resource Usage
- **Kafka**: ~512MB RAM
- **Dask Worker**: 2GB RAM per worker (configurable)
- **Total (with 2 workers)**: ~5GB RAM

## Best Practices

### For Big Data Class Project

1. **Demonstrate Lambda Architecture**:
   - Show real-time processing (Kafka)
   - Show batch processing (Dask)
   - Show serving layer merging both views

2. **Scalability**:
   - Add more Dask workers: `docker-compose up --scale dask-worker=4`
   - Kafka partitioning for parallel consumption
   - Horizontal scaling architecture

3. **Fault Tolerance**:
   - Kafka message persistence (configurable retention)
   - Dask automatic task retry
   - Graceful degradation without infrastructure

4. **Monitoring**:
   - Use `/ingestion/status` endpoint
   - View Dask dashboard at `:8787`
   - Track message counts and processing rates

## Extension Ideas

### For Advanced Projects

1. **Add More Data Sources**:
   - Twitter API for social sentiment
   - News APIs (NewsAPI, Alpha Vantage)
   - Additional market data providers

2. **Machine Learning Pipeline**:
   - Train models on batch data (Dask)
   - Deploy models to speed layer (Kafka Streams)
   - Online learning from real-time events

3. **Time-series Database**:
   - Store processed events in TimescaleDB/InfluxDB
   - Efficient historical queries
   - Downsampling and aggregation

4. **Advanced Analytics**:
   - Sliding window aggregations
   - Pattern detection in streams
   - Anomaly detection using ML models

## Troubleshooting

### Kafka Not Starting
```bash
# Check Zookeeper is running
docker logs neural_oracle_zookeeper

# Check Kafka logs
docker logs neural_oracle_kafka

# Verify network connectivity
docker network inspect option-trading-agent_neural_oracle_network
```

### Dask Cluster Issues
```bash
# Check scheduler logs
docker logs neural_oracle_dask_scheduler

# Check worker logs
docker logs neural_oracle_dask_worker

# Access Dask dashboard
curl http://localhost:8787
```

### Application Not Connecting
```bash
# Verify environment variables
docker exec neural_oracle_api env | grep KAFKA
docker exec neural_oracle_api env | grep DASK

# Check ingestion health
curl http://localhost:8080/api/v1/system/ingestion/health
```

## References

- [Lambda Architecture](http://lambda-architecture.net/)
- [Apache Kafka](https://kafka.apache.org/documentation/)
- [Dask Distributed](https://distributed.dask.org/)
- [aiokafka Documentation](https://aiokafka.readthedocs.io/)

## License

MIT License - See parent project LICENSE file
