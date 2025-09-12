# Neural Options Oracle++ - Docker Deployment Guide

## Docker Architecture Overview

The Neural Options Oracle++ system uses a microservices architecture deployed with Docker containers. Each service runs in its own container with proper networking, volume management, and environment configuration.

## Container Architecture

```mermaid
graph TB
    subgraph "Docker Network"
        A[Nginx Proxy :80/443]
        B[API Gateway :8080]
        C[Agent Orchestrator :8081]
        D[Market Data :8082]
        E[Decision Engine :8083]
        F[Trading Execution :8084]
        G[Educational :8085]
        H[Frontend :3000]
        I[Redis :6379]
        J[PostgreSQL :5432]
        K[InfluxDB :8086]
    end
    
    A --> B
    B --> C
    B --> D
    B --> E
    B --> F
    B --> G
    B --> H
    C --> I
    D --> I
    D --> K
    E --> J
    F --> J
    G --> J
```

## Docker Compose Configuration

### Development Environment

```yaml
# docker-compose.dev.yml
version: '3.8'

services:
  # Reverse Proxy
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.dev.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - api-gateway
      - frontend
    networks:
      - oracle-network

  # API Gateway
  api-gateway:
    build: 
      context: ./services/api-gateway
      dockerfile: Dockerfile
    ports:
      - "8080:8080"
    environment:
      - ENV=development
      - DEBUG=true
      - REDIS_URL=redis://redis:6379/0
      - DATABASE_URL=postgresql://oracle_user:oracle_pass@postgres:5432/oracle_dev
      - JWT_SECRET=${JWT_SECRET}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    depends_on:
      - redis
      - postgres
      - agent-orchestrator
      - decision-engine
      - trading-execution
      - educational
    volumes:
      - ./services/api-gateway:/app
      - /app/node_modules
    networks:
      - oracle-network
    restart: unless-stopped

  # Agent Orchestrator
  agent-orchestrator:
    build: 
      context: ./services/agent-orchestrator
      dockerfile: Dockerfile
    ports:
      - "8081:8081"
    environment:
      - ENV=development
      - REDIS_URL=redis://redis:6379/1
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - GEMINI_API_KEY=${GEMINI_API_KEY}
      - JIGSAWSTACK_API_KEY=${JIGSAWSTACK_API_KEY}
    depends_on:
      - redis
    volumes:
      - ./services/agent-orchestrator:/app
      - agent-logs:/app/logs
    networks:
      - oracle-network
    restart: unless-stopped

  # Market Data Service
  market-data:
    build: 
      context: ./services/market-data
      dockerfile: Dockerfile
    ports:
      - "8082:8082"
    environment:
      - ENV=development
      - REDIS_URL=redis://redis:6379/2
      - INFLUXDB_URL=http://influxdb:8086
      - INFLUXDB_TOKEN=${INFLUXDB_TOKEN}
      - ALPACA_API_KEY=${ALPACA_API_KEY}
      - ALPACA_SECRET_KEY=${ALPACA_SECRET_KEY}
      - STOCKTWITS_API_KEY=${STOCKTWITS_API_KEY}
      - JIGSAWSTACK_API_KEY=${JIGSAWSTACK_API_KEY}
    depends_on:
      - redis
      - influxdb
    volumes:
      - ./services/market-data:/app
      - market-data-cache:/app/cache
    networks:
      - oracle-network
    restart: unless-stopped

  # Decision Engine
  decision-engine:
    build: 
      context: ./services/decision-engine
      dockerfile: Dockerfile
    ports:
      - "8083:8083"
    environment:
      - ENV=development
      - DATABASE_URL=postgresql://oracle_user:oracle_pass@postgres:5432/oracle_dev
      - REDIS_URL=redis://redis:6379/3
    depends_on:
      - postgres
      - redis
    volumes:
      - ./services/decision-engine:/app
      - decision-models:/app/models
    networks:
      - oracle-network
    restart: unless-stopped

  # Trading Execution
  trading-execution:
    build: 
      context: ./services/trading-execution
      dockerfile: Dockerfile
    ports:
      - "8084:8084"
    environment:
      - ENV=development
      - DATABASE_URL=postgresql://oracle_user:oracle_pass@postgres:5432/oracle_dev
      - REDIS_URL=redis://redis:6379/4
      - ALPACA_API_KEY=${ALPACA_API_KEY}
      - ALPACA_SECRET_KEY=${ALPACA_SECRET_KEY}
      - ALPACA_BASE_URL=https://paper-api.alpaca.markets
    depends_on:
      - postgres
      - redis
    volumes:
      - ./services/trading-execution:/app
    networks:
      - oracle-network
    restart: unless-stopped

  # Educational Service
  educational:
    build: 
      context: ./services/educational
      dockerfile: Dockerfile
    ports:
      - "8085:8085"
    environment:
      - ENV=development
      - DATABASE_URL=postgresql://oracle_user:oracle_pass@postgres:5432/oracle_dev
      - REDIS_URL=redis://redis:6379/5
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    depends_on:
      - postgres
      - redis
    volumes:
      - ./services/educational:/app
      - educational-content:/app/content
    networks:
      - oracle-network
    restart: unless-stopped

  # Frontend
  frontend:
    build: 
      context: ./frontend
      dockerfile: Dockerfile.dev
    ports:
      - "3000:3000"
    environment:
      - NODE_ENV=development
      - NEXT_PUBLIC_API_URL=http://localhost:8080
      - NEXT_PUBLIC_WS_URL=ws://localhost:8080/ws
    volumes:
      - ./frontend:/app
      - /app/node_modules
      - /app/.next
    networks:
      - oracle-network
    restart: unless-stopped

  # Redis Cache
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    command: >
      redis-server 
      --appendonly yes 
      --appendfsync everysec 
      --maxmemory 512mb 
      --maxmemory-policy allkeys-lru
    volumes:
      - redis-data:/data
      - ./redis/redis.conf:/etc/redis/redis.conf
    networks:
      - oracle-network
    restart: unless-stopped

  # PostgreSQL Database
  postgres:
    image: postgres:15-alpine
    ports:
      - "5432:5432"
    environment:
      POSTGRES_DB: oracle_dev
      POSTGRES_USER: oracle_user
      POSTGRES_PASSWORD: oracle_pass
      POSTGRES_INITDB_ARGS: "--encoding=UTF8 --lc-collate=C --lc-ctype=C"
    volumes:
      - postgres-data:/var/lib/postgresql/data
      - ./database/init:/docker-entrypoint-initdb.d
      - ./database/schemas:/schemas
    networks:
      - oracle-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U oracle_user -d oracle_dev"]
      interval: 30s
      timeout: 10s
      retries: 5

  # InfluxDB Time Series Database
  influxdb:
    image: influxdb:2.7-alpine
    ports:
      - "8086:8086"
    environment:
      DOCKER_INFLUXDB_INIT_MODE: setup
      DOCKER_INFLUXDB_INIT_USERNAME: oracle_admin
      DOCKER_INFLUXDB_INIT_PASSWORD: oracle_influx_pass
      DOCKER_INFLUXDB_INIT_ORG: neural-oracle
      DOCKER_INFLUXDB_INIT_BUCKET: market-data
      DOCKER_INFLUXDB_INIT_ADMIN_TOKEN: ${INFLUXDB_TOKEN}
    volumes:
      - influxdb-data:/var/lib/influxdb2
      - influxdb-config:/etc/influxdb2
    networks:
      - oracle-network
    restart: unless-stopped

volumes:
  postgres-data:
  redis-data:
  influxdb-data:
  influxdb-config:
  agent-logs:
  market-data-cache:
  decision-models:
  educational-content:

networks:
  oracle-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
```

### Production Environment

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  # Reverse Proxy with SSL
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.prod.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
      - nginx-cache:/var/cache/nginx
    depends_on:
      - api-gateway
      - frontend
    networks:
      - oracle-network
    restart: always
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

  # API Gateway
  api-gateway:
    build: 
      context: ./services/api-gateway
      dockerfile: Dockerfile.prod
    expose:
      - "8080"
    environment:
      - ENV=production
      - DEBUG=false
      - REDIS_URL=redis://redis:6379/0
      - DATABASE_URL=postgresql://oracle_user:${POSTGRES_PASSWORD}@postgres:5432/oracle_prod
      - JWT_SECRET=${JWT_SECRET}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    depends_on:
      - redis
      - postgres
      - agent-orchestrator
      - decision-engine
      - trading-execution
      - educational
    networks:
      - oracle-network
    restart: always
    deploy:
      replicas: 2
      resources:
        limits:
          cpus: '0.5'
          memory: 512M
        reservations:
          cpus: '0.25'
          memory: 256M

  # Agent Orchestrator
  agent-orchestrator:
    build: 
      context: ./services/agent-orchestrator
      dockerfile: Dockerfile.prod
    expose:
      - "8081"
    environment:
      - ENV=production
      - REDIS_URL=redis://redis:6379/1
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - GEMINI_API_KEY=${GEMINI_API_KEY}
      - JIGSAWSTACK_API_KEY=${JIGSAWSTACK_API_KEY}
    depends_on:
      - redis
    volumes:
      - agent-logs:/app/logs
    networks:
      - oracle-network
    restart: always
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 1G
        reservations:
          cpus: '0.5'
          memory: 512M

  # Market Data Service
  market-data:
    build: 
      context: ./services/market-data
      dockerfile: Dockerfile.prod
    expose:
      - "8082"
    environment:
      - ENV=production
      - REDIS_URL=redis://redis:6379/2
      - INFLUXDB_URL=http://influxdb:8086
      - INFLUXDB_TOKEN=${INFLUXDB_TOKEN}
      - ALPACA_API_KEY=${ALPACA_API_KEY}
      - ALPACA_SECRET_KEY=${ALPACA_SECRET_KEY}
      - STOCKTWITS_API_KEY=${STOCKTWITS_API_KEY}
      - JIGSAWSTACK_API_KEY=${JIGSAWSTACK_API_KEY}
    depends_on:
      - redis
      - influxdb
    volumes:
      - market-data-cache:/app/cache
    networks:
      - oracle-network
    restart: always

  # Additional services configured similarly...

  # Production Redis with persistence
  redis:
    image: redis:7-alpine
    expose:
      - "6379"
    command: >
      redis-server 
      --appendonly yes 
      --appendfsync everysec 
      --maxmemory 2gb 
      --maxmemory-policy allkeys-lru
      --save 900 1
      --save 300 10
      --save 60 10000
    volumes:
      - redis-data:/data
      - ./redis/redis.prod.conf:/etc/redis/redis.conf
    networks:
      - oracle-network
    restart: always
    deploy:
      resources:
        limits:
          cpus: '0.5'
          memory: 2G
        reservations:
          cpus: '0.25'
          memory: 1G

  # Production PostgreSQL with backups
  postgres:
    image: postgres:15-alpine
    expose:
      - "5432"
    environment:
      POSTGRES_DB: oracle_prod
      POSTGRES_USER: oracle_user
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres-data:/var/lib/postgresql/data
      - postgres-backups:/backups
      - ./database/init:/docker-entrypoint-initdb.d
    networks:
      - oracle-network
    restart: always
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 2G
        reservations:
          cpus: '0.5'
          memory: 1G

volumes:
  postgres-data:
  postgres-backups:
  redis-data:
  influxdb-data:
  influxdb-config:
  nginx-cache:
  agent-logs:
  market-data-cache:
  decision-models:
  educational-content:

networks:
  oracle-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.21.0.0/16
```

## Individual Service Dockerfiles

### API Gateway Dockerfile

```dockerfile
# services/api-gateway/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY static/ ./static/

# Create non-root user
RUN groupadd -r oracle && useradd -r -g oracle oracle
RUN chown -R oracle:oracle /app
USER oracle

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Start application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "4"]
```

### Agent Orchestrator Dockerfile

```dockerfile
# services/agent-orchestrator/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for AI libraries
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY models/ ./models/
COPY config/ ./config/

# Create logs directory
RUN mkdir -p /app/logs

# Create non-root user
RUN groupadd -r oracle && useradd -r -g oracle oracle
RUN chown -R oracle:oracle /app
USER oracle

# Expose port
EXPOSE 8081

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8081/health')" || exit 1

# Start application
CMD ["python", "-m", "app.main"]
```

### Frontend Dockerfile (Development)

```dockerfile
# frontend/Dockerfile.dev
FROM node:18-alpine

WORKDIR /app

# Copy package files
COPY package.json package-lock.json ./

# Install dependencies
RUN npm ci

# Copy source code
COPY . .

# Create next.js user
RUN addgroup -g 1001 -S nodejs
RUN adduser -S nextjs -u 1001

# Change ownership
RUN chown -R nextjs:nodejs /app
USER nextjs

# Expose port
EXPOSE 3000

# Start development server
CMD ["npm", "run", "dev"]
```

### Frontend Dockerfile (Production)

```dockerfile
# frontend/Dockerfile.prod
FROM node:18-alpine AS builder

WORKDIR /app

# Copy package files
COPY package.json package-lock.json ./

# Install dependencies
RUN npm ci --only=production

# Copy source code
COPY . .

# Build application
RUN npm run build

# Production image
FROM node:18-alpine AS runner

WORKDIR /app

# Create nextjs user
RUN addgroup -g 1001 -S nodejs
RUN adduser -S nextjs -u 1001

# Copy built application
COPY --from=builder --chown=nextjs:nodejs /app/.next ./.next
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package.json ./package.json

USER nextjs

EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:3000/api/health || exit 1

CMD ["npm", "start"]
```

## Nginx Configuration

### Development Nginx Config

```nginx
# nginx/nginx.dev.conf
events {
    worker_connections 1024;
}

http {
    upstream api_gateway {
        server api-gateway:8080;
    }
    
    upstream frontend {
        server frontend:3000;
    }

    server {
        listen 80;
        server_name localhost;

        # API routes
        location /api/ {
            proxy_pass http://api_gateway;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection 'upgrade';
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_cache_bypass $http_upgrade;
        }

        # WebSocket routes
        location /ws {
            proxy_pass http://api_gateway;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
        }

        # Frontend routes
        location / {
            proxy_pass http://frontend;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection 'upgrade';
            proxy_set_header Host $host;
            proxy_cache_bypass $http_upgrade;
        }
    }
}
```

### Production Nginx Config

```nginx
# nginx/nginx.prod.conf
events {
    worker_connections 2048;
    use epoll;
    multi_accept on;
}

http {
    include       /etc/nginx/mime.types;
    default_type  application/octet-stream;

    # Logging
    log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                   '$status $body_bytes_sent "$http_referer" '
                   '"$http_user_agent" "$http_x_forwarded_for"';

    access_log /var/log/nginx/access.log main;
    error_log /var/log/nginx/error.log warn;

    # Performance optimizations
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    types_hash_max_size 2048;

    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 10240;
    gzip_proxied expired no-cache no-store private must-revalidate auth;
    gzip_types
        text/plain
        text/css
        text/xml
        text/javascript
        application/x-javascript
        application/xml+rss
        application/javascript
        application/json;

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=auth:10m rate=5r/s;

    # Upstream servers
    upstream api_gateway {
        least_conn;
        server api-gateway:8080 max_fails=3 fail_timeout=30s;
        keepalive 32;
    }
    
    upstream frontend {
        server frontend:3000 max_fails=3 fail_timeout=30s;
    }

    # HTTPS redirect
    server {
        listen 80;
        server_name api.neural-oracle.com;
        return 301 https://$server_name$request_uri;
    }

    # Main HTTPS server
    server {
        listen 443 ssl http2;
        server_name api.neural-oracle.com;

        # SSL configuration
        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384;
        ssl_prefer_server_ciphers off;
        ssl_session_cache shared:SSL:10m;
        ssl_session_timeout 10m;

        # Security headers
        add_header X-Frame-Options DENY;
        add_header X-Content-Type-Options nosniff;
        add_header X-XSS-Protection "1; mode=block";
        add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload";

        # API routes with rate limiting
        location /api/v1/auth/ {
            limit_req zone=auth burst=10 nodelay;
            proxy_pass http://api_gateway;
            include /etc/nginx/proxy_params;
        }

        location /api/ {
            limit_req zone=api burst=20 nodelay;
            proxy_pass http://api_gateway;
            include /etc/nginx/proxy_params;
        }

        # WebSocket routes
        location /ws {
            proxy_pass http://api_gateway;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            include /etc/nginx/proxy_params;
        }

        # Static assets caching
        location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg|woff|woff2)$ {
            proxy_pass http://frontend;
            expires 1y;
            add_header Cache-Control "public, immutable";
        }

        # Frontend routes
        location / {
            proxy_pass http://frontend;
            include /etc/nginx/proxy_params;
        }
    }
}
```

## Environment Variables

### .env.example

```bash
# Database
POSTGRES_PASSWORD=secure_postgres_password
DATABASE_URL=postgresql://oracle_user:secure_postgres_password@postgres:5432/oracle_prod

# Redis
REDIS_URL=redis://redis:6379

# InfluxDB
INFLUXDB_TOKEN=secure_influxdb_token

# JWT
JWT_SECRET=super_secure_jwt_secret_key_min_32_chars

# OpenAI
OPENAI_API_KEY=sk-your-openai-api-key

# Gemini
GEMINI_API_KEY=your-gemini-api-key

# Alpaca (Paper Trading)
ALPACA_API_KEY=your-alpaca-api-key
ALPACA_SECRET_KEY=your-alpaca-secret-key
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# JigsawStack
JIGSAWSTACK_API_KEY=your-jigsawstack-api-key

# StockTwits
STOCKTWITS_API_KEY=your-stocktwits-api-key

# Monitoring
SENTRY_DSN=your-sentry-dsn
LOG_LEVEL=INFO
```

## Deployment Commands

### Development Setup

```bash
# Clone repository
git clone <repository-url>
cd neural-options-oracle

# Copy environment file
cp .env.example .env
# Edit .env with your API keys

# Build and start development environment
docker-compose -f docker-compose.dev.yml up --build -d

# View logs
docker-compose -f docker-compose.dev.yml logs -f

# Stop services
docker-compose -f docker-compose.dev.yml down
```

### Production Deployment

```bash
# Production deployment
docker-compose -f docker-compose.prod.yml up --build -d

# Scale specific services
docker-compose -f docker-compose.prod.yml up --scale api-gateway=3 -d

# Update specific service
docker-compose -f docker-compose.prod.yml up --no-deps -d api-gateway

# View production logs
docker-compose -f docker-compose.prod.yml logs -f --tail=100

# Backup database
docker-compose -f docker-compose.prod.yml exec postgres pg_dump -U oracle_user oracle_prod > backup.sql

# Stop production
docker-compose -f docker-compose.prod.yml down
```

## Health Checks and Monitoring

### Health Check Endpoints

Each service exposes a `/health` endpoint:

```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T10:00:00Z",
  "version": "1.0.0",
  "dependencies": {
    "database": "healthy",
    "redis": "healthy",
    "external_apis": "healthy"
  }
}
```

### Docker Health Checks

Health checks are configured in Dockerfiles and docker-compose files to ensure service reliability:

```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 40s
```

### Volume Management

```bash
# List volumes
docker volume ls

# Backup volume data
docker run --rm -v oracle_postgres-data:/data -v $(pwd):/backup alpine tar czf /backup/postgres-backup.tar.gz -C /data .

# Restore volume data
docker run --rm -v oracle_postgres-data:/data -v $(pwd):/backup alpine tar xzf /backup/postgres-backup.tar.gz -C /data

# Clean up unused volumes
docker volume prune
```

This Docker deployment configuration provides a scalable, maintainable, and production-ready environment for the Neural Options Oracle++ system with proper networking, security, and monitoring capabilities.