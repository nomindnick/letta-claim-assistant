# Letta Deployment Guide

**Version:** 1.0  
**Last Updated:** 2025-08-19

This guide provides comprehensive instructions for deploying the Letta agent memory system in production environments.

---

## Overview

Letta deployment involves several components:
- **Letta Server**: Core memory and agent management server
- **Client Integration**: Application connection to Letta server  
- **Data Storage**: SQLite database for persistent memory
- **Monitoring**: Health checks and performance monitoring

---

## Deployment Architectures

### 1. Single Server Deployment (Recommended for most cases)

```
┌─────────────────────────────────────┐
│         Ubuntu Server              │
│                                     │
│  ┌──────────────┐ ┌─────────────┐  │
│  │ Application  │ │ Letta Server│  │
│  │              │ │             │  │
│  │   Port 8000  │ │  Port 8283  │  │
│  └──────────────┘ └─────────────┘  │
│                                     │
│  ┌─────────────────────────────┐    │
│  │      SQLite Database        │    │
│  │  ~/.letta-claim/letta.db    │    │
│  └─────────────────────────────┘    │
└─────────────────────────────────────┘
```

**Best for:**
- Individual users or small teams
- Development and staging environments
- Desktop deployments

---

### 2. Containerized Deployment

```
┌─────────────────────────────────────┐
│         Docker Host                 │
│                                     │
│  ┌──────────────┐ ┌─────────────┐  │
│  │ App Container│ │Letta Container│ │
│  │              │ │             │  │
│  │   Port 8000  │ │  Port 8283  │  │
│  └──────────────┘ └─────────────┘  │
│                                     │
│  ┌─────────────────────────────┐    │
│  │    Volume Mount             │    │
│  │  /data/letta.db             │    │
│  └─────────────────────────────┘    │
└─────────────────────────────────────┘
```

**Best for:**
- Production environments
- Multi-environment deployments
- Scalable infrastructure

---

### 3. Distributed Deployment

```
┌─────────────────┐    ┌─────────────────┐
│  App Servers    │    │  Letta Server   │
│                 │    │                 │
│  ┌───────────┐  │    │  ┌───────────┐  │
│  │App Node 1 │──┼────┼─▶│   Letta   │  │
│  └───────────┘  │    │  │  Server   │  │
│  ┌───────────┐  │    │  │Port 8283  │  │
│  │App Node 2 │──┼────┼─▶│           │  │
│  └───────────┘  │    │  └───────────┘  │
│                 │    │                 │
└─────────────────┘    └─────────────────┘
```

**Best for:**
- Large-scale deployments
- High availability requirements
- Team environments

---

## Prerequisites

### System Requirements

#### Minimum Requirements
- **OS**: Ubuntu 20.04+ LTS
- **RAM**: 4GB available
- **CPU**: 2 cores
- **Disk**: 20GB free space
- **Network**: Stable internet for initial setup

#### Recommended Requirements
- **OS**: Ubuntu 22.04 LTS
- **RAM**: 8GB available  
- **CPU**: 4 cores
- **Disk**: 100GB SSD
- **Network**: High-speed connection for LLM downloads

### Software Dependencies

#### Required Packages
```bash
# System packages
sudo apt update
sudo apt install -y \
  python3.11 \
  python3.11-venv \
  python3.11-dev \
  sqlite3 \
  curl \
  wget \
  git

# Optional: Docker for containerized deployment
sudo apt install -y docker.io docker-compose-v2
```

#### Python Environment
```bash
# Create virtual environment (use venv instead of .venv)
python3.11 -m venv /opt/letta-claim-assistant/venv
source /opt/letta-claim-assistant/venv/bin/activate  # On Linux/Mac
# or
/opt/letta-claim-assistant/venv/Scripts/activate      # On Windows

# Install application
pip install letta-claim-assistant[all]
```

---

## Installation Methods

### Method 1: Package Installation (Recommended)

```bash
# Install from package repository
pip install letta-claim-assistant[production]

# Initialize configuration
letta-claim-assistant --init-production

# Start services
systemctl --user enable letta-claim-assistant
systemctl --user start letta-claim-assistant
```

### Method 2: Source Installation

```bash
# Clone repository
git clone https://github.com/your-org/letta-claim-assistant.git
cd letta-claim-assistant

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Linux/Mac
# or
venv\Scripts\activate     # On Windows

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Initialize configuration
./scripts/init-production.sh
```

### Method 3: Docker Installation

```bash
# Pull images
docker pull letta-claim-assistant:latest
docker pull letta/server:latest

# Start with docker-compose
docker-compose -f docker/production.yml up -d
```

---

## Configuration for Production

### Server Configuration

Create production configuration at `/etc/letta-claim-assistant/server.yaml`:

```yaml
# Production Server Configuration
server:
  mode: local
  host: 0.0.0.0
  port: 8283
  auto_start: true
  startup_timeout: 60
  health_check_interval: 30
  max_retries: 5

# Enhanced storage for production
storage:
  type: sqlite
  path: /opt/letta-claim-assistant/data/letta.db
  backup_interval: 1800    # 30 minutes
  max_backups: 48         # 24 hours worth
  vacuum_interval: 86400   # Daily vacuum
  
# Production logging
logging:
  level: info
  file: /var/log/letta-claim-assistant/server.log
  max_size: 100mb
  backup_count: 10
  structured_logging: true

# Performance tuning
performance:
  workers: 6
  batch_size: 250
  request_timeout: 60
  connection_pool_size: 20
  max_connections: 100

# Security settings
security:
  auth_enabled: true
  jwt_secret: "${JWT_SECRET}"
  token_expiration: 3600
  rate_limiting: true
  ssl_enabled: true
  cert_file: /etc/ssl/certs/letta.crt
  key_file: /etc/ssl/private/letta.key

# Memory management for production
memory:
  max_memory_items: 50000
  archival_memory_limit: 200000
  prune_old_memories: true
  prune_after_days: 180
  backup_before_prune: true
```

### Application Configuration

Create production configuration at `/etc/letta-claim-assistant/config.toml`:

```toml
[environment]
mode = "production"
debug = false

[server]
host = "0.0.0.0"
port = 8000
workers = 4

[letta]
server_host = "localhost"
server_port = 8283
connection_timeout = 30
request_timeout = 60
max_retries = 5
fallback_mode = true

[logging]
level = "info"
file = "/var/log/letta-claim-assistant/app.log"
structured = true
correlation_id = true

[security]
cors_origins = ["https://yourdomain.com"]
api_keys_required = true
rate_limiting = true

[monitoring]
enable_metrics = true
metrics_port = 9090
health_check_interval = 30
```

### Environment Variables

Create `/etc/letta-claim-assistant/environment`:

```bash
# Application Environment
ENVIRONMENT=production
LOG_LEVEL=info
DEBUG=false

# Security
JWT_SECRET=your-jwt-secret-here
API_KEY=your-api-key-here
SSL_CERT_PATH=/etc/ssl/certs/letta.crt
SSL_KEY_PATH=/etc/ssl/private/letta.key

# Database
DATABASE_PATH=/opt/letta-claim-assistant/data/letta.db
BACKUP_PATH=/opt/letta-claim-assistant/backups/

# External providers (optional)
GEMINI_API_KEY=your-gemini-key
OPENAI_API_KEY=your-openai-key

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=9090
SENTRY_DSN=your-sentry-dsn

# Resource limits
MAX_MEMORY_MB=4096
MAX_CPU_PERCENT=80
```

---

## SSL/TLS Configuration

### Generate Self-Signed Certificate (Development)

```bash
# Create certificate directory
sudo mkdir -p /etc/ssl/letta-claim-assistant

# Generate private key
sudo openssl genrsa -out /etc/ssl/letta-claim-assistant/key.pem 4096

# Generate certificate
sudo openssl req -new -x509 -key /etc/ssl/letta-claim-assistant/key.pem \
  -out /etc/ssl/letta-claim-assistant/cert.pem -days 365 \
  -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"

# Set permissions
sudo chown root:letta-claim-assistant /etc/ssl/letta-claim-assistant/*
sudo chmod 640 /etc/ssl/letta-claim-assistant/*
```

### Production Certificate (Let's Encrypt)

```bash
# Install certbot
sudo apt install -y certbot

# Generate certificate
sudo certbot certonly --standalone -d your-domain.com

# Link certificates
sudo ln -sf /etc/letsencrypt/live/your-domain.com/fullchain.pem \
  /etc/ssl/letta-claim-assistant/cert.pem
sudo ln -sf /etc/letsencrypt/live/your-domain.com/privkey.pem \
  /etc/ssl/letta-claim-assistant/key.pem

# Setup auto-renewal
sudo crontab -e
# Add: 0 0 1 * * certbot renew --quiet
```

---

## Database Setup

### SQLite Production Configuration

```bash
# Create data directory
sudo mkdir -p /opt/letta-claim-assistant/data
sudo mkdir -p /opt/letta-claim-assistant/backups

# Set ownership
sudo chown -R letta-user:letta-group /opt/letta-claim-assistant/

# Configure SQLite for production
sqlite3 /opt/letta-claim-assistant/data/letta.db << EOF
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;  
PRAGMA cache_size = -64000;
PRAGMA temp_store = MEMORY;
PRAGMA mmap_size = 134217728;
EOF
```

### Database Backup Script

Create `/opt/letta-claim-assistant/scripts/backup-db.sh`:

```bash
#!/bin/bash
set -e

BACKUP_DIR="/opt/letta-claim-assistant/backups"
DB_PATH="/opt/letta-claim-assistant/data/letta.db"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="letta_backup_${TIMESTAMP}.db"

# Create backup
sqlite3 "$DB_PATH" ".backup ${BACKUP_DIR}/${BACKUP_FILE}"

# Compress backup
gzip "${BACKUP_DIR}/${BACKUP_FILE}"

# Clean old backups (keep 48 hours worth)
find "$BACKUP_DIR" -name "letta_backup_*.db.gz" -mtime +2 -delete

echo "Backup completed: ${BACKUP_FILE}.gz"
```

---

## Service Configuration

### Systemd Service

Create `/etc/systemd/system/letta-claim-assistant.service`:

```ini
[Unit]
Description=Letta Construction Claim Assistant
After=network.target
Wants=network-online.target
StartLimitIntervalSec=60
StartLimitBurst=3

[Service]
Type=exec
User=letta-user
Group=letta-group
WorkingDirectory=/opt/letta-claim-assistant
Environment=PYTHONPATH=/opt/letta-claim-assistant
EnvironmentFile=/etc/letta-claim-assistant/environment
ExecStart=/opt/letta-claim-assistant/venv/bin/python -m letta_claim_assistant
ExecReload=/bin/kill -HUP $MAINPID
Restart=always
RestartSec=10
TimeoutStartSec=60
TimeoutStopSec=30

# Security
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/opt/letta-claim-assistant /var/log/letta-claim-assistant
PrivateTmp=true

# Resource limits
LimitNOFILE=65536
MemoryLimit=8G
CPUQuota=400%

[Install]
WantedBy=multi-user.target
```

### Enable and Start Service

```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable service
sudo systemctl enable letta-claim-assistant

# Start service
sudo systemctl start letta-claim-assistant

# Check status
sudo systemctl status letta-claim-assistant
```

---

## Docker Deployment

### Production Docker Compose

Create `docker-compose.production.yml`:

```yaml
version: '3.8'

services:
  letta-server:
    image: letta/server:latest
    container_name: letta-server
    restart: unless-stopped
    ports:
      - "8283:8283"
    volumes:
      - letta-data:/app/data
      - ./config/letta-server.yaml:/app/config.yaml
    environment:
      - LETTA_CONFIG_FILE=/app/config.yaml
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8283/v1/health/"]
      interval: 30s
      timeout: 10s
      retries: 3
    
  letta-claim-assistant:
    image: letta-claim-assistant:latest
    container_name: letta-app
    restart: unless-stopped
    ports:
      - "8000:8000"
    depends_on:
      - letta-server
    volumes:
      - app-data:/app/data
      - ./config/app-config.toml:/app/config.toml
      - ./logs:/app/logs
    environment:
      - LETTA_SERVER_HOST=letta-server
      - LETTA_SERVER_PORT=8283
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  nginx:
    image: nginx:alpine
    container_name: letta-nginx
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./config/nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - letta-claim-assistant

volumes:
  letta-data:
    driver: local
    driver_opts:
      type: bind
      device: /opt/letta-claim-assistant/data
  app-data:
    driver: local
    driver_opts:
      type: bind
      device: /opt/letta-claim-assistant/app-data
```

### Nginx Configuration

Create `config/nginx.conf`:

```nginx
events {
    worker_connections 1024;
}

http {
    upstream app {
        server letta-claim-assistant:8000;
    }

    server {
        listen 80;
        server_name your-domain.com;
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl http2;
        server_name your-domain.com;

        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers HIGH:!aNULL:!MD5;

        client_max_body_size 100M;
        
        location / {
            proxy_pass http://app;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_timeout 60s;
        }

        location /api/matters/*/upload {
            proxy_pass http://app;
            proxy_set_header Host $host;
            proxy_request_buffering off;
            proxy_timeout 300s;
        }
    }
}
```

---

## Monitoring and Logging

### Log Configuration

#### Logrotate Configuration
Create `/etc/logrotate.d/letta-claim-assistant`:

```
/var/log/letta-claim-assistant/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    sharedscripts
    postrotate
        systemctl reload letta-claim-assistant
    endscript
}
```

#### Structured Logging Setup
Update logging configuration to include structured logs:

```yaml
logging:
  level: info
  structured: true
  format: json
  fields:
    - timestamp
    - level
    - message
    - correlation_id
    - user_id
    - matter_id
    - duration
```

### Health Monitoring

#### Health Check Endpoint
The application provides comprehensive health checks at:
- `GET /api/health` - Basic health check
- `GET /api/health/detailed` - Detailed system status
- `GET /api/letta/health` - Letta server health

#### Monitoring Script
Create `/opt/letta-claim-assistant/scripts/health-monitor.sh`:

```bash
#!/bin/bash

HEALTH_URL="http://localhost:8000/api/health/detailed"
LOG_FILE="/var/log/letta-claim-assistant/health-monitor.log"

while true; do
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    RESPONSE=$(curl -s -w "%{http_code}" "$HEALTH_URL")
    HTTP_CODE="${RESPONSE: -3}"
    
    if [ "$HTTP_CODE" != "200" ]; then
        echo "[$TIMESTAMP] ALERT: Health check failed (HTTP $HTTP_CODE)" >> "$LOG_FILE"
        # Send alert (email, Slack, etc.)
    else
        echo "[$TIMESTAMP] OK: Application healthy" >> "$LOG_FILE"
    fi
    
    sleep 300  # Check every 5 minutes
done
```

### Performance Monitoring

#### Prometheus Metrics
Add Prometheus metrics endpoint:

```python
# Application metrics
from prometheus_client import Counter, Histogram, Gauge

REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests')
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')
ACTIVE_AGENTS = Gauge('letta_active_agents', 'Number of active Letta agents')
MEMORY_ITEMS = Gauge('letta_memory_items_total', 'Total memory items stored')
```

#### Grafana Dashboard
Create monitoring dashboard with panels for:
- Request rate and response times
- Memory usage and item counts
- Error rates and types
- Agent health and performance
- Database size and backup status

---

## Security Hardening

### System Security

#### User and Permissions
```bash
# Create dedicated user
sudo useradd -r -s /bin/false letta-user
sudo usermod -a -G letta-group letta-user

# Set file permissions
sudo chown -R letta-user:letta-group /opt/letta-claim-assistant
sudo chmod -R 750 /opt/letta-claim-assistant
sudo chmod -R 640 /opt/letta-claim-assistant/data
```

#### Firewall Configuration
```bash
# Configure UFW firewall
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Allow SSH (adjust port as needed)
sudo ufw allow ssh

# Allow application ports
sudo ufw allow 8000/tcp  # Application
sudo ufw allow 8283/tcp  # Letta server (local only)

# Allow HTTPS
sudo ufw allow 443/tcp

# Enable firewall
sudo ufw --force enable
```

#### Fail2Ban Configuration
Create `/etc/fail2ban/jail.d/letta-claim-assistant.conf`:

```ini
[letta-claim-assistant]
enabled = true
port = 8000
filter = letta-claim-assistant
logpath = /var/log/letta-claim-assistant/app.log
maxretry = 5
bantime = 3600
findtime = 600
```

### Application Security

#### API Security
```yaml
security:
  auth_enabled: true
  api_keys_required: true
  rate_limiting:
    requests_per_minute: 100
    burst_limit: 20
  cors:
    allowed_origins: ["https://your-domain.com"]
    allowed_methods: ["GET", "POST", "PUT", "DELETE"]
    allowed_headers: ["Authorization", "Content-Type"]
```

#### Data Encryption
```yaml
encryption:
  at_rest: true
  in_transit: true
  key_rotation: true
  key_rotation_interval: 2592000  # 30 days
```

---

## Backup and Recovery

### Automated Backup Strategy

#### Daily Backup Script
Create `/opt/letta-claim-assistant/scripts/daily-backup.sh`:

```bash
#!/bin/bash
set -e

BACKUP_DIR="/opt/letta-claim-assistant/backups"
DATE=$(date +%Y%m%d)
BACKUP_NAME="full_backup_${DATE}"

# Create backup directory
mkdir -p "$BACKUP_DIR/$BACKUP_NAME"

# Backup database
sqlite3 /opt/letta-claim-assistant/data/letta.db \
  ".backup $BACKUP_DIR/$BACKUP_NAME/letta.db"

# Backup configuration
cp -r /etc/letta-claim-assistant "$BACKUP_DIR/$BACKUP_NAME/config"

# Backup user data
cp -r /opt/letta-claim-assistant/data "$BACKUP_DIR/$BACKUP_NAME/data"

# Create archive
tar -czf "$BACKUP_DIR/${BACKUP_NAME}.tar.gz" \
  -C "$BACKUP_DIR" "$BACKUP_NAME"

# Remove temporary directory
rm -rf "$BACKUP_DIR/$BACKUP_NAME"

# Upload to cloud storage (optional)
# aws s3 cp "$BACKUP_DIR/${BACKUP_NAME}.tar.gz" s3://your-backup-bucket/

echo "Backup completed: ${BACKUP_NAME}.tar.gz"
```

#### Backup Verification
```bash
#!/bin/bash
BACKUP_FILE="$1"

# Verify archive integrity
if tar -tzf "$BACKUP_FILE" > /dev/null 2>&1; then
    echo "Archive integrity: OK"
else
    echo "Archive integrity: FAILED"
    exit 1
fi

# Verify database integrity
TEMP_DIR=$(mktemp -d)
tar -xzf "$BACKUP_FILE" -C "$TEMP_DIR"

if sqlite3 "$TEMP_DIR"/*/letta.db "PRAGMA integrity_check;" | grep -q "ok"; then
    echo "Database integrity: OK"
else
    echo "Database integrity: FAILED"
    exit 1
fi

rm -rf "$TEMP_DIR"
echo "Backup verification: PASSED"
```

### Disaster Recovery

#### Recovery Procedure
```bash
#!/bin/bash
# Disaster recovery script

BACKUP_FILE="$1"
RECOVERY_DIR="/opt/letta-claim-assistant-recovery"

# Stop services
sudo systemctl stop letta-claim-assistant

# Create recovery directory
sudo mkdir -p "$RECOVERY_DIR"
cd "$RECOVERY_DIR"

# Extract backup
sudo tar -xzf "$BACKUP_FILE"

# Restore database
sudo cp */letta.db /opt/letta-claim-assistant/data/

# Restore configuration
sudo cp -r */config/* /etc/letta-claim-assistant/

# Restore data
sudo cp -r */data/* /opt/letta-claim-assistant/data/

# Fix permissions
sudo chown -R letta-user:letta-group /opt/letta-claim-assistant/

# Start services
sudo systemctl start letta-claim-assistant

echo "Recovery completed. Verify application functionality."
```

---

## Performance Optimization

### Database Optimization

#### SQLite Performance Tuning
```sql
-- Production SQLite settings
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA cache_size = -128000;  -- 128MB cache
PRAGMA temp_store = MEMORY;
PRAGMA mmap_size = 268435456; -- 256MB mmap
PRAGMA optimize;
```

#### Database Maintenance
```bash
# Weekly vacuum script
#!/bin/bash
DB_PATH="/opt/letta-claim-assistant/data/letta.db"

# Vacuum database
sqlite3 "$DB_PATH" "VACUUM;"

# Analyze tables
sqlite3 "$DB_PATH" "ANALYZE;"

# Optimize query planner
sqlite3 "$DB_PATH" "PRAGMA optimize;"

echo "Database maintenance completed"
```

### Application Performance

#### Resource Allocation
```yaml
performance:
  workers: 8              # 2x CPU cores
  worker_connections: 1000
  keepalive_timeout: 65
  client_max_body_size: 100M
  
  # Memory settings
  memory_limit: "4GB"
  max_memory_per_request: "512MB"
  
  # Connection pooling
  db_pool_size: 20
  db_pool_timeout: 30
  
  # Caching
  response_cache_size: 1000
  memory_cache_ttl: 3600
```

#### Load Balancing (Multi-server)
```nginx
upstream letta_app {
    least_conn;
    server app1.internal:8000 max_fails=3 fail_timeout=30s;
    server app2.internal:8000 max_fails=3 fail_timeout=30s;
    server app3.internal:8000 max_fails=3 fail_timeout=30s;
    keepalive 32;
}
```

---

## Scaling Considerations

### Horizontal Scaling

#### Multi-Server Setup
```yaml
# Load balancer configuration
load_balancer:
  algorithm: least_conn
  health_checks: true
  session_affinity: false

# Application servers
app_servers:
  - host: app1.internal
    weight: 1
    max_connections: 100
  - host: app2.internal
    weight: 1
    max_connections: 100

# Shared Letta server
letta_server:
  host: letta.internal
  port: 8283
  connection_pool: 50
```

#### Database Scaling
For high-load scenarios, consider:
- Read replicas for query distribution
- Database clustering (PostgreSQL)
- Sharding by matter ID
- Caching layer (Redis)

### Vertical Scaling

#### Resource Recommendations
| Load Level | CPU | RAM | Disk | Notes |
|------------|-----|-----|------|-------|
| Light (1-10 users) | 2 cores | 4GB | 100GB | Single server |
| Medium (10-50 users) | 4 cores | 8GB | 500GB | Optimized config |
| Heavy (50+ users) | 8+ cores | 16GB | 1TB+ | Multi-server |

---

## Troubleshooting Deployment Issues

### Common Problems

#### Service Won't Start
```bash
# Check service status
sudo systemctl status letta-claim-assistant

# View logs
sudo journalctl -u letta-claim-assistant -f

# Check configuration
letta-claim-assistant --validate-config

# Test database connection
sqlite3 /opt/letta-claim-assistant/data/letta.db "SELECT 1;"
```

#### Performance Issues
```bash
# Monitor resource usage
htop
iotop
netstat -tulpn

# Check application logs
tail -f /var/log/letta-claim-assistant/app.log

# Profile database queries
sqlite3 /opt/letta-claim-assistant/data/letta.db ".timer on" "SELECT count(*) FROM archival_memory;"
```

#### Connection Problems
```bash
# Test Letta server
curl http://localhost:8283/v1/health/

# Test application API
curl http://localhost:8000/api/health

# Check network connectivity
netstat -an | grep 8283
telnet localhost 8283
```

### Log Analysis

#### Key Log Patterns
```bash
# Error patterns to monitor
grep -E "ERROR|CRITICAL|FATAL" /var/log/letta-claim-assistant/app.log

# Performance issues
grep -E "timeout|slow|performance" /var/log/letta-claim-assistant/app.log

# Memory issues
grep -E "memory|oom|killed" /var/log/letta-claim-assistant/app.log

# Connection issues
grep -E "connection|refused|timeout" /var/log/letta-claim-assistant/app.log
```

---

## Maintenance Procedures

### Regular Maintenance Tasks

#### Weekly Tasks
- [ ] Review system logs
- [ ] Check backup integrity
- [ ] Monitor resource usage
- [ ] Update security patches
- [ ] Vacuum database

#### Monthly Tasks
- [ ] Review performance metrics
- [ ] Update SSL certificates
- [ ] Clean old logs and backups
- [ ] Review user access
- [ ] Security audit

#### Quarterly Tasks
- [ ] Full system backup
- [ ] Disaster recovery test
- [ ] Performance optimization review
- [ ] Security assessment
- [ ] Dependency updates

### Update Procedures

#### Application Updates
```bash
# Stop services
sudo systemctl stop letta-claim-assistant

# Backup current installation
sudo tar -czf "/opt/backups/pre-update-$(date +%Y%m%d).tar.gz" \
  /opt/letta-claim-assistant

# Update application
pip install --upgrade letta-claim-assistant

# Migrate configuration if needed
letta-claim-assistant --migrate-config

# Restart services
sudo systemctl start letta-claim-assistant

# Verify functionality
curl http://localhost:8000/api/health
```

#### System Updates
```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Update Python packages
source /opt/letta-claim-assistant/venv/bin/activate  # On Linux/Mac
# or
/opt/letta-claim-assistant/venv/Scripts/activate      # On Windows
pip list --outdated
pip install --upgrade pip

# Restart services after updates
sudo systemctl restart letta-claim-assistant
```

---

## Support and Documentation

### Production Support

#### Emergency Contacts
- **System Administrator**: admin@yourcompany.com
- **Development Team**: dev@yourcompany.com
- **On-call Support**: +1-xxx-xxx-xxxx

#### Support Procedures
1. Check system status and logs
2. Consult troubleshooting documentation
3. Search known issues database
4. Contact appropriate support team
5. Document issue and resolution

### Documentation Maintenance

#### Required Documentation
- [ ] Deployment procedures
- [ ] Configuration management
- [ ] Backup and recovery procedures
- [ ] Troubleshooting guides
- [ ] Security procedures
- [ ] Monitoring and alerting setup

#### Review Schedule
- **Monthly**: Update procedures and configurations
- **Quarterly**: Full documentation review
- **After incidents**: Update based on lessons learned

---

*This deployment guide provides comprehensive instructions for production deployment. For additional assistance, consult the troubleshooting guide and technical support resources.*