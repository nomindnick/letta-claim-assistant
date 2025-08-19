# Letta Troubleshooting Guide

**Version:** 1.0  
**Last Updated:** 2025-08-19

This guide provides solutions to common issues with the Letta agent memory system in the Construction Claim Assistant.

---

## Quick Diagnosis

### Health Check Commands
Start troubleshooting by running these diagnostic commands:

```bash
# Check application health
curl http://localhost:8000/api/health/detailed

# Check Letta server health
curl http://localhost:8283/v1/health/

# View recent logs
tail -n 50 ~/.letta-claim/logs/app.log
tail -n 50 ~/.letta-claim/logs/letta-server.log

# Check process status
ps aux | grep letta
netstat -tulpn | grep -E "8000|8283"
```

### System Status Indicators

| Indicator | Status | Description |
|-----------|--------|-------------|
| 🟢 Green | Healthy | All systems operational |
| 🟡 Yellow | Warning | Minor issues, reduced functionality |
| 🔴 Red | Error | Major problems, core features affected |
| ⚫ Gray | Unknown | Cannot determine status |

---

## Connection Issues

### Problem: Cannot Connect to Letta Server

**Symptoms:**
- Red connection indicator in UI
- "Letta server unavailable" messages
- Memory features not working
- Timeout errors

**Diagnostic Steps:**
```bash
# 1. Check if Letta server is running
ps aux | grep "letta server"

# 2. Check if port is listening
netstat -tulpn | grep 8283

# 3. Test server response
curl http://localhost:8283/v1/health/

# 4. Check server logs
tail -f ~/.letta-claim/logs/letta-server.log
```

**Common Solutions:**

#### Solution 1: Start Letta Server
```bash
# If server is not running, start it manually
letta server --port 8283 --host localhost &

# Or restart the full application
pkill -f letta-claim-assistant
python -m letta_claim_assistant.main
```

#### Solution 2: Port Conflict
```bash
# Check what's using port 8283
sudo lsof -i :8283

# Kill conflicting process (if safe)
sudo kill -9 <PID>

# Or configure different port
# Edit ~/.letta-claim/config.toml
[letta]
server_port = 8284
```

#### Solution 3: Configuration Issues
```bash
# Reset to default configuration
mv ~/.letta-claim/letta-server.yaml ~/.letta-claim/letta-server.yaml.backup
letta-claim-assistant --reset-letta-config

# Validate configuration
letta config validate
```

#### Solution 4: Firewall/Network Issues
```bash
# Check firewall status
sudo ufw status

# Allow Letta server port (if needed)
sudo ufw allow 8283/tcp

# Test local connectivity
telnet localhost 8283
```

---

### Problem: Connection Timeouts

**Symptoms:**
- Yellow connection indicator
- Slow response times
- Intermittent connection failures
- "Request timeout" errors

**Diagnostic Steps:**
```bash
# Check server response time
time curl http://localhost:8283/v1/health/

# Monitor system resources
htop
iotop

# Check for memory issues
free -h
df -h
```

**Solutions:**

#### Solution 1: Increase Timeout Values
Edit matter configuration (`~/LettaClaims/Matter_<slug>/config.json`):
```json
{
  "letta": {
    "connection_timeout": 30,
    "request_timeout": 60,
    "max_retries": 5
  }
}
```

#### Solution 2: Optimize Server Performance
Edit `~/.letta-claim/letta-server.yaml`:
```yaml
performance:
  workers: 4
  batch_size: 100
  request_timeout: 30
  connection_pool_size: 10
```

#### Solution 3: Check System Resources
```bash
# If memory is low, restart services
sudo systemctl restart letta-claim-assistant

# If disk is full, clean up space
df -h
sudo find ~/.letta-claim/logs -name "*.log" -mtime +7 -delete
```

---

## Memory Operation Issues

### Problem: Memory Not Storing

**Symptoms:**
- Memory item count not increasing
- "Storage failed" notifications
- No memory context in responses
- Empty memory dashboard

**Diagnostic Steps:**
```bash
# Check memory operation logs
grep -i "memory\|store\|recall" ~/.letta-claim/logs/app.log

# Test memory storage manually
curl -X POST http://localhost:8000/api/matters/<matter-id>/memory/store \
  -H "Content-Type: application/json" \
  -d '{"items": [{"type": "Test", "label": "Test item"}]}'

# Check database permissions
ls -la ~/.letta-claim/letta.db
sqlite3 ~/.letta-claim/letta.db "SELECT count(*) FROM archival_memory;"
```

**Solutions:**

#### Solution 1: Database Issues
```bash
# Check database integrity
sqlite3 ~/.letta-claim/letta.db "PRAGMA integrity_check;"

# Repair database if corrupted
sqlite3 ~/.letta-claim/letta.db "VACUUM;"

# Restore from backup if needed
cp ~/.letta-claim/backups/letta_backup_latest.db ~/.letta-claim/letta.db
```

#### Solution 2: Permission Issues
```bash
# Fix database permissions
chmod 664 ~/.letta-claim/letta.db
chown $USER:$USER ~/.letta-claim/letta.db

# Fix directory permissions
chmod 755 ~/.letta-claim/
```

#### Solution 3: Memory Limits Exceeded
```bash
# Check memory usage
sqlite3 ~/.letta-claim/letta.db "SELECT count(*) FROM archival_memory;"

# Clean old memories
sqlite3 ~/.letta-claim/letta.db "DELETE FROM archival_memory WHERE created_at < date('now', '-90 days');"
```

#### Solution 4: Agent Configuration Issues
Check matter configuration for correct agent settings:
```json
{
  "letta": {
    "enabled": true,
    "agent_id": "agent_12345",
    "memory_settings": {
      "max_items": 10000,
      "pruning_enabled": true
    }
  }
}
```

---

### Problem: Poor Memory Recall

**Symptoms:**
- Responses lack context from previous conversations
- Memory search returns irrelevant results
- "No memory context found" messages
- Memory seems to forget recent interactions

**Diagnostic Steps:**
```bash
# Check memory search functionality
curl -X POST http://localhost:8000/api/matters/<matter-id>/memory/recall \
  -H "Content-Type: application/json" \
  -d '{"query": "test search", "max_results": 5}'

# Check memory item quality
sqlite3 ~/.letta-claim/letta.db "SELECT * FROM archival_memory ORDER BY created_at DESC LIMIT 10;"

# Verify agent is configured correctly
curl http://localhost:8000/api/matters/<matter-id>/agent/status
```

**Solutions:**

#### Solution 1: Improve Memory Quality
```bash
# Enable importance scoring
# Edit matter config.json:
{
  "letta": {
    "memory_settings": {
      "importance_scoring": true,
      "quality_threshold": 0.5
    }
  }
}
```

#### Solution 2: Adjust Search Parameters
```python
# In Python code, adjust recall parameters
recall_context = await letta_adapter.recall_with_context(
    query=query,
    k=10,  # Increase number of results
    recency_weight=0.3,  # Adjust recency vs relevance
    similarity_threshold=0.6  # Lower threshold for more results
)
```

#### Solution 3: Memory Fragmentation
```bash
# Defragment memory database
sqlite3 ~/.letta-claim/letta.db "VACUUM;"
sqlite3 ~/.letta-claim/letta.db "ANALYZE;"

# Rebuild memory index if available
curl -X POST http://localhost:8000/api/matters/<matter-id>/memory/rebuild-index
```

---

## Agent Management Issues

### Problem: Agent Not Found

**Symptoms:**
- "Agent not found" errors
- Unable to create new conversations
- Missing agent in matter settings
- Agent status shows as "unknown"

**Diagnostic Steps:**
```bash
# Check agent configuration
cat ~/LettaClaims/Matter_*/config.json | grep -A 5 '"letta"'

# List agents on server
curl http://localhost:8283/v1/agents/

# Check agent creation logs
grep -i "agent.*create\|agent.*error" ~/.letta-claim/logs/app.log
```

**Solutions:**

#### Solution 1: Recreate Agent
```bash
# Delete agent configuration
rm ~/LettaClaims/Matter_<slug>/agent_config.json

# Restart application to trigger agent creation
pkill -f letta-claim-assistant
python -m letta_claim_assistant.main

# Or recreate via API
curl -X POST http://localhost:8000/api/matters/<matter-id>/agent/create
```

#### Solution 2: Agent ID Mismatch
```bash
# Check for orphaned agents
curl http://localhost:8283/v1/agents/ | jq '.agents[] | .id'

# Update matter configuration with correct agent ID
# Edit ~/LettaClaims/Matter_<slug>/config.json
{
  "letta": {
    "agent_id": "correct-agent-id-here"
  }
}
```

#### Solution 3: Server State Issues
```bash
# Reset Letta server state
sudo systemctl stop letta-claim-assistant
rm ~/.letta-claim/letta.db
sudo systemctl start letta-claim-assistant

# Note: This will delete all memory data
```

---

### Problem: Agent Configuration Errors

**Symptoms:**
- Agent fails to respond
- LLM provider errors
- Configuration validation failures
- Agent status shows "error"

**Diagnostic Steps:**
```bash
# Validate agent configuration
curl http://localhost:8000/api/matters/<matter-id>/agent/config

# Check LLM provider status
curl http://localhost:11434/api/tags  # For Ollama

# Test provider connectivity
curl -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-oss:20b", "prompt": "Test", "stream": false}'
```

**Solutions:**

#### Solution 1: Fix Provider Configuration
```bash
# Check if Ollama is running
sudo systemctl status ollama

# Start Ollama if needed
sudo systemctl start ollama

# Pull required models
ollama pull gpt-oss:20b
ollama pull nomic-embed-text
```

#### Solution 2: Update Agent Configuration
Edit matter configuration:
```json
{
  "letta": {
    "llm_provider": {
      "type": "ollama",
      "model": "gpt-oss:20b",
      "host": "localhost",
      "port": 11434
    }
  }
}
```

#### Solution 3: Reset to Default Configuration
```bash
# Backup current configuration
cp ~/LettaClaims/Matter_<slug>/config.json ~/LettaClaims/Matter_<slug>/config.json.backup

# Reset to defaults
letta-claim-assistant --reset-matter-config <matter-id>
```

---

## Performance Issues

### Problem: Slow Response Times

**Symptoms:**
- Long delays for memory operations
- UI feels sluggish
- Timeout warnings
- High CPU/memory usage

**Diagnostic Steps:**
```bash
# Monitor system resources
htop
iotop -o

# Check application performance
curl -w "@curl-format.txt" http://localhost:8000/api/health

# Profile database queries
sqlite3 ~/.letta-claim/letta.db ".timer on" ".explain on" "SELECT count(*) FROM archival_memory;"

# Check memory usage
free -h
ps aux | grep letta | awk '{print $6}'
```

**Solutions:**

#### Solution 1: Optimize Database
```bash
# Vacuum and analyze database
sqlite3 ~/.letta-claim/letta.db "VACUUM;"
sqlite3 ~/.letta-claim/letta.db "ANALYZE;"

# Update SQLite settings for performance
sqlite3 ~/.letta-claim/letta.db << EOF
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA cache_size = -64000;
PRAGMA temp_store = MEMORY;
EOF
```

#### Solution 2: Increase System Resources
```bash
# Increase file descriptors
ulimit -n 65536

# Configure swap if needed
sudo swapon --show
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

#### Solution 3: Optimize Application Settings
Edit `~/.letta-claim/config.toml`:
```toml
[performance]
workers = 4
batch_size = 100
connection_pool_size = 10
request_timeout = 30

[caching]
enabled = true
ttl = 300
max_size = 1000
```

#### Solution 4: Memory Management
```bash
# Enable memory pruning
# Edit matter config.json:
{
  "letta": {
    "memory_settings": {
      "pruning_enabled": true,
      "prune_after_days": 60,
      "max_items": 5000
    }
  }
}

# Manual memory cleanup
curl -X POST http://localhost:8000/api/matters/<matter-id>/memory/prune
```

---

## Installation and Setup Issues

### Problem: Letta Not Installed

**Symptoms:**
- "Letta import failed" warnings
- Features automatically disabled
- Missing Letta commands
- Import errors in logs

**Solutions:**

#### Solution 1: Install Letta
```bash
# Install Letta with pip
pip install letta==0.10.0

# Verify installation
letta --version
python -c "import letta; print('Letta installed successfully')"
```

#### Solution 2: Fix Python Environment
```bash
# Check Python environment
which python
echo $PYTHONPATH

# Reinstall in correct environment
source .venv/bin/activate  # If using virtual environment
pip install --force-reinstall letta==0.10.0
```

#### Solution 3: Dependency Conflicts
```bash
# Check for conflicts
pip check

# Create clean environment
python -m venv fresh_env
source fresh_env/bin/activate
pip install letta-claim-assistant[letta]
```

---

### Problem: Version Incompatibility

**Symptoms:**
- API errors with new Letta versions
- Deprecated method warnings
- Database migration errors
- Feature not working after update

**Diagnostic Steps:**
```bash
# Check installed versions
letta --version
pip list | grep letta

# Check compatibility requirements
cat requirements.txt | grep letta
```

**Solutions:**

#### Solution 1: Downgrade to Compatible Version
```bash
# Install specific compatible version
pip install letta==0.10.0

# Verify downgrade
letta --version
```

#### Solution 2: Update Application Code
```bash
# Pull latest application updates
git pull origin main
pip install --upgrade letta-claim-assistant

# Run migration if needed
letta-claim-assistant --migrate-data
```

#### Solution 3: Check Breaking Changes
```bash
# Review changelog for breaking changes
curl -s https://api.github.com/repos/letta-ai/letta/releases | jq '.[].body'

# Update configuration for new version
letta-claim-assistant --update-config-for-version <version>
```

---

## Data and Storage Issues

### Problem: Database Corruption

**Symptoms:**
- SQLite errors in logs
- Memory operations failing
- Data inconsistencies
- "Database is locked" errors

**Diagnostic Steps:**
```bash
# Check database integrity
sqlite3 ~/.letta-claim/letta.db "PRAGMA integrity_check;"

# Check for locks
lsof ~/.letta-claim/letta.db

# Check database size
ls -lah ~/.letta-claim/letta.db
```

**Solutions:**

#### Solution 1: Repair Database
```bash
# Stop application
sudo systemctl stop letta-claim-assistant

# Backup corrupted database
cp ~/.letta-claim/letta.db ~/.letta-claim/letta.db.corrupted

# Attempt repair
sqlite3 ~/.letta-claim/letta.db "VACUUM INTO '~/.letta-claim/letta.db.repaired';"
mv ~/.letta-claim/letta.db.repaired ~/.letta-claim/letta.db

# Restart application
sudo systemctl start letta-claim-assistant
```

#### Solution 2: Restore from Backup
```bash
# Find latest backup
ls -la ~/.letta-claim/backups/

# Restore backup
cp ~/.letta-claim/backups/letta_backup_latest.db ~/.letta-claim/letta.db

# Verify restoration
sqlite3 ~/.letta-claim/letta.db "SELECT count(*) FROM archival_memory;"
```

#### Solution 3: Rebuild Database
```bash
# Export data if possible
sqlite3 ~/.letta-claim/letta.db ".dump" > database_dump.sql

# Remove corrupted database
rm ~/.letta-claim/letta.db

# Recreate database
sqlite3 ~/.letta-claim/letta.db < database_dump.sql
```

---

### Problem: Disk Space Issues

**Symptoms:**
- "No space left on device" errors
- Database growth warnings
- Backup failures
- Log files growing too large

**Diagnostic Steps:**
```bash
# Check disk usage
df -h
du -sh ~/.letta-claim/

# Check largest files
find ~/.letta-claim/ -type f -size +100M -exec ls -lh {} \;

# Check log sizes
du -sh ~/.letta-claim/logs/
```

**Solutions:**

#### Solution 1: Clean Up Logs
```bash
# Rotate logs manually
cd ~/.letta-claim/logs/
for log in *.log; do
    mv "$log" "${log}.old"
    touch "$log"
done

# Remove old logs
find ~/.letta-claim/logs/ -name "*.log.*" -mtime +7 -delete
```

#### Solution 2: Clean Up Backups
```bash
# Remove old backups (keep last 10)
cd ~/.letta-claim/backups/
ls -t *.db | tail -n +11 | xargs -r rm

# Compress backups
gzip ~/.letta-claim/backups/*.db
```

#### Solution 3: Database Maintenance
```bash
# Vacuum database to reclaim space
sqlite3 ~/.letta-claim/letta.db "VACUUM;"

# Prune old memory items
sqlite3 ~/.letta-claim/letta.db "DELETE FROM archival_memory WHERE created_at < date('now', '-90 days');"
```

---

## Security and Permissions Issues

### Problem: Permission Denied Errors

**Symptoms:**
- Cannot write to database
- Log file creation failures
- Configuration file access denied
- "Permission denied" in logs

**Diagnostic Steps:**
```bash
# Check file permissions
ls -la ~/.letta-claim/
ls -la ~/.letta-claim/letta.db

# Check user/group ownership
id
whoami
```

**Solutions:**

#### Solution 1: Fix File Permissions
```bash
# Fix directory permissions
chmod 755 ~/.letta-claim/
chmod 644 ~/.letta-claim/*.yaml ~/.letta-claim/*.toml

# Fix database permissions
chmod 664 ~/.letta-claim/letta.db

# Fix log directory
chmod 755 ~/.letta-claim/logs/
chmod 644 ~/.letta-claim/logs/*.log
```

#### Solution 2: Fix Ownership
```bash
# Change ownership to current user
chown -R $USER:$USER ~/.letta-claim/

# Fix SELinux context if needed (RHEL/CentOS)
restorecon -R ~/.letta-claim/
```

#### Solution 3: Check Directory Structure
```bash
# Recreate directory structure if needed
mkdir -p ~/.letta-claim/{logs,backups,data}
chmod 755 ~/.letta-claim ~/.letta-claim/{logs,backups,data}
```

---

## Provider-Specific Issues

### Problem: Ollama Connection Issues

**Symptoms:**
- "Ollama not available" errors
- Model loading failures
- Generation timeouts
- Empty responses from Ollama

**Diagnostic Steps:**
```bash
# Check Ollama status
sudo systemctl status ollama
curl http://localhost:11434/api/tags

# Test model availability
ollama list
ollama show gpt-oss:20b
```

**Solutions:**

#### Solution 1: Start Ollama Service
```bash
# Start Ollama
sudo systemctl start ollama
sudo systemctl enable ollama

# Verify it's running
curl http://localhost:11434/api/tags
```

#### Solution 2: Pull Required Models
```bash
# Pull generation model
ollama pull gpt-oss:20b

# Pull embedding model
ollama pull nomic-embed-text

# Verify models are available
ollama list
```

#### Solution 3: Configure Ollama Settings
```bash
# Check Ollama configuration
cat /etc/systemd/system/ollama.service

# Adjust memory limits if needed
sudo systemctl edit ollama

# Add to override.conf:
[Service]
Environment="OLLAMA_NUM_PARALLEL=2"
Environment="OLLAMA_MAX_LOADED_MODELS=4"
```

---

### Problem: Gemini API Issues

**Symptoms:**
- API key authentication failures
- Rate limit exceeded errors
- Quota exhausted messages
- Network connectivity issues

**Solutions:**

#### Solution 1: Check API Key Configuration
```bash
# Verify API key is set
echo $GEMINI_API_KEY

# Test API key manually
curl -H "Authorization: Bearer $GEMINI_API_KEY" \
  https://generativelanguage.googleapis.com/v1/models
```

#### Solution 2: Handle Rate Limits
Edit matter configuration:
```json
{
  "letta": {
    "llm_provider": {
      "type": "gemini",
      "rate_limit": {
        "requests_per_minute": 50,
        "retry_backoff": 2.0
      }
    }
  }
}
```

#### Solution 3: Monitor Quota Usage
```bash
# Check API usage (if available)
curl -H "Authorization: Bearer $GEMINI_API_KEY" \
  https://generativelanguage.googleapis.com/v1/usage

# Set spending limits
curl -X POST http://localhost:8000/api/matters/<matter-id>/provider/limits \
  -d '{"daily_limit": 10.00, "monthly_limit": 100.00}'
```

---

## Logging and Debugging

### Enable Debug Logging

Edit `~/.letta-claim/config.toml`:
```toml
[logging]
level = "debug"
structured = true
include_traces = true

[letta]
debug = true
log_requests = true
log_responses = false  # Don't log responses (may contain sensitive data)
```

### Useful Log Patterns

#### Connection Issues
```bash
grep -i "connection\|timeout\|refused" ~/.letta-claim/logs/app.log
```

#### Memory Operations
```bash
grep -i "memory\|store\|recall\|agent" ~/.letta-claim/logs/app.log
```

#### Performance Issues
```bash
grep -i "slow\|timeout\|performance" ~/.letta-claim/logs/app.log
```

#### Error Patterns
```bash
grep -E "ERROR|CRITICAL|FATAL" ~/.letta-claim/logs/app.log
```

### Log Analysis Tools

#### Parse Structured Logs
```bash
# If using JSON logging
cat ~/.letta-claim/logs/app.log | jq 'select(.level=="ERROR")'

# Filter by component
cat ~/.letta-claim/logs/app.log | jq 'select(.component=="letta")'
```

#### Monitor Logs in Real-Time
```bash
# Follow application logs
tail -f ~/.letta-claim/logs/app.log

# Follow Letta server logs  
tail -f ~/.letta-claim/logs/letta-server.log

# Filter for specific patterns
tail -f ~/.letta-claim/logs/app.log | grep -i memory
```

---

## Emergency Procedures

### Complete System Reset

⚠️ **Warning**: This will delete all memory data and configuration.

```bash
# 1. Stop all services
sudo systemctl stop letta-claim-assistant
pkill -f letta

# 2. Backup critical data
cp -r ~/LettaClaims ~/LettaClaims.backup.$(date +%Y%m%d)
cp -r ~/.letta-claim ~/.letta-claim.backup.$(date +%Y%m%d)

# 3. Remove all Letta data
rm -rf ~/.letta-claim/
rm -rf ~/LettaClaims/*/agent_config.json

# 4. Reinstall application
pip uninstall -y letta-claim-assistant letta
pip install letta-claim-assistant[letta]

# 5. Restart and reconfigure
letta-claim-assistant --init
```

### Recovery from Backup

```bash
# 1. Stop services
sudo systemctl stop letta-claim-assistant

# 2. Restore from backup
cp ~/.letta-claim.backup/letta.db ~/.letta-claim/
cp ~/.letta-claim.backup/config.toml ~/.letta-claim/

# 3. Restore matter data
cp -r ~/LettaClaims.backup/* ~/LettaClaims/

# 4. Fix permissions
chown -R $USER:$USER ~/.letta-claim ~/LettaClaims

# 5. Restart services
sudo systemctl start letta-claim-assistant
```

---

## Getting Help

### Before Contacting Support

1. **Gather System Information**:
   ```bash
   # System info
   uname -a
   python --version
   letta --version
   pip list | grep letta
   
   # Error logs
   tail -n 50 ~/.letta-claim/logs/app.log
   tail -n 50 ~/.letta-claim/logs/letta-server.log
   
   # Configuration
   cat ~/.letta-claim/config.toml
   ```

2. **Run Diagnostics**:
   ```bash
   # Health check
   curl http://localhost:8000/api/health/detailed
   
   # System resources
   df -h
   free -h
   ps aux | grep letta
   ```

3. **Try Common Solutions**:
   - Restart the application
   - Check server connectivity
   - Review recent log entries
   - Verify configuration files

### Support Channels

- **Documentation**: Check other documentation files
- **GitHub Issues**: Report bugs and feature requests
- **GitHub Discussions**: Community support and questions
- **Technical Support**: For enterprise users

### Issue Reporting Template

When reporting issues, include:

```
**System Information:**
- OS: Ubuntu 22.04
- Python: 3.11.5
- Letta Version: 0.10.0
- Application Version: 1.0.0

**Issue Description:**
Brief description of the problem

**Steps to Reproduce:**
1. Step one
2. Step two
3. Step three

**Expected Behavior:**
What should happen

**Actual Behavior:**
What actually happens

**Error Logs:**
```
[Paste relevant log entries]
```

**Configuration:**
[Paste relevant configuration sections]
```

---

*This troubleshooting guide covers common issues. If your problem isn't addressed here, please consult the other documentation files or contact support with detailed information about your issue.*