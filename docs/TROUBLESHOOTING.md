# Letta Construction Claim Assistant - Troubleshooting Guide

This guide helps you diagnose and resolve common issues with the Letta Construction Claim Assistant.

## Quick Diagnostics

### Health Check

Run the health check first to identify issues:

```bash
# Check system health via API
curl http://localhost:8000/api/health/detailed | python3 -m json.tool

# Or run startup validation
python -c "
from app.startup_checks import validate_startup, format_check_results
import asyncio

async def main():
    success, results = await validate_startup()
    print(format_check_results(results))

asyncio.run(main())
"
```

### System Status

Check if core services are running:

```bash
# Check Ollama
ollama list

# Check Python environment
python --version
pip list | grep -E "(nicegui|chromadb|ollama|letta)"

# Check disk space
df -h ~/LettaClaims

# Check memory
free -h
```

## Letta Memory Issues

For comprehensive troubleshooting of Letta agent memory features, see the **[Letta Troubleshooting Guide](LETTA_TROUBLESHOOTING.md)**.

### Quick Letta Diagnostics

#### Check Letta Server Status
```bash
# Test Letta server connectivity
curl http://localhost:8283/v1/health/

# Check if Letta server process is running
ps aux | grep "letta server"

# Check Letta server logs
tail -f ~/.letta-claim/logs/letta-server.log
```

#### Memory Dashboard Indicators
- **Green Connection**: Letta server connected and operational
- **Yellow Warning**: Connection issues or degraded performance
- **Red Error**: Server disconnected or critical errors
- **Memory Item Count**: Should increase as you upload documents and chat
- **Last Sync**: Should show recent timestamps ("Just now", "2m ago")

#### Common Quick Fixes
```bash
# Restart application (fixes most Letta connection issues)
pkill -f letta-claim-assistant
python main.py

# Reset Letta server configuration
rm ~/.letta-claim/letta-server.yaml
# Application will recreate with defaults on next start

# Check memory operations in recent logs
grep -i "memory\|letta\|agent" ~/.letta-claim/logs/app.log | tail -20
```

#### When to Use the Full Letta Troubleshooting Guide
- Memory features not working (no purple badges on responses)
- Agent health indicator showing red/yellow status
- Memory dashboard showing disconnected status
- Memory operations timing out or failing
- Agent not remembering previous conversations
- California domain features not working

*See [Letta Troubleshooting Guide](LETTA_TROUBLESHOOTING.md) for detailed solutions to memory-specific issues.*

## Installation Issues

### Issue: System Dependencies Missing

**Error Messages:**
- `command not found: ocrmypdf`
- `ModuleNotFoundError: No module named 'tesseract'`
- `FileNotFoundError: ghostscript`

**Diagnosis:**
```bash
# Check what's missing
which ocrmypdf
which tesseract
which gs
dpkg -l | grep -E "(ocrmypdf|tesseract|poppler)"
```

**Solution:**
```bash
# Install missing system packages
sudo apt-get update
sudo apt-get install -y \
    ocrmypdf \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-osd \
    poppler-utils \
    ghostscript

# Verify installation
ocrmypdf --version
tesseract --version
```

### Issue: Python Dependencies Conflict

**Error Messages:**
- `pip install` fails with dependency conflicts
- Import errors for core packages

**Diagnosis:**
```bash
# Check virtual environment
echo $VIRTUAL_ENV
pip check
pip list --outdated
```

**Solution:**
```bash
# Recreate virtual environment
deactivate
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Verify critical packages
python -c "import nicegui, chromadb, ollama; print('Core packages OK')"
```

### Issue: Ollama Installation Problems

**Error Messages:**
- `Connection refused` when contacting Ollama
- `ollama: command not found`

**Diagnosis:**
```bash
# Check Ollama installation
which ollama
systemctl status ollama  # If using systemd
pgrep ollama
```

**Solution:**
```bash
# Install or reinstall Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama service
ollama serve &

# Test basic functionality
ollama list
ollama pull llama3.1  # Test model download
```

## Startup Issues

### Issue: Application Won't Start

**Error Messages:**
- `Permission denied` errors
- `Address already in use`
- Import errors

**Diagnosis:**
```bash
# Check for port conflicts
netstat -tulpn | grep :8000
lsof -i :8000

# Check permissions
ls -la ~/.letta-claim/
ls -la ~/LettaClaims/

# Check Python path
python -c "import sys; print('\n'.join(sys.path))"
```

**Solution:**
```bash
# Fix port conflicts
# Kill existing process or change port
pkill -f "python main.py"

# Fix permissions
chmod 700 ~/.letta-claim/
chmod -R 755 ~/LettaClaims/

# Fix Python path (run from project root)
cd /path/to/letta-claim-assistant
python main.py
```

### Issue: Native UI Mode Fails

**Error Messages:**
- `Failed to create native window`
- Falls back to browser mode unexpectedly

**Diagnosis:**
```bash
# Check display environment
echo $DISPLAY
echo $WAYLAND_DISPLAY

# Check GUI dependencies
dpkg -l | grep -E "(python3-tk|python3-pil)"
```

**Solution:**
```bash
# Install GUI dependencies
sudo apt-get install python3-tk python3-pil

# Set display if needed
export DISPLAY=:0

# Or use browser mode explicitly
export LETTA_UI_NATIVE=false
python main.py
```

## Runtime Issues

### Issue: Slow Performance

**Symptoms:**
- Long response times (>30 seconds)
- UI feels sluggish
- High memory usage

**Diagnosis:**
```bash
# Check system resources
top -p $(pgrep -f "python main.py")
free -h
df -h

# Check metrics endpoint
curl http://localhost:8000/api/metrics | python3 -m json.tool
```

**Solutions:**

1. **Switch to Lighter Model:**
```bash
# Edit config to use faster model
sed -i 's/gpt-oss:20b/llama3.1/' ~/.letta-claim/config.toml
```

2. **Reduce Memory Usage:**
   - Close other applications
   - Process fewer documents at once
   - Restart the application periodically

3. **Check Disk Space:**
```bash
# Clean up if needed
du -sh ~/LettaClaims/*/
# Remove unnecessary matters or documents
```

### Issue: High Memory Usage

**Symptoms:**
- System becomes unresponsive
- "Out of memory" errors
- Application crashes

**Diagnosis:**
```bash
# Monitor memory usage
ps aux | grep python
watch -n 5 free -h

# Check application metrics
curl http://localhost:8000/api/health | grep -A5 memory
```

**Solutions:**

1. **Immediate Relief:**
```bash
# Restart application
pkill -f "python main.py"
python main.py
```

2. **Configuration Changes:**
   - Reduce `max_tokens` in settings
   - Switch to `llama3.1` model
   - Process documents in smaller batches

3. **System Optimization:**
```bash
# Increase swap if needed
sudo swapon --show
# Add swap file if necessary
```

### Issue: Document Processing Failures

**Error Messages:**
- "OCR processing failed"
- "PDF parsing error"
- "Embedding generation failed"

**Diagnosis:**
```bash
# Check document format
file /path/to/problematic.pdf
pdfinfo /path/to/problematic.pdf

# Test OCR manually
ocrmypdf --version
ocrmypdf test.pdf test_ocr.pdf

# Check logs
tail -f ~/LettaClaims/Matter_*/logs/app.log
```

**Solutions:**

1. **OCR Issues:**
```bash
# Test with force OCR
# In UI: Enable "Force OCR" option

# Or manually test
ocrmypdf --force-ocr input.pdf output.pdf
```

2. **PDF Format Issues:**
   - Try saving PDF in different format
   - Use PDF repair tools if corrupted
   - Split large PDFs into smaller files

3. **Permission Issues:**
```bash
# Fix file permissions
chmod 644 /path/to/document.pdf
chown $USER:$USER /path/to/document.pdf
```

## Model and AI Issues

### Issue: Models Not Available

**Error Messages:**
- "Model not found"
- "Failed to connect to Ollama"
- Empty model list in settings

**Diagnosis:**
```bash
# Check Ollama status
ollama list
ollama ps

# Check model downloads
du -sh ~/.ollama/models/
```

**Solution:**
```bash
# Ensure Ollama is running
ollama serve &

# Pull required models
ollama pull gpt-oss:20b
ollama pull nomic-embed-text

# Alternative lighter models
ollama pull llama3.1
ollama pull mxbai-embed-large
```

### Issue: Poor AI Response Quality

**Symptoms:**
- Vague or incorrect answers
- Missing citations
- Irrelevant responses

**Diagnosis:**
- Check if documents processed successfully
- Verify question relates to uploaded content
- Review source quality and OCR accuracy

**Solutions:**

1. **Improve Document Quality:**
   - Re-upload with "Force OCR" enabled
   - Ensure documents are readable
   - Check OCR language settings

2. **Adjust Settings:**
   - Lower temperature for more focused responses
   - Increase context size for more relevant chunks
   - Try different embedding models

3. **Question Refinement:**
   - Be more specific in questions
   - Use domain-specific terminology
   - Reference specific documents or dates

### Issue: External API Problems (Gemini)

**Error Messages:**
- "API key invalid"
- "Rate limit exceeded"
- "Service unavailable"

**Diagnosis:**
```bash
# Test API key
curl -H "Authorization: Bearer YOUR_API_KEY" \
     https://generativelanguage.googleapis.com/v1/models

# Check rate limits and quotas
```

**Solutions:**

1. **API Key Issues:**
   - Verify key is correct and active
   - Check API permissions
   - Regenerate key if necessary

2. **Rate Limiting:**
   - Reduce query frequency
   - Switch to local models temporarily
   - Upgrade API plan if needed

3. **Service Issues:**
   - Check Google AI status page
   - Switch to local Ollama temporarily
   - Retry after delay

## Data and Storage Issues

### Issue: Disk Space Problems

**Error Messages:**
- "No space left on device"
- "Failed to write file"
- Slow performance

**Diagnosis:**
```bash
# Check available space
df -h ~/LettaClaims/
du -sh ~/LettaClaims/*/

# Check for large files
find ~/LettaClaims/ -size +100M -ls
```

**Solutions:**

1. **Clean Up Data:**
```bash
# Remove old matters
rm -rf ~/LettaClaims/Matter_old_project/

# Clean up processed files
find ~/LettaClaims/ -name "*.log" -mtime +30 -delete

# Compress old documents
gzip ~/LettaClaims/*/docs_ocr/*.pdf
```

2. **Move Data Location:**
   - Change `paths.root` in configuration
   - Move `~/LettaClaims/` to larger drive
   - Create symlink to new location

### Issue: Data Corruption

**Symptoms:**
- Cannot load matters
- Missing chat history
- Vector search failures

**Diagnosis:**
```bash
# Check file integrity
find ~/LettaClaims/ -name "*.json" -exec python3 -m json.tool {} \; > /dev/null

# Check database files
sqlite3 ~/LettaClaims/Matter_*/knowledge/graph.sqlite3 ".schema"
```

**Solutions:**

1. **Restore from Backup:**
   - Restore from recent backup if available
   - Recreate matter if no backup exists

2. **Repair Data:**
```bash
# Rebuild vector database
# In UI: Remove and re-upload documents

# Reset AI memory
rm -rf ~/LettaClaims/Matter_*/knowledge/letta_state/
# Restart application to rebuild
```

## Network and Connectivity Issues

### Issue: Cannot Access UI

**Error Messages:**
- "Connection refused"
- "This site can't be reached"
- Blank page in browser

**Diagnosis:**
```bash
# Check if application is running
pgrep -f "python main.py"
netstat -tulpn | grep :8000

# Check firewall
sudo ufw status
iptables -L | grep 8000
```

**Solutions:**

1. **Application Not Running:**
```bash
# Start application
cd /path/to/letta-claim-assistant
source .venv/bin/activate
python main.py
```

2. **Port Issues:**
```bash
# Use different port
export LETTA_PORT=8080
python main.py

# Or kill conflicting process
lsof -ti:8000 | xargs kill
```

3. **Browser Issues:**
   - Try different browser
   - Clear browser cache
   - Disable browser extensions

### Issue: External Service Connectivity

**Error Messages:**
- "Connection timeout"
- "DNS resolution failed"
- "SSL verification failed"

**Diagnosis:**
```bash
# Test internet connectivity
ping -c 3 google.com
curl -I https://api.anthropic.com

# Check DNS
nslookup generativelanguage.googleapis.com
```

**Solutions:**

1. **Network Connectivity:**
   - Check internet connection
   - Verify DNS settings
   - Try different network

2. **Proxy/Firewall:**
   - Configure proxy settings if needed
   - Check corporate firewall rules
   - Use VPN if required

## Advanced Troubleshooting

### Debug Mode

Enable debug logging for detailed information:

```bash
# Set debug environment
export DEBUG=1
export LETTA_LOG_LEVEL=DEBUG

# Start with verbose logging
python main.py
```

### Log Analysis

Examine logs for error patterns:

```bash
# Application logs
tail -f ~/.letta-claim/logs/app.log

# Error logs
grep -i error ~/.letta-claim/logs/error.log

# Performance logs
grep -i performance ~/.letta-claim/logs/performance.log
```

### Database Debugging

Check database states:

```bash
# ChromaDB status
python3 -c "
import chromadb
client = chromadb.PersistentClient('~/LettaClaims/Matter_test/vectors/chroma')
print(f'Collections: {client.list_collections()}')
"

# Letta agent status
ls -la ~/LettaClaims/Matter_*/knowledge/letta_state/
```

### Performance Profiling

Profile application performance:

```bash
# Check API endpoints
time curl http://localhost:8000/api/health
time curl http://localhost:8000/api/metrics

# Monitor resources during operation
top -p $(pgrep -f "python main.py")
```

## Recovery Procedures

### Complete Application Reset

If all else fails, perform a complete reset:

```bash
# Stop application
pkill -f "python main.py"

# Backup data
cp -r ~/LettaClaims ~/LettaClaims.backup

# Reset environment
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Reset configuration
rm -rf ~/.letta-claim
cp config.toml.example ~/.letta-claim/config.toml

# Restart application
python main.py
```

### Matter Recovery

Recover a specific matter:

```bash
# Backup matter
cp -r ~/LettaClaims/Matter_name ~/LettaClaims/Matter_name.backup

# Reset matter processing
rm -rf ~/LettaClaims/Matter_name/vectors/
rm -rf ~/LettaClaims/Matter_name/parsed/
rm -rf ~/LettaClaims/Matter_name/knowledge/

# Re-upload documents through UI
```

## Getting Help

### Self-Service Resources

1. **System Health Check:**
   ```bash
   curl http://localhost:8000/api/health/detailed
   ```

2. **Validation Report:**
   ```bash
   curl http://localhost:8000/api/system/validation
   ```

3. **Log Files:**
   - Application: `~/.letta-claim/logs/app.log`
   - Errors: `~/.letta-claim/logs/error.log`
   - Performance: `~/.letta-claim/logs/performance.log`

### Community Support

- **GitHub Issues**: Report bugs and request features
- **Discussions**: Community Q&A and tips
- **Documentation**: Wiki and guides

### Professional Support

For business-critical issues:
- Include health check output
- Provide relevant log excerpts
- Describe steps to reproduce
- Specify system configuration

## Prevention

### Regular Maintenance

1. **Weekly:**
   - Check disk space
   - Review error logs
   - Update models if needed

2. **Monthly:**
   - Run system validation
   - Clean up old logs
   - Backup important matters

3. **Quarterly:**
   - Update application
   - Review configuration
   - Archive completed matters

### Monitoring

Set up monitoring for:
- Disk space usage
- Memory consumption
- Error rates
- Response times

### Best Practices

1. **Resource Management:**
   - Monitor system resources
   - Process documents in batches
   - Clean up old data regularly

2. **Data Protection:**
   - Regular backups
   - Test restore procedures
   - Document recovery steps

3. **Performance Optimization:**
   - Use appropriate models for hardware
   - Monitor and tune settings
   - Keep system updated

---

**Still Having Issues?**
- Check the [Configuration Guide](CONFIGURATION.md) for detailed settings
- Review the [Installation Guide](INSTALLATION.md) for setup verification
- Consult the [User Guide](USER_GUIDE.md) for usage questions