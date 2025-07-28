# 🚨 Incident Response Runbook

## Overview
This runbook provides step-by-step procedures for responding to incidents in the AI Scientist v2 system.

## 📞 Emergency Contacts
- **On-call Engineer**: [TBD]
- **Team Lead**: [TBD]
- **Security Team**: [TBD]

## 🔍 Incident Classification

### Severity Levels

#### P0 - Critical
- Complete system outage
- Data loss or corruption
- Security breach
- **Response Time**: 15 minutes

#### P1 - High
- Significant feature degradation
- Performance severely impacted
- **Response Time**: 1 hour

#### P2 - Medium
- Minor feature issues
- Some users affected
- **Response Time**: 4 hours

#### P3 - Low
- Cosmetic issues
- Minimal user impact
- **Response Time**: 24 hours

## 🔧 Common Incident Procedures

### 1. System Outage Response

#### Immediate Actions (0-15 minutes)
1. **Acknowledge the incident** in monitoring system
2. **Check system status**:
   ```bash
   # Check service health
   curl -f http://localhost:8080/health
   
   # Check resource usage
   htop
   docker stats
   ```
3. **Notify stakeholders** via incident channel
4. **Begin investigation**

#### Investigation Steps
1. **Check logs**:
   ```bash
   # Application logs
   tail -f /var/log/ai-scientist/app.log
   
   # System logs
   journalctl -u ai-scientist -f
   ```

2. **Check infrastructure**:
   - CPU, memory, disk usage
   - Network connectivity
   - Database connectivity
   - External API status

3. **Check recent deployments**:
   ```bash
   git log --oneline -10
   ```

#### Recovery Actions
1. **Immediate fixes**:
   - Restart services if needed
   - Scale resources if capacity issue
   - Rollback if recent deployment caused issue

2. **Service restart**:
   ```bash
   # Docker restart
   docker-compose restart ai-scientist
   
   # System service restart
   sudo systemctl restart ai-scientist
   ```

3. **Rollback procedure**:
   ```bash
   # Rollback to previous release
   git checkout <previous-tag>
   docker-compose up -d --build
   ```

### 2. Performance Degradation

#### Diagnosis
1. **Check resource usage**:
   ```bash
   # CPU and memory
   top -p $(pgrep -f ai-scientist)
   
   # GPU usage (if applicable)
   nvidia-smi
   ```

2. **Check database performance**:
   ```sql
   -- Check slow queries
   SELECT * FROM pg_stat_activity WHERE state = 'active';
   ```

3. **Check external dependencies**:
   - API response times
   - Rate limiting status
   - Network latency

#### Mitigation
1. **Scale resources** if needed
2. **Optimize queries** or processes
3. **Enable circuit breakers** for external services
4. **Implement caching** where appropriate

### 3. Security Incident Response

#### Immediate Actions
1. **Isolate affected systems**
2. **Preserve evidence**
3. **Notify security team**
4. **Change credentials** if compromised

#### Investigation
1. **Check access logs**
2. **Review system changes**
3. **Scan for malware**
4. **Analyze network traffic**

## 📋 Post-Incident Procedures

### 1. Resolution Confirmation
- [ ] System fully operational
- [ ] All monitoring green
- [ ] User access restored
- [ ] Performance within SLA

### 2. Communication
- [ ] Update stakeholders
- [ ] Close incident ticket
- [ ] Send resolution notification

### 3. Post-Mortem
- [ ] Schedule post-mortem within 48 hours
- [ ] Document timeline of events
- [ ] Identify root cause
- [ ] Create action items for improvement

## 📊 Monitoring and Alerting

### Key Metrics to Monitor
- **Response Time**: < 2 seconds average
- **Error Rate**: < 1% 
- **Uptime**: > 99.9%
- **Resource Usage**: < 80% capacity

### Alert Thresholds
- **High response time**: > 5 seconds
- **High error rate**: > 5%
- **Resource usage**: > 90%
- **Service down**: Health check fails

## 🔗 Useful Commands

### Docker Commands
```bash
# View container logs
docker logs ai-scientist --tail 100 -f

# Check container status
docker ps -a

# Restart specific service
docker-compose restart <service-name>
```

### System Commands
```bash
# Check disk space
df -h

# Check memory usage
free -h

# Check network connections
netstat -tuln

# Check system load
uptime
```

### Database Commands
```sql
-- Check database connections
SELECT count(*) FROM pg_stat_activity;

-- Check database size
SELECT pg_size_pretty(pg_database_size('ai_scientist'));

-- Kill long-running queries
SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE query_start < now() - interval '1 hour';
```

## 📚 Additional Resources
- [Architecture Documentation](../ARCHITECTURE.md)
- [Deployment Guide](DEPLOYMENT.md)
- [Monitoring Dashboard](http://monitoring.example.com)
- [Log Aggregation](http://logs.example.com)