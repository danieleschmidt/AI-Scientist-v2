# AI Scientist v2 - Kubernetes Deployment

This directory contains production-ready Kubernetes manifests for deploying AI Scientist v2 with full enterprise features including auto-scaling, monitoring, cost optimization, and security hardening.

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     AI Scientist v2 - Kubernetes               │
├─────────────────────────────────────────────────────────────────┤
│  Ingress Layer                                                  │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐   │
│  │   Load Balancer │ │   SSL/TLS       │ │   Rate Limiting │   │
│  │   (NGINX)       │ │   Termination   │ │   & Security    │   │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│  Application Layer                                              │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐   │
│  │   AI Scientist  │ │   Auto-scaling  │ │   Health Checks │   │
│  │   Pods (3-10)   │ │   (HPA/VPA)     │ │   & Monitoring  │   │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│  Cache Layer                                                    │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐   │
│  │   Redis Cluster │ │   Distributed   │ │   Circuit       │   │
│  │   (1-3 nodes)   │ │   Caching       │ │   Breakers      │   │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│  Storage Layer                                                  │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐   │
│  │   Persistent    │ │   Experiment    │ │   Log           │   │
│  │   Data (100GB)  │ │   Results       │ │   Storage       │   │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## 📋 Prerequisites

### Required Infrastructure
- **Kubernetes Cluster**: v1.24+ with GPU support
- **GPU Nodes**: NVIDIA GPUs with CUDA 12.4+
- **Storage**: Dynamic provisioning with SSD storage classes
- **Networking**: Ingress controller (NGINX recommended)
- **Monitoring**: Prometheus & Grafana (optional but recommended)

### Required Operators/Controllers
```bash
# Install NVIDIA GPU Operator
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/gpu-operator/v23.9.1/deployments/gpu-operator/gpu-operator.yaml

# Install NGINX Ingress Controller
helm install ingress-nginx ingress-nginx/ingress-nginx

# Install cert-manager for SSL certificates
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml

# Install Prometheus Operator (optional)
helm install prometheus prometheus-community/kube-prometheus-stack
```

## 🚀 Quick Deployment

### 1. Prepare Secrets
```bash
# Create API key secrets
kubectl create secret generic ai-scientist-api-keys \
  --from-literal=OPENAI_API_KEY="your-openai-key" \
  --from-literal=ANTHROPIC_API_KEY="your-anthropic-key" \
  --from-literal=GEMINI_API_KEY="your-gemini-key" \
  --from-literal=S2_API_KEY="your-semantic-scholar-key" \
  --from-literal=REDIS_PASSWORD="$(openssl rand -base64 32)" \
  -n ai-scientist

# Create TLS certificate (Let's Encrypt)
kubectl create secret tls ai-scientist-tls \
  --cert=path/to/tls.crt \
  --key=path/to/tls.key \
  -n ai-scientist
```

### 2. Deploy with Kustomize
```bash
# Deploy to production
kubectl apply -k .

# Verify deployment
kubectl get pods -n ai-scientist
kubectl get svc -n ai-scientist
kubectl get ingress -n ai-scientist
```

### 3. Check Status
```bash
# Monitor deployment progress
kubectl rollout status deployment/ai-scientist -n ai-scientist

# Check auto-scaling
kubectl get hpa -n ai-scientist

# View logs
kubectl logs -f deployment/ai-scientist -n ai-scientist
```

## 🔧 Configuration

### Environment Variables
The deployment supports environment-specific configuration through ConfigMaps and Secrets:

```yaml
# Production environment variables
ENVIRONMENT: production
LOG_LEVEL: INFO
WORKER_PROCESSES: 8
MAX_CONCURRENT_EXPERIMENTS: 5
CACHE_TTL: 7200
COST_TRACKING_ENABLED: true
PROMETHEUS_METRICS_ENABLED: true
```

### Resource Requirements

#### Minimum Requirements
- **CPU**: 1 core per pod
- **Memory**: 2GB per pod  
- **GPU**: 1 GPU per pod
- **Storage**: 100GB for data, 50GB for logs

#### Production Requirements
- **CPU**: 2-8 cores per pod
- **Memory**: 4-16GB per pod
- **GPU**: 1-2 GPUs per pod
- **Storage**: 500GB+ for data, 200GB+ for logs

## 📊 Monitoring & Observability

### Prometheus Metrics
The deployment exposes comprehensive metrics on port 8080:

```
# Cost metrics
ai_scientist_daily_cost
ai_scientist_weekly_cost
ai_scientist_monthly_cost
ai_scientist_cost_by_category

# Performance metrics
ai_scientist_request_duration_seconds
ai_scientist_requests_total
ai_scientist_errors_total

# GPU metrics
ai_scientist_gpu_utilization
ai_scientist_gpu_memory_usage

# Cache metrics
ai_scientist_cache_hit_rate
ai_scientist_cache_operations_total

# Circuit breaker metrics
ai_scientist_circuit_breaker_open_total
```

### Health Checks
- **Liveness Probe**: `/health` endpoint
- **Readiness Probe**: `/ready` endpoint  
- **Startup Probe**: `/startup` endpoint

### Alerts
Pre-configured Prometheus alerts for:
- High daily costs (>$100, >$200)
- High error rates (>10%)
- Low GPU utilization (<30%)
- Application downtime
- Circuit breaker activation

## 🔒 Security

### Network Policies
- Ingress: Only from ingress controller and monitoring
- Egress: Redis, external APIs, and Kubernetes API only
- Pod-to-pod: Restricted to same namespace

### Pod Security
- **Non-root user**: Runs as UID 1000
- **Read-only filesystem**: Except for mounted volumes
- **No privilege escalation**: Security contexts enforced
- **Capabilities dropped**: All unnecessary capabilities removed

### RBAC
- **Service Account**: Minimal permissions
- **Role**: Read-only access to configs, secrets, pods
- **ClusterRole**: Metrics reader only

## 💰 Cost Optimization

### Auto-scaling Configuration
```yaml
# Horizontal Pod Autoscaler
minReplicas: 2
maxReplicas: 10
targetCPU: 70%
targetMemory: 80%
targetGPU: 75%
```

### Cost Controls
- **Budget limits**: Daily ($100), Weekly ($500), Monthly ($2000)
- **Auto-throttling**: Enabled at 80% budget utilization
- **Model optimization**: Automatic cheaper model recommendations
- **Circuit breakers**: Prevent runaway costs from failed API calls

### Resource Optimization
- **Vertical Pod Autoscaler**: Automatic resource right-sizing
- **GPU sharing**: Multiple experiments per GPU when possible
- **Efficient caching**: Multi-layer Redis + memory caching
- **Request batching**: Optimize LLM API call efficiency

## 🔄 Operations

### Scaling Operations
```bash
# Manual scaling
kubectl scale deployment ai-scientist --replicas=5 -n ai-scientist

# Update resource limits
kubectl patch deployment ai-scientist -n ai-scientist -p '{"spec":{"template":{"spec":{"containers":[{"name":"ai-scientist","resources":{"limits":{"memory":"8Gi"}}}]}}}}'

# Check auto-scaling status
kubectl describe hpa ai-scientist-hpa -n ai-scientist
```

### Maintenance Operations
```bash
# Rolling update
kubectl set image deployment/ai-scientist ai-scientist=ai-scientist:v2.1.0 -n ai-scientist

# Restart deployment
kubectl rollout restart deployment/ai-scientist -n ai-scientist

# Rollback deployment
kubectl rollout undo deployment/ai-scientist -n ai-scientist

# Scale down for maintenance
kubectl scale deployment ai-scientist --replicas=0 -n ai-scientist
```

### Backup Operations
```bash
# Backup persistent data
kubectl create job --from=cronjob/backup-data backup-$(date +%Y%m%d-%H%M%S) -n ai-scientist

# Export configuration
kubectl get configmap ai-scientist-config -o yaml > backup-config.yaml

# Backup secrets (encrypted)
kubectl get secret ai-scientist-api-keys -o yaml > backup-secrets.yaml
```

## 🐛 Troubleshooting

### Common Issues

#### Pods Stuck in Pending
```bash
# Check node resources
kubectl describe nodes

# Check GPU availability
kubectl get nodes -l nvidia.com/gpu.present=true

# Check persistent volume claims
kubectl get pvc -n ai-scientist
```

#### High Memory Usage
```bash
# Check memory usage by pod
kubectl top pods -n ai-scientist

# Check for memory leaks
kubectl logs -f deployment/ai-scientist -n ai-scientist | grep -i memory

# Restart high-memory pods
kubectl delete pod -l app.kubernetes.io/name=ai-scientist -n ai-scientist
```

#### API Rate Limiting
```bash
# Check circuit breaker status
kubectl logs -f deployment/ai-scientist -n ai-scientist | grep -i "circuit.*open"

# Reset circuit breakers (if needed)
kubectl exec -it deployment/ai-scientist -n ai-scientist -- curl -X POST http://localhost:8000/admin/circuit-breakers/reset
```

### Debug Commands
```bash
# Get detailed pod information
kubectl describe pod -l app.kubernetes.io/name=ai-scientist -n ai-scientist

# Check resource usage
kubectl top pod -n ai-scientist --containers

# View recent events
kubectl get events --sort-by=.metadata.creationTimestamp -n ai-scientist

# Access pod shell for debugging
kubectl exec -it deployment/ai-scientist -n ai-scientist -- /bin/bash
```

## 📚 Additional Resources

### Dashboards
- **Grafana Dashboards**: Pre-configured in `monitoring.yaml`
- **Cost Analysis**: Real-time cost tracking and budget alerts
- **Performance Metrics**: Request rates, response times, error rates
- **GPU Utilization**: GPU usage, memory, temperature monitoring

### Documentation
- [Kubernetes GPU Guide](https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/)
- [NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/overview.html)
- [Prometheus Monitoring](https://prometheus.io/docs/kubernetes/kubernetes/)
- [Cost Optimization Best Practices](docs/cost-optimization.md)

### Support
- **Issues**: Report issues via GitHub Issues
- **Community**: Join our Slack workspace
- **Enterprise Support**: Contact terragon-labs-support@terragonlabs.ai

---

**Note**: This deployment is production-ready but requires customization for your specific environment. Always review and test configurations before deploying to production.