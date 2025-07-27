# AI Scientist v2 - Deployment Guide

This guide covers deploying AI Scientist v2 in various environments, from local development to production clusters.

## Quick Start

### Local Development
```bash
# Clone and setup
git clone https://github.com/SakanaAI/AI-Scientist-v2.git
cd AI-Scientist-v2
make setup

# Start with Docker Compose
docker-compose up -d

# Access services
# - AI Scientist: http://localhost:8000
# - Jupyter Lab: http://localhost:8888
# - Grafana: http://localhost:3000
```

## Environment Setup

### Development Environment
```bash
# Start development stack
make dev

# Run with custom configuration
export ENVIRONMENT=development
export DEBUG=true
docker-compose -f docker-compose.dev.yml up
```

### Staging Environment
```bash
# Deploy to staging
export ENVIRONMENT=staging
docker-compose -f docker-compose.staging.yml up -d

# Run smoke tests
python scripts/smoke_tests.py --env staging
```

### Production Environment
```bash
# Production deployment with Kubernetes
kubectl apply -f k8s/production/

# Verify deployment
kubectl rollout status deployment/ai-scientist-production
```

## Configuration Management

### Environment Variables

#### Required Variables
```bash
# LLM API Keys
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"

# Database Configuration
export DATABASE_URL="postgresql://user:pass@host:5432/db"
export REDIS_URL="redis://localhost:6379"

# Security Settings
export SECRET_KEY="your-secret-key"
export SANDBOX_ENABLED="true"
```

#### Optional Variables
```bash
# Monitoring and Observability
export PROMETHEUS_ENABLED="true"
export GRAFANA_PASSWORD="secure-password"

# Performance Tuning
export MAX_CONCURRENT_EXPERIMENTS="5"
export EXPERIMENT_TIMEOUT="3600"

# Cost Management
export MAX_DAILY_COST="100"
export COST_TRACKING_ENABLED="true"
```

### Configuration Files

#### Main Configuration (`config/production.yaml`)
```yaml
environment: production
debug: false

# Database settings
database:
  url: ${DATABASE_URL}
  pool_size: 20
  max_overflow: 0

# Redis settings
cache:
  url: ${REDIS_URL}
  default_ttl: 3600

# Model configuration
models:
  default: gpt-4
  fallback: gpt-3.5-turbo
  providers:
    openai:
      api_key: ${OPENAI_API_KEY}
      rate_limit: 60
    anthropic:
      api_key: ${ANTHROPIC_API_KEY}
      rate_limit: 30

# Security settings
security:
  sandbox_enabled: true
  max_execution_time: 300
  max_memory_usage: 4096
  network_access: false

# Monitoring
monitoring:
  prometheus:
    enabled: true
    port: 9090
  grafana:
    enabled: true
    port: 3000
```

## Container Deployment

### Docker

#### Building Images
```bash
# Build production image
docker build -t ai-scientist:latest .

# Build development image
docker build -t ai-scientist:dev --target development .

# Multi-architecture build
docker buildx build --platform linux/amd64,linux/arm64 -t ai-scientist:latest .
```

#### Running Containers
```bash
# Run single container
docker run -d \
  --name ai-scientist \
  -p 8000:8000 \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
  ai-scientist:latest

# Run with GPU support
docker run -d \
  --name ai-scientist-gpu \
  --gpus all \
  -p 8000:8000 \
  ai-scientist:latest
```

### Docker Compose

#### Production Stack
```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  ai-scientist:
    image: ai-scientist:latest
    environment:
      - ENVIRONMENT=production
      - DATABASE_URL=postgresql://ai_scientist:${DB_PASSWORD}@postgres:5432/ai_scientist
      - REDIS_URL=redis://redis:6379
    depends_on:
      - postgres
      - redis
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=ai_scientist
      - POSTGRES_USER=ai_scientist
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
```

## Kubernetes Deployment

### Namespace Setup
```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: ai-scientist
  labels:
    name: ai-scientist
```

### ConfigMap
```yaml
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: ai-scientist-config
  namespace: ai-scientist
data:
  config.yaml: |
    environment: production
    debug: false
    models:
      default: gpt-4
      providers:
        openai:
          rate_limit: 60
```

### Secrets
```yaml
# k8s/secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: ai-scientist-secrets
  namespace: ai-scientist
type: Opaque
stringData:
  openai-api-key: "your-openai-key"
  anthropic-api-key: "your-anthropic-key"
  database-url: "postgresql://user:pass@postgres:5432/ai_scientist"
```

### Deployment
```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-scientist
  namespace: ai-scientist
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ai-scientist
  template:
    metadata:
      labels:
        app: ai-scientist
    spec:
      containers:
      - name: ai-scientist
        image: ai-scientist:latest
        ports:
        - containerPort: 8000
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: ai-scientist-secrets
              key: openai-api-key
        resources:
          requests:
            cpu: 500m
            memory: 1Gi
            nvidia.com/gpu: 1
          limits:
            cpu: 2
            memory: 4Gi
            nvidia.com/gpu: 1
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

### Service
```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: ai-scientist-service
  namespace: ai-scientist
spec:
  selector:
    app: ai-scientist
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

## Monitoring Setup

### Prometheus Configuration
```yaml
# monitoring/prometheus-k8s.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
data:
  prometheus.yml: |
    scrape_configs:
    - job_name: 'ai-scientist'
      kubernetes_sd_configs:
      - role: pod
      relabel_configs:
      - source_labels: [__meta_kubernetes_pod_label_app]
        action: keep
        regex: ai-scientist
```

### Grafana Dashboards
```bash
# Deploy monitoring stack
kubectl apply -f monitoring/k8s/

# Import dashboards
kubectl create configmap grafana-dashboards \
  --from-file=monitoring/grafana/dashboards/
```

## Security Considerations

### Network Security
```yaml
# k8s/network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: ai-scientist-netpol
spec:
  podSelector:
    matchLabels:
      app: ai-scientist
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: ingress-controller
  egress:
  - to: []
    ports:
    - protocol: TCP
      port: 443  # HTTPS only
```

### RBAC
```yaml
# k8s/rbac.yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: ai-scientist
  namespace: ai-scientist
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: ai-scientist-role
rules:
- apiGroups: [""]
  resources: ["pods", "configmaps"]
  verbs: ["get", "list", "watch"]
```

## Scaling and Performance

### Horizontal Pod Autoscaler
```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ai-scientist-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ai-scientist
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Vertical Pod Autoscaler
```yaml
# k8s/vpa.yaml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: ai-scientist-vpa
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ai-scientist
  updatePolicy:
    updateMode: "Auto"
```

## Backup and Recovery

### Database Backup
```bash
# Create backup
kubectl exec -it postgres-pod -- pg_dump ai_scientist > backup.sql

# Automated backup with CronJob
kubectl apply -f k8s/backup-cronjob.yaml
```

### Disaster Recovery
```bash
# Backup configurations
kubectl get all -n ai-scientist -o yaml > ai-scientist-backup.yaml

# Restore from backup
kubectl apply -f ai-scientist-backup.yaml
```

## Troubleshooting

### Common Issues

#### Pod Not Starting
```bash
# Check pod status
kubectl describe pod ai-scientist-xxx

# Check logs
kubectl logs ai-scientist-xxx

# Common solutions:
# 1. Check resource limits
# 2. Verify secrets and configmaps
# 3. Check image pull policy
```

#### High Memory Usage
```bash
# Check memory usage
kubectl top pod ai-scientist-xxx

# Solutions:
# 1. Increase memory limits
# 2. Enable memory monitoring
# 3. Check for memory leaks
```

#### GPU Not Available
```bash
# Check GPU nodes
kubectl get nodes -l accelerator=nvidia-tesla-k80

# Check GPU allocation
kubectl describe node gpu-node

# Solutions:
# 1. Install GPU device plugin
# 2. Check GPU driver
# 3. Verify resource requests
```

### Debugging Commands
```bash
# Shell into container
kubectl exec -it ai-scientist-xxx -- /bin/bash

# Port forward for local access
kubectl port-forward svc/ai-scientist-service 8000:80

# View real-time logs
kubectl logs -f ai-scientist-xxx

# Check resource usage
kubectl top pod ai-scientist-xxx
```

## Performance Optimization

### Database Tuning
```sql
-- PostgreSQL optimizations
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
```

### Redis Configuration
```bash
# Redis optimizations
redis-cli CONFIG SET maxmemory 512mb
redis-cli CONFIG SET maxmemory-policy allkeys-lru
```

### Application Tuning
```python
# Python optimizations
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
```

## Cost Management

### Resource Optimization
```yaml
# Set appropriate resource limits
resources:
  requests:
    cpu: 100m
    memory: 256Mi
  limits:
    cpu: 500m
    memory: 1Gi
```

### Monitoring Costs
```bash
# Track resource usage
kubectl cost-analyzer --namespace ai-scientist

# Set up cost alerts
kubectl apply -f monitoring/cost-alerts.yaml
```

---

For more deployment assistance, contact our DevOps team at devops@terragonlabs.ai