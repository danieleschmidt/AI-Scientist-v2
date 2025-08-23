# AI Scientist v2 Production Deployment Guide

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Architecture](#architecture)
4. [Quick Start](#quick-start)
5. [Detailed Deployment](#detailed-deployment)
6. [Configuration Management](#configuration-management)
7. [Monitoring and Observability](#monitoring-and-observability)
8. [Security](#security)
9. [Scaling](#scaling)
10. [Troubleshooting](#troubleshooting)
11. [Maintenance](#maintenance)

## Overview

This guide provides comprehensive instructions for deploying AI Scientist v2 in production environments. The deployment uses Docker containers orchestrated by Kubernetes, with comprehensive monitoring, security, and scalability features.

### Key Features

- **Multi-stage Docker builds** for optimized production images
- **Kubernetes-native deployment** with auto-scaling and self-healing
- **Comprehensive monitoring** with Prometheus and Grafana
- **Security hardening** with RBAC, network policies, and container security
- **High availability** with multi-replica deployments and load balancing
- **GPU support** for accelerated machine learning workloads
- **Automated health checks** and recovery mechanisms

## Prerequisites

### Infrastructure Requirements

#### Minimum Requirements
- **Kubernetes cluster**: v1.24+ with 3+ nodes
- **Node specifications**: 
  - CPU: 4+ cores per node
  - Memory: 16GB+ RAM per node
  - Storage: 100GB+ SSD per node
- **GPU nodes**: NVIDIA GPUs with CUDA support (recommended)
- **Network**: High-speed internal networking (10Gbps+)

#### Production Requirements
- **Kubernetes cluster**: v1.24+ with 5+ nodes
- **Node specifications**: 
  - CPU: 8+ cores per node
  - Memory: 32GB+ RAM per node
  - Storage: 500GB+ NVMe SSD per node
- **GPU nodes**: NVIDIA A100/V100 with 16GB+ VRAM
- **Network**: Redundant 25Gbps+ networking
- **Load balancer**: External load balancer with SSL termination

### Software Requirements

#### Required Tools
```bash
# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
chmod +x kubectl && sudo mv kubectl /usr/local/bin/

# Install Helm
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# Install Docker
curl -fsSL https://get.docker.com | sh

# Install docker-compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

#### Cluster Add-ons
- **NVIDIA GPU Operator** (for GPU support)
- **Cert-manager** (for SSL certificates)
- **Ingress controller** (NGINX recommended)
- **External DNS** (for automatic DNS management)
- **Cluster autoscaler** (for dynamic scaling)

### Access Requirements

#### API Keys
- **Anthropic API Key**: Claude model access
- **OpenAI API Key**: GPT model access

#### Registry Access
- **Docker registry** credentials for image storage
- **Kubernetes RBAC** permissions for deployment

#### External Services
- **DNS management** access for domain configuration
- **SSL certificate** management (Let's Encrypt recommended)
- **Monitoring storage** (Prometheus remote storage)
- **Log aggregation** service (ELK stack or similar)

## Architecture

### Deployment Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Load Balancer                            │
│              (SSL Termination)                              │
└─────────────────┬───────────────────────────────────────────┘
                  │
┌─────────────────┴───────────────────────────────────────────┐
│                 Ingress Controller                          │
│              (NGINX/Traefik)                                │
└─────────────────┬───────────────────────────────────────────┘
                  │
    ┌─────────────┼─────────────┐
    │             │             │
┌───▼───┐    ┌────▼────┐   ┌────▼────┐
│ App-1 │    │ App-2   │   │ App-3   │
│       │    │         │   │         │
└───┬───┘    └────┬────┘   └────┬────┘
    │             │             │
    └─────────────┼─────────────┘
                  │
┌─────────────────┴───────────────────────────────────────────┐
│                  Worker Nodes                               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐     │
│  │Worker-1  │  │Worker-2  │  │Worker-3  │  │Worker-4  │     │
│  │(GPU)     │  │(GPU)     │  │(GPU)     │  │(CPU)     │     │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘     │
└─────────────────────────────────────────────────────────────┘
                  │
┌─────────────────┴───────────────────────────────────────────┐
│                Support Services                             │
│  ┌─────────┐  ┌─────────┐  ┌───────────┐  ┌──────────────┐   │
│  │ Redis   │  │Postgres │  │Prometheus │  │  Grafana     │   │
│  │ Cache   │  │Database │  │Monitoring │  │ Dashboard    │   │
│  └─────────┘  └─────────┘  └───────────┘  └──────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Component Overview

#### Core Application
- **AI Scientist Main App**: Primary application containers
- **Worker Processes**: Background task processors
- **API Gateway**: Request routing and load balancing

#### Infrastructure
- **Redis**: Caching and message broker
- **PostgreSQL**: Primary database (optional)
- **Storage**: Persistent volumes for data and logs

#### Monitoring Stack
- **Prometheus**: Metrics collection and alerting
- **Grafana**: Visualization and dashboards
- **AlertManager**: Alert routing and notifications
- **Jaeger**: Distributed tracing (optional)

## Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/your-org/ai-scientist-v2.git
cd ai-scientist-v2
```

### 2. Configure Environment
```bash
# Copy and edit configuration
cp deployment/configs/production.yaml.template deployment/configs/production.yaml
```

### 3. Set Required Environment Variables
```bash
export ANTHROPIC_API_KEY="your-anthropic-key"
export OPENAI_API_KEY="your-openai-key"
export SECRET_KEY="your-secret-key"
export JWT_SECRET_KEY="your-jwt-secret"
```

### 4. Build and Deploy
```bash
# Build Docker image
./deployment/scripts/build.sh --tag v2.0.0 --push

# Deploy to Kubernetes
./deployment/scripts/deploy.sh --environment production --tag v2.0.0
```

### 5. Verify Deployment
```bash
# Check pod status
kubectl get pods -n ai-scientist

# Check service status
kubectl get services -n ai-scientist

# Test health endpoint
kubectl port-forward -n ai-scientist service/ai-scientist 8000:8000
curl http://localhost:8000/health
```

## Detailed Deployment

### Step 1: Prepare Infrastructure

#### 1.1 Kubernetes Cluster Setup
```bash
# Verify cluster access
kubectl cluster-info

# Check node resources
kubectl get nodes -o wide

# Verify GPU nodes (if applicable)
kubectl get nodes -l nvidia.com/gpu.present=true
```

#### 1.2 Install Required Add-ons
```bash
# Install NVIDIA GPU Operator
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/gpu-operator/master/deployments/gpu-operator.yaml

# Install cert-manager
kubectl apply -f https://github.com/jetstack/cert-manager/releases/download/v1.12.0/cert-manager.yaml

# Install NGINX ingress
helm upgrade --install ingress-nginx ingress-nginx \
  --repo https://kubernetes.github.io/ingress-nginx \
  --namespace ingress-nginx --create-namespace
```

### Step 2: Configure Secrets and ConfigMaps

#### 2.1 Create Namespace
```bash
kubectl apply -f deployment/kubernetes/namespace.yaml
```

#### 2.2 Configure Secrets
```bash
# Edit secrets template with actual values
cp deployment/kubernetes/secrets.yaml deployment/kubernetes/secrets-prod.yaml

# Apply secrets (ensure you've updated the values)
kubectl apply -f deployment/kubernetes/secrets-prod.yaml
```

#### 2.3 Apply ConfigMaps
```bash
kubectl apply -f deployment/kubernetes/configmap.yaml
```

### Step 3: Build and Push Images

#### 3.1 Build Production Image
```bash
# Set registry and image details
export REGISTRY="your-registry.com"
export IMAGE_TAG="v2.0.0"

# Build and push image
./deployment/scripts/build.sh \
  --registry "$REGISTRY" \
  --tag "$IMAGE_TAG" \
  --target production \
  --push
```

#### 3.2 Verify Image
```bash
# Pull and test image
docker pull "$REGISTRY/ai-scientist:$IMAGE_TAG"
docker run --rm "$REGISTRY/ai-scientist:$IMAGE_TAG" --help
```

### Step 4: Deploy Application

#### 4.1 Deploy Core Components
```bash
# Apply in order
kubectl apply -f deployment/kubernetes/rbac.yaml
kubectl apply -f deployment/kubernetes/pvc.yaml
kubectl apply -f deployment/kubernetes/service.yaml
kubectl apply -f deployment/kubernetes/deployment.yaml
```

#### 4.2 Configure Auto-scaling
```bash
kubectl apply -f deployment/kubernetes/hpa.yaml
```

#### 4.3 Set up Ingress
```bash
# Update ingress with your domain
sed -i 's/ai-scientist.yourdomain.com/your-actual-domain.com/g' deployment/kubernetes/ingress.yaml
kubectl apply -f deployment/kubernetes/ingress.yaml
```

### Step 5: Deploy Monitoring

#### 5.1 Install Prometheus Stack
```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace ai-scientist \
  --values deployment/monitoring/prometheus-values.yaml
```

#### 5.2 Configure Grafana Dashboards
```bash
kubectl create configmap grafana-dashboards \
  --from-file=deployment/monitoring/grafana-dashboard.json \
  -n ai-scientist
```

## Configuration Management

### Environment-Specific Configurations

#### Production Configuration
```yaml
# deployment/configs/production.yaml
application:
  environment: production
  debug: false
  log_level: INFO

security:
  ssl_redirect: true
  secure_cookies: true
  
performance:
  workers: 8
  max_memory_usage: 85
```

#### Staging Configuration
```yaml
# deployment/configs/staging.yaml
application:
  environment: staging
  debug: true
  log_level: DEBUG

security:
  ssl_redirect: false
  secure_cookies: false
  
performance:
  workers: 4
  max_memory_usage: 80
```

### Secret Management

#### Using External Secret Operator
```yaml
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: ai-scientist-secrets
spec:
  secretStoreRef:
    name: vault-backend
    kind: SecretStore
  target:
    name: ai-scientist-secrets
  data:
  - secretKey: ANTHROPIC_API_KEY
    remoteRef:
      key: ai-scientist/api-keys
      property: anthropic_key
```

#### Using Kubernetes Secrets Directly
```bash
# Create secret from command line
kubectl create secret generic ai-scientist-secrets \
  --from-literal=ANTHROPIC_API_KEY="your-key" \
  --from-literal=OPENAI_API_KEY="your-key" \
  -n ai-scientist
```

## Monitoring and Observability

### Prometheus Metrics

#### Application Metrics
- `ai_scientist_experiments_total`: Total experiments run
- `ai_scientist_experiments_duration`: Experiment execution time
- `ai_scientist_active_experiments`: Currently running experiments
- `ai_scientist_queue_size`: Experiment queue backlog

#### Infrastructure Metrics
- CPU, memory, disk usage per node
- GPU utilization and memory usage
- Network I/O and latency
- Kubernetes pod and container metrics

### Grafana Dashboards

#### Main Dashboard Panels
1. **System Overview**: Application status and key metrics
2. **Performance**: Response times, throughput, error rates
3. **Resource Usage**: CPU, memory, GPU utilization
4. **Experiments**: Success rates, duration, queue status
5. **Infrastructure**: Node status, storage usage

### Alerting Rules

#### Critical Alerts
- Application down for > 1 minute
- High error rate (>10%) for > 2 minutes
- GPU temperature > 85°C
- Disk space > 95%
- Memory usage > 90%

#### Warning Alerts
- High response time (>2s 95th percentile)
- Low experiment success rate (<80%)
- High queue backlog (>50 items)
- Certificate expiring (<30 days)

### Log Aggregation

#### Centralized Logging Setup
```bash
# Install ELK stack
helm repo add elastic https://helm.elastic.co
helm install elasticsearch elastic/elasticsearch -n ai-scientist
helm install kibana elastic/kibana -n ai-scientist
helm install filebeat elastic/filebeat -n ai-scientist
```

#### Log Structure
```json
{
  "timestamp": "2024-01-01T12:00:00Z",
  "level": "INFO",
  "logger": "ai_scientist.experiments",
  "message": "Experiment completed successfully",
  "experiment_id": "exp-123",
  "duration": 1800,
  "success": true
}
```

## Security

### Container Security

#### Security Scanning
```bash
# Scan image for vulnerabilities
trivy image your-registry.com/ai-scientist:v2.0.0

# Generate security report
trivy image --format json --output security-report.json your-registry.com/ai-scientist:v2.0.0
```

#### Runtime Security
- Non-root user execution
- Read-only root filesystem
- Dropped capabilities
- seccomp and AppArmor profiles

### Network Security

#### Network Policies
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: ai-scientist-netpol
spec:
  podSelector:
    matchLabels:
      app.kubernetes.io/name: ai-scientist
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8000
```

#### SSL/TLS Configuration
```yaml
apiVersion: cert-manager.io/v1
kind: Certificate
metadata:
  name: ai-scientist-tls
spec:
  secretName: ai-scientist-tls
  issuerRef:
    name: letsencrypt-prod
    kind: ClusterIssuer
  dnsNames:
  - ai-scientist.yourdomain.com
```

### RBAC Configuration

#### Service Account
```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: ai-scientist
  namespace: ai-scientist
```

#### Role and RoleBinding
```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: ai-scientist-role
rules:
- apiGroups: [""]
  resources: ["pods", "services", "configmaps"]
  verbs: ["get", "list", "watch"]
```

## Scaling

### Horizontal Pod Autoscaler

#### CPU-based Scaling
```yaml
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
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

#### Custom Metrics Scaling
```yaml
metrics:
- type: Pods
  pods:
    metric:
      name: ai_scientist_queue_size
    target:
      type: AverageValue
      averageValue: "10"
```

### Vertical Pod Autoscaler

#### VPA Configuration
```yaml
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

### Cluster Autoscaler

#### Node Pool Configuration
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: cluster-autoscaler-status
  namespace: kube-system
data:
  nodes.max: "100"
  nodes.min: "3"
  scale-down-delay-after-add: "10m"
  scale-down-unneeded-time: "10m"
```

## Troubleshooting

### Common Issues

#### 1. Pod Stuck in Pending State
```bash
# Check pod events
kubectl describe pod <pod-name> -n ai-scientist

# Check node resources
kubectl top nodes

# Check PVC status
kubectl get pvc -n ai-scientist
```

**Solutions:**
- Insufficient node resources: Add more nodes or increase node capacity
- PVC pending: Check storage class and provisioner
- Node affinity issues: Review node selectors and tolerations

#### 2. ImagePullBackOff Error
```bash
# Check image pull secrets
kubectl get secrets -n ai-scientist | grep docker

# Test image pull manually
docker pull your-registry.com/ai-scientist:v2.0.0
```

**Solutions:**
- Registry authentication: Update image pull secrets
- Image not found: Verify image tag and registry URL
- Network issues: Check registry connectivity

#### 3. Application Startup Failures
```bash
# Check application logs
kubectl logs deployment/ai-scientist -n ai-scientist

# Check configuration
kubectl get configmap ai-scientist-config -n ai-scientist -o yaml
```

**Solutions:**
- Missing environment variables: Update secrets and configmaps
- Configuration errors: Validate YAML syntax and values
- Dependency issues: Check database and Redis connectivity

#### 4. High Memory Usage
```bash
# Check memory usage
kubectl top pods -n ai-scientist

# Check memory limits
kubectl describe deployment ai-scientist -n ai-scientist
```

**Solutions:**
- Increase memory limits in deployment
- Optimize application memory usage
- Enable memory profiling to identify leaks

#### 5. GPU Not Available
```bash
# Check GPU nodes
kubectl get nodes -l nvidia.com/gpu.present=true

# Check GPU resources
kubectl describe node <gpu-node-name>

# Check NVIDIA device plugin
kubectl get daemonset -n gpu-operator
```

**Solutions:**
- Install NVIDIA GPU Operator
- Verify GPU drivers on nodes
- Check CUDA compatibility

### Debug Commands

#### Pod Debugging
```bash
# Get pod details
kubectl describe pod <pod-name> -n ai-scientist

# Check pod logs
kubectl logs <pod-name> -n ai-scientist --previous

# Execute into pod
kubectl exec -it <pod-name> -n ai-scientist -- /bin/bash

# Port forward for local testing
kubectl port-forward <pod-name> -n ai-scientist 8000:8000
```

#### Service Debugging
```bash
# Check service endpoints
kubectl get endpoints -n ai-scientist

# Test service connectivity
kubectl run debug --image=busybox --rm -it --restart=Never -- wget -qO- http://ai-scientist:8000/health
```

#### Network Debugging
```bash
# Check ingress status
kubectl get ingress -n ai-scientist

# Test DNS resolution
kubectl run debug --image=busybox --rm -it --restart=Never -- nslookup ai-scientist.ai-scientist.svc.cluster.local
```

### Performance Tuning

#### Application Performance
```yaml
# Optimize resource requests/limits
resources:
  requests:
    cpu: 2000m
    memory: 4Gi
  limits:
    cpu: 8000m
    memory: 16Gi
```

#### Database Performance
```yaml
# Redis optimization
redis:
  maxmemory: "4gb"
  maxmemory-policy: "allkeys-lru"
  tcp-keepalive: 300
```

#### GPU Performance
```yaml
# GPU resource allocation
resources:
  limits:
    nvidia.com/gpu: 1
env:
- name: CUDA_VISIBLE_DEVICES
  value: "0"
```

## Maintenance

### Regular Maintenance Tasks

#### Daily Tasks
- Monitor application health and performance metrics
- Check log files for errors and warnings
- Verify backup completion
- Review security alerts

#### Weekly Tasks
- Update container images with security patches
- Review and rotate secrets
- Analyze resource usage trends
- Test disaster recovery procedures

#### Monthly Tasks
- Update Kubernetes cluster and add-ons
- Review and update monitoring alerts
- Performance optimization review
- Security audit and vulnerability assessment

### Backup and Recovery

#### Data Backup
```bash
# Create backup script
#!/bin/bash
kubectl exec deployment/postgres -n ai-scientist -- pg_dump -U postgres ai_scientist_db | gzip > backup-$(date +%Y%m%d).sql.gz

# Redis backup
kubectl exec deployment/redis -n ai-scientist -- redis-cli BGSAVE
kubectl cp ai-scientist/redis-pod:/data/dump.rdb ./redis-backup-$(date +%Y%m%d).rdb
```

#### Configuration Backup
```bash
# Backup all configurations
kubectl get all,configmap,secret,pvc,ingress -n ai-scientist -o yaml > ai-scientist-config-backup-$(date +%Y%m%d).yaml
```

#### Disaster Recovery
1. **Data Recovery**: Restore from latest backups
2. **Configuration Recovery**: Apply backed-up manifests
3. **Service Recovery**: Redeploy applications in correct order
4. **Validation**: Run health checks and functional tests

### Updates and Upgrades

#### Application Updates
```bash
# Update to new version
./deployment/scripts/deploy.sh \
  --environment production \
  --tag v2.1.0 \
  --timeout 1200

# Rollback if needed
./deployment/scripts/deploy.sh \
  --environment production \
  --rollback
```

#### Kubernetes Updates
```bash
# Update cluster (specific to your platform)
# AWS EKS
eksctl upgrade cluster --name ai-scientist-cluster

# Google GKE
gcloud container clusters upgrade ai-scientist-cluster

# Azure AKS
az aks upgrade --resource-group myResourceGroup --name ai-scientist-cluster
```

### Monitoring and Alerting Maintenance

#### Alert Rule Updates
```bash
# Test alert rules
promtool query instant 'up{job="ai-scientist-app"} == 0'

# Update alert rules
kubectl apply -f deployment/monitoring/alerting-rules.yml
```

#### Dashboard Updates
```bash
# Import updated dashboard
curl -X POST \
  http://admin:password@grafana:3000/api/dashboards/db \
  -H 'Content-Type: application/json' \
  -d @deployment/monitoring/grafana-dashboard.json
```

This comprehensive deployment guide provides everything needed to successfully deploy AI Scientist v2 in production. For additional support, refer to the troubleshooting section or contact the development team.