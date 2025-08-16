# 🚀 Autonomous SDLC Deployment Guide

## Table of Contents

1. [Quick Start](#quick-start)
2. [Production Deployment](#production-deployment)
3. [Cloud Platforms](#cloud-platforms)
4. [Container Orchestration](#container-orchestration)
5. [Security Configuration](#security-configuration)
6. [Monitoring & Observability](#monitoring--observability)
7. [Scaling & Performance](#scaling--performance)
8. [Troubleshooting](#troubleshooting)

## Quick Start

### Prerequisites

```bash
# System requirements
- Python 3.11+
- 4GB+ RAM (8GB recommended)
- 2+ CPU cores (4+ recommended)
- 20GB+ storage space
- Linux/Unix environment preferred
```

### Local Development Setup

```bash
# 1. Clone and setup
git clone <repository-url>
cd ai-scientist-v2
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Quick test
python3 autonomous_cli.py --topic "Test Research" --output test_output

# 4. Verify installation
python3 autonomous_quality_validator.py
```

## Production Deployment

### Environment Setup

```bash
# Create production environment
export TERRAGON_ENV=production
export TERRAGON_LOG_LEVEL=INFO
export TERRAGON_REGION=na
export TERRAGON_LANGUAGE=en

# Security settings
export TERRAGON_SECURITY_LEVEL=high
export TERRAGON_ENABLE_SANDBOX=true
export TERRAGON_VALIDATE_INPUTS=true

# Performance settings
export TERRAGON_MAX_CONCURRENT=8
export TERRAGON_ENABLE_CACHING=true
export TERRAGON_AUTO_SCALE=true
```

### Production Configuration

Create `production.json`:

```json
{
  "environment": "production",
  "research": {
    "max_experiments": 20,
    "timeout_hours": 48.0,
    "enable_distributed": true
  },
  "performance": {
    "max_concurrent_experiments": 16,
    "enable_parallel_stages": true,
    "enable_caching": true,
    "cache_strategy": "hybrid",
    "cache_size_mb": 4096,
    "enable_auto_scaling": true,
    "optimize_memory": true,
    "enable_gpu_acceleration": true
  },
  "security": {
    "level": "critical",
    "sandbox_mode": true,
    "validate_inputs": true,
    "max_file_size_mb": 500,
    "enable_audit_logging": true
  },
  "compliance": {
    "frameworks": ["gdpr", "ccpa", "pdpa"],
    "data_retention_days": 2555,
    "enable_consent_management": true,
    "enable_data_anonymization": true,
    "require_explicit_consent": true
  },
  "monitoring": {
    "enable_metrics": true,
    "enable_health_checks": true,
    "metrics_interval": 30,
    "health_check_interval": 60
  }
}
```

### Production Startup Script

Create `start_production.sh`:

```bash
#!/bin/bash

# Production startup script
set -e

echo "🚀 Starting Autonomous SDLC Production System"

# Check prerequisites
python3 --version || { echo "Python 3.11+ required"; exit 1; }

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✅ Virtual environment activated"
fi

# Check dependencies
python3 -c "import ai_scientist.unified_autonomous_executor" || { echo "❌ Dependencies not installed"; exit 1; }

# Create necessary directories
mkdir -p logs production_output monitoring

# Set production environment
export TERRAGON_ENV=production
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Start with production configuration
echo "🔧 Starting with production configuration..."
python3 scalable_cli.py \
    --config production.json \
    --output production_output \
    --parallel \
    --auto-scale \
    --cache hybrid \
    --security critical \
    --monitor-resources \
    --performance-monitoring \
    --optimize-memory

echo "✅ Production system started successfully"
```

## Cloud Platforms

### AWS Deployment

#### ECS Fargate Deployment

```yaml
# ecs-task-definition.json
{
  "family": "autonomous-research",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "8192",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::account:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "research-engine",
      "image": "terragon/autonomous-research:latest",
      "portMappings": [
        {
          "containerPort": 8080,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {"name": "TERRAGON_ENV", "value": "production"},
        {"name": "TERRAGON_REGION", "value": "na"},
        {"name": "TERRAGON_SECURITY_LEVEL", "value": "high"}
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/autonomous-research",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "python3 -c 'import ai_scientist; print(\"healthy\")'"],
        "interval": 30,
        "timeout": 5,
        "retries": 3
      }
    }
  ]
}
```

#### CloudFormation Template

```yaml
# cloudformation-template.yaml
AWSTemplateFormatVersion: '2010-09-09'
Description: 'Autonomous SDLC Infrastructure'

Parameters:
  VpcId:
    Type: AWS::EC2::VPC::Id
  SubnetIds:
    Type: List<AWS::EC2::Subnet::Id>
  
Resources:
  ECSCluster:
    Type: AWS::ECS::Cluster
    Properties:
      ClusterName: autonomous-research-cluster
      CapacityProviders:
        - FARGATE
        - FARGATE_SPOT
      
  ECSService:
    Type: AWS::ECS::Service
    Properties:
      Cluster: !Ref ECSCluster
      TaskDefinition: !Ref TaskDefinition
      DesiredCount: 2
      LaunchType: FARGATE
      NetworkConfiguration:
        AwsvpcConfiguration:
          SecurityGroups:
            - !Ref SecurityGroup
          Subnets: !Ref SubnetIds
          AssignPublicIp: ENABLED
          
  LoadBalancer:
    Type: AWS::ElasticLoadBalancingV2::LoadBalancer
    Properties:
      Type: application
      Scheme: internet-facing
      Subnets: !Ref SubnetIds
      SecurityGroups:
        - !Ref SecurityGroup
        
  SecurityGroup:
    Type: AWS::EC2::SecurityGroup
    Properties:
      GroupDescription: Security group for Autonomous Research
      VpcId: !Ref VpcId
      SecurityGroupIngress:
        - IpProtocol: tcp
          FromPort: 8080
          ToPort: 8080
          CidrIp: 0.0.0.0/0
```

#### Deployment Commands

```bash
# Deploy to AWS
aws cloudformation create-stack \
  --stack-name autonomous-research \
  --template-body file://cloudformation-template.yaml \
  --parameters ParameterKey=VpcId,ParameterValue=vpc-12345 \
              ParameterKey=SubnetIds,ParameterValue=subnet-123,subnet-456

# Register task definition
aws ecs register-task-definition --cli-input-json file://ecs-task-definition.json

# Create service
aws ecs create-service \
  --cluster autonomous-research-cluster \
  --service-name research-service \
  --task-definition autonomous-research:1 \
  --desired-count 2 \
  --launch-type FARGATE
```

### Google Cloud Platform

#### Cloud Run Deployment

```yaml
# cloudrun.yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: autonomous-research
  annotations:
    run.googleapis.com/ingress: all
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/maxScale: "10"
        run.googleapis.com/cpu-throttling: "false"
        run.googleapis.com/memory: "8Gi"
        run.googleapis.com/cpu: "4"
    spec:
      containers:
      - image: gcr.io/project-id/autonomous-research:latest
        ports:
        - containerPort: 8080
        env:
        - name: TERRAGON_ENV
          value: "production"
        - name: TERRAGON_REGION
          value: "na"
        resources:
          limits:
            memory: "8Gi"
            cpu: "4"
          requests:
            memory: "4Gi"
            cpu: "2"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
```

#### Deployment Commands

```bash
# Build and push image
gcloud builds submit --tag gcr.io/project-id/autonomous-research:latest

# Deploy to Cloud Run
gcloud run services replace cloudrun.yaml --region=us-central1

# Set up domain mapping
gcloud run domain-mappings create --service=autonomous-research --domain=research.example.com
```

### Microsoft Azure

#### Container Instances

```yaml
# azure-container-group.yaml
apiVersion: 2019-12-01
location: eastus
name: autonomous-research
properties:
  containers:
  - name: research-engine
    properties:
      image: terragon/autonomous-research:latest
      ports:
      - port: 8080
        protocol: TCP
      resources:
        requests:
          cpu: 2.0
          memoryInGB: 8.0
        limits:
          cpu: 4.0
          memoryInGB: 16.0
      environmentVariables:
      - name: TERRAGON_ENV
        value: production
      - name: TERRAGON_REGION
        value: na
  osType: Linux
  restartPolicy: Always
  ipAddress:
    type: Public
    ports:
    - port: 8080
      protocol: TCP
    dnsNameLabel: autonomous-research
```

#### Deployment Commands

```bash
# Create resource group
az group create --name autonomous-research-rg --location eastus

# Deploy container group
az container create \
  --resource-group autonomous-research-rg \
  --file azure-container-group.yaml

# Get logs
az container logs --resource-group autonomous-research-rg --name autonomous-research
```

## Container Orchestration

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  research-engine:
    build: .
    ports:
      - "8080:8080"
    environment:
      - TERRAGON_ENV=production
      - TERRAGON_REGION=na
      - TERRAGON_SECURITY_LEVEL=high
    volumes:
      - ./production_output:/app/output
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python3", "-c", "import ai_scientist; print('healthy')"]
      interval: 30s
      timeout: 10s
      retries: 3
    deploy:
      resources:
        limits:
          cpus: '4.0'
          memory: 8G
        reservations:
          cpus: '2.0'
          memory: 4G

  monitoring:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'

  cache:
    image: redis:alpine
    ports:
      - "6379:6379"
    volumes:
      - cache_data:/data
    restart: unless-stopped

volumes:
  cache_data:
```

### Kubernetes

#### Namespace and ConfigMap

```yaml
# k8s-namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: autonomous-research
  labels:
    name: autonomous-research

---
apiVersion: v1
kind: ConfigMap
metadata:
  name: research-config
  namespace: autonomous-research
data:
  TERRAGON_ENV: "production"
  TERRAGON_REGION: "na"
  TERRAGON_SECURITY_LEVEL: "high"
  TERRAGON_AUTO_SCALE: "true"
  TERRAGON_ENABLE_CACHING: "true"
```

#### Deployment and Service

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: autonomous-research
  namespace: autonomous-research
  labels:
    app: autonomous-research
spec:
  replicas: 3
  selector:
    matchLabels:
      app: autonomous-research
  template:
    metadata:
      labels:
        app: autonomous-research
    spec:
      containers:
      - name: research-engine
        image: terragon/autonomous-research:latest
        ports:
        - containerPort: 8080
        envFrom:
        - configMapRef:
            name: research-config
        resources:
          requests:
            memory: "4Gi"
            cpu: "2000m"
          limits:
            memory: "8Gi"
            cpu: "4000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: output-storage
          mountPath: /app/output
        - name: cache-volume
          mountPath: /app/cache
      volumes:
      - name: output-storage
        persistentVolumeClaim:
          claimName: research-output-pvc
      - name: cache-volume
        emptyDir: {}

---
apiVersion: v1
kind: Service
metadata:
  name: autonomous-research-service
  namespace: autonomous-research
spec:
  selector:
    app: autonomous-research
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8080
  type: LoadBalancer

---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: research-output-pvc
  namespace: autonomous-research
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 100Gi
```

#### Horizontal Pod Autoscaler

```yaml
# k8s-hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: autonomous-research-hpa
  namespace: autonomous-research
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: autonomous-research
  minReplicas: 2
  maxReplicas: 20
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
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 0
      policies:
      - type: Percent
        value: 100
        periodSeconds: 15
```

#### Deployment Commands

```bash
# Apply all Kubernetes manifests
kubectl apply -f k8s-namespace.yaml
kubectl apply -f k8s-deployment.yaml
kubectl apply -f k8s-hpa.yaml

# Check deployment status
kubectl get pods -n autonomous-research
kubectl get services -n autonomous-research

# View logs
kubectl logs -f deployment/autonomous-research -n autonomous-research

# Scale manually
kubectl scale deployment autonomous-research --replicas=5 -n autonomous-research
```

## Security Configuration

### SSL/TLS Configuration

```yaml
# k8s-ingress-tls.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: autonomous-research-ingress
  namespace: autonomous-research
  annotations:
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/force-ssl-redirect: "true"
spec:
  tls:
  - hosts:
    - research.example.com
    secretName: research-tls-secret
  rules:
  - host: research.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: autonomous-research-service
            port:
              number: 80
```

### Network Policies

```yaml
# k8s-network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: autonomous-research-network-policy
  namespace: autonomous-research
spec:
  podSelector:
    matchLabels:
      app: autonomous-research
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
      port: 8080
  egress:
  - to: []
    ports:
    - protocol: TCP
      port: 443  # HTTPS
    - protocol: TCP
      port: 53   # DNS
    - protocol: UDP
      port: 53   # DNS
```

### RBAC Configuration

```yaml
# k8s-rbac.yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: autonomous-research-sa
  namespace: autonomous-research

---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: autonomous-research-role
  namespace: autonomous-research
rules:
- apiGroups: [""]
  resources: ["pods", "configmaps", "secrets"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["apps"]
  resources: ["deployments"]
  verbs: ["get", "list", "watch"]

---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: autonomous-research-rolebinding
  namespace: autonomous-research
subjects:
- kind: ServiceAccount
  name: autonomous-research-sa
  namespace: autonomous-research
roleRef:
  kind: Role
  name: autonomous-research-role
  apiGroup: rbac.authorization.k8s.io
```

## Monitoring & Observability

### Prometheus Configuration

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

scrape_configs:
  - job_name: 'autonomous-research'
    static_configs:
      - targets: ['localhost:8080']
    scrape_interval: 5s
    metrics_path: /metrics

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['localhost:9100']

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "Autonomous Research System",
    "panels": [
      {
        "title": "Research Execution Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(research_executions_total[5m])",
            "legendFormat": "Executions/sec"
          }
        ]
      },
      {
        "title": "Success Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "rate(research_executions_success_total[5m]) / rate(research_executions_total[5m]) * 100",
            "legendFormat": "Success %"
          }
        ]
      },
      {
        "title": "Resource Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "process_resident_memory_bytes",
            "legendFormat": "Memory"
          },
          {
            "expr": "rate(process_cpu_seconds_total[5m]) * 100",
            "legendFormat": "CPU %"
          }
        ]
      }
    ]
  }
}
```

### Alert Rules

```yaml
# monitoring/alert_rules.yml
groups:
- name: autonomous_research_alerts
  rules:
  - alert: HighErrorRate
    expr: rate(research_executions_error_total[5m]) / rate(research_executions_total[5m]) > 0.1
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "High error rate in research executions"
      description: "Error rate is {{ $value }}% for the last 5 minutes"

  - alert: HighMemoryUsage
    expr: process_resident_memory_bytes / (1024*1024*1024) > 6
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "High memory usage"
      description: "Memory usage is {{ $value }}GB"

  - alert: ServiceDown
    expr: up{job="autonomous-research"} == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "Autonomous research service is down"
      description: "Service has been down for more than 1 minute"
```

## Scaling & Performance

### Horizontal Scaling

```bash
# Manual scaling
kubectl scale deployment autonomous-research --replicas=10 -n autonomous-research

# Auto-scaling based on CPU
kubectl autoscale deployment autonomous-research --cpu-percent=70 --min=2 --max=20 -n autonomous-research

# Custom metrics scaling
kubectl apply -f - <<EOF
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: research-custom-hpa
  namespace: autonomous-research
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: autonomous-research
  minReplicas: 2
  maxReplicas: 50
  metrics:
  - type: Pods
    pods:
      metric:
        name: research_queue_length
      target:
        type: AverageValue
        averageValue: "10"
EOF
```

### Vertical Scaling

```yaml
# Vertical Pod Autoscaler
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: autonomous-research-vpa
  namespace: autonomous-research
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: autonomous-research
  updatePolicy:
    updateMode: "Auto"
  resourcePolicy:
    containerPolicies:
    - containerName: research-engine
      maxAllowed:
        cpu: 8
        memory: 16Gi
      minAllowed:
        cpu: 1
        memory: 2Gi
```

### Performance Optimization

```yaml
# Performance-optimized deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: autonomous-research-optimized
  namespace: autonomous-research
spec:
  replicas: 5
  template:
    spec:
      containers:
      - name: research-engine
        image: terragon/autonomous-research:latest
        resources:
          requests:
            memory: "8Gi"
            cpu: "4000m"
          limits:
            memory: "16Gi"
            cpu: "8000m"
        env:
        - name: TERRAGON_MAX_CONCURRENT
          value: "16"
        - name: TERRAGON_CACHE_SIZE_MB
          value: "4096"
        - name: TERRAGON_OPTIMIZE_MEMORY
          value: "true"
        - name: TERRAGON_ENABLE_GPU
          value: "true"
      nodeSelector:
        instance-type: compute-optimized
      tolerations:
      - key: "research-workload"
        operator: "Equal"
        value: "true"
        effect: "NoSchedule"
```

## Troubleshooting

### Common Issues

#### Pod Crashes

```bash
# Check pod status
kubectl get pods -n autonomous-research

# Get pod logs
kubectl logs <pod-name> -n autonomous-research

# Describe pod for events
kubectl describe pod <pod-name> -n autonomous-research

# Check resource usage
kubectl top pods -n autonomous-research
```

#### Performance Issues

```bash
# Monitor resource usage
kubectl top nodes
kubectl top pods -n autonomous-research

# Check HPA status
kubectl get hpa -n autonomous-research

# View detailed HPA metrics
kubectl describe hpa autonomous-research-hpa -n autonomous-research
```

#### Network Issues

```bash
# Test service connectivity
kubectl exec -it <pod-name> -n autonomous-research -- curl http://autonomous-research-service

# Check network policies
kubectl get networkpolicy -n autonomous-research

# Test DNS resolution
kubectl exec -it <pod-name> -n autonomous-research -- nslookup autonomous-research-service
```

### Debug Mode

```bash
# Enable debug logging
kubectl set env deployment/autonomous-research TERRAGON_LOG_LEVEL=DEBUG -n autonomous-research

# Port forward for local debugging
kubectl port-forward service/autonomous-research-service 8080:80 -n autonomous-research

# Execute commands in pod
kubectl exec -it <pod-name> -n autonomous-research -- bash
```

### Health Checks

```bash
# Manual health check
curl -f http://localhost:8080/health || echo "Service unhealthy"

# Readiness check
curl -f http://localhost:8080/ready || echo "Service not ready"

# Metrics endpoint
curl http://localhost:8080/metrics
```

### Performance Profiling

```bash
# CPU profiling
kubectl exec -it <pod-name> -n autonomous-research -- python3 -m cProfile -o profile.stats scalable_cli.py

# Memory profiling
kubectl exec -it <pod-name> -n autonomous-research -- python3 -m memory_profiler scalable_cli.py
```

### Backup and Recovery

```bash
# Backup persistent volumes
kubectl get pvc -n autonomous-research
kubectl get pv

# Create volume snapshot
kubectl apply -f - <<EOF
apiVersion: snapshot.storage.k8s.io/v1
kind: VolumeSnapshot
metadata:
  name: research-output-snapshot
  namespace: autonomous-research
spec:
  source:
    persistentVolumeClaimName: research-output-pvc
EOF

# Restore from snapshot
kubectl apply -f - <<EOF
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: research-output-restored
  namespace: autonomous-research
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 100Gi
  dataSource:
    name: research-output-snapshot
    kind: VolumeSnapshot
    apiGroup: snapshot.storage.k8s.io
EOF
```

## Security Best Practices

### Image Security

```dockerfile
# Multi-stage secure Dockerfile
FROM python:3.11-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.11-slim
RUN groupadd -r research && useradd -r -g research research
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY . .
RUN chown -R research:research /app
USER research
EXPOSE 8080
CMD ["python3", "scalable_cli.py"]
```

### Secret Management

```yaml
# External secrets operator
apiVersion: external-secrets.io/v1beta1
kind: SecretStore
metadata:
  name: vault-secret-store
  namespace: autonomous-research
spec:
  provider:
    vault:
      server: "https://vault.example.com"
      path: "secret"
      version: "v2"
      auth:
        kubernetes:
          mountPath: "kubernetes"
          role: "autonomous-research"

---
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: research-secrets
  namespace: autonomous-research
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: vault-secret-store
    kind: SecretStore
  target:
    name: research-api-keys
    creationPolicy: Owner
  data:
  - secretKey: openai-api-key
    remoteRef:
      key: research/api-keys
      property: openai
```

### Pod Security

```yaml
# Pod Security Standards
apiVersion: v1
kind: Namespace
metadata:
  name: autonomous-research
  labels:
    pod-security.kubernetes.io/enforce: restricted
    pod-security.kubernetes.io/audit: restricted
    pod-security.kubernetes.io/warn: restricted

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: autonomous-research-secure
  namespace: autonomous-research
spec:
  template:
    spec:
      securityContext:
        runAsNonRoot: true
        runAsUser: 65534
        fsGroup: 65534
        seccompProfile:
          type: RuntimeDefault
      containers:
      - name: research-engine
        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
          runAsNonRoot: true
          runAsUser: 65534
          capabilities:
            drop:
            - ALL
        volumeMounts:
        - name: tmp
          mountPath: /tmp
        - name: cache
          mountPath: /app/cache
      volumes:
      - name: tmp
        emptyDir: {}
      - name: cache
        emptyDir: {}
```

---

**🚀 This deployment guide provides comprehensive instructions for deploying the Autonomous SDLC system across various environments and platforms. For additional support, refer to the main documentation or contact the Terragon Labs support team.**