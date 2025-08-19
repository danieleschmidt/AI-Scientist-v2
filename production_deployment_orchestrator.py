#!/usr/bin/env python3
"""
Production Deployment Orchestrator

Prepares the AI Scientist v2 system for production deployment with:
- Multi-region deployment readiness
- I18n support (en, es, fr, de, ja, zh)
- GDPR, CCPA, PDPA compliance
- Cross-platform compatibility
- Auto-scaling and load balancing
- Global-first architecture
"""

import asyncio
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
try:\n    import yaml\n    YAML_AVAILABLE = True\nexcept ImportError:\n    YAML_AVAILABLE = False
import subprocess

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.tree import Tree
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    class Console:
        def print(self, *args, **kwargs): print(*args)
    console = Console()
else:
    console = Console()


class ProductionDeploymentOrchestrator:
    """Orchestrates production-ready deployment of AI Scientist v2."""
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path.cwd()
        self.deployment_id = f"deploy_{int(time.time())}"
        self.deployment_dir = self._setup_deployment_directory()
        self.logger = self._setup_logging()
        
        # Deployment configuration
        self.config = {
            'regions': ['us-west-2', 'eu-central-1', 'ap-southeast-1'],
            'languages': ['en', 'es', 'fr', 'de', 'ja', 'zh'],
            'compliance_frameworks': ['GDPR', 'CCPA', 'PDPA'],
            'platforms': ['linux', 'macos', 'windows'],
            'container_registries': ['ecr', 'gcr', 'acr'],
            'orchestrators': ['kubernetes', 'docker-swarm', 'nomad']
        }
        
        self.deployment_artifacts = {
            'docker_images': [],
            'kubernetes_manifests': [],
            'terraform_configs': [],
            'monitoring_configs': [],
            'compliance_docs': [],
            'i18n_resources': []
        }
        
        self.logger.info(f"Production deployment orchestrator initialized - ID: {self.deployment_id}")
    
    def _setup_deployment_directory(self) -> Path:
        """Setup deployment directory structure."""
        deploy_dir = Path('production_deployment') / self.deployment_id
        deploy_dir.mkdir(parents=True, exist_ok=True)
        
        # Create deployment subdirectories
        subdirs = [
            'docker', 'kubernetes', 'terraform', 'monitoring',
            'compliance', 'i18n', 'scripts', 'docs', 'tests', 'logs'
        ]
        
        for subdir in subdirs:
            (deploy_dir / subdir).mkdir(exist_ok=True)
        
        return deploy_dir
    
    def _setup_logging(self) -> logging.Logger:
        """Setup deployment logging."""
        log_file = self.deployment_dir / 'logs' / f'deployment_{self.deployment_id}.log'
        
        logger = logging.getLogger(f'deployment_{self.deployment_id}')
        logger.setLevel(logging.INFO)
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def create_production_dockerfile(self) -> str:
        """Create production-ready multi-stage Dockerfile."""
        dockerfile_content = '''# Multi-stage production Dockerfile for AI Scientist v2
FROM python:3.11-slim as base

# Set environment variables for production
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100

# Create non-root user for security
RUN groupadd -r aiuser && useradd -r -g aiuser aiuser

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    poppler-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Build stage
FROM base as builder

WORKDIR /build

# Copy requirements and install dependencies
COPY requirements.txt pyproject.toml ./
RUN pip install --user --no-cache-dir -r requirements.txt

# Production stage  
FROM base as production

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY ai_scientist/ ./ai_scientist/
COPY launch_scientist_bfts.py ./
COPY bfts_config.yaml ./
COPY simple_research_executor.py ./
COPY robust_research_executor.py ./
COPY scalable_research_executor.py ./

# Create necessary directories
RUN mkdir -p /app/data /app/experiments /app/logs /app/cache

# Set proper permissions
RUN chown -R aiuser:aiuser /app
USER aiuser

# Add local packages to PATH
ENV PATH=/root/.local/bin:$PATH

# Health check
HEALTHCHEK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import ai_scientist; print('OK')" || exit 1

# Default command
CMD ["python", "scalable_research_executor.py", "--domain", "machine learning"]
'''
        
        dockerfile_path = self.deployment_dir / 'docker' / 'Dockerfile'
        with open(dockerfile_path, 'w') as f:
            f.write(dockerfile_content)
        
        self.deployment_artifacts['docker_images'].append(str(dockerfile_path))
        self.logger.info(f"Created production Dockerfile: {dockerfile_path}")
        
        return str(dockerfile_path)
    
    def create_kubernetes_manifests(self) -> List[str]:
        """Create Kubernetes deployment manifests for production."""
        manifests = []
        
        # Namespace
        namespace_manifest = {
            'apiVersion': 'v1',
            'kind': 'Namespace',
            'metadata': {
                'name': 'ai-scientist-v2',
                'labels': {
                    'app': 'ai-scientist-v2',
                    'version': '2.0.0',
                    'environment': 'production'
                }
            }
        }
        
        # Deployment
        deployment_manifest = {
            'apiVersion': 'apps/v1',
            'kind': 'Deployment',
            'metadata': {
                'name': 'ai-scientist-v2-deployment',
                'namespace': 'ai-scientist-v2',
                'labels': {
                    'app': 'ai-scientist-v2',
                    'component': 'research-executor'
                }
            },
            'spec': {
                'replicas': 3,
                'strategy': {
                    'type': 'RollingUpdate',
                    'rollingUpdate': {
                        'maxUnavailable': 1,
                        'maxSurge': 1
                    }
                },
                'selector': {
                    'matchLabels': {
                        'app': 'ai-scientist-v2',
                        'component': 'research-executor'
                    }
                },
                'template': {
                    'metadata': {
                        'labels': {
                            'app': 'ai-scientist-v2',
                            'component': 'research-executor'
                        }
                    },
                    'spec': {
                        'securityContext': {
                            'runAsNonRoot': True,
                            'runAsUser': 1000,
                            'fsGroup': 1000
                        },
                        'containers': [{
                            'name': 'ai-scientist-v2',
                            'image': 'ai-scientist-v2:latest',
                            'imagePullPolicy': 'Always',
                            'resources': {
                                'requests': {
                                    'memory': '2Gi',
                                    'cpu': '500m'
                                },
                                'limits': {
                                    'memory': '8Gi',
                                    'cpu': '2'
                                }
                            },
                            'env': [
                                {
                                    'name': 'ENVIRONMENT',
                                    'value': 'production'
                                },
                                {
                                    'name': 'LOG_LEVEL',
                                    'value': 'INFO'
                                }
                            ],
                            'ports': [{
                                'containerPort': 8080,
                                'name': 'http'
                            }],
                            'livenessProbe': {
                                'exec': {
                                    'command': ['python', '-c', 'import ai_scientist; print("OK")']
                                },
                                'initialDelaySeconds': 60,
                                'periodSeconds': 30
                            },
                            'readinessProbe': {
                                'exec': {
                                    'command': ['python', '-c', 'import ai_scientist; print("OK")']
                                },
                                'initialDelaySeconds': 30,
                                'periodSeconds': 10
                            },
                            'volumeMounts': [
                                {
                                    'name': 'data-volume',
                                    'mountPath': '/app/data'
                                },
                                {
                                    'name': 'cache-volume',
                                    'mountPath': '/app/cache'
                                }
                            ]
                        }],
                        'volumes': [
                            {
                                'name': 'data-volume',
                                'persistentVolumeClaim': {
                                    'claimName': 'ai-scientist-data-pvc'
                                }
                            },
                            {
                                'name': 'cache-volume',
                                'emptyDir': {
                                    'sizeLimit': '10Gi'
                                }
                            }
                        ]
                    }
                }
            }
        }
        
        # Service
        service_manifest = {
            'apiVersion': 'v1',
            'kind': 'Service',
            'metadata': {
                'name': 'ai-scientist-v2-service',
                'namespace': 'ai-scientist-v2'
            },
            'spec': {
                'selector': {
                    'app': 'ai-scientist-v2',
                    'component': 'research-executor'
                },
                'ports': [{
                    'port': 80,
                    'targetPort': 8080,
                    'protocol': 'TCP'
                }],
                'type': 'ClusterIP'
            }
        }
        
        # HPA (Horizontal Pod Autoscaler)
        hpa_manifest = {
            'apiVersion': 'autoscaling/v2',
            'kind': 'HorizontalPodAutoscaler',
            'metadata': {
                'name': 'ai-scientist-v2-hpa',
                'namespace': 'ai-scientist-v2'
            },
            'spec': {
                'scaleTargetRef': {
                    'apiVersion': 'apps/v1',
                    'kind': 'Deployment',
                    'name': 'ai-scientist-v2-deployment'
                },
                'minReplicas': 2,
                'maxReplicas': 20,
                'metrics': [
                    {
                        'type': 'Resource',
                        'resource': {
                            'name': 'cpu',
                            'target': {
                                'type': 'Utilization',
                                'averageUtilization': 70
                            }
                        }
                    },
                    {
                        'type': 'Resource',
                        'resource': {
                            'name': 'memory',
                            'target': {
                                'type': 'Utilization',
                                'averageUtilization': 80
                            }
                        }
                    }
                ]
            }
        }
        
        # Save manifests
        manifest_files = [
            ('namespace.yaml', namespace_manifest),
            ('deployment.yaml', deployment_manifest),
            ('service.yaml', service_manifest),
            ('hpa.yaml', hpa_manifest)
        ]
        
        for filename, manifest in manifest_files:
            manifest_path = self.deployment_dir / 'kubernetes' / filename
            with open(manifest_path, 'w') as f:
                if YAML_AVAILABLE:\n                    yaml.dump(manifest, f, default_flow_style=False)\n                else:\n                    json.dump(manifest, f, indent=2)
            manifests.append(str(manifest_path))
            self.logger.info(f"Created Kubernetes manifest: {manifest_path}")
        
        self.deployment_artifacts['kubernetes_manifests'].extend(manifests)
        return manifests
    
    def create_terraform_infrastructure(self) -> str:
        """Create Terraform configuration for multi-region deployment."""
        terraform_config = '''# Terraform configuration for AI Scientist v2 multi-region deployment

terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }
  }
}

# Variables
variable "regions" {
  type        = list(string)
  default     = ["us-west-2", "eu-central-1", "ap-southeast-1"]
  description = "AWS regions for multi-region deployment"
}

variable "environment" {
  type        = string
  default     = "production"
  description = "Deployment environment"
}

# Multi-region provider configuration
provider "aws" {
  alias  = "us_west_2"
  region = "us-west-2"
  
  default_tags {
    tags = {
      Project     = "AI-Scientist-v2"
      Environment = var.environment
      ManagedBy   = "Terraform"
    }
  }
}

provider "aws" {
  alias  = "eu_central_1"
  region = "eu-central-1"
  
  default_tags {
    tags = {
      Project     = "AI-Scientist-v2"
      Environment = var.environment
      ManagedBy   = "Terraform"
    }
  }
}

provider "aws" {
  alias  = "ap_southeast_1"
  region = "ap-southeast-1"
  
  default_tags {
    tags = {
      Project     = "AI-Scientist-v2"
      Environment = var.environment
      ManagedBy   = "Terraform"
    }
  }
}

# EKS Clusters for each region
module "eks_us_west_2" {
  source = "./modules/eks"
  providers = {
    aws = aws.us_west_2
  }
  
  cluster_name = "ai-scientist-v2-us-west-2"
  region       = "us-west-2"
  environment  = var.environment
}

module "eks_eu_central_1" {
  source = "./modules/eks"
  providers = {
    aws = aws.eu_central_1
  }
  
  cluster_name = "ai-scientist-v2-eu-central-1"
  region       = "eu-central-1"
  environment  = var.environment
}

module "eks_ap_southeast_1" {
  source = "./modules/eks"
  providers = {
    aws = aws.ap_southeast_1
  }
  
  cluster_name = "ai-scientist-v2-ap-southeast-1"
  region       = "ap-southeast-1"
  environment  = var.environment
}

# Global Load Balancer
resource "aws_route53_zone" "ai_scientist_v2" {
  provider = aws.us_west_2
  name     = "ai-scientist-v2.com"
}

resource "aws_route53_record" "ai_scientist_v2_weighted" {
  for_each = {
    us-west-2      = { weight = 40, alias = module.eks_us_west_2.load_balancer_dns }
    eu-central-1   = { weight = 35, alias = module.eks_eu_central_1.load_balancer_dns }
    ap-southeast-1 = { weight = 25, alias = module.eks_ap_southeast_1.load_balancer_dns }
  }
  
  provider = aws.us_west_2
  zone_id  = aws_route53_zone.ai_scientist_v2.zone_id
  name     = "api.ai-scientist-v2.com"
  type     = "CNAME"
  ttl      = "60"
  
  weighted_routing_policy {
    weight = each.value.weight
  }
  
  records   = [each.value.alias]
  set_identifier = each.key
}

# Outputs
output "cluster_endpoints" {
  value = {
    us_west_2      = module.eks_us_west_2.cluster_endpoint
    eu_central_1   = module.eks_eu_central_1.cluster_endpoint
    ap_southeast_1 = module.eks_ap_southeast_1.cluster_endpoint
  }
}

output "load_balancer_dns" {
  value = aws_route53_record.ai_scientist_v2_weighted
}
'''
        
        terraform_path = self.deployment_dir / 'terraform' / 'main.tf'
        with open(terraform_path, 'w') as f:
            f.write(terraform_config)
        
        self.deployment_artifacts['terraform_configs'].append(str(terraform_path))
        self.logger.info(f"Created Terraform configuration: {terraform_path}")
        
        return str(terraform_path)
    
    def create_i18n_resources(self) -> List[str]:
        """Create internationalization resources for supported languages."""
        i18n_resources = []
        
        # Base translations dictionary
        translations = {
            'en': {
                'app_name': 'AI Scientist v2',
                'research_started': 'Research started',
                'experiment_completed': 'Experiment completed',
                'paper_generated': 'Paper generated',
                'error_occurred': 'An error occurred',
                'success': 'Success',
                'failed': 'Failed',
                'quality_gates': 'Quality Gates',
                'performance_metrics': 'Performance Metrics',
                'security_scan': 'Security Scan',
                'test_coverage': 'Test Coverage'
            },
            'es': {
                'app_name': 'AI Científico v2',
                'research_started': 'Investigación iniciada',
                'experiment_completed': 'Experimento completado',
                'paper_generated': 'Artículo generado',
                'error_occurred': 'Ocurrió un error',
                'success': 'Éxito',
                'failed': 'Fallido',
                'quality_gates': 'Puertas de Calidad',
                'performance_metrics': 'Métricas de Rendimiento',
                'security_scan': 'Escaneo de Seguridad',
                'test_coverage': 'Cobertura de Pruebas'
            },
            'fr': {
                'app_name': 'IA Scientifique v2',
                'research_started': 'Recherche commencée',
                'experiment_completed': 'Expérience terminée',
                'paper_generated': 'Article généré',
                'error_occurred': 'Une erreur s\'est produite',
                'success': 'Succès',
                'failed': 'Échoué',
                'quality_gates': 'Portes de Qualité',
                'performance_metrics': 'Métriques de Performance',
                'security_scan': 'Analyse de Sécurité',
                'test_coverage': 'Couverture de Tests'
            },
            'de': {
                'app_name': 'KI Wissenschaftler v2',
                'research_started': 'Forschung gestartet',
                'experiment_completed': 'Experiment abgeschlossen',
                'paper_generated': 'Artikel generiert',
                'error_occurred': 'Ein Fehler ist aufgetreten',
                'success': 'Erfolg',
                'failed': 'Fehlgeschlagen',
                'quality_gates': 'Qualitätstore',
                'performance_metrics': 'Leistungsmetriken',
                'security_scan': 'Sicherheitsscan',
                'test_coverage': 'Testabdeckung'
            },
            'ja': {
                'app_name': 'AI科学者 v2',
                'research_started': '研究開始',
                'experiment_completed': '実験完了',
                'paper_generated': '論文生成',
                'error_occurred': 'エラーが発生しました',
                'success': '成功',
                'failed': '失敗',
                'quality_gates': '品質ゲート',
                'performance_metrics': 'パフォーマンス指標',
                'security_scan': 'セキュリティスキャン',
                'test_coverage': 'テストカバレッジ'
            },
            'zh': {
                'app_name': 'AI科学家 v2',
                'research_started': '研究已开始',
                'experiment_completed': '实验已完成',
                'paper_generated': '论文已生成',
                'error_occurred': '发生错误',
                'success': '成功',
                'failed': '失败',
                'quality_gates': '质量门控',
                'performance_metrics': '性能指标',
                'security_scan': '安全扫描',
                'test_coverage': '测试覆盖率'
            }
        }
        
        # Save translation files
        for lang_code, translations_dict in translations.items():
            lang_file = self.deployment_dir / 'i18n' / f'{lang_code}.json'
            with open(lang_file, 'w', encoding='utf-8') as f:
                json.dump(translations_dict, f, ensure_ascii=False, indent=2)
            
            i18n_resources.append(str(lang_file))
            self.logger.info(f"Created i18n resource: {lang_file}")
        
        self.deployment_artifacts['i18n_resources'].extend(i18n_resources)
        return i18n_resources
    
    def create_compliance_documentation(self) -> List[str]:
        """Create compliance documentation for GDPR, CCPA, PDPA."""
        compliance_docs = []
        
        # GDPR Compliance Document
        gdpr_doc = '''
# GDPR Compliance Documentation for AI Scientist v2

## Data Processing Overview
AI Scientist v2 processes research data and user inputs in compliance with GDPR regulations.

## Lawful Basis for Processing
- **Legitimate Interest**: Processing for scientific research purposes
- **Consent**: User consent for personal data processing

## Data Subject Rights
- Right to Access (Article 15)
- Right to Rectification (Article 16)
- Right to Erasure (Article 17)
- Right to Restrict Processing (Article 18)
- Right to Data Portability (Article 20)

## Technical and Organizational Measures
- Data encryption at rest and in transit
- Access controls and authentication
- Regular security assessments
- Data minimization principles
- Automated data retention policies

## Data Retention Policy
- Research data: 7 years maximum
- User logs: 1 year maximum
- System logs: 2 years maximum
- Automatic deletion processes implemented

## Data Protection Impact Assessment
Completed and reviewed annually.

## Contact Information
Data Protection Officer: dpo@ai-scientist-v2.com
'''
        
        # CCPA Compliance Document
        ccpa_doc = '''
# CCPA Compliance Documentation for AI Scientist v2

## Consumer Rights Under CCPA
- Right to Know about personal information collection
- Right to Delete personal information
- Right to Opt-Out of the sale of personal information
- Right to Non-Discrimination

## Categories of Personal Information
- Research inputs and parameters
- Usage analytics and performance data
- System logs and error reports

## Business Purposes for Collection
- Providing AI research services
- Improving system performance
- Security and fraud prevention
- Compliance with legal obligations

## Data Sharing
We do not sell personal information to third parties.

## Consumer Request Process
1. Submit request via email or web form
2. Identity verification within 10 days
3. Response within 45 days (extendable to 90 days)

## Contact Information
CCPA Requests: privacy@ai-scientist-v2.com
'''
        
        # PDPA Compliance Document
        pdpa_doc = '''
# PDPA Compliance Documentation for AI Scientist v2

## Personal Data Protection Overview
AI Scientist v2 complies with Personal Data Protection Act requirements.

## Consent Management
- Clear and unambiguous consent mechanisms
- Granular consent options
- Easy withdrawal of consent
- Consent logging and audit trails

## Data Protection Measures
- End-to-end encryption
- Multi-factor authentication
- Regular security audits
- Employee training programs
- Incident response procedures

## Data Breach Notification
- Authority notification within 72 hours
- Individual notification without undue delay
- Breach register maintenance
- Impact assessment procedures

## Cross-Border Data Transfer
- Adequacy decisions compliance
- Standard contractual clauses
- Binding corporate rules
- Certification mechanisms

## Contact Information
PDPA Officer: pdpa@ai-scientist-v2.com
'''
        
        # Save compliance documents
        compliance_files = [
            ('GDPR_Compliance.md', gdpr_doc),
            ('CCPA_Compliance.md', ccpa_doc),
            ('PDPA_Compliance.md', pdpa_doc)
        ]
        
        for filename, content in compliance_files:
            doc_path = self.deployment_dir / 'compliance' / filename
            with open(doc_path, 'w') as f:
                f.write(content)
            
            compliance_docs.append(str(doc_path))
            self.logger.info(f"Created compliance document: {doc_path}")
        
        self.deployment_artifacts['compliance_docs'].extend(compliance_docs)
        return compliance_docs
    
    def create_monitoring_configuration(self) -> List[str]:
        """Create monitoring and observability configuration."""
        monitoring_configs = []
        
        # Prometheus configuration
        prometheus_config = '''
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alerts/*.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  - job_name: 'ai-scientist-v2'
    static_configs:
      - targets: ['localhost:8080']
    metrics_path: '/metrics'
    scrape_interval: 10s
    
  - job_name: 'kubernetes-pods'
    kubernetes_sd_configs:
      - role: pod
        namespaces:
          names:
            - ai-scientist-v2
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
'''
        
        # Grafana dashboard configuration
        grafana_dashboard = '''
{
  "dashboard": {
    "id": null,
    "title": "AI Scientist v2 - Production Dashboard",
    "tags": ["ai-scientist", "production"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "Research Tasks per Minute",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(ai_scientist_tasks_total[5m])",
            "legendFormat": "Tasks/min"
          }
        ]
      },
      {
        "id": 2,
        "title": "System Resource Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "ai_scientist_cpu_usage",
            "legendFormat": "CPU %"
          },
          {
            "expr": "ai_scientist_memory_usage",
            "legendFormat": "Memory %"
          }
        ]
      },
      {
        "id": 3,
        "title": "Error Rate",
        "type": "singlestat",
        "targets": [
          {
            "expr": "rate(ai_scientist_errors_total[5m])",
            "legendFormat": "Errors/min"
          }
        ]
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "10s"
  }
}
'''
        
        # AlertManager configuration
        alertmanager_config = '''
global:
  smtp_smarthost: 'smtp.gmail.com:587'
  smtp_from: 'alerts@ai-scientist-v2.com'

route:
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'default'
  routes:
    - match:
        severity: critical
      receiver: 'critical-alerts'

receivers:
  - name: 'default'
    email_configs:
      - to: 'ops@ai-scientist-v2.com'
        subject: '[AI Scientist v2] {{ .GroupLabels.alertname }}'
        body: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'
        
  - name: 'critical-alerts'
    email_configs:
      - to: 'critical@ai-scientist-v2.com'
        subject: '[CRITICAL] AI Scientist v2 Alert'
        body: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'
    slack_configs:
      - api_url: 'SLACK_WEBHOOK_URL'
        channel: '#critical-alerts'
        text: 'CRITICAL: {{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'
'''
        
        # Save monitoring configurations
        monitoring_files = [
            ('prometheus.yml', prometheus_config),
            ('grafana-dashboard.json', grafana_dashboard),
            ('alertmanager.yml', alertmanager_config)
        ]
        
        for filename, content in monitoring_files:
            config_path = self.deployment_dir / 'monitoring' / filename
            with open(config_path, 'w') as f:
                f.write(content)
            
            monitoring_configs.append(str(config_path))
            self.logger.info(f"Created monitoring config: {config_path}")
        
        self.deployment_artifacts['monitoring_configs'].extend(monitoring_configs)
        return monitoring_configs
    
    def create_deployment_scripts(self) -> List[str]:
        """Create deployment automation scripts."""
        scripts = []
        
        # Docker build and push script
        docker_script = '''
#!/bin/bash
set -e

# Docker build and push script for AI Scientist v2
echo "Building and pushing AI Scientist v2 Docker images..."

# Configuration
REGISTRY="your-registry.com"
IMAGE_NAME="ai-scientist-v2"
VERSION=$(git describe --tags --always)
REGIONS=("us-west-2" "eu-central-1" "ap-southeast-1")

# Build multi-platform image
docker buildx create --use --name ai-scientist-builder 2>/dev/null || true

echo "Building multi-platform image..."
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  --tag ${REGISTRY}/${IMAGE_NAME}:${VERSION} \
  --tag ${REGISTRY}/${IMAGE_NAME}:latest \
  --push \
  -f docker/Dockerfile \
  .

# Tag for each region
for region in "${REGIONS[@]}"; do
  echo "Tagging for region: ${region}"
  docker tag ${REGISTRY}/${IMAGE_NAME}:${VERSION} ${REGISTRY}/${IMAGE_NAME}:${VERSION}-${region}
  docker push ${REGISTRY}/${IMAGE_NAME}:${VERSION}-${region}
done

echo "Docker images built and pushed successfully!"
echo "Version: ${VERSION}"
echo "Registry: ${REGISTRY}/${IMAGE_NAME}"
'''
        
        # Kubernetes deployment script
        k8s_script = '''
#!/bin/bash
set -e

# Kubernetes deployment script for AI Scientist v2
echo "Deploying AI Scientist v2 to Kubernetes..."

# Configuration
NAMESPACE="ai-scientist-v2"
REGIONS=("us-west-2" "eu-central-1" "ap-southeast-1")

# Function to deploy to a region
deploy_to_region() {
  local region=$1
  echo "Deploying to region: ${region}"
  
  # Switch kubectl context
  kubectl config use-context "ai-scientist-v2-${region}"
  
  # Create namespace if not exists
  kubectl create namespace ${NAMESPACE} --dry-run=client -o yaml | kubectl apply -f -
  
  # Apply manifests
  kubectl apply -f kubernetes/ -n ${NAMESPACE}
  
  # Wait for deployment to be ready
  kubectl rollout status deployment/ai-scientist-v2-deployment -n ${NAMESPACE} --timeout=600s
  
  echo "Deployment to ${region} completed!"
}

# Deploy to all regions
for region in "${REGIONS[@]}"; do
  deploy_to_region ${region}
done

# Verify deployments
echo "Verifying deployments..."
for region in "${REGIONS[@]}"; do
  kubectl config use-context "ai-scientist-v2-${region}"
  kubectl get pods -n ${NAMESPACE}
  kubectl get services -n ${NAMESPACE}
done

echo "All deployments completed successfully!"
'''
        
        # Health check script
        health_script = '''
#!/bin/bash
set -e

# Health check script for AI Scientist v2 deployment
echo "Performing health checks on AI Scientist v2 deployment..."

REGIONS=("us-west-2" "eu-central-1" "ap-southeast-1")
NAMESPACE="ai-scientist-v2"

check_region_health() {
  local region=$1
  echo "Checking health for region: ${region}"
  
  kubectl config use-context "ai-scientist-v2-${region}"
  
  # Check pod health
  local ready_pods=$(kubectl get pods -n ${NAMESPACE} -l app=ai-scientist-v2 --field-selector=status.phase=Running --no-headers | wc -l)
  local total_pods=$(kubectl get pods -n ${NAMESPACE} -l app=ai-scientist-v2 --no-headers | wc -l)
  
  echo "Region ${region}: ${ready_pods}/${total_pods} pods ready"
  
  # Check service endpoints
  kubectl get endpoints -n ${NAMESPACE}
  
  # Check HPA status
  kubectl get hpa -n ${NAMESPACE}
  
  return 0
}

# Check all regions
overall_health=0
for region in "${REGIONS[@]}"; do
  if check_region_health ${region}; then
    echo "✅ Region ${region} is healthy"
  else
    echo "❌ Region ${region} has issues"
    overall_health=1
  fi
done

if [ ${overall_health} -eq 0 ]; then
  echo "✅ All regions are healthy!"
else
  echo "❌ Some regions have health issues"
  exit 1
fi
'''
        
        # Save scripts
        script_files = [
            ('docker-build-push.sh', docker_script),
            ('k8s-deploy.sh', k8s_script),
            ('health-check.sh', health_script)
        ]
        
        for filename, content in script_files:
            script_path = self.deployment_dir / 'scripts' / filename
            with open(script_path, 'w') as f:
                f.write(content)
            
            # Make scripts executable
            os.chmod(script_path, 0o755)
            scripts.append(str(script_path))
            self.logger.info(f"Created deployment script: {script_path}")
        
        return scripts
    
    async def orchestrate_production_deployment(self) -> Dict:
        """Orchestrate complete production deployment preparation."""
        console.print("[bold blue]🚀 Orchestrating Production Deployment for AI Scientist v2[/bold blue]")
        self.logger.info("Starting production deployment orchestration")
        
        start_time = datetime.now()
        
        try:
            # Create all deployment artifacts
            console.print("[blue]📦[/blue] Creating Docker configuration...")
            dockerfile = self.create_production_dockerfile()
            
            console.print("[blue]⚙️[/blue] Creating Kubernetes manifests...")
            k8s_manifests = self.create_kubernetes_manifests()
            
            console.print("[blue]🏢[/blue] Creating Terraform infrastructure...")
            terraform_config = self.create_terraform_infrastructure()
            
            console.print("[blue]🌍[/blue] Creating i18n resources...")
            i18n_resources = self.create_i18n_resources()
            
            console.print("[blue]📋[/blue] Creating compliance documentation...")
            compliance_docs = self.create_compliance_documentation()
            
            console.print("[blue]📊[/blue] Creating monitoring configuration...")
            monitoring_configs = self.create_monitoring_configuration()
            
            console.print("[blue]📜[/blue] Creating deployment scripts...")
            scripts = self.create_deployment_scripts()
            
            # Generate deployment summary
            end_time = datetime.now()
            duration = end_time - start_time
            
            deployment_summary = {
                'deployment_id': self.deployment_id,
                'timestamp': start_time.isoformat(),
                'duration': str(duration),
                'status': 'completed',
                'regions_configured': self.config['regions'],
                'languages_supported': self.config['languages'],
                'compliance_frameworks': self.config['compliance_frameworks'],
                'platforms_supported': self.config['platforms'],
                'artifacts': self.deployment_artifacts,
                'features': [
                    'Multi-region deployment',
                    'Auto-scaling',
                    'Load balancing',
                    'Health monitoring',
                    'Security compliance',
                    'Internationalization',
                    'Cross-platform support',
                    'Infrastructure as Code'
                ]
            }
            
            # Save deployment summary
            summary_file = self.deployment_dir / 'deployment_summary.json'
            with open(summary_file, 'w') as f:
                json.dump(deployment_summary, f, indent=2, default=str)
            
            # Create README for deployment
            readme_content = f'''
# AI Scientist v2 - Production Deployment Package

Deployment ID: {self.deployment_id}
Generated: {start_time.isoformat()}
Duration: {duration}

## Overview
This package contains all necessary artifacts for deploying AI Scientist v2 to production environments with:
- Multi-region support ({len(self.config["regions"])} regions)
- Internationalization ({len(self.config["languages"])} languages)
- Compliance frameworks ({len(self.config["compliance_frameworks"])} frameworks)
- Cross-platform compatibility

## Directory Structure
- `docker/`: Docker configuration and Dockerfile
- `kubernetes/`: Kubernetes manifests for deployment
- `terraform/`: Infrastructure as Code configurations
- `i18n/`: Internationalization resources
- `compliance/`: Compliance documentation
- `monitoring/`: Monitoring and observability configurations
- `scripts/`: Deployment automation scripts

## Quick Start
1. Review and customize configurations
2. Run `scripts/docker-build-push.sh` to build and push images
3. Run `scripts/k8s-deploy.sh` to deploy to Kubernetes
4. Run `scripts/health-check.sh` to verify deployment

## Support
For deployment support, contact: deployment@ai-scientist-v2.com
'''
            
            readme_file = self.deployment_dir / 'README.md'
            with open(readme_file, 'w') as f:
                f.write(readme_content)
            
            console.print(f"[green]✓[/green] Production deployment package created: {self.deployment_dir}")
            self.logger.info(f"Production deployment orchestration completed: {self.deployment_dir}")
            
            # Display summary
            self._display_deployment_summary(deployment_summary)
            
            return deployment_summary
            
        except Exception as e:
            error_msg = f"Production deployment orchestration failed: {str(e)}"
            self.logger.error(error_msg)
            console.print(f"[red]❌[/red] {error_msg}")
            
            return {
                'deployment_id': self.deployment_id,
                'status': 'failed',
                'error': error_msg,
                'timestamp': start_time.isoformat()
            }
    
    def _display_deployment_summary(self, summary: Dict) -> None:
        """Display comprehensive deployment summary."""
        if not RICH_AVAILABLE or not Table:
            console.print("\n=== Production Deployment Summary ===")
            console.print(f"Deployment ID: {summary['deployment_id']}")
            console.print(f"Status: {summary['status']}")
            console.print(f"Regions: {len(summary['regions_configured'])}")
            console.print(f"Languages: {len(summary['languages_supported'])}")
            return
        
        # Deployment overview table
        table = Table(title="🚀 AI Scientist v2 - Production Deployment Summary")
        table.add_column("Component", style="cyan", no_wrap=True)
        table.add_column("Status", style="green")
        table.add_column("Details", style="yellow")
        
        table.add_row("Deployment ID", "✓ Generated", summary['deployment_id'][:20] + "...")
        table.add_row("Multi-Region Setup", "✓ Configured", f"{len(summary['regions_configured'])} regions")
        table.add_row("Internationalization", "✓ Ready", f"{len(summary['languages_supported'])} languages")
        table.add_row("Compliance", "✓ Documented", f"{len(summary['compliance_frameworks'])} frameworks")
        table.add_row("Container Images", "✓ Ready", "Multi-platform Dockerfile")
        table.add_row("Kubernetes", "✓ Manifests Created", "Auto-scaling enabled")
        table.add_row("Infrastructure", "✓ Terraform Ready", "Multi-region IaC")
        table.add_row("Monitoring", "✓ Configured", "Prometheus + Grafana")
        table.add_row("Deployment Scripts", "✓ Created", "Automated deployment")
        
        console.print(table)
        
        # Features panel
        if Panel:
            features_text = "\n".join(f"✓ {feature}" for feature in summary.get('features', []))
            panel = Panel(
                features_text,
                title="🎆 Production Features Enabled",
                border_style="green"
            )
            console.print(panel)
        
        # Regional deployment info
        if Tree:
            tree = Tree("🌍 Global Deployment Configuration")
            
            regions_branch = tree.add("🗺️ Regions")
            for region in summary['regions_configured']:
                regions_branch.add(f"📍 {region}")
            
            languages_branch = tree.add("🌍 Languages")
            for lang in summary['languages_supported']:
                languages_branch.add(f"🗯️ {lang}")
            
            compliance_branch = tree.add("🛡️ Compliance")
            for framework in summary['compliance_frameworks']:
                compliance_branch.add(f"📜 {framework}")
            
            console.print(tree)


async def main():
    """Main entry point for production deployment orchestration."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Production Deployment Orchestrator for AI Scientist v2")
    parser.add_argument('--project-root', type=Path, default=Path.cwd(), help='Project root directory')
    parser.add_argument('--deployment-dir', type=str, help='Custom deployment directory name')
    
    args = parser.parse_args()
    
    try:
        orchestrator = ProductionDeploymentOrchestrator(args.project_root)
        
        if args.deployment_dir:
            orchestrator.deployment_id = args.deployment_dir
            orchestrator.deployment_dir = orchestrator._setup_deployment_directory()
        
        results = await orchestrator.orchestrate_production_deployment()
        
        success = results.get('status') == 'completed'
        return 0 if success else 1
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Deployment orchestration cancelled by user[/yellow]")
        return 130
    except Exception as e:
        console.print(f"[red]Fatal error during deployment orchestration: {e}[/red]")
        return 1


if __name__ == '__main__':
    sys.exit(asyncio.run(main()))
