# DevOps Guide - Transcription Service Backend

## Overview

This guide provides comprehensive information for deploying, monitoring, and maintaining the Transcription Service Backend in production environments.

## Architecture

The system is built with a microservices architecture using:

- **FastAPI Application**: Main REST API service
- **Celery Workers**: Background task processing for transcription jobs
- **PostgreSQL**: Primary database for persistent data
- **Redis**: Caching layer and message broker for Celery
- **Nginx**: Reverse proxy and load balancer
- **Prometheus/Grafana**: Monitoring and observability

## Infrastructure Requirements

### Minimum System Requirements

#### Production Environment
- **CPU**: 8 cores minimum (16 cores recommended)
- **RAM**: 16GB minimum (32GB recommended)
- **Storage**: 500GB SSD (for audio files and models)
- **Network**: 1Gbps bandwidth

#### Development Environment
- **CPU**: 4 cores minimum
- **RAM**: 8GB minimum
- **Storage**: 100GB SSD
- **Network**: 100Mbps bandwidth

### Container Resources

| Service | CPU Request | CPU Limit | Memory Request | Memory Limit |
|---------|------------|-----------|----------------|--------------|
| App | 250m | 1 | 512Mi | 2Gi |
| Worker | 500m | 2 | 2Gi | 4Gi |
| PostgreSQL | 250m | 1 | 512Mi | 2Gi |
| Redis | 100m | 500m | 256Mi | 512Mi |
| Nginx | 50m | 100m | 64Mi | 128Mi |

## Deployment

### Docker Deployment

#### Production
```bash
# Build and deploy with Docker Compose
docker-compose -f docker-compose.yml up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f app
```

#### Development
```bash
# Start development environment
docker-compose -f docker-compose.dev.yml up -d

# Run tests
docker-compose -f docker-compose.test.yml run --rm unit-tests
```

### Kubernetes Deployment

#### Prerequisites
- Kubernetes cluster (v1.24+)
- kubectl configured
- NGINX Ingress Controller
- cert-manager (for TLS certificates)

#### Quick Deploy
```bash
cd deployment/scripts
./deploy.sh production deploy
```

#### Manual Deployment
```bash
# Apply in order
kubectl apply -f deployment/kubernetes/namespace.yaml
kubectl apply -f deployment/kubernetes/secrets.yaml
kubectl apply -f deployment/kubernetes/configmap.yaml
kubectl apply -f deployment/kubernetes/persistent-volumes.yaml
kubectl apply -f deployment/kubernetes/postgres.yaml
kubectl apply -f deployment/kubernetes/redis.yaml
kubectl apply -f deployment/kubernetes/app.yaml
kubectl apply -f deployment/kubernetes/ingress.yaml
```

#### Scaling
```bash
# Scale application
kubectl scale deployment transcription-service --replicas=5 -n transcription-service

# Scale workers
kubectl scale deployment celery-worker --replicas=3 -n transcription-service
```

## Configuration Management

### Environment Variables

#### Application Configuration
| Variable | Description | Default |
|----------|-------------|---------|
| `ENVIRONMENT` | Environment name | `production` |
| `SECRET_KEY` | Application secret key | Required |
| `DB_URL` | Database connection string | Required |
| `REDIS_URL` | Redis connection string | Required |
| `HF_TOKEN` | HuggingFace API token | Required |

#### AI Model Configuration
| Variable | Description | Default |
|----------|-------------|---------|
| `AI_DEVICE` | Device for AI models (cpu/cuda) | `cpu` |
| `WHISPER_MODEL_PATH` | Whisper model identifier | `openai/whisper-large-v3` |
| `DIARIZATION_MODEL_PATH` | Diarization model path | `pyannote/speaker-diarization` |

#### Storage Configuration
| Variable | Description | Default |
|----------|-------------|---------|
| `STORAGE_TYPE` | Storage backend type | `local` |
| `STORAGE_BASE_PATH` | Base path for file storage | `/app/data` |
| `AUDIO_MAX_FILE_SIZE` | Maximum audio file size | `104857600` (100MB) |

### Secrets Management

Secrets are managed through Kubernetes secrets:

```bash
# Create secrets
kubectl create secret generic transcription-service-secrets \
  --from-literal=SECRET_KEY=your-secret-key \
  --from-literal=HF_TOKEN=your-huggingface-token \
  -n transcription-service
```

### ConfigMaps

Configuration is managed through ConfigMaps. Update configuration:

```bash
kubectl edit configmap transcription-service-config -n transcription-service
```

## Monitoring and Observability

### Metrics Collection

The application exposes Prometheus metrics at `/metrics`. Key metrics include:

- **Application Metrics**: Request rate, response time, error rate
- **Business Metrics**: Job completion rate, queue length, processing time
- **Infrastructure Metrics**: CPU, memory, disk usage
- **Custom Metrics**: AI model performance, audio processing metrics

### Dashboards

Grafana dashboards are available for:

1. **Application Overview**: High-level service health
2. **API Performance**: Request metrics and response times
3. **Job Processing**: Transcription job metrics
4. **Infrastructure**: System resource usage
5. **AI Models**: Model loading and performance metrics

### Alerting

Prometheus alerting rules are configured for:

- Service availability (< 99.9% uptime)
- High error rate (> 10% errors)
- High response time (> 2s 95th percentile)
- Queue backlog (> 100 pending jobs)
- Resource usage (CPU > 80%, Memory > 90%)

### Log Management

Logs are structured in JSON format and include:

- **Access logs**: API request/response
- **Application logs**: Business logic events
- **Error logs**: Exceptions and errors
- **Audit logs**: Security-related events
- **Performance logs**: Timing and performance data

Log rotation is configured with 10MB files, keeping 5 backups.

## Database Management

### Migrations

Database migrations are handled by Alembic:

```bash
# Run migrations
kubectl run migration-job --rm -i --restart=Never \
  --image=ghcr.io/yourusername/transcription-service:latest \
  --env="DB_URL=postgresql://user:pass@postgres:5432/db" \
  -n transcription-service \
  -- alembic upgrade head
```

### Backup Strategy

#### Automated Backups
- Daily full backups at 2 AM UTC
- Continuous WAL archiving
- Cross-region backup replication
- 30-day retention policy

#### Manual Backup
```bash
# Create backup
kubectl exec -it postgres-pod -n transcription-service -- \
  pg_dump -U transcription_user transcription_db > backup.sql

# Restore backup
kubectl exec -i postgres-pod -n transcription-service -- \
  psql -U transcription_user transcription_db < backup.sql
```

### Performance Optimization

#### Database Configuration
- Shared buffers: 25% of available RAM
- Effective cache size: 75% of available RAM
- Work mem: 4MB
- Maintenance work mem: 64MB
- Max connections: 200

#### Index Management
Key indexes for performance:
- Jobs by user_id and status
- Jobs by created_at (for pagination)
- Users by email (unique)
- Segments by job_id

## Security

### Network Security

- All inter-service communication encrypted
- Network policies restrict pod-to-pod communication
- WAF configured on ingress
- Rate limiting: 100 requests/minute per IP

### Data Security

- Encryption at rest for database and file storage
- TLS 1.3 for all HTTPS traffic
- Secrets encrypted in Kubernetes
- No sensitive data in logs

### Access Control

- RBAC configured for Kubernetes access
- Service accounts with minimal permissions
- No root containers in production
- Security contexts enforced

### Vulnerability Management

- Container images scanned with Trivy
- Dependencies checked with Safety
- Regular security updates
- Penetration testing quarterly

## Disaster Recovery

### Backup Strategy

#### Data Backups
- Database: Daily full + continuous WAL
- Audio files: Daily incremental sync
- Configuration: Version controlled in Git

#### Recovery Objectives
- **RTO (Recovery Time Objective)**: 1 hour
- **RPO (Recovery Point Objective)**: 15 minutes

### Failover Procedures

#### Database Failover
1. Promote standby database
2. Update connection strings
3. Restart application pods
4. Verify data consistency

#### Application Failover
1. Deploy to backup region
2. Update DNS records
3. Verify service health
4. Monitor for issues

## Performance Tuning

### Application Optimization

#### FastAPI Configuration
- Worker processes: CPU cores × 2
- Keep-alive timeout: 75 seconds
- Max request size: 100MB
- Connection pool: 20 connections

#### Celery Configuration
- Worker concurrency: CPU cores ÷ 2
- Task acknowledgment: Late
- Result backend: Redis
- Task compression: gzip

### Infrastructure Optimization

#### Resource Allocation
- CPU requests set to 80% of limits
- Memory requests set to 90% of limits
- Horizontal Pod Autoscaler configured
- Node affinity for AI workloads

#### Storage Optimization
- SSD storage for database
- NFS for shared audio files
- Lifecycle policies for old files
- Compression for archived data

## Troubleshooting

### Common Issues

#### High Memory Usage
```bash
# Check memory usage
kubectl top pods -n transcription-service

# Check for memory leaks
kubectl logs deployment/transcription-service -n transcription-service | grep -i memory
```

#### Database Connection Issues
```bash
# Check database connectivity
kubectl exec -it app-pod -n transcription-service -- \
  pg_isready -h postgres-service -p 5432

# Check connection pool
kubectl logs deployment/transcription-service -n transcription-service | grep "connection pool"
```

#### Queue Backlog
```bash
# Check Celery queue length
kubectl exec -it redis-pod -n transcription-service -- \
  redis-cli llen celery

# Check worker status
kubectl logs deployment/celery-worker -n transcription-service
```

### Debug Commands

#### Application Debug
```bash
# Enter application container
kubectl exec -it deployment/transcription-service -n transcription-service -- bash

# Check application health
curl http://localhost:8000/health

# View real-time logs
kubectl logs -f deployment/transcription-service -n transcription-service
```

#### Database Debug
```bash
# Connect to database
kubectl exec -it deployment/postgres -n transcription-service -- \
  psql -U transcription_user -d transcription_db

# Check active connections
SELECT * FROM pg_stat_activity;

# Check database size
SELECT pg_size_pretty(pg_database_size('transcription_db'));
```

## Maintenance

### Regular Maintenance Tasks

#### Weekly
- Review error logs and metrics
- Check disk space usage
- Verify backup integrity
- Update security patches

#### Monthly
- Database maintenance (VACUUM, REINDEX)
- Clean up old audio files
- Review and update monitoring alerts
- Performance optimization review

#### Quarterly
- Security audit and penetration testing
- Disaster recovery testing
- Capacity planning review
- Update dependencies

### Update Procedures

#### Application Updates
1. Test in staging environment
2. Create database backup
3. Deploy with rolling update
4. Monitor metrics and logs
5. Rollback if issues detected

#### Infrastructure Updates
1. Plan maintenance window
2. Notify stakeholders
3. Apply updates to staging first
4. Monitor system health
5. Document changes

## Contact Information

### On-Call Procedures

#### Escalation Matrix
1. **Level 1**: DevOps Team
2. **Level 2**: Backend Engineering Team
3. **Level 3**: Infrastructure Team
4. **Level 4**: VP Engineering

#### Emergency Contacts
- **DevOps Team**: devops@company.com
- **Backend Team**: backend@company.com
- **Infrastructure**: infrastructure@company.com

### Support Channels

- **Slack**: #transcription-service
- **Email**: support@company.com
- **Ticketing**: JIRA Service Desk
- **Documentation**: Confluence Wiki