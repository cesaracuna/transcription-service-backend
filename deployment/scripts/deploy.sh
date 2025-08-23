#!/bin/bash

# Transcription Service Deployment Script
# Usage: ./deploy.sh [environment] [action]
# Example: ./deploy.sh production deploy

set -e

ENVIRONMENT=${1:-staging}
ACTION=${2:-deploy}
NAMESPACE="transcription-service"
IMAGE_TAG=${3:-latest}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
    error "kubectl is not installed or not in PATH"
fi

# Check if we're connected to the cluster
if ! kubectl cluster-info &> /dev/null; then
    error "Not connected to Kubernetes cluster"
fi

# Function to deploy resources
deploy() {
    log "Starting deployment to $ENVIRONMENT environment..."
    
    # Create namespace if it doesn't exist
    log "Creating namespace..."
    kubectl apply -f ../kubernetes/namespace.yaml
    
    # Apply secrets (these should be updated separately for security)
    log "Applying secrets..."
    kubectl apply -f ../kubernetes/secrets.yaml
    
    # Apply ConfigMaps
    log "Applying configuration..."
    kubectl apply -f ../kubernetes/configmap.yaml
    
    # Apply persistent volumes
    log "Setting up storage..."
    kubectl apply -f ../kubernetes/persistent-volumes.yaml
    
    # Deploy database
    log "Deploying PostgreSQL..."
    kubectl apply -f ../kubernetes/postgres.yaml
    
    # Deploy Redis
    log "Deploying Redis..."
    kubectl apply -f ../kubernetes/redis.yaml
    
    # Wait for database and Redis to be ready
    log "Waiting for database to be ready..."
    kubectl wait --for=condition=ready pod -l app=postgres -n $NAMESPACE --timeout=300s
    
    log "Waiting for Redis to be ready..."
    kubectl wait --for=condition=ready pod -l app=redis -n $NAMESPACE --timeout=300s
    
    # Update image tag in deployment
    if [ "$IMAGE_TAG" != "latest" ]; then
        log "Updating image tag to $IMAGE_TAG..."
        sed -i.bak "s|:latest|:$IMAGE_TAG|g" ../kubernetes/app.yaml
    fi
    
    # Deploy application
    log "Deploying application..."
    kubectl apply -f ../kubernetes/app.yaml
    
    # Deploy ingress
    log "Setting up ingress..."
    kubectl apply -f ../kubernetes/ingress.yaml
    
    # Wait for deployment to be ready
    log "Waiting for application to be ready..."
    kubectl wait --for=condition=available deployment/transcription-service -n $NAMESPACE --timeout=600s
    kubectl wait --for=condition=available deployment/celery-worker -n $NAMESPACE --timeout=600s
    
    # Restore original app.yaml if we modified it
    if [ "$IMAGE_TAG" != "latest" ] && [ -f "../kubernetes/app.yaml.bak" ]; then
        mv ../kubernetes/app.yaml.bak ../kubernetes/app.yaml
    fi
    
    log "Deployment completed successfully!"
    
    # Show deployment status
    status
}

# Function to check deployment status
status() {
    log "Checking deployment status..."
    
    echo -e "\n${GREEN}Pods:${NC}"
    kubectl get pods -n $NAMESPACE
    
    echo -e "\n${GREEN}Services:${NC}"
    kubectl get services -n $NAMESPACE
    
    echo -e "\n${GREEN}Ingress:${NC}"
    kubectl get ingress -n $NAMESPACE
    
    echo -e "\n${GREEN}Persistent Volume Claims:${NC}"
    kubectl get pvc -n $NAMESPACE
    
    # Check if all pods are running
    NOT_RUNNING=$(kubectl get pods -n $NAMESPACE --field-selector=status.phase!=Running --no-headers 2>/dev/null | wc -l)
    if [ "$NOT_RUNNING" -gt 0 ]; then
        warn "Some pods are not running. Check with: kubectl get pods -n $NAMESPACE"
    else
        log "All pods are running successfully!"
    fi
}

# Function to rollback deployment
rollback() {
    log "Rolling back deployment..."
    
    # Rollback main application
    kubectl rollout undo deployment/transcription-service -n $NAMESPACE
    kubectl rollout undo deployment/celery-worker -n $NAMESPACE
    kubectl rollout undo deployment/nginx -n $NAMESPACE
    
    # Wait for rollback to complete
    kubectl rollout status deployment/transcription-service -n $NAMESPACE
    kubectl rollout status deployment/celery-worker -n $NAMESPACE
    kubectl rollout status deployment/nginx -n $NAMESPACE
    
    log "Rollback completed!"
}

# Function to scale deployment
scale() {
    REPLICAS=${3:-3}
    log "Scaling deployment to $REPLICAS replicas..."
    
    kubectl scale deployment transcription-service --replicas=$REPLICAS -n $NAMESPACE
    kubectl scale deployment celery-worker --replicas=$REPLICAS -n $NAMESPACE
    
    kubectl rollout status deployment/transcription-service -n $NAMESPACE
    kubectl rollout status deployment/celery-worker -n $NAMESPACE
    
    log "Scaling completed!"
}

# Function to clean up deployment
cleanup() {
    log "Cleaning up deployment..."
    
    # Delete all resources in order
    kubectl delete -f ../kubernetes/ingress.yaml --ignore-not-found=true
    kubectl delete -f ../kubernetes/app.yaml --ignore-not-found=true
    kubectl delete -f ../kubernetes/redis.yaml --ignore-not-found=true
    kubectl delete -f ../kubernetes/postgres.yaml --ignore-not-found=true
    kubectl delete -f ../kubernetes/persistent-volumes.yaml --ignore-not-found=true
    kubectl delete -f ../kubernetes/configmap.yaml --ignore-not-found=true
    kubectl delete -f ../kubernetes/secrets.yaml --ignore-not-found=true
    kubectl delete -f ../kubernetes/namespace.yaml --ignore-not-found=true
    
    log "Cleanup completed!"
}

# Function to show logs
logs() {
    COMPONENT=${3:-app}
    log "Showing logs for $COMPONENT..."
    
    case $COMPONENT in
        app)
            kubectl logs -f deployment/transcription-service -n $NAMESPACE
            ;;
        worker)
            kubectl logs -f deployment/celery-worker -n $NAMESPACE
            ;;
        beat)
            kubectl logs -f deployment/celery-beat -n $NAMESPACE
            ;;
        postgres)
            kubectl logs -f deployment/postgres -n $NAMESPACE
            ;;
        redis)
            kubectl logs -f deployment/redis -n $NAMESPACE
            ;;
        nginx)
            kubectl logs -f deployment/nginx -n $NAMESPACE
            ;;
        *)
            error "Unknown component: $COMPONENT. Available: app, worker, beat, postgres, redis, nginx"
            ;;
    esac
}

# Function to run database migrations
migrate() {
    log "Running database migrations..."
    
    kubectl run migration-job --rm -i --restart=Never \
        --image=ghcr.io/yourusername/transcription-service:$IMAGE_TAG \
        --env="DB_URL=postgresql://transcription_user:transcription_pass@postgres-service:5432/transcription_db" \
        -n $NAMESPACE \
        -- alembic upgrade head
    
    log "Database migration completed!"
}

# Function to run health check
health() {
    log "Running health check..."
    
    # Get service IP
    SERVICE_IP=$(kubectl get service transcription-service -n $NAMESPACE -o jsonpath='{.spec.clusterIP}')
    
    # Run health check pod
    kubectl run health-check --rm -i --restart=Never \
        --image=curlimages/curl:latest \
        -n $NAMESPACE \
        -- curl -f http://$SERVICE_IP:8000/health
    
    log "Health check completed!"
}

# Main script logic
case $ACTION in
    deploy)
        deploy
        ;;
    status)
        status
        ;;
    rollback)
        rollback
        ;;
    scale)
        scale
        ;;
    cleanup)
        cleanup
        ;;
    logs)
        logs
        ;;
    migrate)
        migrate
        ;;
    health)
        health
        ;;
    *)
        echo "Usage: $0 [environment] [action] [options]"
        echo ""
        echo "Actions:"
        echo "  deploy   - Deploy the application"
        echo "  status   - Check deployment status"
        echo "  rollback - Rollback to previous version"
        echo "  scale    - Scale deployment (requires replicas count)"
        echo "  cleanup  - Remove all resources"
        echo "  logs     - Show logs (requires component name)"
        echo "  migrate  - Run database migrations"
        echo "  health   - Run health check"
        echo ""
        echo "Examples:"
        echo "  $0 production deploy"
        echo "  $0 staging status"
        echo "  $0 production scale 5"
        echo "  $0 staging logs app"
        echo "  $0 production migrate"
        exit 1
        ;;
esac