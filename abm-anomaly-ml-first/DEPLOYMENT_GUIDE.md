# ABM ML-First Anomaly Detection System - Flexible Deployment Guide

This guide explains how to deploy the ABM ML-First Anomaly Detection System in both local development and production environments using the new flexible deployment architecture.

## Quick Start

### Local Development
```bash
# Clone the repository
git clone <repository-url>
cd abm-anomaly-ml-first

# Start local development environment
./deploy.sh local up

# Access the application
# Dashboard: http://localhost
# API: http://localhost/api
# Grafana: http://localhost:3001
# Prometheus: http://localhost:9090
```

### Production Deployment
```bash
# Deploy to production (DigitalOcean or any server)
./deploy.sh production up
```

## Architecture Overview

The system uses a unified Docker Compose architecture that supports both local development and production deployments through environment-specific configuration files.

### Services

1. **API Service** (`services/api/`)
   - FastAPI backend with ML endpoints
   - Continuous learning and retraining capabilities
   - Database integration for labeled anomalies

2. **Anomaly Detector** (`services/anomaly-detector/`)
   - ML-First anomaly detection with multiple models:
     - Isolation Forest
     - One-Class SVM
     - Supervised Classifier
     - Expert Rules Engine
   - Real-time processing and batch analysis

3. **Dashboard** (`services/dashboard/`)
   - React-based web interface
   - Real-time anomaly visualization
   - Expert labeling system

4. **Database** (PostgreSQL)
   - ML sessions and results storage
   - Labeled anomalies for continuous learning
   - Expert feedback system

5. **Redis**
   - Caching and session management
   - Real-time data processing

6. **Nginx**
   - Reverse proxy and load balancer
   - Single entry point for all services

7. **Monitoring Stack**
   - Prometheus: Metrics collection
   - Grafana: Visualization and dashboards

## Environment Configuration

### Local Development (`.env.local`)
- Debug mode enabled
- Local database credentials
- Development-friendly logging
- Port 80 for easy access

### Production (`.env.production`)
- Production-optimized settings
- Secure credentials (via environment variables)
- Performance logging
- SSL/TLS ready

## Deployment Commands

The `deploy.sh` script provides a unified interface for all deployment operations:

### Basic Commands
```bash
# Start services
./deploy.sh [local|production] up

# Stop services
./deploy.sh [local|production] down

# Build services
./deploy.sh [local|production] build

# View logs
./deploy.sh [local|production] logs

# Restart services
./deploy.sh [local|production] restart

# Check status
./deploy.sh [local|production] status
```

### Examples
```bash
# Local development
./deploy.sh local up          # Start local environment
./deploy.sh local logs        # View local logs
./deploy.sh local down        # Stop local environment

# Production deployment
./deploy.sh production build  # Build for production
./deploy.sh production up     # Deploy to production
./deploy.sh production status # Check production status
```

## Database Setup

The system includes automatic database migration and schema setup:

1. **Initial Schema**: Base tables for ML operations
2. **ML Schema**: Enhanced tables for continuous learning
3. **Migration Support**: Automatic schema updates

### Manual Database Operations
```bash
# Access database directly
docker exec -it abm-ml-postgres psql -U abm_user -d abm_anomaly_detection

# Run migrations manually
docker exec -it abm-ml-api python apply_migration.py
```

## ML Model Management

### Continuous Learning Workflow
1. **Data Collection**: ABM logs processed and sessionized
2. **Anomaly Detection**: Multi-model analysis with ensemble scoring
3. **Expert Labeling**: Human feedback through dashboard interface
4. **Model Retraining**: Automatic retraining with labeled data
5. **Performance Monitoring**: Continuous model evaluation

### Model Files
- `data/models/isolation_forest.pkl`
- `data/models/one_class_svm.pkl`
- `data/models/expert_rules.json`

### Retraining Process
The system supports both manual and automatic retraining:
- **Manual**: Click "Retrain Models" in dashboard
- **Automatic**: Scheduled retraining based on new labeled data
- **Feedback Integration**: Uses database-stored expert feedback

## API Endpoints

### Core Endpoints
- `GET /api/health` - Health check
- `POST /api/detect-anomalies` - Process ABM logs for anomalies
- `POST /api/retrain` - Trigger model retraining
- `GET /api/anomalies` - Retrieve anomaly reports
- `POST /api/label-anomaly` - Submit expert feedback

### Monitoring Endpoints
- `GET /api/metrics` - System metrics
- `GET /api/models/status` - Model training status
- `GET /api/continuous-learning/status` - Learning system status

## Monitoring and Observability

### Grafana Dashboards
- **System Overview**: Service health and performance
- **ML Metrics**: Model accuracy and retraining events
- **Anomaly Trends**: Detection patterns over time

### Prometheus Metrics
- Custom ML metrics for model performance
- System resource utilization
- API response times and error rates

## Troubleshooting

### Common Issues

1. **Services Not Starting**
   ```bash
   # Check service status
   ./deploy.sh [environment] status
   
   # View logs for errors
   ./deploy.sh [environment] logs
   ```

2. **Database Connection Issues**
   ```bash
   # Verify database is running
   docker exec -it abm-ml-postgres pg_isready
   
   # Check database logs
   docker logs abm-ml-postgres
   ```

3. **Model Loading Errors**
   ```bash
   # Check if model files exist
   ls -la data/models/
   
   # Retrain models if missing
   curl -X POST http://localhost/api/retrain
   ```

4. **Port Conflicts**
   ```bash
   # Check what's using port 80
   sudo lsof -i :80
   
   # Modify NGINX_PORT in environment file if needed
   ```

### Log Locations
- **Application Logs**: `./deploy.sh [env] logs`
- **Individual Service**: `docker logs [container-name]`
- **System Logs**: `data/logs/`

## Development Workflow

### Local Development Setup
1. Clone repository
2. Start local environment: `./deploy.sh local up`
3. Make code changes
4. Rebuild specific service: `docker-compose build [service-name]`
5. Restart: `./deploy.sh local restart`

### Testing Changes
```bash
# Run tests locally
python -m pytest tests/

# Test specific components
python test_anomaly_detection.py
python test_comprehensive.py
```

### Production Deployment
1. Test locally first
2. Commit changes to main branch
3. GitHub Actions automatically deploys to production
4. Monitor deployment: `./deploy.sh production status`

## Security Considerations

### Production Security
- Use strong passwords in `.env.production`
- Set up SSL/TLS certificates
- Configure firewall rules
- Regular security updates
- Database access restrictions

### Environment Variables
Never commit sensitive data to version control. Use environment variables for:
- Database passwords
- API keys
- JWT secrets
- External service credentials

## Performance Optimization

### Resource Allocation
- **Development**: Minimal resources for fast iteration
- **Production**: Optimized for throughput and stability

### Scaling Considerations
- Redis for session management
- Database connection pooling
- Load balancing with Nginx
- Horizontal scaling ready

## Support and Maintenance

### Regular Maintenance
1. **Log Rotation**: Monitor disk usage
2. **Model Updates**: Periodic retraining
3. **Security Updates**: Keep containers updated
4. **Backup Strategy**: Database and model backups

### Getting Help
- Check logs first: `./deploy.sh [env] logs`
- Review this documentation
- Examine error messages carefully
- Test in local environment first

This flexible deployment system ensures consistent behavior across development and production environments while maintaining the full feature set of the ML-First ABM Anomaly Detection System.
