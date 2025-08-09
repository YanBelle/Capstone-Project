# Development Environment Quick Reference

## Service Ports (Development vs Production)

### Core Services
- **API**: http://64.227.16.180:8001 (dev) vs http://64.227.16.180:8000 (prod)
- **Dashboard**: http://64.227.16.180:3000 (dev) vs http://64.227.16.180 (prod - via nginx)
- **PostgreSQL**: 5434 (dev) vs 5433 (prod)
- **Redis**: 6380 (dev) vs 6379 (prod)

### Development Tools
- **Jupyter Notebook**: http://64.227.16.180:8889 (dev) vs http://64.227.16.180:8888 (prod)
  - Token: `dev-token`
- **Grafana**: http://64.227.16.180:3002 (dev) vs http://64.227.16.180:3001 (prod)
  - Username: `admin`
  - Password: `dev-admin`
- **Prometheus**: http://64.227.16.180:9091 (dev) vs http://64.227.16.180:9090 (prod)

## Quick Commands

### Start Development Environment
```bash
docker-compose -f docker-compose.dev.yml up -d
```

### Stop Development Environment
```bash
docker-compose -f docker-compose.dev.yml down
```

### View Development Logs
```bash
docker-compose -f docker-compose.dev.yml logs -f
```

### Rebuild Development Services
```bash
docker-compose -f docker-compose.dev.yml up -d --build
```

### Check Development Status
```bash
docker-compose -f docker-compose.dev.yml ps
```

## TF-IDF Visualization Development

### API Endpoints (Development)
- **Vocabulary**: http://64.227.16.180:8001/api/v1/svm-tfidf/vocabulary
- **Session Analysis**: http://64.227.16.180:8001/api/v1/svm-tfidf/session/{session_id}

### Dashboard Access
- **TF-IDF Visualization**: http://64.227.16.180:3000 (React component integrated)

### Database Connection (Development)
```bash
psql -h 64.227.16.180 -p 5434 -U abmuser -d abmdb_dev
```

## Environment Separation

This development environment runs alongside production on the same DigitalOcean server using different ports. All data volumes and networks are isolated with `_dev` suffix to prevent conflicts.

### Container Names
All development containers use `_dev` suffix:
- `abm-ml-anomaly-detector-dev`
- `postgres_dev`
- `redis_dev`
- `api_dev`
- `dashboard_dev`
- `jupyter_dev`
- `grafana_dev`
- `prometheus_dev`

### Network Isolation
Development uses `abm_network_dev` network, completely separate from production `abm_network`.
