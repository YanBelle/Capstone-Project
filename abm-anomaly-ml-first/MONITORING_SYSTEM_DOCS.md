# Real-time Monitoring System Documentation

## Overview

The real-time monitoring system provides comprehensive visibility into the ABM anomaly detection pipeline, tracking parsing, sessionization, and ML training activities in real-time.

## Architecture

### Components

1. **Frontend Monitoring Interface** (`RealtimeMonitoringInterface.js`)
   - React component with real-time dashboard
   - WebSocket connectivity for live updates
   - Performance charts and system metrics
   - Activity logs with filtering capabilities

2. **Backend API Endpoints** (`services/api/main.py`)
   - `/api/v1/monitoring/status` - Current monitoring statistics
   - `/api/v1/monitoring/logs` - Recent system logs
   - `/api/v1/monitoring/performance` - Detailed performance metrics
   - `/ws/monitoring` - WebSocket endpoint for real-time updates

3. **Monitoring Collection** (`monitoring_utils.py`)
   - Centralized statistics collection
   - Redis-based data storage
   - Component activity logging

4. **ML Integration** (`monitoring_integration.py`)
   - Hooks into ML analyzer operations
   - Training progress tracking
   - Error monitoring and reporting

## Features

### Real-time Statistics
- **Parsing**: Transaction processing rate, errors, status
- **Sessionization**: Active sessions, creation rate, status  
- **ML Training**: Model accuracy, training time, status
- **System**: CPU, memory, disk usage, uptime

### Activity Logging
- Component-specific activity tracking
- Session-based filtering
- Error tracking and alerts
- Performance metrics

### Performance Monitoring
- System resource utilization
- Database connection monitoring
- Redis performance metrics
- Processing throughput tracking

## API Endpoints

### GET /api/v1/monitoring/status
Returns current monitoring statistics for all components.

**Response:**
```json
{
  "parsing": {
    "rate": 120.5,
    "processed": 15000,
    "errors": 3,
    "status": "active"
  },
  "sessionization": {
    "rate": 15.2,
    "sessions_created": 1200,
    "active_sessions": 45,
    "status": "active"
  },
  "ml_training": {
    "accuracy": 0.923,
    "models_trained": 5,
    "training_time": 45.2,
    "status": "idle"
  },
  "system": {
    "cpu": 45.2,
    "memory": 62.1,
    "disk": 23.4,
    "uptime": 86400
  },
  "timestamp": "2025-01-06T10:30:00Z"
}
```

### GET /api/v1/monitoring/logs
Returns recent system logs with optional filtering.

**Parameters:**
- `level` (optional): Filter by log level (INFO, ERROR, DEBUG)
- `component` (optional): Filter by component name
- `limit` (optional): Maximum number of logs to return (default: 100)

**Response:**
```json
{
  "status": "success",
  "logs": [
    {
      "timestamp": "2025-01-06T10:29:45Z",
      "level": "INFO",
      "component": "ml_training",
      "message": "Completed Isolation Forest training",
      "session_id": "abc123"
    }
  ],
  "total": 45
}
```

### WebSocket /ws/monitoring
Real-time monitoring updates pushed every 5 seconds.

**Message Format:**
```json
{
  "parsing": {...},
  "sessionization": {...},
  "ml_training": {...},
  "system": {...},
  "timestamp": "2025-01-06T10:30:00Z"
}
```

## Integration Guide

### Adding Monitoring to ML Components

1. **Import monitoring functions:**
```python
from monitoring_integration import (
    mark_ml_training_start, mark_ml_training_complete,
    mark_ml_detection_run, log_ml_activity
)
```

2. **Track training operations:**
```python
# Start training
mark_ml_training_start("isolation_forest")

# During training
log_ml_activity("Training epoch completed", details={"epoch": 5, "accuracy": 0.92})

# Complete training  
mark_ml_training_complete(accuracy=0.95, training_time=120.5, model_type="isolation_forest")
```

3. **Track detection runs:**
```python
# After detection
mark_ml_detection_run(session_count=1000, anomaly_count=45)
```

### Adding Monitoring to Processing Components

1. **Import utilities:**
```python
from monitoring_utils import update_parsing_stats, log_component_activity
```

2. **Update statistics:**
```python
# Update parsing stats
update_parsing_stats(
    processed_count=100,
    error_count=2,
    rate=50.0,  # per minute
    status="active"
)

# Log activity
log_component_activity(
    component="parsing",
    activity="Processed batch of transactions",
    details={"batch_size": 100}
)
```

## Dashboard Usage

### Navigation
The monitoring interface is accessible via the "Monitoring" tab in the main dashboard.

### Real-time Updates
- Statistics update every 5 seconds via WebSocket
- Charts show last 20 data points (covering ~100 seconds)
- Logs refresh automatically with new entries

### Filtering
- Filter logs by component (parsing, sessionization, ml_training)
- Filter by log level (INFO, ERROR, DEBUG)
- Search logs by keyword

### System Metrics
- CPU usage percentage
- Memory utilization
- Disk space usage
- System uptime

## Configuration

### Environment Variables
```bash
REDIS_HOST=redis
REDIS_PASSWORD=your_password
POSTGRES_HOST=postgres
POSTGRES_USER=your_user
POSTGRES_PASSWORD=your_password
```

### Dependencies
Add to `requirements.txt`:
```
psutil==5.9.5
websockets==11.0.3
```

## Testing

Run the monitoring test script:
```bash
python test_monitoring.py
```

This will simulate:
- 30 seconds of parsing activity
- 20 iterations of sessionization
- Complete ML training cycle

## Troubleshooting

### Common Issues

1. **WebSocket Connection Failed**
   - Check if API server is running
   - Verify firewall/port settings
   - Check browser console for errors

2. **No Monitoring Data**
   - Verify Redis connection
   - Check if monitoring integration is imported
   - Ensure monitoring functions are being called

3. **Performance Issues**
   - Monitor system resources
   - Check database connection pool
   - Verify Redis performance

### Debug Commands
```bash
# Check Redis connectivity
redis-cli ping

# Monitor Redis keys
redis-cli keys "monitoring:*"

# Check API logs
docker logs abm-api

# Test monitoring endpoints
curl http://localhost:8000/api/v1/monitoring/status
```

## Future Enhancements

1. **Alerting System**
   - Email/Slack notifications for errors
   - Threshold-based alerts
   - Performance degradation detection

2. **Historical Analytics**
   - Long-term trend analysis
   - Performance regression detection
   - Capacity planning metrics

3. **Advanced Visualizations**
   - Heat maps for error patterns
   - Geographic distribution (if applicable)
   - Anomaly pattern visualization

4. **Integration Monitoring**
   - External API response times
   - Database query performance
   - Message queue metrics
