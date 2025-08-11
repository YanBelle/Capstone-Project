#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main Dashboard Integration Script
================================

Script to integrate the cash forecasting service with the main ABM dashboard
"""

import os
import json
import requests
from datetime import datetime


def get_cash_forecasting_widget_config():
    """Get widget configuration for main dashboard"""
    return {
        'id': 'cash-forecasting',
        'title': 'Cash Forecasting',
        'description': 'ATM cash depletion predictions and alerts',
        'icon': 'fas fa-money-bill-wave',
        'color': 'success',
        'type': 'iframe',
        'iframe_url': '/cash-forecasting/',
        'width': 12,  # Full width
        'height': 600,
        'refresh_interval': 300,  # 5 minutes
        'api_endpoints': {
            'status': '/api/cash-forecasting/terminal-status',
            'alerts': '/api/cash-forecasting/alerts',
            'predictions': '/api/cash-forecasting/predictions'
        }
    }


def create_dashboard_link_html():
    """Create HTML snippet for main dashboard navigation"""
    return """
<!-- Cash Forecasting Navigation Link -->
<li class="nav-item">
    <a class="nav-link" href="/cash-forecasting/" target="_blank" title="Cash Forecasting Dashboard">
        <i class="fas fa-money-bill-wave me-2"></i>
        Cash Forecasting
        <span id="cash-alerts-badge" class="badge bg-danger d-none"></span>
    </a>
</li>

<script>
// Cash Forecasting Integration
async function updateCashForecastingBadge() {
    try {
        const response = await fetch('/api/cash-forecasting/alerts');
        const data = await response.json();
        const badge = document.getElementById('cash-alerts-badge');
        
        if (data.alerts && data.alerts.length > 0) {
            badge.textContent = data.alerts.length;
            badge.classList.remove('d-none');
        } else {
            badge.classList.add('d-none');
        }
    } catch (error) {
        console.error('Failed to update cash forecasting badge:', error);
    }
}

// Update badge every 5 minutes
setInterval(updateCashForecastingBadge, 5 * 60 * 1000);
updateCashForecastingBadge(); // Initial update
</script>
"""


def create_dashboard_widget_html():
    """Create widget HTML for main dashboard"""
    return """
<!-- Cash Forecasting Widget -->
<div class="col-12 mb-4">
    <div class="card border-success">
        <div class="card-header bg-success text-white d-flex justify-content-between align-items-center">
            <h5 class="mb-0">
                <i class="fas fa-money-bill-wave me-2"></i>
                Cash Forecasting Status
            </h5>
            <div>
                <button class="btn btn-sm btn-outline-light" onclick="refreshCashForecasting()">
                    <i class="fas fa-sync-alt"></i>
                </button>
                <a href="/cash-forecasting/" target="_blank" class="btn btn-sm btn-light">
                    <i class="fas fa-external-link-alt"></i> Open
                </a>
            </div>
        </div>
        <div class="card-body">
            <div class="row" id="cash-forecasting-summary">
                <div class="col-md-3">
                    <div class="text-center">
                        <h3 class="text-success" id="cf-total-terminals">-</h3>
                        <small class="text-muted">Total Terminals</small>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="text-center">
                        <h3 class="text-danger" id="cf-high-risk">-</h3>
                        <small class="text-muted">High Risk</small>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="text-center">
                        <h3 class="text-warning" id="cf-medium-risk">-</h3>
                        <small class="text-muted">Medium Risk</small>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="text-center">
                        <h3 class="text-success" id="cf-low-risk">-</h3>
                        <small class="text-muted">Low Risk</small>
                    </div>
                </div>
            </div>
            
            <hr>
            
            <div id="cash-forecasting-alerts">
                <!-- Alerts will be populated here -->
            </div>
            
            <div class="text-end">
                <small class="text-muted" id="cf-last-updated">Last updated: -</small>
            </div>
        </div>
    </div>
</div>

<script>
// Cash Forecasting Widget Functions
async function loadCashForecastingData() {
    try {
        // Load terminal status
        const statusResponse = await fetch('/api/cash-forecasting/terminal-status');
        const statusData = await statusResponse.json();
        
        // Update summary
        document.getElementById('cf-total-terminals').textContent = statusData.summary.total_terminals;
        document.getElementById('cf-high-risk').textContent = statusData.summary.high_risk;
        document.getElementById('cf-medium-risk').textContent = statusData.summary.medium_risk;
        document.getElementById('cf-low-risk').textContent = statusData.summary.low_risk;
        document.getElementById('cf-last-updated').textContent = 'Last updated: ' + new Date(statusData.timestamp).toLocaleString();
        
        // Load alerts
        const alertsResponse = await fetch('/api/cash-forecasting/alerts');
        const alertsData = await alertsResponse.json();
        
        const alertsContainer = document.getElementById('cash-forecasting-alerts');
        alertsContainer.innerHTML = '';
        
        if (alertsData.alerts && alertsData.alerts.length > 0) {
            alertsContainer.innerHTML = '<h6>Active Alerts:</h6>';
            alertsData.alerts.slice(0, 3).forEach(alert => { // Show top 3 alerts
                const alertDiv = document.createElement('div');
                alertDiv.className = 'alert alert-' + (alert.risk_level.toLowerCase() === 'high' ? 'danger' : 'warning') + ' alert-sm';
                alertDiv.innerHTML = `
                    <strong>Terminal ${alert.terminal_id}</strong>: ${alert.message}
                    <span class="badge bg-${alert.risk_level.toLowerCase() === 'high' ? 'danger' : 'warning'} float-end">${alert.risk_level}</span>
                `;
                alertsContainer.appendChild(alertDiv);
            });
            
            if (alertsData.alerts.length > 3) {
                const moreDiv = document.createElement('div');
                moreDiv.className = 'text-center';
                moreDiv.innerHTML = `<small class="text-muted">... and ${alertsData.alerts.length - 3} more alerts</small>`;
                alertsContainer.appendChild(moreDiv);
            }
        } else {
            alertsContainer.innerHTML = '<p class="text-muted text-center mb-0">No active alerts</p>';
        }
        
    } catch (error) {
        console.error('Failed to load cash forecasting data:', error);
        document.getElementById('cash-forecasting-alerts').innerHTML = 
            '<div class="alert alert-danger">Failed to load cash forecasting data</div>';
    }
}

function refreshCashForecasting() {
    loadCashForecastingData();
}

// Auto-refresh every 5 minutes
setInterval(loadCashForecastingData, 5 * 60 * 1000);
loadCashForecastingData(); // Initial load
</script>
"""


def create_integration_instructions():
    """Create integration instructions for the main dashboard"""
    return """
# Cash Forecasting Dashboard Integration Instructions

## Integration Steps

### 1. Add Navigation Link
Add the following HTML snippet to your main dashboard navigation:

```html
<!-- In your main navigation menu -->
<li class="nav-item">
    <a class="nav-link" href="/cash-forecasting/" target="_blank">
        <i class="fas fa-money-bill-wave me-2"></i>
        Cash Forecasting
        <span id="cash-alerts-badge" class="badge bg-danger d-none"></span>
    </a>
</li>
```

### 2. Add Dashboard Widget
Include the cash forecasting widget in your main dashboard:

```html
<!-- Add this widget to your dashboard grid -->
<div class="col-12 mb-4">
    <div class="card border-success">
        <div class="card-header bg-success text-white">
            <h5 class="mb-0">
                <i class="fas fa-money-bill-wave me-2"></i>
                Cash Forecasting Status
            </h5>
        </div>
        <!-- Widget content here -->
    </div>
</div>
```

### 3. Update Nginx Configuration
Ensure your nginx.conf includes the cash forecasting routes:

```nginx
# Cash forecasting service
location /cash-forecasting/ {
    proxy_pass http://cash-forecasting:5000/;
}

# API endpoints
location /api/cash-forecasting/ {
    proxy_pass http://cash-forecasting:5000/api/;
}
```

### 4. Deploy with Docker Compose
Run the integrated system:

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f cash-forecasting

# Scale if needed
docker-compose up -d --scale cash-forecasting=2
```

## API Endpoints

### Available Endpoints
- `GET /api/terminal-status` - Get terminal status summary
- `GET /api/alerts` - Get active alerts
- `GET /api/predictions` - Get all predictions
- `GET /api/predictions/{terminal_id}` - Get specific terminal prediction
- `GET /api/performance` - Get model performance metrics
- `POST /api/retrain` - Trigger model retraining

### Response Format
```json
{
    "terminals": [...],
    "summary": {
        "total_terminals": 5,
        "high_risk": 1,
        "medium_risk": 2,
        "low_risk": 2
    },
    "timestamp": "2025-01-27T10:30:00Z"
}
```

## Styling Integration

### CSS Classes
```css
.cash-forecasting-widget {
    border-left: 4px solid #28a745;
}

.risk-indicator {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    display: inline-block;
}

.risk-high { background-color: #dc3545; }
.risk-medium { background-color: #ffc107; }
.risk-low { background-color: #28a745; }
```

## Auto-Refresh Configuration

### JavaScript Integration
```javascript
// Auto-refresh cash forecasting data
setInterval(async () => {
    try {
        const response = await fetch('/api/cash-forecasting/terminal-status');
        const data = await response.json();
        updateCashForecastingWidget(data);
    } catch (error) {
        console.error('Cash forecasting update failed:', error);
    }
}, 5 * 60 * 1000); // 5 minutes
```

## Alert Integration

### Notification System
```javascript
// Check for high-priority alerts
async function checkCashAlerts() {
    const response = await fetch('/api/cash-forecasting/alerts');
    const data = await response.json();
    
    const highRiskAlerts = data.alerts.filter(a => a.risk_level === 'HIGH');
    
    if (highRiskAlerts.length > 0) {
        // Show notification
        showNotification(
            'Cash Alert', 
            `${highRiskAlerts.length} terminal(s) require immediate refill`,
            'error'
        );
    }
}
```

## Mobile Responsiveness

### Responsive Design
The cash forecasting dashboard is fully responsive and will adapt to:
- Desktop browsers
- Tablet displays
- Mobile devices

### PWA Support
Consider adding Progressive Web App features:
```html
<link rel="manifest" href="/manifest.json">
<meta name="theme-color" content="#28a745">
```

## Security Considerations

### API Security
- Implement API authentication if needed
- Use HTTPS in production
- Rate limiting for API endpoints
- CORS configuration for cross-origin requests

### Environment Variables
```env
DATABASE_URL=postgresql://user:pass@host:5432/db
MODEL_RETRAIN_HOURS=24
DASHBOARD_REFRESH_MINUTES=15
FLASK_ENV=production
```

## Monitoring & Logging

### Health Checks
- Service health: `GET /health`
- Model status: `GET /api/performance`
- Database connectivity: Included in health check

### Log Monitoring
```bash
# View service logs
docker-compose logs -f cash-forecasting

# Monitor model performance
tail -f logs/cash_forecasting.log
```

## Production Deployment Checklist

- [ ] Database connection configured
- [ ] Nginx reverse proxy set up
- [ ] SSL certificates installed (if HTTPS)
- [ ] Docker containers running
- [ ] Health checks passing
- [ ] API endpoints responding
- [ ] Dashboard integration complete
- [ ] Alert notifications working
- [ ] Auto-refresh functioning
- [ ] Mobile responsiveness tested
- [ ] Performance monitoring active

---

**Status**: Ready for Production Integration
**Service URL**: http://localhost:5001/cash-forecasting/
**API Base**: http://localhost:5001/api/
"""


def main():
    """Generate integration files and instructions"""
    print("Generating Cash Forecasting Dashboard Integration Files...")
    
    # Create integration directory
    integration_dir = "/Users/christopherpearson/Projects/abm_ej_exporter_docker_cleaned/EJAnomalyDetectionV3/integration"
    if not os.path.exists(integration_dir):
        os.makedirs(integration_dir)
    
    # Generate widget configuration
    widget_config = get_cash_forecasting_widget_config()
    with open("{}/widget_config.json".format(integration_dir), 'w') as f:
        json.dump(widget_config, f, indent=2)
    
    # Generate HTML snippets
    with open("{}/navigation_link.html".format(integration_dir), 'w') as f:
        f.write(create_dashboard_link_html())
    
    with open("{}/dashboard_widget.html".format(integration_dir), 'w') as f:
        f.write(create_dashboard_widget_html())
    
    # Generate integration instructions
    with open("{}/INTEGRATION_INSTRUCTIONS.md".format(integration_dir), 'w') as f:
        f.write(create_integration_instructions())
    
    print("Integration files generated successfully!")
    print("Files created in: {}".format(integration_dir))
    print("- widget_config.json")
    print("- navigation_link.html") 
    print("- dashboard_widget.html")
    print("- INTEGRATION_INSTRUCTIONS.md")


if __name__ == "__main__":
    main()
