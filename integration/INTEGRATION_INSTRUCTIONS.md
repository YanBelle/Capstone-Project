
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
