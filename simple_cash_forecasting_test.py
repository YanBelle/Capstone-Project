#!/usr/bin/env python3
"""
Simple Cash Forecasting Test Application
"""

from flask import Flask, jsonify, render_template_string
import logging
import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Simple HTML template for testing
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Cash Forecasting Dashboard - Test</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .container { max-width: 1200px; margin: 0 auto; }
        .card { border: 1px solid #ddd; padding: 20px; margin: 20px 0; border-radius: 5px; }
        .success { background-color: #d4edda; border-color: #c3e6cb; }
        .warning { background-color: #fff3cd; border-color: #ffeaa7; }
        .danger { background-color: #f8d7da; border-color: #f5c6cb; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; }
        h1, h2 { color: #333; }
        .status { font-weight: bold; padding: 5px 10px; border-radius: 3px; }
        .status.healthy { background-color: #28a745; color: white; }
        .status.warning { background-color: #ffc107; color: black; }
        .status.critical { background-color: #dc3545; color: white; }
    </style>
</head>
<body>
    <div class="container">
        <h1>💰 Cash Forecasting Dashboard</h1>
        <p>Production ML System for ATM Cash Depletion Prediction</p>
        
        <div class="card success">
            <h2>✅ System Status</h2>
            <p><strong>Service:</strong> <span class="status healthy">RUNNING</span></p>
            <p><strong>Timestamp:</strong> {{ timestamp }}</p>
            <p><strong>Version:</strong> 1.0.0</p>
        </div>
        
        <div class="grid">
            <div class="card">
                <h3>🏛️ Terminal ATM001</h3>
                <p><strong>Risk Level:</strong> <span class="status healthy">LOW</span></p>
                <p><strong>Cash Level:</strong> 85%</p>
                <p><strong>Predicted Depletion:</strong> 7 days</p>
            </div>
            
            <div class="card">
                <h3>🏛️ Terminal ATM002</h3>
                <p><strong>Risk Level:</strong> <span class="status warning">MEDIUM</span></p>
                <p><strong>Cash Level:</strong> 45%</p>
                <p><strong>Predicted Depletion:</strong> 3 days</p>
            </div>
            
            <div class="card">
                <h3>🏛️ Terminal ATM003</h3>
                <p><strong>Risk Level:</strong> <span class="status critical">HIGH</span></p>
                <p><strong>Cash Level:</strong> 15%</p>
                <p><strong>Predicted Depletion:</strong> 1 day</p>
            </div>
        </div>
        
        <div class="card warning">
            <h2>⚠️ Active Alerts</h2>
            <ul>
                <li><strong>ATM003:</strong> Critical cash level - refill required within 24 hours</li>
                <li><strong>ATM002:</strong> Medium risk - monitor closely</li>
            </ul>
        </div>
        
        <div class="card">
            <h2>📊 API Endpoints</h2>
            <ul>
                <li><a href="/api/terminal-status">GET /api/terminal-status</a> - Terminal status summary</li>
                <li><a href="/api/alerts">GET /api/alerts</a> - Active alerts</li>
                <li><a href="/api/predictions">GET /api/predictions</a> - All predictions</li>
                <li><a href="/health">GET /health</a> - Health check</li>
            </ul>
        </div>
    </div>
</body>
</html>
"""

@app.route('/')
def dashboard():
    """Main dashboard"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return render_template_string(HTML_TEMPLATE, timestamp=timestamp)

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'cash-forecasting',
        'timestamp': datetime.datetime.now().isoformat(),
        'version': '1.0.0'
    })

@app.route('/api/terminal-status')
def terminal_status():
    """Get terminal status summary"""
    return jsonify({
        'terminals': [
            {
                'id': 'ATM001',
                'cash_level': 85,
                'risk_level': 'LOW',
                'predicted_depletion_days': 7,
                'last_refill': '2025-01-20',
                'location': 'Main Branch'
            },
            {
                'id': 'ATM002', 
                'cash_level': 45,
                'risk_level': 'MEDIUM',
                'predicted_depletion_days': 3,
                'last_refill': '2025-01-18',
                'location': 'Shopping Mall'
            },
            {
                'id': 'ATM003',
                'cash_level': 15,
                'risk_level': 'HIGH',
                'predicted_depletion_days': 1,
                'last_refill': '2025-01-15',
                'location': 'Airport Terminal'
            }
        ],
        'summary': {
            'total_terminals': 3,
            'healthy': 1,
            'warning': 1,
            'critical': 1
        },
        'timestamp': datetime.datetime.now().isoformat()
    })

@app.route('/api/alerts')
def alerts():
    """Get active alerts"""
    return jsonify({
        'alerts': [
            {
                'terminal_id': 'ATM003',
                'level': 'CRITICAL',
                'message': 'Cash level critically low - refill required within 24 hours',
                'created_at': '2025-01-27T09:00:00Z',
                'priority': 1
            },
            {
                'terminal_id': 'ATM002',
                'level': 'WARNING',
                'message': 'Cash level medium risk - monitor closely',
                'created_at': '2025-01-27T08:30:00Z',
                'priority': 2
            }
        ],
        'total_alerts': 2,
        'timestamp': datetime.datetime.now().isoformat()
    })

@app.route('/api/predictions')
def predictions():
    """Get all predictions"""
    return jsonify({
        'predictions': [
            {
                'terminal_id': 'ATM001',
                'predicted_depletion_date': '2025-02-03',
                'confidence': 0.89,
                'factors': ['low_traffic', 'recent_refill']
            },
            {
                'terminal_id': 'ATM002',
                'predicted_depletion_date': '2025-01-30',
                'confidence': 0.92,
                'factors': ['medium_traffic', 'weekend_pattern']
            },
            {
                'terminal_id': 'ATM003',
                'predicted_depletion_date': '2025-01-28',
                'confidence': 0.95,
                'factors': ['high_traffic', 'airport_location']
            }
        ],
        'model_info': {
            'algorithm': 'Random Forest + LSTM Ensemble',
            'last_trained': '2025-01-27T06:00:00Z',
            'accuracy': 0.91
        },
        'timestamp': datetime.datetime.now().isoformat()
    })

if __name__ == '__main__':
    logger.info("🚀 Starting Cash Forecasting Test Application")
    logger.info("📊 Dashboard available at: http://localhost:5001/")
    logger.info("🔍 Health check at: http://localhost:5001/health")
    logger.info("📡 API endpoints at: http://localhost:5001/api/")
    
    app.run(host='0.0.0.0', port=5001, debug=True)
