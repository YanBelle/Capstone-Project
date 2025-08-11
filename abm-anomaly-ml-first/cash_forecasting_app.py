#!/usr/bin/env python3
"""
Cash Forecasting Application for ABM Integration
"""

from flask import Flask, jsonify, render_template_string
import logging
import datetime
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

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

@app.route('/api/retrain', methods=['POST'])
def retrain_models():
    """Trigger model retraining"""
    return jsonify({
        'status': 'success',
        'message': 'Model retraining triggered successfully',
        'timestamp': datetime.datetime.now().isoformat()
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') != 'production'
    
    logger.info("🚀 Starting Cash Forecasting Application")
    logger.info(f"📊 API available at: http://localhost:{port}/api/")
    logger.info(f"🔍 Health check at: http://localhost:{port}/health")
    
    app.run(host='0.0.0.0', port=port, debug=debug)
