#!/usr/bin/env python
"""
Cash Forecasting Web Application
================================

Production Flask application for ATM cash forecasting with:
- Random Forest + LSTM ensemble models
- Real-time dashboard
- API endpoints for predictions
- Integration with existing ABM system
"""

import os
import json
import logging
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify, request, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import schedule
import threading
import time

# Import our forecasting system
from cash_forecasting_system import CashForecastingSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/cash_forecasting.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
class Config:
    # Database configuration
    DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://user:password@localhost:5432/ejdb')
    
    # Model configuration
    MODEL_RETRAIN_INTERVAL = int(os.getenv('MODEL_RETRAIN_HOURS', '24'))  # Hours
    FORECAST_HORIZONS = [1, 3, 7]  # Days
    
    # Alert thresholds
    HIGH_RISK_THRESHOLD = 2  # Days
    MEDIUM_RISK_THRESHOLD = 5  # Days
    
    # Dashboard refresh
    DASHBOARD_REFRESH_MINUTES = int(os.getenv('DASHBOARD_REFRESH_MINUTES', '15'))

app.config.from_object(Config)

# Global forecasting system instance
forecasting_system = None
last_predictions = {}
model_performance = {}

def initialize_forecasting_system():
    """Initialize the cash forecasting system"""
    global forecasting_system
    
    try:
        logger.info("Initializing cash forecasting system...")
        forecasting_system = CashForecastingSystem(app.config['DATABASE_URL'])
        
        # Load and train initial models
        df = forecasting_system.load_cassette_data()
        if df is not None and len(df) > 0:
            df_features = forecasting_system.prepare_features(df)
            forecasting_system.train_models(df_features)
            logger.info("Cash forecasting system initialized successfully")
        else:
            logger.warning("No cassette data available for training")
            
    except Exception as e:
        logger.error(f"Failed to initialize forecasting system: {e}")
        forecasting_system = None

def update_predictions():
    """Update cash depletion predictions for all terminals"""
    global last_predictions, model_performance
    
    if not forecasting_system or not forecasting_system.models:
        logger.warning("Forecasting system not available for predictions")
        return
    
    try:
        logger.info("Updating cash depletion predictions...")
        predictions = {}
        
        for terminal_id in forecasting_system.models.keys():
            prediction = forecasting_system.predict_cash_depletion(terminal_id)
            if prediction:
                predictions[terminal_id] = prediction
        
        last_predictions = predictions
        model_performance = forecasting_system.performance_metrics
        
        logger.info(f"Updated predictions for {len(predictions)} terminals")
        
    except Exception as e:
        logger.error(f"Failed to update predictions: {e}")

def retrain_models():
    """Retrain forecasting models with latest data"""
    if not forecasting_system:
        logger.warning("Forecasting system not available for retraining")
        return
    
    try:
        logger.info("Retraining forecasting models...")
        
        # Load fresh data
        df = forecasting_system.load_cassette_data()
        if df is not None and len(df) > 0:
            df_features = forecasting_system.prepare_features(df)
            forecasting_system.train_models(df_features)
            
            # Update predictions after retraining
            update_predictions()
            
            logger.info("Model retraining completed successfully")
        else:
            logger.warning("No data available for retraining")
            
    except Exception as e:
        logger.error(f"Failed to retrain models: {e}")

def schedule_tasks():
    """Schedule background tasks"""
    # Schedule model retraining
    schedule.every(app.config['MODEL_RETRAIN_INTERVAL']).hours.do(retrain_models)
    
    # Schedule prediction updates
    schedule.every(app.config['DASHBOARD_REFRESH_MINUTES']).minutes.do(update_predictions)
    
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute

# Routes

@app.route('/')
def dashboard():
    """Main cash forecasting dashboard"""
    return render_template('cash_forecasting_dashboard.html')

@app.route('/health')
def health_check():
    """Health check endpoint"""
    status = {
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'forecasting_system': forecasting_system is not None,
        'models_trained': len(forecasting_system.models) if forecasting_system else 0,
        'last_predictions': len(last_predictions)
    }
    return jsonify(status)

@app.route('/api/predictions')
def get_predictions():
    """Get current cash depletion predictions"""
    return jsonify({
        'predictions': last_predictions,
        'timestamp': datetime.utcnow().isoformat(),
        'status': 'success'
    })

@app.route('/api/predictions/<int:terminal_id>')
def get_terminal_prediction(terminal_id):
    """Get prediction for specific terminal"""
    if str(terminal_id) in last_predictions:
        prediction = last_predictions[str(terminal_id)]
        prediction['timestamp'] = datetime.utcnow().isoformat()
        return jsonify(prediction)
    else:
        return jsonify({'error': 'Terminal not found', 'terminal_id': terminal_id}), 404

@app.route('/api/performance')
def get_model_performance():
    """Get model performance metrics"""
    return jsonify({
        'performance': model_performance,
        'timestamp': datetime.utcnow().isoformat()
    })

@app.route('/api/retrain', methods=['POST'])
def trigger_retrain():
    """Manually trigger model retraining"""
    try:
        # Run retraining in background thread
        thread = threading.Thread(target=retrain_models)
        thread.start()
        
        return jsonify({
            'status': 'success',
            'message': 'Model retraining started',
            'timestamp': datetime.utcnow().isoformat()
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@app.route('/api/alerts')
def get_alerts():
    """Get current risk alerts"""
    alerts = []
    
    for terminal_id, prediction in last_predictions.items():
        days_until_depletion = prediction.get('days_until_depletion', 999)
        
        if days_until_depletion <= app.config['HIGH_RISK_THRESHOLD']:
            alerts.append({
                'terminal_id': terminal_id,
                'risk_level': 'HIGH',
                'days_until_depletion': days_until_depletion,
                'message': f'Terminal {terminal_id} requires immediate refill',
                'priority': 1
            })
        elif days_until_depletion <= app.config['MEDIUM_RISK_THRESHOLD']:
            alerts.append({
                'terminal_id': terminal_id,
                'risk_level': 'MEDIUM',
                'days_until_depletion': days_until_depletion,
                'message': f'Terminal {terminal_id} requires refill within 2-3 days',
                'priority': 2
            })
    
    return jsonify({
        'alerts': sorted(alerts, key=lambda x: x['priority']),
        'total_alerts': len(alerts),
        'timestamp': datetime.utcnow().isoformat()
    })

@app.route('/api/visualizations')
def get_visualizations():
    """Generate and return visualization data"""
    try:
        if not forecasting_system:
            return jsonify({'error': 'Forecasting system not available'}), 503
        
        # Generate performance visualizations
        forecasting_system.visualize_performance()
        forecasting_system.create_forecasting_dashboard()
        
        return jsonify({
            'status': 'success',
            'message': 'Visualizations generated',
            'performance_chart': '/static/images/cash_forecasting_performance.png',
            'dashboard_chart': '/static/images/cash_forecasting_dashboard.png',
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Failed to generate visualizations: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/terminal-status')
def get_terminal_status():
    """Get comprehensive terminal status"""
    status_data = []
    
    for terminal_id, prediction in last_predictions.items():
        days_until_depletion = prediction.get('days_until_depletion', 999)
        
        # Determine risk level
        if days_until_depletion <= app.config['HIGH_RISK_THRESHOLD']:
            risk_level = 'HIGH'
            status_color = 'red'
        elif days_until_depletion <= app.config['MEDIUM_RISK_THRESHOLD']:
            risk_level = 'MEDIUM'
            status_color = 'orange'
        else:
            risk_level = 'LOW'
            status_color = 'green'
        
        # Get model performance for this terminal
        performance = model_performance.get(terminal_id, {})
        
        status_data.append({
            'terminal_id': terminal_id,
            'current_cash': prediction.get('current_cash', 0),
            'days_until_depletion': round(days_until_depletion, 1),
            'predicted_depletion_date': prediction.get('predicted_depletion_date', '').strftime('%Y-%m-%d') if prediction.get('predicted_depletion_date') else '',
            'risk_level': risk_level,
            'status_color': status_color,
            'confidence': prediction.get('confidence', 0),
            'model_accuracy': performance.get('ensemble_r2', 0),
            'last_updated': datetime.utcnow().isoformat()
        })
    
    return jsonify({
        'terminals': status_data,
        'summary': {
            'total_terminals': len(status_data),
            'high_risk': len([t for t in status_data if t['risk_level'] == 'HIGH']),
            'medium_risk': len([t for t in status_data if t['risk_level'] == 'MEDIUM']),
            'low_risk': len([t for t in status_data if t['risk_level'] == 'LOW'])
        },
        'timestamp': datetime.utcnow().isoformat()
    })

@app.route('/static/images/<filename>')
def serve_images(filename):
    """Serve generated visualization images"""
    return send_file(f'/app/static/images/{filename}')

@app.route('/integration')
def integration_info():
    """Provide integration information for main dashboard"""
    return jsonify({
        'service': 'Cash Forecasting',
        'version': '1.0.0',
        'endpoints': {
            'dashboard': '/cash-forecasting/',
            'api_predictions': '/cash-forecasting/api/predictions',
            'api_alerts': '/cash-forecasting/api/alerts',
            'api_status': '/cash-forecasting/api/terminal-status'
        },
        'iframe_url': '/cash-forecasting/',
        'widget_config': {
            'title': 'Cash Forecasting',
            'description': 'ATM cash depletion predictions and alerts',
            'icon': 'money-bill-wave',
            'color': 'success'
        }
    })

# Initialize system on startup
def startup():
    """Initialize system on first request"""
    initialize_forecasting_system()
    update_predictions()
    
    # Start background scheduler
    scheduler_thread = threading.Thread(target=schedule_tasks)
    scheduler_thread.daemon = True
    scheduler_thread.start()
    return scheduler_thread

# Initialize on app startup
with app.app_context():
    background_thread = startup()

if __name__ == '__main__':
    # Initialize system
    initialize_forecasting_system()
    update_predictions()
    
    # Start background scheduler
    scheduler_thread = threading.Thread(target=schedule_tasks)
    scheduler_thread.daemon = True
    scheduler_thread.start()
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)
