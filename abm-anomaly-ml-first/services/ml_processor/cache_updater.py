import redis
import json
import os
from datetime import datetime
from sqlalchemy import create_engine, text
from loguru import logger

def update_ml_cache():
    """Update Redis cache with latest ML processing summary"""
    try:
        # Redis connection
        redis_client = redis.Redis(
            host=os.getenv('REDIS_HOST', 'redis'),
            port=6379,
            password=os.getenv('REDIS_PASSWORD'),
            decode_responses=True
        )
        
        # Database connection
        db_engine = create_engine(
            f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}"
            f"@{os.getenv('POSTGRES_HOST', 'postgres')}:5432/{os.getenv('POSTGRES_DB')}"
        )
        
        with db_engine.connect() as conn:
            # Get total sessions
            total_sessions = conn.execute(text("SELECT COUNT(*) FROM ml_sessions")).scalar()
            
            # Get total anomalies
            total_anomalies = conn.execute(text("SELECT COUNT(*) FROM ml_sessions WHERE is_anomaly = true")).scalar()
            
            # Get high risk count
            high_risk_count = conn.execute(text("SELECT COUNT(*) FROM ml_sessions WHERE is_anomaly = true AND anomaly_score > 0.8")).scalar()
            
            # Calculate anomaly rate
            anomaly_rate = (total_anomalies / total_sessions) if total_sessions > 0 else 0.0
            
            summary = {
                'total_transactions': total_sessions,
                'total_anomalies': total_anomalies,
                'anomaly_rate': anomaly_rate,
                'high_risk_count': high_risk_count,
                'last_updated': datetime.now().isoformat(),
                'updated_by': 'ml_processor'
            }
            
            # Update Redis cache (expire after 1 hour)
            redis_client.set('latest_ml_summary', json.dumps(summary), ex=3600)
            
            logger.info(f"ML cache updated: {summary}")
            return summary
            
    except Exception as e:
        logger.error(f"Error updating ML cache: {str(e)}")
        return None

if __name__ == "__main__":
    update_ml_cache()
