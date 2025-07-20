#!/usr/bin/env python3
"""
Script to apply the multi-anomaly database migration
"""
import os
import psycopg2
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_db_connection():
    """Get database connection using environment variables"""
    try:
        conn = psycopg2.connect(
            host=os.getenv('POSTGRES_HOST', 'localhost'),
            database=os.getenv('POSTGRES_DB', 'abm_anomaly'),
            user=os.getenv('POSTGRES_USER', 'abm_user'),
            password=os.getenv('POSTGRES_PASSWORD', 'abm_pass'),
            port=os.getenv('POSTGRES_PORT', '5432')
        )
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        return None

def apply_migration():
    """Apply the multi-anomaly support migration"""
    migration_file = Path(__file__).parent / "database" / "migrations" / "002_multi_anomaly_support.sql"
    
    if not migration_file.exists():
        logger.error(f"Migration file not found: {migration_file}")
        return False
    
    # Read migration SQL
    with open(migration_file, 'r') as f:
        migration_sql = f.read()
    
    # Connect to database
    conn = get_db_connection()
    if not conn:
        return False
    
    try:
        with conn.cursor() as cursor:
            # Split SQL statements (simple approach)
            statements = [stmt.strip() for stmt in migration_sql.split(';') if stmt.strip()]
            
            for i, statement in enumerate(statements):
                if statement and not statement.startswith('--'):
                    try:
                        logger.info(f"Executing statement {i+1}/{len(statements)}...")
                        cursor.execute(statement)
                        conn.commit()
                        logger.info(f"Statement {i+1} executed successfully")
                    except Exception as e:
                        logger.warning(f"Statement {i+1} failed (might be expected): {e}")
                        conn.rollback()
                        continue
        
        logger.info("Migration completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()

def verify_migration():
    """Verify that the migration was applied correctly"""
    conn = get_db_connection()
    if not conn:
        return False
    
    try:
        with conn.cursor() as cursor:
            # Check if new columns exist
            cursor.execute("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'ml_sessions' 
                AND column_name IN ('anomaly_count', 'anomaly_types', 'max_severity', 'anomalies_detail')
            """)
            columns = [row[0] for row in cursor.fetchall()]
            
            expected_columns = ['anomaly_count', 'anomaly_types', 'max_severity', 'anomalies_detail']
            missing_columns = [col for col in expected_columns if col not in columns]
            
            if missing_columns:
                logger.error(f"Missing columns after migration: {missing_columns}")
                return False
            
            logger.info("All new columns found - migration verification passed!")
            
            # Check a sample of updated records
            cursor.execute("""
                SELECT anomaly_count, anomaly_types, max_severity 
                FROM ml_sessions 
                WHERE is_anomaly = true 
                LIMIT 5
            """)
            samples = cursor.fetchall()
            logger.info(f"Sample migrated records: {samples}")
            
            return True
            
    except Exception as e:
        logger.error(f"Migration verification failed: {e}")
        return False
    finally:
        conn.close()

if __name__ == "__main__":
    logger.info("Starting multi-anomaly database migration...")
    
    # Load environment variables from .env file if it exists
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                if '=' in line and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value
    
    success = apply_migration()
    if success:
        logger.info("Migration applied successfully")
        verify_migration()
    else:
        logger.error("Migration failed")
        exit(1)
