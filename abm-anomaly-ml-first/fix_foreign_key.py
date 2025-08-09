#!/usr/bin/env python3
"""
Fix foreign key constraint issue by clearing data in proper order
"""

import asyncio
import asyncpg
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def fix_foreign_key_constraint():
    """Fix the foreign key constraint issue by clearing data properly"""
    try:
        # Connect to database
        conn = await asyncpg.connect(
            host=os.getenv('DB_HOST', 'postgres'),
            port=5432,
            database=os.getenv('DB_NAME', 'abm_anomaly'),
            user=os.getenv('DB_USER', 'postgres'),
            password=os.getenv('DB_PASSWORD', 'postgres123')
        )
        
        logger.info("Connected to database successfully")
        
        # METHOD 1: Clear in dependency order
        logger.info("Attempting Method 1: Clearing in dependency order")
        try:
            async with conn.transaction():
                deletion_order = [
                    'ml_anomalies',         # Child table referencing ml_sessions
                    'expert_feedback',      # Child table referencing ml_sessions 
                    'labeled_anomalies',    # Child table referencing ml_sessions
                    'anomaly_detections',   # Independent table
                    'ml_summaries',         # Independent table
                    'ml_sessions'           # Parent table
                ]
                
                for table_name in deletion_order:
                    result = await conn.execute(f'DELETE FROM {table_name}')
                    logger.info(f'✅ Cleared {table_name}: {result}')
            
            logger.info("✅ Method 1 succeeded: Dependency order clearing")
            return True
            
        except Exception as method1_error:
            logger.warning(f"Method 1 failed: {method1_error}")
        
        # METHOD 2: TRUNCATE CASCADE
        logger.info("Attempting Method 2: TRUNCATE CASCADE")
        try:
            # TRUNCATE with CASCADE automatically handles foreign key constraints
            await conn.execute('TRUNCATE TABLE ml_sessions RESTART IDENTITY CASCADE')
            logger.info("✅ Method 2 succeeded: TRUNCATE CASCADE")
            return True
            
        except Exception as method2_error:
            logger.warning(f"Method 2 failed: {method2_error}")
        
        # METHOD 3: Drop constraints, delete, recreate
        logger.info("Attempting Method 3: Temporary constraint removal")
        try:
            # Drop foreign key constraint temporarily
            await conn.execute("ALTER TABLE ml_anomalies DROP CONSTRAINT IF EXISTS ml_anomalies_session_id_fkey")
            logger.info("Dropped foreign key constraint")
            
            # Clear all tables
            all_tables = ['ml_sessions', 'ml_anomalies', 'expert_feedback', 
                         'labeled_anomalies', 'anomaly_detections', 'ml_summaries']
            
            for table in all_tables:
                result = await conn.execute(f'DELETE FROM {table}')
                logger.info(f'Cleared {table}: {result}')
            
            # Recreate foreign key constraint
            await conn.execute("""
                ALTER TABLE ml_anomalies 
                ADD CONSTRAINT ml_anomalies_session_id_fkey 
                FOREIGN KEY (session_id) REFERENCES ml_sessions(session_id)
            """)
            logger.info("Recreated foreign key constraint")
            logger.info("✅ Method 3 succeeded: Temporary constraint removal")
            return True
            
        except Exception as method3_error:
            logger.error(f"Method 3 failed: {method3_error}")
            return False
        
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return False
    
    finally:
        if 'conn' in locals():
            await conn.close()
            logger.info("Database connection closed")

if __name__ == "__main__":
    print("🔥 Foreign Key Constraint Fix")
    print("============================")
    
    result = asyncio.run(fix_foreign_key_constraint())
    
    if result:
        print("✅ SUCCESS: Foreign key constraint issue has been resolved!")
        print("✅ All data has been cleared from the database.")
    else:
        print("❌ FAILED: Could not resolve foreign key constraint issue.")
        print("❌ Manual intervention may be required.")
