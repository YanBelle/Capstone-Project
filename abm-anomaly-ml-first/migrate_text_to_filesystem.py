#!/usr/bin/env python3
"""
Migration script to move existing raw_text and cleaned_text from database to file system
Script: migrate_text_to_filesystem.py
"""

import os
import sys
import logging
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

def migrate_text_to_filesystem():
    """Migrate existing text data from database to file system"""
    
    # Database connection
    db_engine = create_engine(
        f"postgresql://{os.getenv('POSTGRES_USER', 'abm_user')}:{os.getenv('POSTGRES_PASSWORD', 'anomaly_detection_123')}"
        f"@{os.getenv('POSTGRES_HOST', 'localhost')}:5432/{os.getenv('POSTGRES_DB', 'abm_anomaly_detection')}"
    )
    
    # Create data directory structure
    base_dir = "/app/data/sessions"
    os.makedirs(base_dir, exist_ok=True)
    for i in range(100):
        subdir = f"{i:02d}"
        os.makedirs(f"{base_dir}/{subdir}", exist_ok=True)
    
    logger.info("Created file system directory structure")
    
    try:
        with db_engine.connect() as conn:
            # Check if text columns exist
            result = conn.execute(text("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'ml_sessions' 
                AND column_name IN ('raw_text', 'cleaned_text')
            """))
            
            columns = [row[0] for row in result.fetchall()]
            
            if not columns:
                logger.info("No text columns found in database. Migration not needed.")
                return
            
            logger.info(f"Found text columns: {columns}")
            
            # Get all sessions with text data
            query = """
                SELECT session_id, raw_text, cleaned_text 
                FROM ml_sessions 
                WHERE raw_text IS NOT NULL OR cleaned_text IS NOT NULL
            """
            
            result = conn.execute(text(query))
            sessions = result.fetchall()
            
            logger.info(f"Found {len(sessions)} sessions with text data to migrate")
            
            migrated_count = 0
            error_count = 0
            
            for session in sessions:
                session_id = session[0]
                raw_text = session[1] if len(session) > 1 else None
                cleaned_text = session[2] if len(session) > 2 else None
                
                try:
                    # Create file paths
                    output_dir = f"{base_dir}/{session_id[:2]}"
                    
                    # Store raw text if available
                    if raw_text:
                        raw_file_path = f"{output_dir}/{session_id}_raw.txt"
                        with open(raw_file_path, 'w', encoding='utf-8') as f:
                            f.write(raw_text)
                        logger.debug(f"Migrated raw text for session {session_id}")
                    
                    # Store cleaned text if available
                    if cleaned_text:
                        cleaned_file_path = f"{output_dir}/{session_id}_cleaned.txt"
                        with open(cleaned_file_path, 'w', encoding='utf-8') as f:
                            f.write(cleaned_text)
                        logger.debug(f"Migrated cleaned text for session {session_id}")
                    
                    migrated_count += 1
                    
                    if migrated_count % 100 == 0:
                        logger.info(f"Migrated {migrated_count} sessions...")
                
                except Exception as e:
                    logger.error(f"Error migrating session {session_id}: {e}")
                    error_count += 1
            
            logger.info(f"Migration complete: {migrated_count} sessions migrated, {error_count} errors")
            
            if error_count == 0:
                logger.info("All sessions migrated successfully. You can now run the database migration to remove text columns.")
            else:
                logger.warning(f"Migration completed with {error_count} errors. Review logs before proceeding with database schema changes.")
    
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    logger.info("Starting text data migration from database to file system...")
    success = migrate_text_to_filesystem()
    
    if success:
        logger.info("Migration completed successfully!")
        print("\nNext steps:")
        print("1. Verify the migrated files in /app/data/sessions/")
        print("2. Run the database migration: ./apply_filesystem_migration.sh")
        print("3. Test the application to ensure text retrieval works correctly")
    else:
        logger.error("Migration failed!")
        sys.exit(1)
