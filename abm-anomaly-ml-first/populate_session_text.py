#!/usr/bin/env python3
"""
Script to populate missing raw_text and cleaned_text for sessions in the database
using available EJ files in the processed directory.
"""

import asyncio
import asyncpg
import os
import sys
import logging
import re
from pathlib import Path

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'abm_ml_db',
    'user': 'abm_user',
    'password': 'secure_ml_password123'
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def get_db_connection():
    """Get database connection"""
    return await asyncpg.connect(**DB_CONFIG)

def clean_ej_text(raw_text):
    """Clean EJ text by removing escape sequences and normalizing"""
    if not raw_text:
        return ""
    
    # Remove ANSI escape sequences
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    cleaned = ansi_escape.sub('', raw_text)
    
    # Replace \r\n with \n and normalize line endings
    cleaned = cleaned.replace('\r\n', '\n').replace('\r', '\n')
    
    # Remove excessive whitespace but preserve structure
    lines = cleaned.split('\n')
    cleaned_lines = []
    for line in lines:
        # Remove trailing spaces but keep line structure
        cleaned_line = line.rstrip()
        cleaned_lines.append(cleaned_line)
    
    return '\n'.join(cleaned_lines)

async def update_session_text(conn, session_id, raw_text):
    """Update a session with raw and cleaned text"""
    try:
        cleaned_text = clean_ej_text(raw_text)
        
        await conn.execute("""
            UPDATE ml_sessions 
            SET raw_text = $1, cleaned_text = $2, updated_at = NOW()
            WHERE session_id = $3
        """, raw_text, cleaned_text, session_id)
        
        logger.info(f"Updated session {session_id} with text data (raw: {len(raw_text)} chars, cleaned: {len(cleaned_text)} chars)")
        return True
        
    except Exception as e:
        logger.error(f"Error updating session {session_id}: {str(e)}")
        return False

def extract_sessions_from_ej_file(file_path):
    """Extract session data from an EJ file"""
    sessions = {}
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Look for session patterns in the content
        # Session IDs typically follow patterns like ABM25_20250613_SESSION_1_b6e09174_20250806_201225
        session_pattern = r'(ABM\d+_\d{8}_SESSION_\d+_[a-f0-9]{8}_\d{14})'
        
        # Split content into potential sessions
        lines = content.split('\n')
        current_session_id = None
        current_session_content = []
        
        for line in lines:
            # Check if line contains a session ID
            session_matches = re.findall(session_pattern, line)
            
            if session_matches:
                # Save previous session if exists
                if current_session_id and current_session_content:
                    sessions[current_session_id] = '\n'.join(current_session_content)
                
                # Start new session
                current_session_id = session_matches[0]
                current_session_content = [line]
            elif current_session_id:
                # Add line to current session
                current_session_content.append(line)
        
        # Save final session
        if current_session_id and current_session_content:
            sessions[current_session_id] = '\n'.join(current_session_content)
            
        logger.info(f"Extracted {len(sessions)} sessions from {file_path}")
        return sessions
        
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {str(e)}")
        return {}

async def main():
    """Main function to populate session text data"""
    try:
        # Connect to database
        conn = await get_db_connection()
        logger.info("Connected to database")
        
        # Get sessions without text data
        sessions_without_text = await conn.fetch("""
            SELECT session_id, created_at 
            FROM ml_sessions 
            WHERE raw_text IS NULL 
            ORDER BY created_at DESC
            LIMIT 1000
        """)
        
        logger.info(f"Found {len(sessions_without_text)} sessions without text data")
        
        if not sessions_without_text:
            logger.info("No sessions need text data updates")
            await conn.close()
            return
        
        # Get available EJ files (using docker path structure)
        ej_files = [
            "/app/input/processed/ABM25EJ_20250613_20250613.txt",
            "/app/input/processed/ABM163EJ_20240501_20240531.txt", 
            "/app/input/processed/ABM163EJ_20250101_20250626.txt",
            "/app/input/processed/ABM175EJ_20250624_20250624.txt",
            "/app/input/processed/ABM357EJ_20250101_20250430.txt",
            "/app/input/processed/ABM357EJ_20250101_20250430_new.txt"
        ]
        
        # Check which files exist 
        available_files = []
        for file_path in ej_files:
            if os.path.exists(file_path):
                available_files.append(file_path)
                logger.info(f"Found EJ file: {file_path}")
        
        if not available_files:
            logger.error("No EJ files found in docker container. Checking mounted volumes...")
            logger.info(f"Checked paths: {ej_files}")
            await conn.close()
            return
        
        # Process each EJ file and extract sessions
        all_extracted_sessions = {}
        for file_path in available_files:
            extracted_sessions = extract_sessions_from_ej_file(file_path)
            all_extracted_sessions.update(extracted_sessions)
        
        logger.info(f"Total extracted sessions from all files: {len(all_extracted_sessions)}")
        
        # Match sessions from database with extracted sessions
        updates_made = 0
        for session_row in sessions_without_text:
            session_id = session_row['session_id']
            
            if session_id in all_extracted_sessions:
                # Update session with text data
                success = await update_session_text(
                    conn, 
                    session_id, 
                    all_extracted_sessions[session_id]
                )
                if success:
                    updates_made += 1
            else:
                # Try to find session by partial match (in case of ID variations)
                found_match = False
                for extracted_id, extracted_text in all_extracted_sessions.items():
                    # Match by the base part (before timestamp)
                    base_session_id = '_'.join(session_id.split('_')[:4])  # ABM25_20250613_SESSION_1
                    base_extracted_id = '_'.join(extracted_id.split('_')[:4])
                    
                    if base_session_id == base_extracted_id:
                        success = await update_session_text(conn, session_id, extracted_text)
                        if success:
                            updates_made += 1
                            found_match = True
                            break
                
                if not found_match:
                    logger.debug(f"No matching EJ content found for session {session_id}")
        
        logger.info(f"Successfully updated {updates_made} sessions with text data")
        
        # Close database connection
        await conn.close()
        logger.info("Database connection closed")
        
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
