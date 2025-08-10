"""
EJ Processing API Endpoints
Additional FastAPI endpoints for EJ processing and cleaning
"""

from fastapi import HTTPException
from typing import Dict, Any
import json
import logging

logger = logging.getLogger(__name__)

# EJ Processing and Cleaning API Endpoints
EJ_ENDPOINTS_CODE = '''
@app.post("/api/v1/ej/process-session")
async def process_ej_session_endpoint(request: Dict[str, Any]):
    """Process and store a single EJ session"""
    try:
        session_id = request.get('session_id')
        raw_content = request.get('raw_content')
        
        if not session_id or not raw_content:
            raise HTTPException(status_code=400, detail="session_id and raw_content are required")
        
        result = await process_and_store_ej_session(session_id, raw_content)
        return result
        
    except Exception as e:
        logger.error(f"Error processing EJ session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/ej/session/{session_id}/raw")
async def get_session_raw_text_endpoint(session_id: str):
    """Get raw EJ text for a session"""
    try:
        raw_text = await get_session_raw_text(session_id)
        return {
            'status': 'success',
            'session_id': session_id,
            'raw_text': raw_text,
            'available': raw_text != "Raw text not available"
        }
    except Exception as e:
        logger.error(f"Error retrieving raw text: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/ej/session/{session_id}/cleaned")
async def get_session_cleaned_text_endpoint(session_id: str):
    """Get cleaned EJ text for a session"""
    try:
        cleaned_text = await get_session_cleaned_text(session_id)
        return {
            'status': 'success',
            'session_id': session_id,
            'cleaned_text': cleaned_text,
            'available': cleaned_text != "Cleaned text not available"
        }
    except Exception as e:
        logger.error(f"Error retrieving cleaned text: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/ej/session/{session_id}/events")
async def get_session_events_endpoint(session_id: str):
    """Get structured events for a session"""
    try:
        events = await get_session_events(session_id)
        return {
            'status': 'success',
            'session_id': session_id,
            'events': events,
            'event_count': len(events)
        }
    except Exception as e:
        logger.error(f"Error retrieving session events: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/ej/clean")
async def clean_ej_text_endpoint(request: Dict[str, Any]):
    """Clean EJ text without storing"""
    try:
        if not EJ_CLEANER_AVAILABLE:
            raise HTTPException(status_code=503, detail="EJ Cleaner not available")
        
        raw_text = request.get('raw_text')
        if not raw_text:
            raise HTTPException(status_code=400, detail="raw_text is required")
        
        result = ej_cleaner.clean_ej_log(raw_text)
        
        return {
            'status': 'success',
            'original_length': len(raw_text),
            'cleaned_text': result['cleaned_text'],
            'normalized_tokens': result['normalized_tokens'],
            'structured_events': json.loads(result['structured_events']),
            'cleaning_stats': json.loads(result['cleaning_stats'])
        }
        
    except Exception as e:
        logger.error(f"Error cleaning EJ text: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/ej/sessions/summary")
async def get_ej_sessions_summary():
    """Get summary of stored EJ sessions"""
    try:
        async with get_db_connection() as conn:
            # Count total sessions
            total_sessions = await conn.fetchval("SELECT COUNT(*) FROM ml_sessions")
            
            # Count sessions with raw text
            sessions_with_raw = await conn.fetchval(
                "SELECT COUNT(*) FROM ml_sessions WHERE raw_text IS NOT NULL AND raw_text != \\'\\'"
            )
            
            # Count sessions with cleaned text
            sessions_with_cleaned = await conn.fetchval(
                "SELECT COUNT(*) FROM ml_sessions WHERE cleaned_text IS NOT NULL AND cleaned_text != \\'\\'"
            )
            
            # Count sessions with events
            sessions_with_events = await conn.fetchval(
                "SELECT COUNT(*) FROM ml_sessions WHERE processed_events IS NOT NULL"
            )
            
            # Get recent sessions
            recent_sessions = await conn.fetch("""
                SELECT session_id, 
                       LENGTH(raw_text) as raw_length,
                       LENGTH(cleaned_text) as cleaned_length,
                       created_at
                FROM ml_sessions 
                WHERE raw_text IS NOT NULL 
                ORDER BY created_at DESC 
                LIMIT 10
            """)
            
            recent_list = []
            for session in recent_sessions:
                recent_list.append({
                    'session_id': session['session_id'],
                    'raw_length': session['raw_length'] or 0,
                    'cleaned_length': session['cleaned_length'] or 0,
                    'created_at': session['created_at'].isoformat() if session['created_at'] else None
                })
        
        return {
            'status': 'success',
            'summary': {
                'total_sessions': total_sessions,
                'sessions_with_raw_text': sessions_with_raw,
                'sessions_with_cleaned_text': sessions_with_cleaned,
                'sessions_with_events': sessions_with_events,
                'ej_cleaner_available': EJ_CLEANER_AVAILABLE
            },
            'recent_sessions': recent_list
        }
        
    except Exception as e:
        logger.error(f"Error getting EJ sessions summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/ej/batch-process")
async def batch_process_ej_endpoint():
    """Batch process EJ files from input directory"""
    try:
        result = await batch_process_ej_files()
        return result
        
    except Exception as e:
        logger.error(f"Error in batch EJ processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))
'''

def add_ej_endpoints_to_app(app):
    """Add EJ endpoints to the FastAPI app"""
    # This is a placeholder function
    # The actual endpoints should be copy-pasted into main.py
    pass
