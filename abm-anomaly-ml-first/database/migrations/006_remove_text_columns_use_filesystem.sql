-- Remove raw_text and cleaned_text columns from ml_sessions table
-- Migration: 006_remove_text_columns_use_filesystem.sql
-- These texts are now stored on the file system for better performance

-- Drop text search indexes first
DROP INDEX IF EXISTS idx_ml_sessions_raw_text_search;
DROP INDEX IF EXISTS idx_ml_sessions_cleaned_text_search;

-- Remove the text columns since we're now using file system storage
-- Note: In production, you might want to backup this data first
ALTER TABLE ml_sessions DROP COLUMN IF EXISTS raw_text;
ALTER TABLE ml_sessions DROP COLUMN IF EXISTS cleaned_text;

-- Add comment to document the change
COMMENT ON TABLE ml_sessions IS 'ML Sessions table - raw_text and cleaned_text now stored on file system under /app/data/sessions/{session_id[:2]}/{session_id}_raw.txt and {session_id}_cleaned.txt for improved performance';

-- Add a flag to indicate file system storage (optional, for tracking)
ALTER TABLE ml_sessions ADD COLUMN IF NOT EXISTS uses_filesystem_storage BOOLEAN DEFAULT TRUE;
COMMENT ON COLUMN ml_sessions.uses_filesystem_storage IS 'Indicates whether session texts are stored on file system (TRUE) or in database (FALSE)';

-- Update existing records to indicate they use filesystem storage
UPDATE ml_sessions SET uses_filesystem_storage = TRUE WHERE uses_filesystem_storage IS NULL;
