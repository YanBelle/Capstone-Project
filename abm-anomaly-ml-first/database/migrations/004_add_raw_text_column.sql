-- Add raw_text and cleaned_text columns to ml_sessions table
-- Migration: 004_add_raw_text_column.sql

ALTER TABLE ml_sessions ADD COLUMN IF NOT EXISTS raw_text TEXT;
ALTER TABLE ml_sessions ADD COLUMN IF NOT EXISTS cleaned_text TEXT;
ALTER TABLE ml_sessions ADD COLUMN IF NOT EXISTS processed_events JSONB;

-- Add indexes for better performance
CREATE INDEX IF NOT EXISTS idx_ml_sessions_session_id ON ml_sessions(session_id);
CREATE INDEX IF NOT EXISTS idx_ml_sessions_timestamp ON ml_sessions(timestamp);
CREATE INDEX IF NOT EXISTS idx_ml_sessions_raw_text_search ON ml_sessions USING gin(to_tsvector('english', raw_text));
CREATE INDEX IF NOT EXISTS idx_ml_sessions_cleaned_text_search ON ml_sessions USING gin(to_tsvector('english', cleaned_text));

-- Add comments
COMMENT ON COLUMN ml_sessions.raw_text IS 'Original raw EJ log content as received';
COMMENT ON COLUMN ml_sessions.cleaned_text IS 'Cleaned and preprocessed EJ log content for analysis';
COMMENT ON COLUMN ml_sessions.processed_events IS 'Processed and parsed events from the session';
