-- Migration to ensure terminal_id column exists and is properly indexed
-- This migration ensures the terminal_id extraction and storage works properly

-- Ensure terminal_id column exists in ml_sessions table (already exists from previous migration)
ALTER TABLE ml_sessions 
ADD COLUMN IF NOT EXISTS terminal_id VARCHAR(20);

-- Ensure index exists for terminal_id for better query performance
CREATE INDEX IF NOT EXISTS idx_ml_sessions_terminal_id ON ml_sessions(terminal_id);

-- Add terminal_id to anomaly_sessions table if it exists
ALTER TABLE anomaly_sessions 
ADD COLUMN IF NOT EXISTS terminal_id VARCHAR(20);

CREATE INDEX IF NOT EXISTS idx_anomaly_sessions_terminal_id ON anomaly_sessions(terminal_id);

-- Update existing sessions to extract terminal_id from session_id if possible
-- This handles cases where the session_id contains the terminal ID pattern
UPDATE ml_sessions 
SET terminal_id = 
    CASE 
        -- Extract terminal ID from session_id patterns like ABM416_20250101_SESSION_*
        WHEN session_id ~ '^ABM(\d+)_' THEN 
            substring(session_id from '^ABM(\d+)_')
        -- Extract from other patterns that might contain ABM followed by numbers
        WHEN session_id ~ 'ABM(\d+)' THEN 
            substring(session_id from 'ABM(\d+)')
        ELSE NULL
    END
WHERE terminal_id IS NULL 
AND session_id IS NOT NULL;

-- Create a view for terminal statistics with terminal_id
CREATE OR REPLACE VIEW terminal_statistics AS
SELECT 
    terminal_id,
    COUNT(*) as total_sessions,
    COUNT(CASE WHEN is_anomaly THEN 1 END) as anomaly_sessions,
    ROUND(
        CASE 
            WHEN COUNT(*) > 0 THEN 
                (COUNT(CASE WHEN is_anomaly THEN 1 END)::decimal / COUNT(*)) * 100 
            ELSE 0 
        END, 2
    ) as anomaly_rate_percent,
    AVG(CASE WHEN is_anomaly THEN anomaly_score ELSE 0 END) as avg_anomaly_score,
    MAX(timestamp) as last_session_time,
    MIN(timestamp) as first_session_time
FROM ml_sessions 
WHERE terminal_id IS NOT NULL
GROUP BY terminal_id
ORDER BY anomaly_rate_percent DESC, total_sessions DESC;

-- Create a function to get sessions by terminal ID
CREATE OR REPLACE FUNCTION get_sessions_by_terminal(terminal_id_param VARCHAR(20), limit_count integer DEFAULT 100)
RETURNS TABLE (
    session_id VARCHAR(50),
    timestamp TIMESTAMP,
    is_anomaly BOOLEAN,
    anomaly_score DECIMAL(5,4),
    anomaly_type VARCHAR(50)
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        ms.session_id,
        ms.timestamp,
        ms.is_anomaly,
        ms.anomaly_score,
        ms.anomaly_type
    FROM ml_sessions ms
    WHERE ms.terminal_id = terminal_id_param
    ORDER BY ms.timestamp DESC
    LIMIT limit_count;
END;
$$ LANGUAGE plpgsql;

-- Add a comment to the terminal_id column for documentation
COMMENT ON COLUMN ml_sessions.terminal_id IS 'ABM Terminal ID extracted from filename (e.g., 416 from ABM416EJ_20250101_20250630.txt)';

-- Create indexes for better performance on queries filtering by terminal_id and anomaly status
CREATE INDEX IF NOT EXISTS idx_ml_sessions_terminal_anomaly ON ml_sessions(terminal_id, is_anomaly);
CREATE INDEX IF NOT EXISTS idx_ml_sessions_terminal_timestamp ON ml_sessions(terminal_id, timestamp);
