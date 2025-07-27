-- Add Terminal/Machine ID and Location support to ml_sessions
-- This extends the multi-anomaly migration to include location information

-- Add machine/location columns to ml_sessions table
ALTER TABLE ml_sessions 
ADD COLUMN IF NOT EXISTS terminal_id VARCHAR(20),
ADD COLUMN IF NOT EXISTS location VARCHAR(100);

-- Create a view to analyze anomalies by terminal/location
CREATE OR REPLACE VIEW terminal_anomaly_summary AS
SELECT 
    terminal_id,
    location,
    COUNT(*) as total_sessions,
    COUNT(CASE WHEN is_anomaly THEN 1 END) as anomaly_sessions,
    SUM(anomaly_count) as total_anomalies,
    AVG(CASE WHEN is_anomaly THEN overall_anomaly_score ELSE 0 END) as avg_anomaly_score,
    COUNT(CASE WHEN max_severity = 'critical' THEN 1 END) as critical_anomalies,
    COUNT(CASE WHEN max_severity = 'high' THEN 1 END) as high_anomalies,
    COUNT(CASE WHEN max_severity = 'medium' THEN 1 END) as medium_anomalies
FROM ml_sessions
WHERE terminal_id IS NOT NULL
GROUP BY terminal_id, location
ORDER BY total_anomalies DESC;

-- Create index for terminal_id for better query performance
CREATE INDEX IF NOT EXISTS idx_ml_sessions_terminal_id ON ml_sessions(terminal_id);
CREATE INDEX IF NOT EXISTS idx_ml_sessions_location ON ml_sessions(location);

-- Update data from transactions table if available (populate terminal_id)
UPDATE ml_sessions ms
SET terminal_id = t.terminal_id
FROM transactions t
WHERE ms.session_id = t.session_id
AND ms.terminal_id IS NULL
AND t.terminal_id IS NOT NULL;

-- A function to get most problematic terminals
CREATE OR REPLACE FUNCTION get_most_problematic_terminals(limit_count integer DEFAULT 5)
RETURNS TABLE (
    terminal_id VARCHAR(20),
    location VARCHAR(100),
    total_anomalies INTEGER,
    critical_count INTEGER,
    high_count INTEGER,
    avg_score FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        tas.terminal_id,
        tas.location,
        tas.total_anomalies,
        tas.critical_anomalies as critical_count,
        tas.high_anomalies as high_count,
        tas.avg_anomaly_score as avg_score
    FROM terminal_anomaly_summary tas
    ORDER BY 
        tas.critical_anomalies DESC,
        tas.high_anomalies DESC,
        tas.total_anomalies DESC
    LIMIT limit_count;
END;
$$ LANGUAGE plpgsql;
