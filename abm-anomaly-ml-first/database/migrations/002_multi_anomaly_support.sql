-- Multi-Anomaly Support Database Schema Update
-- This script adds support for multiple anomalies per session

-- Add new columns to ml_sessions table for multi-anomaly support
ALTER TABLE ml_sessions 
ADD COLUMN IF NOT EXISTS anomaly_count INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS anomaly_types TEXT, -- JSON array of anomaly types
ADD COLUMN IF NOT EXISTS max_severity VARCHAR(20) DEFAULT 'normal',
ADD COLUMN IF NOT EXISTS overall_anomaly_score FLOAT DEFAULT 0.0,
ADD COLUMN IF NOT EXISTS critical_anomalies_count INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS high_severity_anomalies_count INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS detection_methods TEXT, -- JSON array of detection methods
ADD COLUMN IF NOT EXISTS anomalies_detail TEXT; -- JSON array of detailed anomaly information

-- Update existing records to have default multi-anomaly values
UPDATE ml_sessions 
SET 
    anomaly_count = CASE WHEN is_anomaly THEN 1 ELSE 0 END,
    anomaly_types = CASE 
        WHEN is_anomaly AND anomaly_type IS NOT NULL THEN '["' || anomaly_type || '"]'
        ELSE '[]'
    END,
    max_severity = CASE 
        WHEN is_anomaly THEN 'medium'
        ELSE 'normal'
    END,
    overall_anomaly_score = COALESCE(anomaly_score, 0.0),
    critical_anomalies_count = 0,
    high_severity_anomalies_count = CASE WHEN is_anomaly THEN 1 ELSE 0 END,
    detection_methods = CASE 
        WHEN is_anomaly THEN '["isolation_forest"]'
        ELSE '[]'
    END,
    anomalies_detail = CASE 
        WHEN is_anomaly AND anomaly_type IS NOT NULL THEN 
            '[{"type": "' || anomaly_type || '", "confidence": ' || COALESCE(anomaly_score, 0.5) || ', "method": "isolation_forest", "severity": "medium", "description": "Legacy anomaly detection"}]'
        ELSE '[]'
    END
WHERE anomaly_count IS NULL;

-- Create indexes for better performance on new columns
CREATE INDEX IF NOT EXISTS idx_ml_sessions_anomaly_count ON ml_sessions(anomaly_count);
CREATE INDEX IF NOT EXISTS idx_ml_sessions_max_severity ON ml_sessions(max_severity);
CREATE INDEX IF NOT EXISTS idx_ml_sessions_overall_score ON ml_sessions(overall_anomaly_score);

-- Create a view for easy multi-anomaly analysis
CREATE OR REPLACE VIEW multi_anomaly_summary AS
SELECT 
    COUNT(*) as total_sessions,
    COUNT(CASE WHEN is_anomaly THEN 1 END) as anomaly_sessions,
    COUNT(CASE WHEN anomaly_count > 1 THEN 1 END) as multi_anomaly_sessions,
    AVG(CASE WHEN is_anomaly THEN anomaly_count ELSE 0 END) as avg_anomalies_per_session,
    COUNT(CASE WHEN max_severity = 'critical' THEN 1 END) as critical_sessions,
    COUNT(CASE WHEN max_severity = 'high' THEN 1 END) as high_severity_sessions,
    COUNT(CASE WHEN max_severity = 'medium' THEN 1 END) as medium_severity_sessions,
    COUNT(CASE WHEN max_severity = 'low' THEN 1 END) as low_severity_sessions
FROM ml_sessions;

-- Create a function to parse JSON arrays for anomaly types
CREATE OR REPLACE FUNCTION parse_anomaly_types(anomaly_types_json TEXT)
RETURNS TEXT[] AS $$
BEGIN
    IF anomaly_types_json IS NULL OR anomaly_types_json = '' THEN
        RETURN ARRAY[]::TEXT[];
    END IF;
    
    BEGIN
        RETURN ARRAY(SELECT json_array_elements_text(anomaly_types_json::json));
    EXCEPTION WHEN OTHERS THEN
        -- Fallback for non-JSON strings
        RETURN ARRAY[anomaly_types_json];
    END;
END;
$$ LANGUAGE plpgsql;

-- Create indexes for JSON columns to improve query performance
CREATE INDEX IF NOT EXISTS idx_ml_sessions_anomaly_types_gin ON ml_sessions USING gin((anomaly_types::jsonb));
CREATE INDEX IF NOT EXISTS idx_ml_sessions_detection_methods_gin ON ml_sessions USING gin((detection_methods::jsonb));
