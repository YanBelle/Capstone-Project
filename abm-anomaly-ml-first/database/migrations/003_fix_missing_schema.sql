-- Migration to fix missing database schema columns and tables
-- This fixes the errors we're seeing in the logs

-- Add missing last_activity column to ml_sessions
ALTER TABLE ml_sessions ADD COLUMN IF NOT EXISTS last_activity TIMESTAMP;

-- Create anomaly_sessions table (referenced but missing)
CREATE TABLE IF NOT EXISTS anomaly_sessions (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(100) UNIQUE NOT NULL,
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP,
    anomaly_score DECIMAL(5, 4),
    anomaly_type VARCHAR(100),
    detected_patterns JSONB,
    critical_events JSONB,
    is_anomaly BOOLEAN DEFAULT FALSE,
    expert_override_applied BOOLEAN DEFAULT FALSE,
    expert_override_reason TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_anomaly_sessions_anomaly ON anomaly_sessions(is_anomaly);
CREATE INDEX IF NOT EXISTS idx_anomaly_sessions_score ON anomaly_sessions(anomaly_score);
CREATE INDEX IF NOT EXISTS idx_anomaly_sessions_start_time ON anomaly_sessions(start_time);

-- Create expert_feedback table (referenced but missing)
CREATE TABLE IF NOT EXISTS expert_feedback (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(100),
    feedback_type VARCHAR(50),
    expert_label VARCHAR(100),
    expert_confidence DECIMAL(3, 2),
    expert_explanation TEXT,
    original_ml_prediction VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_expert_feedback_session ON expert_feedback(session_id);
CREATE INDEX IF NOT EXISTS idx_expert_feedback_type ON expert_feedback(feedback_type);

-- Create model_retraining_events table (referenced in retraining code)
CREATE TABLE IF NOT EXISTS model_retraining_events (
    id SERIAL PRIMARY KEY,
    trigger_type VARCHAR(50),
    feedback_samples INTEGER,
    trigger_time TIMESTAMP,
    status VARCHAR(50),
    performance_improvement DECIMAL(5, 4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_retraining_events_trigger_time ON model_retraining_events(trigger_time);
CREATE INDEX IF NOT EXISTS idx_retraining_events_status ON model_retraining_events(status);

-- Update existing ml_sessions to have last_activity set to created_at if NULL
UPDATE ml_sessions 
SET last_activity = created_at 
WHERE last_activity IS NULL;

-- Create a trigger to automatically update last_activity
CREATE OR REPLACE FUNCTION update_last_activity() 
RETURNS TRIGGER AS $$
BEGIN
    NEW.last_activity = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER IF NOT EXISTS trigger_update_last_activity
    BEFORE UPDATE ON ml_sessions
    FOR EACH ROW
    EXECUTE FUNCTION update_last_activity();

-- Add some sample data if tables are empty (for testing)
INSERT INTO ml_sessions (session_id, timestamp, session_length, is_anomaly, anomaly_score, anomaly_type, last_activity)
SELECT 
    'test_session_' || generate_series(1, 5),
    NOW() - INTERVAL '1 hour' * generate_series(1, 5),
    300 + random() * 600,
    CASE WHEN random() > 0.7 THEN TRUE ELSE FALSE END,
    random() * 1.0,
    CASE WHEN random() > 0.7 THEN 'test_anomaly' ELSE NULL END,
    NOW() - INTERVAL '1 hour' * generate_series(1, 5)
WHERE NOT EXISTS (SELECT 1 FROM ml_sessions LIMIT 1);

-- Add some sample labeled anomalies for testing
INSERT INTO labeled_anomalies (session_id, anomaly_label, label_confidence, labeled_by, label_reason)
SELECT 
    'test_session_' || generate_series(1, 3),
    'anomaly',
    0.9,
    'test_user',
    'Test labeled anomaly for retraining'
WHERE NOT EXISTS (SELECT 1 FROM labeled_anomalies LIMIT 1);

COMMIT;
