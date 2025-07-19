-- ML-specific schema
CREATE TABLE IF NOT EXISTS ml_sessions (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(100) UNIQUE NOT NULL,
    timestamp TIMESTAMP,
    session_length INTEGER,
    is_anomaly BOOLEAN DEFAULT FALSE,
    anomaly_score DECIMAL(5, 4),
    anomaly_type VARCHAR(100),
    detected_patterns JSONB,
    critical_events JSONB,
    embedding_vector BYTEA,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_ml_sessions_anomaly ON ml_sessions(is_anomaly);
CREATE INDEX idx_ml_sessions_score ON ml_sessions(anomaly_score);
CREATE INDEX idx_ml_sessions_type ON ml_sessions(anomaly_type);

-- ML anomalies table
CREATE TABLE IF NOT EXISTS ml_anomalies (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(100) REFERENCES ml_sessions(session_id),
    anomaly_type VARCHAR(100),
    anomaly_score DECIMAL(5, 4),
    cluster_id INTEGER,
    detected_patterns JSONB,
    critical_events JSONB,
    error_codes JSONB,
    model_name VARCHAR(100),
    model_version VARCHAR(20),
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Anomaly clusters
CREATE TABLE IF NOT EXISTS anomaly_clusters (
    id SERIAL PRIMARY KEY,
    cluster_id INTEGER UNIQUE NOT NULL,
    cluster_name VARCHAR(100),
    cluster_description TEXT,
    typical_patterns JSONB,
    member_count INTEGER DEFAULT 0,
    centroid_vector BYTEA,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Labeled training data
CREATE TABLE IF NOT EXISTS labeled_anomalies (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(100) REFERENCES ml_sessions(session_id),
    anomaly_label VARCHAR(100) NOT NULL,
    label_confidence DECIMAL(3, 2),
    labeled_by VARCHAR(100),
    label_reason TEXT,
    is_verified BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_labeled_anomalies_label ON labeled_anomalies(anomaly_label);

-- ML model metadata
CREATE TABLE IF NOT EXISTS ml_models (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    model_type VARCHAR(50),
    model_version VARCHAR(20),
    training_date TIMESTAMP,
    training_samples INTEGER,
    anomaly_threshold DECIMAL(5, 4),
    performance_metrics JSONB,
    model_parameters JSONB,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Pattern definitions
CREATE TABLE IF NOT EXISTS anomaly_patterns (
    id SERIAL PRIMARY KEY,
    pattern_name VARCHAR(100) UNIQUE NOT NULL,
    pattern_regex TEXT,
    pattern_description TEXT,
    severity_level VARCHAR(20),
    recommended_action TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert predefined patterns
INSERT INTO anomaly_patterns (pattern_name, pattern_regex, pattern_description, severity_level, recommended_action) VALUES
('supervisor_mode', 'SUPERVISOR\s+MODE\s+(ENTRY|EXIT)', 'Supervisor mode activity detected', 'MEDIUM', 'Review supervisor access logs'),
('unable_to_dispense', 'UNABLE\s+TO\s+DISPENSE', 'ATM unable to dispense cash', 'HIGH', 'Check cash cassettes and dispensing mechanism'),
('device_error', 'DEVICE\s+ERROR', 'Hardware device error', 'HIGH', 'Schedule immediate maintenance'),
('power_reset', 'POWER-UP/RESET', 'Power reset or restart', 'HIGH', 'Check power supply and UPS status'),
('cash_retract', 'CASHIN\s+RETRACT\s+STARTED', 'Cash retraction initiated', 'HIGH', 'Review deposit module functionality'),
('no_dispense', 'NO\s+DISPENSE\s+SUCCESS', 'Cash dispensing failed', 'HIGH', 'Inspect cash handling mechanism'),
('note_error', 'NOTE\s+ERROR\s+OCCURRED', 'Note processing error', 'MEDIUM', 'Check note reader and validator'),
('recovery_failed', 'RECOVERY\s+FAILED', 'Recovery operation failed', 'CRITICAL', 'Immediate technical intervention required')
ON CONFLICT (pattern_name) DO NOTHING;

-- Views for analysis
CREATE OR REPLACE VIEW ml_anomaly_summary AS
SELECT 
    DATE(s.timestamp) as date,
    COUNT(DISTINCT s.id) as total_sessions,
    COUNT(DISTINCT CASE WHEN s.is_anomaly THEN s.id END) as anomaly_count,
    AVG(CASE WHEN s.is_anomaly THEN s.anomaly_score END) as avg_anomaly_score,
    COUNT(DISTINCT a.cluster_id) as unique_clusters,
    ARRAY_AGG(DISTINCT a.anomaly_type) as anomaly_types
FROM ml_sessions s
LEFT JOIN ml_anomalies a ON s.session_id = a.session_id
GROUP BY DATE(s.timestamp);

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO abm_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO abm_user;
