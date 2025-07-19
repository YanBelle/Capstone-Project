-- Base schema for ABM anomaly detection
CREATE TABLE IF NOT EXISTS transactions (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    card_number VARCHAR(20),
    transaction_type VARCHAR(50),
    amount DECIMAL(10, 2),
    status VARCHAR(20),
    error_type VARCHAR(50),
    response_time INTEGER,
    terminal_id VARCHAR(20),
    session_id VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_transactions_timestamp ON transactions(timestamp);
CREATE INDEX idx_transactions_session ON transactions(session_id);

-- Alerts table
CREATE TABLE IF NOT EXISTS alerts (
    id SERIAL PRIMARY KEY,
    anomaly_id INTEGER,
    alert_level VARCHAR(20),
    message TEXT,
    is_resolved BOOLEAN DEFAULT FALSE,
    resolved_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_alerts_level ON alerts(alert_level);
CREATE INDEX idx_alerts_resolved ON alerts(is_resolved);
