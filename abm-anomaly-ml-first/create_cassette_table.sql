-- Create cassette_counters table for cash forecasting
CREATE TABLE IF NOT EXISTS cassette_counters (
    id SERIAL PRIMARY KEY,
    terminal_id VARCHAR(50) NOT NULL,
    session_id INTEGER REFERENCES ml_sessions(id),
    transaction_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    total_dispensed DECIMAL(10,2) DEFAULT 0,
    cassette_1_remaining DECIMAL(10,2) DEFAULT 50000,
    cassette_2_remaining DECIMAL(10,2) DEFAULT 50000,
    cassette_3_remaining DECIMAL(10,2) DEFAULT 50000,
    cassette_4_remaining DECIMAL(10,2) DEFAULT 50000,
    total_remaining_cash DECIMAL(10,2) DEFAULT 200000,
    withdrawal_successful BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert sample data for testing
INSERT INTO cassette_counters (terminal_id, session_id, transaction_timestamp, total_dispensed, cassette_1_remaining, cassette_2_remaining, cassette_3_remaining, cassette_4_remaining, total_remaining_cash, withdrawal_successful) 
SELECT 
    'ATM' || LPAD((ROW_NUMBER() OVER () % 5 + 1)::text, 3, '0'),
    s.id,
    s.session_date + INTERVAL '1 hour' * (ROW_NUMBER() OVER () % 24),
    CASE WHEN random() > 0.8 THEN 0 ELSE (random() * 500 + 100)::DECIMAL(10,2) END,
    50000 - (random() * 30000)::DECIMAL(10,2),
    50000 - (random() * 25000)::DECIMAL(10,2),
    50000 - (random() * 35000)::DECIMAL(10,2),
    50000 - (random() * 20000)::DECIMAL(10,2),
    200000 - (random() * 100000)::DECIMAL(10,2),
    random() > 0.1
FROM ml_sessions s
WHERE s.id <= 100
ORDER BY random()
LIMIT 200;
