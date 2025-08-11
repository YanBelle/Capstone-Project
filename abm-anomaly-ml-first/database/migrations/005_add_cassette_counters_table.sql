-- Migration to add cassette counter tracking for cash forecasting
-- This table stores cassette counter information after each withdrawal transaction

-- Create cassette_counters table for cash forecasting
CREATE TABLE IF NOT EXISTS cassette_counters (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(100) NOT NULL,
    terminal_id VARCHAR(50),
    transaction_datetime TIMESTAMP NOT NULL,
    
    -- Cassette remaining counts (after withdrawal)
    cassette_1_remaining INTEGER,
    cassette_2_remaining INTEGER,
    cassette_3_remaining INTEGER,
    cassette_4_remaining INTEGER,
    
    -- Cassette denominations (note values)
    cassette_1_denomination INTEGER,
    cassette_2_denomination INTEGER,
    cassette_3_denomination INTEGER,
    cassette_4_denomination INTEGER,
    
    -- Dispensed amounts for this transaction
    cassette_1_dispensed INTEGER DEFAULT 0,
    cassette_2_dispensed INTEGER DEFAULT 0,
    cassette_3_dispensed INTEGER DEFAULT 0,
    cassette_4_dispensed INTEGER DEFAULT 0,
    
    -- Rejected amounts for this transaction
    cassette_1_rejected INTEGER DEFAULT 0,
    cassette_2_rejected INTEGER DEFAULT 0,
    cassette_3_rejected INTEGER DEFAULT 0,
    cassette_4_rejected INTEGER DEFAULT 0,
    
    -- Total transaction amount and metadata
    total_dispensed_amount INTEGER,
    total_rejected_amount INTEGER,
    withdrawal_successful BOOLEAN DEFAULT TRUE,
    
    -- Source information
    source_file VARCHAR(255),
    raw_cassette_data TEXT,  -- Store raw cassette section for debugging
    
    -- Audit fields
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Foreign key relationship to sessions
    FOREIGN KEY (session_id) REFERENCES ml_sessions(session_id) ON DELETE CASCADE
);

-- Create indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_cassette_counters_session_id ON cassette_counters(session_id);
CREATE INDEX IF NOT EXISTS idx_cassette_counters_terminal_id ON cassette_counters(terminal_id);
CREATE INDEX IF NOT EXISTS idx_cassette_counters_datetime ON cassette_counters(transaction_datetime);
CREATE INDEX IF NOT EXISTS idx_cassette_counters_terminal_datetime ON cassette_counters(terminal_id, transaction_datetime);

-- Create a view for cash forecasting analytics
CREATE OR REPLACE VIEW cassette_forecasting_view AS
SELECT 
    cc.terminal_id,
    cc.transaction_datetime,
    cc.cassette_1_remaining,
    cc.cassette_2_remaining,
    cc.cassette_3_remaining,
    cc.cassette_4_remaining,
    cc.cassette_1_denomination,
    cc.cassette_2_denomination,
    cc.cassette_3_denomination,
    cc.cassette_4_denomination,
    (cc.cassette_1_remaining * cc.cassette_1_denomination + 
     cc.cassette_2_remaining * cc.cassette_2_denomination + 
     cc.cassette_3_remaining * cc.cassette_3_denomination + 
     cc.cassette_4_remaining * cc.cassette_4_denomination) as total_cash_remaining,
    cc.total_dispensed_amount,
    ms.is_anomaly,
    ms.anomaly_type
FROM cassette_counters cc
LEFT JOIN ml_sessions ms ON cc.session_id = ms.session_id
ORDER BY cc.terminal_id, cc.transaction_datetime;

-- Create a view for terminal cash status summary
CREATE OR REPLACE VIEW terminal_cash_status AS
SELECT 
    terminal_id,
    MAX(transaction_datetime) as last_transaction_time,
    COUNT(*) as total_transactions,
    SUM(total_dispensed_amount) as total_dispensed,
    -- Latest cassette levels
    MAX(CASE WHEN rn = 1 THEN cassette_1_remaining END) as current_cassette_1_remaining,
    MAX(CASE WHEN rn = 1 THEN cassette_2_remaining END) as current_cassette_2_remaining,
    MAX(CASE WHEN rn = 1 THEN cassette_3_remaining END) as current_cassette_3_remaining,
    MAX(CASE WHEN rn = 1 THEN cassette_4_remaining END) as current_cassette_4_remaining,
    MAX(CASE WHEN rn = 1 THEN cassette_1_denomination END) as cassette_1_denomination,
    MAX(CASE WHEN rn = 1 THEN cassette_2_denomination END) as cassette_2_denomination,
    MAX(CASE WHEN rn = 1 THEN cassette_3_denomination END) as cassette_3_denomination,
    MAX(CASE WHEN rn = 1 THEN cassette_4_denomination END) as cassette_4_denomination,
    -- Current total cash available
    MAX(CASE WHEN rn = 1 THEN 
        (cassette_1_remaining * cassette_1_denomination + 
         cassette_2_remaining * cassette_2_denomination + 
         cassette_3_remaining * cassette_3_denomination + 
         cassette_4_remaining * cassette_4_denomination)
    END) as current_total_cash
FROM (
    SELECT *,
           ROW_NUMBER() OVER (PARTITION BY terminal_id ORDER BY transaction_datetime DESC) as rn
    FROM cassette_counters
    WHERE terminal_id IS NOT NULL
) ranked_counters
GROUP BY terminal_id;

-- Add trigger to automatically update updated_at timestamp
CREATE OR REPLACE FUNCTION update_cassette_counters_updated_at() 
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER IF NOT EXISTS trigger_update_cassette_counters_updated_at
    BEFORE UPDATE ON cassette_counters
    FOR EACH ROW
    EXECUTE FUNCTION update_cassette_counters_updated_at();

-- Add some sample data for testing if table is empty
INSERT INTO cassette_counters (
    session_id, terminal_id, transaction_datetime,
    cassette_1_remaining, cassette_2_remaining, cassette_3_remaining, cassette_4_remaining,
    cassette_1_denomination, cassette_2_denomination, cassette_3_denomination, cassette_4_denomination,
    total_dispensed_amount, source_file
)
SELECT 
    'test_cassette_session_' || generate_series(1, 3),
    '416',
    NOW() - INTERVAL '1 hour' * generate_series(1, 3),
    500 - generate_series(1, 3) * 5,  -- Decreasing cassette 1
    800 - generate_series(1, 3) * 3,  -- Decreasing cassette 2
    300 - generate_series(1, 3) * 2,  -- Decreasing cassette 3
    600 - generate_series(1, 3) * 4,  -- Decreasing cassette 4
    20,  -- $20 notes
    50,  -- $50 notes
    100, -- $100 notes
    20,  -- $20 notes
    (generate_series(1, 3) * 100),  -- Increasing dispensed amounts
    'test_abm416_ej.txt'
WHERE NOT EXISTS (SELECT 1 FROM cassette_counters LIMIT 1);

COMMIT;
