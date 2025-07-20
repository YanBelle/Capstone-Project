INSERT INTO anomaly_patterns (pattern_name, pattern_regex, pattern_description, severity_level, recommended_action) VALUES
('supervisor_mode', 'SUPERVISOR\\s+MODE\\s+(ENTRY|EXIT)', 'Supervisor mode activity detected', 'MEDIUM', 'Review supervisor access logs'),
('unable_to_dispense', 'UNABLE\\s+TO\\s+DISPENSE', 'ATM unable to dispense cash', 'HIGH', 'Check cash cassettes and dispensing mechanism'),
('device_error', 'DEVICE\\s+ERROR', 'Hardware device error', 'HIGH', 'Schedule immediate maintenance'),
('power_reset', 'POWER-UP/RESET', 'Power reset or restart', 'HIGH', 'Check power supply and UPS status'),
('cash_retract', 'CASHIN\\s+RETRACT\\s+STARTED', 'Cash retraction initiated', 'HIGH', 'Review deposit module functionality'),
('no_dispense', 'NO\\s+DISPENSE\\s+SUCCESS', 'Cash dispensing failed', 'HIGH', 'Inspect cash handling mechanism'),
('note_error', 'NOTE\\s+ERROR\\s+OCCURRED', 'Note processing error', 'MEDIUM', 'Check note reader and validator'),
('recovery_failed', 'RECOVERY\\s+FAILED', 'Recovery operation failed', 'CRITICAL', 'Immediate technical intervention required')
ON CONFLICT (pattern_name) DO NOTHING;