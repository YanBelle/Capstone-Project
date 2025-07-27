#!/bin/bash
set -e

echo "Applying location support migration..."

# Copy migration file to the flyway SQL folder
mkdir -p ./data/flyway/sql
cp ./database/migrations/003_add_location_support.sql ./data/flyway/sql/

# Run Flyway migration
docker-compose -f docker-compose-flyway.yml up

echo "Migration completed!"

# Generate some sample location data
echo "Generating sample location data..."
docker-compose exec -T postgres psql -U abmuser -d abmdb << 'SQLSCRIPT'
-- Add some sample location data to existing sessions
UPDATE ml_sessions
SET 
    terminal_id = CASE 
        WHEN id % 5 = 0 THEN 'ATM0163'
        WHEN id % 5 = 1 THEN 'ATM0275'
        WHEN id % 5 = 2 THEN 'ATM0381'
        WHEN id % 5 = 3 THEN 'ATM0192'
        WHEN id % 5 = 4 THEN 'ATM0447'
    END,
    location = CASE 
        WHEN id % 5 = 0 THEN 'Main Street Branch'
        WHEN id % 5 = 1 THEN 'Downtown Mall'
        WHEN id % 5 = 2 THEN 'Airport Terminal'
        WHEN id % 5 = 3 THEN 'University Campus'
        WHEN id % 5 = 4 THEN 'Shopping Center'
    END
WHERE terminal_id IS NULL;
SQLSCRIPT

echo "✓ Location data added successfully!"
