#!/bin/bash

echo "Starting multi-anomaly database migration..."

# Check if Docker services are running
if ! docker-compose ps | grep -q "Up"; then
    echo "Starting Docker services..."
    docker-compose up -d
    sleep 30
fi

# Apply the migration using the postgres container
echo "Applying database migration..."
docker-compose exec -T postgres psql -U abm_user -d abm_ml_db -f /docker-entrypoint-initdb.d/migrations/002_multi_anomaly_support.sql

# Verify the migration
echo "Verifying migration..."
docker-compose exec -T postgres psql -U abm_user -d abm_ml_db -c "
SELECT column_name 
FROM information_schema.columns 
WHERE table_name = 'ml_sessions' 
AND column_name IN ('anomaly_count', 'anomaly_types', 'max_severity', 'anomalies_detail');
"

echo "Migration complete!"
