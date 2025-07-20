#!/bin/bash

echo "🧹 Clearing ML Sessions and State..."

# Clear all session data
docker exec abm-ml-anomaly-detector rm -rf /app/data/sessions/*

# Clear processed file tracking  
docker exec abm-ml-anomaly-detector rm -rf /app/input/processed/*

docker exec abm-ml-postgres psql -U abm_user -d abm_ml_db -c "
SELECT count(*) from ml_sessions;"

docker exec abm-ml-postgres psql -U abm_user -d abm_ml_db -c "
SELECT column_name,data_type,character_maximum_length FROM information_schema.columns WHERE table_name = 'ml_sessions';"

docker exec abm-ml-postgres psql -U abm_user -d abm_ml_db -c "
SELECT column_name,data_type,character_maximum_length FROM information_schema.columns WHERE table_name = 'ml_anomalies';"

# Clear PostgreSQL database data
echo "🗄️  Clearing PostgreSQL database data..."
docker exec abm-ml-postgres psql -U abm_user -d abm_ml_db -c "
    -- Clear anomaly detection results (only if tables exist)
    DROP TABLE IF EXISTS ml_sessions CASCADE;
    DROP TABLE IF EXISTS ml_anomalies CASCADE;
    DROP TABLE IF EXISTS labeled_anomalies CASCADE;
    DROP TABLE IF EXISTS ml_training_data CASCADE;
    DROP TABLE IF EXISTS ml_training_results CASCADE;
    DROP TABLE IF EXISTS ml_training_sessions CASCADE;
    DROP TABLE IF EXISTS ml_training_status CASCADE;
    DROP TABLE IF EXISTS ml_training_models CASCADE;
    DROP TABLE IF EXISTS ml_training_expert_labels CASCADE;
    DROP TABLE IF EXISTS ml_training_processing_status CASCADE;
    DROP TABLE IF EXISTS ml_training_session_tracking CASCADE;
    DROP TABLE IF EXISTS anomaly_sessions CASCADE;
    DROP TABLE IF EXISTS anomaly_results CASCADE;
    DROP TABLE IF EXISTS ml_models CASCADE;
    DROP TABLE IF EXISTS expert_labels CASCADE;
    DROP TABLE IF EXISTS processing_status CASCADE;
    DROP TABLE IF EXISTS session_tracking CASCADE;
    
    -- Confirm clearing
    SELECT 'Database tables dropped successfully' as status;
"

if [ $? -eq 0 ]; then
    echo "✅ PostgreSQL database cleared successfully"
    
    # Recreate the database schema
    echo "🔧 Recreating database schema..."
    docker exec abm-ml-postgres psql -U abm_user -d abm_ml_db -f /docker-entrypoint-initdb.d/01-base-schema.sql
    docker exec abm-ml-postgres psql -U abm_user -d abm_ml_db -f /docker-entrypoint-initdb.d/02-ml-schema.sql
    
    if [ $? -eq 0 ]; then
        echo "✅ Database schema recreated successfully"
    else
        echo "⚠️  Warning: Could not recreate database schema"
    fi
else
    echo "⚠️  Warning: Could not clear PostgreSQL database (may not exist yet)"
fi

# Optionally clear trained models (uncomment if needed)
 docker exec abm-ml-anomaly-detector rm -rf /app/models/*

# Clear any output files (optional)
 docker exec abm-ml-anomaly-detector rm -rf /app/output/*

 rm -rf ./data/sessions/*
rm -rf ./data/output/*
rm -rf ./data/models/*
rm -rf ./data/logs/*

pwd

ls -lrt ./data/logs/
ls -lrt ./data/sessions/

echo "✅ ML sessions and database cleared. New files will be processed fresh."
echo ""
echo "💡 To trigger reprocessing of existing files:"
echo "   - Copy new EJ log files to ./data/input/"
echo "   - Or restart the service: docker compose restart abm-ml-anomaly-detector"
echo ""
echo "📊 What was cleared and recreated:"
echo "   - File system sessions and processed tracking"
echo "   - PostgreSQL database tables (dropped and recreated)"
echo "   - Fresh database schema with all required tables"
echo "   - Ready for fresh ML training and detection"
