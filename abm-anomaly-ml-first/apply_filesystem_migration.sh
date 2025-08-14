#!/bin/bash

# Apply migration to remove text columns and use filesystem storage
# Script: apply_filesystem_migration.sh

echo "Applying filesystem migration..."

# Set database connection variables
DB_HOST=${POSTGRES_HOST:-localhost}
DB_USER=${POSTGRES_USER:-abm_user}
DB_NAME=${POSTGRES_DB:-abm_anomaly_detection}
DB_PASSWORD=${POSTGRES_PASSWORD:-anomaly_detection_123}
DB_PORT=${POSTGRES_PORT:-5432}

# Export password for psql
export PGPASSWORD=$DB_PASSWORD

MIGRATION_FILE="database/migrations/006_remove_text_columns_use_filesystem.sql"

echo "Connecting to database: $DB_HOST:$DB_PORT/$DB_NAME as $DB_USER"

# Check if migration file exists
if [ ! -f "$MIGRATION_FILE" ]; then
    echo "Error: Migration file $MIGRATION_FILE not found!"
    exit 1
fi

echo "Applying migration: $MIGRATION_FILE"

# Apply the migration
psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -f $MIGRATION_FILE

if [ $? -eq 0 ]; then
    echo "Migration applied successfully!"
    echo "Raw text and cleaned text are now stored on the file system."
    echo "File locations:"
    echo "  - Raw text: /app/data/sessions/{session_id[:2]}/{session_id}_raw.txt"
    echo "  - Cleaned text: /app/data/sessions/{session_id[:2]}/{session_id}_cleaned.txt"
else
    echo "Error applying migration!"
    exit 1
fi

# Create the data directory structure if it doesn't exist
echo "Creating data directory structure..."
mkdir -p /app/data/sessions
for i in {00..99}; do
    mkdir -p /app/data/sessions/$i
done

echo "File system storage setup complete!"
