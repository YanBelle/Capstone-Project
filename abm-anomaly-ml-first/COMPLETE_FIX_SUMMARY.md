# Database Schema and Retraining System Fix Summary

## Issues Identified from Logs

### 1. Database Schema Issues
- **Missing `last_activity` column** in `ml_sessions` table
- **Missing `anomaly_sessions` table** referenced in queries
- **Missing `expert_feedback` table** referenced in queries
- **Missing `model_retraining_events` table** referenced in retraining code

### 2. Dependency Issues
- **Missing `spacy` dependency** in API service causing continuous learning status errors
- **Missing NLP dependencies** causing import errors

### 3. Retraining Logic Issues
- **Labeled anomalies not loaded** from database into feedback buffer
- **Retraining status check** only looking at in-memory buffer, not database
- **Minimum threshold too high** (was 10, should be 5)

## Fixes Applied

### 1. Database Schema Fixes
✅ **Created migration** `database/migrations/003_fix_missing_schema.sql`:
- Added `last_activity` column to `ml_sessions` table
- Created `anomaly_sessions` table with proper structure
- Created `expert_feedback` table with proper structure
- Created `model_retraining_events` table for tracking retraining
- Added trigger to auto-update `last_activity` on session updates
- Added sample data for testing

### 2. Dependency Fixes
✅ **Updated API requirements.txt**:
- Added `spacy==3.4.4` to fix NLP dependency error
- This resolves the "No module named 'spacy'" error

### 3. Retraining Logic Fixes
✅ **Modified MLFirstAnomalyDetector constructor**:
- Added `db_engine` parameter to accept database connection
- File: `services/anomaly-detector/ml_analyzer.py`

✅ **Enhanced get_continuous_learning_status() method**:
- Now checks both in-memory feedback buffer AND database labeled_anomalies
- Returns combined count as `feedback_buffer_size`
- Added separate counters for debugging

✅ **Added load_labeled_anomalies_from_database() method**:
- Loads all labeled anomalies from database into feedback buffer
- Converts database format to feedback buffer format
- Handles different label types appropriately

✅ **Updated continuous_model_retraining() method**:
- Now calls database loading method first
- Reduced minimum threshold from 10 to 5 samples
- Will now use your 66 labeled anomalies

✅ **Updated service instantiation**:
- Modified both `main.py` files to pass database connection to MLFirstAnomalyDetector

## Expected Results

### 1. Database Errors Should Stop
- No more "column 'last_activity' does not exist" errors
- No more "relation 'anomaly_sessions' does not exist" errors
- No more "relation 'expert_feedback' does not exist" errors

### 2. API Errors Should Stop
- No more "No module named 'spacy'" errors
- Continuous learning status endpoint should work
- Retraining endpoint should work

### 3. Retraining Should Work
- System will now load your 66 labeled anomalies from database
- Retraining button should trigger actual model retraining
- Status endpoint should show correct feedback counts

## Next Steps

### 1. Apply Database Migration
```bash
# Apply the database migration
cd /Users/christopherpearson/Projects/abm_ej_exporter_docker_cleaned/EJAnomalyDetectionV3/abm-anomaly-ml-first
cat database/migrations/003_fix_missing_schema.sql | docker compose exec -T postgres psql -U abm_user -d abm_db
```

### 2. Rebuild and Restart Services
```bash
# Rebuild services with updated dependencies
docker compose build

# Restart services
docker compose up -d
```

### 3. Test the System
```bash
# Test continuous learning status
curl http://localhost:8000/api/v1/continuous-learning/status

# Test retraining trigger
curl -X POST http://localhost:8000/api/v1/continuous-learning/trigger-retraining
```

### 4. Check Logs
```bash
# Check for errors
docker compose logs api | grep -i error
docker compose logs anomaly-detector | grep -i error
```

## Key Changes Made

1. **Database Schema**: Complete schema with all referenced tables
2. **Dependencies**: Added missing spacy dependency
3. **Retraining Logic**: Now connects database labeled anomalies to retraining process
4. **Error Handling**: Better error handling for database operations
5. **Configuration**: Proper database connection passing

Your 66 labeled anomalies should now be properly loaded and used for retraining when you click the retraining button!
