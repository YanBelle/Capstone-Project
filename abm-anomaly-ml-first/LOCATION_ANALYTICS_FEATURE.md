# Location Analytics Feature

This feature adds terminal/location tracking and analytics to the ML-First ABM system. It provides insights into which ATMs (terminals) are experiencing the most anomalies.

## Features Added

1. **Terminal and Location Tracking**
   - Added `terminal_id` and `location` fields to the `ml_sessions` table
   - Created indexes for efficient querying
   - Added migration script for database updates

2. **Terminal Analytics**
   - Created `terminal_anomaly_summary` view to aggregate anomaly data by terminal/location
   - Added `get_most_problematic_terminals()` function for easy retrieval of high-risk terminals

3. **Dashboard Updates**
   - Added "Most Affected ATMs/Locations" section to the dashboard
   - Shows terminals with highest anomaly counts, critical issues, and risk scores
   - Provides color-coded risk visualization

## How to Apply the Changes

1. Execute the installation script:
   ```
   ./add_location_dashboard_feature.sh
   ```

2. Apply the database migration:
   ```
   ./apply_location_migration.sh
   ```

3. Restart the API and Dashboard services:
   ```
   docker-compose restart api dashboard
   ```

## Dashboard Features

The updated dashboard now shows:
- Terminal IDs with the most anomalies
- Location information for each terminal
- Count of anomalies per terminal
- Count of critical anomalies
- Average risk score (color-coded)

This information helps identify problematic ATM locations that may need maintenance, security measures, or other attention.

## Technical Implementation

The implementation includes:

1. **Database Changes**:
   - Migration file `003_add_location_support.sql`
   - New columns and views for terminal tracking
   - SQL function for aggregating terminal statistics

2. **API Updates**:
   - Extended `DashboardStats` model to include terminal data
   - Added `get_problematic_terminals()` function

3. **Frontend Updates**:
   - Added "Most Affected ATMs/Locations" table to the dashboard
   - Color-coded risk scoring for visual identification
   - Responsive table layout compatible with the existing UI

## Future Enhancements

Possible future enhancements for the location analytics feature:

1. Interactive map visualization of terminal locations
2. Time-based analytics to show terminal performance over time
3. Alerts specific to recurring issues at certain terminals
4. Location-based anomaly detection models
