#!/bin/bash

# Add terminal and location information to the DashboardStats endpoint
echo "Updating API to include location data in the dashboard..."

# Update API file
API_FILE="services/api/app.py"

# Check if API file exists
if [ ! -f "$API_FILE" ]; then
    echo "API file not found at $API_FILE. Please ensure the file exists."
    exit 1
fi

# 1. Update the DashboardStats class to include problematic_terminals
sed -i '/class DashboardStats(BaseModel):/,/hourly_trend: List\[Dict\[str, Any\]\]/ s/hourly_trend: List\[Dict\[str, Any\]\]/hourly_trend: List\[Dict\[str, Any\]\]\n    problematic_terminals: List\[Dict\[str, Any\]\]/' "$API_FILE"

# 2. Update the dashboard stats endpoint to fetch problematic terminals
sed -i '/return DashboardStats(/,/hourly_trend=hourly_trend/ s/hourly_trend=hourly_trend/hourly_trend=hourly_trend,\n            problematic_terminals=get_problematic_terminals()/' "$API_FILE"

# 3. Add the get_problematic_terminals function
cat >> "$API_FILE" << 'EOF'

def get_problematic_terminals():
    """Get the most problematic terminals/locations"""
    try:
        query = """
        SELECT * FROM get_most_problematic_terminals(5)
        """
        
        with db_engine.connect() as conn:
            result = conn.execute(text(query))
            
        terminals = []
        for row in result:
            terminals.append({
                'terminal_id': row[0],
                'location': row[1] or 'Unknown Location',
                'total_anomalies': row[2],
                'critical_count': row[3],
                'high_count': row[4],
                'avg_score': float(row[5])
            })
        
        return terminals
    except Exception as e:
        logger.error(f"Error getting problematic terminals: {str(e)}")
        return []

# Include SVM Debug API routes
try:
    from svm_debug_api import router as svm_debug_router
    app.include_router(svm_debug_router)
    logger.info("SVM Debug API routes loaded successfully")
except ImportError:
    logger.warning("SVM Debug API not available - install required dependencies")
except Exception as e:
    logger.error(f"Error loading SVM Debug API: {str(e)}")
EOF

echo "✓ API updated to include location data"

# Update the Dashboard component to include the Most Affected ATM section
DASHBOARD_FILE="services/dashboard/src/Dashboard.js"

# Check if Dashboard file exists
if [ ! -f "$DASHBOARD_FILE" ]; then
    echo "Dashboard file not found at $DASHBOARD_FILE. Please ensure the file exists."
    exit 1
fi

# Add the new component for Most Affected ATMs to the Dashboard
# First, find the position where we need to add the component (in the overview tab)
sed -i '/className="grid grid-cols-1 md:grid-cols-2 gap-6">/,/<\/div>/ s/<\/div>/  <\/div>\n            <div className="mt-6 bg-white rounded-lg shadow-md p-6">\n              <h3 className="text-lg font-semibold mb-4">Most Affected ATMs\/Locations<\/h3>\n              <div className="overflow-x-auto">\n                <table className="min-w-full divide-y divide-gray-200">\n                  <thead className="bg-gray-50">\n                    <tr>\n                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">ATM ID<\/th>\n                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Location<\/th>\n                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Anomalies<\/th>\n                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Critical<\/th>\n                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Risk Score<\/th>\n                    <\/tr>\n                  <\/thead>\n                  <tbody className="bg-white divide-y divide-gray-200">\n                    {stats.problematic_terminals?.length > 0 ? (\n                      stats.problematic_terminals.map((terminal, idx) => (\n                        <tr key={idx}>\n                          <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">{terminal.terminal_id}<\/td>\n                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{terminal.location}<\/td>\n                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{terminal.total_anomalies}<\/td>\n                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{terminal.critical_count}<\/td>\n                          <td className="px-6 py-4 whitespace-nowrap">\n                            <span className={`px-2 py-1 inline-flex text-xs leading-5 font-semibold rounded-full ${\n                              terminal.avg_score > 0.8 \n                                ? \'bg-red-100 text-red-800\'\n                                : terminal.avg_score > 0.6\n                                ? \'bg-yellow-100 text-yellow-800\'\n                                : \'bg-green-100 text-green-800\'\n                            }`}>\n                              {terminal.avg_score.toFixed(2)}\n                            <\/span>\n                          <\/td>\n                        <\/tr>\n                      ))\n                    ) : (\n                      <tr>\n                        <td colSpan="5" className="px-6 py-4 text-center text-sm text-gray-500">No terminal data available<\/td>\n                      <\/tr>\n                    )}\n                  <\/tbody>\n                <\/table>\n              <\/div>\n            <\/div>/' "$DASHBOARD_FILE"

# Initialize stats.problematic_terminals in the state
sed -i '/const \[stats, setStats\] = useState({/,/hourly_trend: \[\]/ s/hourly_trend: \[\]/hourly_trend: \[\],\n    problematic_terminals: \[\]/' "$DASHBOARD_FILE"

echo "✓ Dashboard updated with Most Affected ATMs/Locations section"

# Create a script to apply the migration
cat > apply_location_migration.sh << 'SCRIPT'
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
SCRIPT

chmod +x apply_location_migration.sh

echo "✅ Created all necessary components for the Most Affected ATM/Location feature"
echo "To apply the changes and migration, run:"
echo "  ./apply_location_migration.sh"
