// Dashboard Data Fix Script
// This script fixes the dashboard data display by fetching current API data
// and updating the DOM elements that show outdated cached values

console.log('Dashboard Fix Script Loading...');

// Function to fetch fresh data from API
async function fetchFreshData() {
    try {
        console.log('Fetching fresh data from API...');
        const response = await fetch('/api/v1/dashboard/stats');
        const data = await response.json();
        console.log('Fresh API data:', data);
        return data;
    } catch (error) {
        console.error('Error fetching fresh data:', error);
        return null;
    }
}

// Function to fetch anomalies data
async function fetchAnomaliesData() {
    try {
        console.log('Fetching anomalies data from API...');
        const response = await fetch('/api/v1/anomalies?limit=50');
        const data = await response.json();
        console.log('Fresh anomalies data:', data);
        return data;
    } catch (error) {
        console.error('Error fetching anomalies data:', error);
        return null;
    }
}

// Function to update dashboard numbers
function updateDashboardNumbers(data) {
    if (!data) return;
    
    console.log('Updating dashboard with:', {
        transactions: data.total_transactions,
        anomalies: data.total_anomalies
    });
    
    // Find and update all elements that contain the old values
    const allElements = document.querySelectorAll('*');
    
    allElements.forEach(element => {
        // Update text content for transaction count
        if (element.textContent && element.textContent.includes('1,250')) {
            console.log('Found 1,250 in element:', element);
            element.textContent = element.textContent.replace('1,250', data.total_transactions.toString());
        }
        
        // Update text content for anomaly count
        if (element.textContent && element.textContent.includes('23')) {
            // Be more specific to avoid replacing other instances of 23
            if (element.textContent.includes('Total Anomalies') || 
                element.textContent.includes('Anomalies Detected') ||
                element.closest('.anomaly-card, .stats-card')) {
                console.log('Found 23 in anomaly element:', element);
                element.textContent = element.textContent.replace('23', data.total_anomalies.toString());
            }
        }
    });
    
    // Also update any specific class-based elements
    const transactionElements = document.querySelectorAll('.total-transactions, .transaction-count');
    transactionElements.forEach(el => {
        if (el.textContent.includes('1,250')) {
            el.textContent = el.textContent.replace('1,250', data.total_transactions.toString());
        }
    });
    
    const anomalyElements = document.querySelectorAll('.total-anomalies, .anomaly-count');
    anomalyElements.forEach(el => {
        if (el.textContent.includes('23')) {
            el.textContent = el.textContent.replace('23', data.total_anomalies.toString());
        }
    });
}

// Function to populate the anomalies tab
function populateAnomaliesTab(anomaliesData) {
    if (!anomaliesData || !anomaliesData.anomalies) return;
    
    console.log('Populating anomalies tab with:', anomaliesData.anomalies.length, 'anomalies');
    
    // Find the anomalies table or container
    const anomaliesTable = document.querySelector('table');
    const anomaliesContainer = document.querySelector('[class*="anomal"]');
    
    if (anomaliesTable) {
        // Check if this is the anomalies tab table
        const tableHeaders = anomaliesTable.querySelectorAll('th');
        const hasAnomalyHeaders = Array.from(tableHeaders).some(th => 
            th.textContent.includes('Session') || 
            th.textContent.includes('Anomaly') ||
            th.textContent.includes('Score')
        );
        
        if (hasAnomalyHeaders) {
            console.log('Found anomalies table, updating...');
            
            // Clear existing rows (except header)
            const tbody = anomaliesTable.querySelector('tbody');
            if (tbody) {
                tbody.innerHTML = '';
                
                // Add fresh anomaly data
                anomaliesData.anomalies.forEach(anomaly => {
                    const row = document.createElement('tr');
                    row.className = 'hover:bg-gray-50';
                    
                    row.innerHTML = `
                        <td class="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                            ${anomaly.session_id}
                        </td>
                        <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                            ${new Date(anomaly.timestamp).toLocaleString()}
                        </td>
                        <td class="px-6 py-4 whitespace-nowrap">
                            <span class="inline-flex px-2 py-1 text-xs font-semibold rounded-full bg-red-100 text-red-800">
                                ${anomaly.anomaly_type || 'Unknown'}
                            </span>
                        </td>
                        <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                            ${(anomaly.anomaly_score || 0).toFixed(3)}
                        </td>
                        <td class="px-6 py-4 whitespace-nowrap">
                            <span class="inline-flex px-2 py-1 text-xs font-semibold rounded-full ${
                                anomaly.max_severity === 'critical' ? 'bg-red-100 text-red-800' :
                                anomaly.max_severity === 'high' ? 'bg-orange-100 text-orange-800' :
                                'bg-yellow-100 text-yellow-800'
                            }">
                                ${anomaly.max_severity || 'Medium'}
                            </span>
                        </td>
                    `;
                    
                    tbody.appendChild(row);
                });
                
                console.log('Anomalies table updated with', anomaliesData.anomalies.length, 'rows');
            }
        }
    }
    
    // Also update any "No anomalies" messages
    const noAnomaliesMessages = document.querySelectorAll('*');
    noAnomaliesMessages.forEach(element => {
        if (element.textContent && element.textContent.includes('No active alerts')) {
            if (anomaliesData.anomalies.length > 0) {
                element.style.display = 'none';
            }
        }
    });
}

// Function to add refresh button
function addRefreshButton() {
    console.log('Adding refresh button...');
    
    // Check if button already exists
    if (document.getElementById('manual-refresh-btn')) {
        console.log('Refresh button already exists');
        return;
    }
    
    // Create refresh button
    const refreshBtn = document.createElement('button');
    refreshBtn.id = 'manual-refresh-btn';
    refreshBtn.textContent = 'Refresh Data';
    refreshBtn.style.cssText = `
        position: fixed;
        top: 10px;
        right: 10px;
        z-index: 9999;
        padding: 10px 15px;
        background: #007bff;
        color: white;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        font-size: 14px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    `;
    
    refreshBtn.onclick = async function() {
        console.log('Manual refresh triggered');
        refreshBtn.textContent = 'Refreshing...';
        refreshBtn.disabled = true;
        
        const freshData = await fetchFreshData();
        if (freshData) {
            updateDashboardNumbers(freshData);
        }
        
        const anomaliesData = await fetchAnomaliesData();
        if (anomaliesData) {
            populateAnomaliesTab(anomaliesData);
        }
        
        refreshBtn.textContent = 'Refresh Data';
        refreshBtn.disabled = false;
    };
    
    document.body.appendChild(refreshBtn);
    console.log('Refresh button added successfully');
}

// Function to create anomalies tab content if it doesn't exist
function createAnomaliesTabContent() {
    // Check if we're on the anomalies tab
    const currentUrl = window.location.href;
    if (!currentUrl.includes('anomalies')) {
        return;
    }
    
    console.log('Creating anomalies tab content...');
    
    // Find the main content area
    const mainContent = document.querySelector('[class*="p-6"]') || document.querySelector('main') || document.body;
    
    // Create anomalies table if it doesn't exist
    if (!document.querySelector('table')) {
        const anomaliesContainer = document.createElement('div');
        anomaliesContainer.className = 'bg-white rounded-lg shadow-md p-6';
        
        anomaliesContainer.innerHTML = `
            <h3 class="text-lg font-semibold mb-4">Detected Anomalies</h3>
            <div class="overflow-x-auto">
                <table class="min-w-full divide-y divide-gray-200">
                    <thead class="bg-gray-50">
                        <tr>
                            <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Session ID</th>
                            <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Timestamp</th>
                            <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Type</th>
                            <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Score</th>
                            <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Severity</th>
                        </tr>
                    </thead>
                    <tbody class="bg-white divide-y divide-gray-200">
                        <!-- Content will be populated by JavaScript -->
                    </tbody>
                </table>
            </div>
        `;
        
        mainContent.appendChild(anomaliesContainer);
        console.log('Anomalies table structure created');
    }
}

// Main function to initialize the fix
async function initializeDashboardFix() {
    console.log('Initializing dashboard fix...');
    
    // Wait for DOM to be ready
    if (document.readyState === 'loading') {
        await new Promise(resolve => document.addEventListener('DOMContentLoaded', resolve));
    }
    
    // Add refresh button immediately
    addRefreshButton();
    
    // Wait a bit for React to render, then apply fix
    setTimeout(async () => {
        console.log('Applying initial data fix...');
        
        // Create anomalies tab content if needed
        createAnomaliesTabContent();
        
        const freshData = await fetchFreshData();
        if (freshData) {
            updateDashboardNumbers(freshData);
        }
        
        const anomaliesData = await fetchAnomaliesData();
        if (anomaliesData) {
            populateAnomaliesTab(anomaliesData);
        }
    }, 2000);
    
    // Set up periodic refresh every 30 seconds
    setInterval(async () => {
        console.log('Periodic refresh...');
        const freshData = await fetchFreshData();
        if (freshData) {
            updateDashboardNumbers(freshData);
        }
        
        const anomaliesData = await fetchAnomaliesData();
        if (anomaliesData) {
            populateAnomaliesTab(anomaliesData);
        }
    }, 30000);
}

// Start the fix when script loads
initializeDashboardFix();

console.log('Dashboard Fix Script Loaded Successfully');
