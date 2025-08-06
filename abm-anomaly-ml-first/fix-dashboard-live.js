console.log("Live Dashboard Fix Script Loaded");

// Function to wait for elements to be available
function waitForElement(selector, timeout = 10000) {
    return new Promise((resolve, reject) => {
        const startTime = Date.now();
        function check() {
            const element = document.querySelector(selector);
            if (element) {
                resolve(element);
            } else if (Date.now() - startTime > timeout) {
                reject(new Error(`Timeout waiting for ${selector}`));
            } else {
                setTimeout(check, 100);
            }
        }
        check();
    });
}

// Fetch current data from API
async function fetchCurrentData() {
    try {
        console.log("Fetching current dashboard data...");
        const response = await fetch("/api/v1/dashboard/stats");
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        console.log("Current API data:", data);
        return data;
    } catch (error) {
        console.error("Error fetching current data:", error);
        return null;
    }
}

// Update the dashboard values
function updateDashboard(apiData) {
    console.log("Updating dashboard with live data:", apiData);
    
    // Find and update Total Transactions
    const allElements = document.querySelectorAll('*');
    allElements.forEach(element => {
        if (element.textContent === '1,250' || element.textContent === '1250') {
            console.log("Updating Total Transactions:", element.textContent, "->", apiData.total_transactions);
            element.textContent = apiData.total_transactions.toString();
        }
        
        if (element.textContent === '23' && !element.textContent.includes('2023')) {
            console.log("Updating Total Anomalies:", element.textContent, "->", apiData.total_anomalies);
            element.textContent = apiData.total_anomalies.toString();
        }
        
        if (element.textContent === '1.84%') {
            const newRate = (apiData.anomaly_rate * 100).toFixed(2) + '%';
            console.log("Updating Anomaly Rate:", element.textContent, "->", newRate);
            element.textContent = newRate;
        }
        
        if (element.textContent === '5' && element.parentElement && 
            element.parentElement.textContent.toLowerCase().includes('high risk')) {
            console.log("Updating High Risk Count:", element.textContent, "->", apiData.high_risk_count);
            element.textContent = apiData.high_risk_count.toString();
        }
    });
    
    console.log("Dashboard update completed");
}

// Main function to fix the dashboard
async function fixDashboardLive() {
    try {
        console.log("Starting live dashboard fix...");
        
        // Wait for page to load
        await waitForElement('body', 5000);
        
        // Small delay to ensure React has rendered
        await new Promise(resolve => setTimeout(resolve, 2000));
        
        // Fetch and apply current data
        const apiData = await fetchCurrentData();
        if (apiData) {
            updateDashboard(apiData);
            
            // Set up periodic updates every 15 seconds
            setInterval(async () => {
                const freshData = await fetchCurrentData();
                if (freshData) {
                    updateDashboard(freshData);
                }
            }, 15000);
            
        } else {
            console.error("Failed to fetch API data");
        }
        
    } catch (error) {
        console.error("Dashboard fix failed:", error);
    }
}

// Start the fix when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', fixDashboardLive);
} else {
    fixDashboardLive();
}

// Also try to run after a delay in case React takes time to render
setTimeout(fixDashboardLive, 3000);
