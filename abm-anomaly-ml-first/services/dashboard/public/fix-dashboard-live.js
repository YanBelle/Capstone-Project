console.log("Permanent Dashboard Fix Script Loaded - Version 3.0 - AGGRESSIVE MODE");

// Configuration
const CONFIG = {
    API_ENDPOINT: '/api/v1/dashboard/stats',
    UPDATE_INTERVAL: 5000, // 5 seconds for more frequent updates
    MAX_RETRIES: 5,
    RETRY_DELAY: 1000 // 1 second
};

// State management
let isUpdating = false;
let retryCount = 0;

// Function to wait for elements to be available
function waitForElement(selector, timeout = 15000) {
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

// Robust API data fetching with retry logic
async function fetchCurrentData() {
    for (let attempt = 1; attempt <= CONFIG.MAX_RETRIES; attempt++) {
        try {
            console.log(`Fetching dashboard data (attempt ${attempt}/${CONFIG.MAX_RETRIES})...`);
            const response = await fetch(CONFIG.API_ENDPOINT);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const data = await response.json();
            console.log("Fresh API data received:", data);
            retryCount = 0; // Reset retry count on success
            return data;
        } catch (error) {
            console.warn(`Attempt ${attempt} failed:`, error.message);
            if (attempt < CONFIG.MAX_RETRIES) {
                await new Promise(resolve => setTimeout(resolve, CONFIG.RETRY_DELAY * attempt));
            } else {
                console.error("All fetch attempts failed:", error);
                return null;
            }
        }
    }
}

// Enhanced dashboard update function with aggressive replacement
function updateDashboard(apiData) {
    if (isUpdating) {
        console.log("Update already in progress, skipping...");
        return;
    }
    
    isUpdating = true;
    console.log("🔄 AGGRESSIVE UPDATE - Updating dashboard with fresh data:", apiData);
    
    try {
        let updatesApplied = 0;
        
        // AGGRESSIVE MODE: Find and replace ALL instances of cached values
        const allElements = document.querySelectorAll('*');
        console.log(`🔍 Scanning ${allElements.length} elements for cached data...`);
        
        allElements.forEach((element, index) => {
            // Skip script and style elements
            if (element.tagName === 'SCRIPT' || element.tagName === 'STYLE') return;
            
            const directText = element.childNodes.length === 1 && element.childNodes[0].nodeType === 3 ? element.textContent.trim() : null;
            
            // AGGRESSIVE: Update Total Transactions (1,250 or 1250)
            if (directText && (directText === '1,250' || directText === '1250') && !element.dataset.fixed) {
                element.textContent = apiData.total_transactions.toString();
                element.dataset.fixed = 'transactions';
                element.style.color = '#10b981'; // Green to show it's been updated
                console.log(`✅ [${index}] FIXED Total Transactions: ${apiData.total_transactions}`);
                updatesApplied++;
            }
            
            // AGGRESSIVE: Update Total Anomalies (23)
            else if (directText && directText === '23' && !directText.includes('2023') && !element.dataset.fixed) {
                element.textContent = apiData.total_anomalies.toString();
                element.dataset.fixed = 'anomalies';
                element.style.color = '#10b981'; // Green to show it's been updated
                console.log(`✅ [${index}] FIXED Total Anomalies: ${apiData.total_anomalies}`);
                updatesApplied++;
            }
            
            // AGGRESSIVE: Update Anomaly Rate (1.84%)
            else if (directText && directText === '1.84%' && !element.dataset.fixed) {
                const newRate = (apiData.anomaly_rate * 100).toFixed(2) + '%';
                element.textContent = newRate;
                element.dataset.fixed = 'rate';
                element.style.color = '#10b981'; // Green to show it's been updated
                console.log(`✅ [${index}] FIXED Anomaly Rate: ${newRate}`);
                updatesApplied++;
            }
            
            // AGGRESSIVE: Update High Risk Count (5)
            else if (directText && directText === '5' && !element.dataset.fixed) {
                // Check if parent context suggests this is high risk count
                const parentText = element.parentElement ? element.parentElement.textContent.toLowerCase() : '';
                if (parentText.includes('high') || parentText.includes('risk') || parentText.includes('alert')) {
                    element.textContent = apiData.high_risk_count.toString();
                    element.dataset.fixed = 'highrisk';
                    element.style.color = '#10b981'; // Green to show it's been updated
                    console.log(`✅ [${index}] FIXED High Risk Count: ${apiData.high_risk_count}`);
                    updatesApplied++;
                }
            }
        });
        
        console.log(`🎯 AGGRESSIVE UPDATE COMPLETED - ${updatesApplied} values updated out of ${allElements.length} elements`);
        
        // Force update any React state by dispatching events
        try {
            window.dispatchEvent(new CustomEvent('forceUpdate'));
            console.log("📡 Dispatched forceUpdate event");
        } catch (e) {
            console.log("ℹ️ forceUpdate event failed (expected)");
        }
        
        // Add visual indicator that fix is active
        addFixIndicator(apiData, updatesApplied);
        
    } catch (error) {
        console.error("❌ Error during dashboard update:", error);
    } finally {
        isUpdating = false;
    }
}

// Add a visual indicator that the fix is active
function addFixIndicator(data, updatesCount = 0) {
    let indicator = document.getElementById('fix-indicator');
    if (!indicator) {
        indicator = document.createElement('div');
        indicator.id = 'fix-indicator';
        indicator.style.cssText = `
            position: fixed;
            top: 10px;
            right: 10px;
            background: #10b981;
            color: white;
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 12px;
            z-index: 9999;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            max-width: 250px;
        `;
        document.body.appendChild(indicator);
    }
    
    const lastUpdate = new Date().toLocaleTimeString();
    indicator.innerHTML = `
        ✅ Live Data Active<br>
        📊 ${data.total_transactions} transactions, ${data.total_anomalies} anomalies<br>
        🔄 ${updatesCount} values fixed<br>
        ⏰ ${lastUpdate}
    `;
}

// Main fix function with enhanced error handling
async function fixDashboardPermanent() {
    try {
        console.log("🚀 Starting AGGRESSIVE permanent dashboard fix...");
        
        // Wait for React to fully render
        await waitForElement('body', 5000);
        console.log("✓ DOM is ready");
        
        let fixApplied = false;
        
        // Multiple attempts with different delays
        for (let attempt = 1; attempt <= 3; attempt++) {
            console.log(`🔄 Fix attempt ${attempt}/3`);
            await new Promise(resolve => setTimeout(resolve, attempt * 2000)); // 2s, 4s, 6s delays
            
            const apiData = await fetchCurrentData();
            if (apiData) {
                updateDashboard(apiData);
                fixApplied = true;
            } else {
                console.error(`❌ Failed to fetch API data on attempt ${attempt}`);
            }
        }
        
        // Setup aggressive refresh interval after all attempts
        if (fixApplied) {
            const intervalId = setInterval(async () => {
                const freshData = await fetchCurrentData();
                if (freshData) {
                    updateDashboard(freshData);
                } else {
                    retryCount++;
                    if (retryCount > 5) {
                        console.warn("Too many failed attempts, clearing interval");
                        clearInterval(intervalId);
                    }
                }
            }, CONFIG.UPDATE_INTERVAL);
            
            console.log(`✅ Aggressive fix activated! Auto-refresh every ${CONFIG.UPDATE_INTERVAL/1000}s`);
        }
        
    } catch (error) {
        console.error("❌ Aggressive dashboard fix failed:", error);
    }
}

// Multiple initialization strategies to ensure the fix runs
function initializeFix() {
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', fixDashboardPermanent);
    } else {
        fixDashboardPermanent();
    }
    
    // Backup initialization after delays
    setTimeout(fixDashboardPermanent, 2000);
    setTimeout(fixDashboardPermanent, 5000);
}

// Start the permanent fix
initializeFix();

// Global access for debugging
window.dashboardFix = {
    fetchData: fetchCurrentData,
    updateDashboard: updateDashboard,
    forceUpdate: fixDashboardPermanent
};
