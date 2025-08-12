console.log("Enhanced Dashboard Fix Script with Charts - Version 4.0");

// Configuration
const CONFIG = {
    API_ENDPOINT: '/api/v1/dashboard/stats',
    CASH_FORECASTING_ENDPOINT: '/api/cash-forecasting',
    UPDATE_INTERVAL: 5000, // 5 seconds for more frequent updates
    MAX_RETRIES: 5,
    RETRY_DELAY: 1000 // 1 second
};

// State management
let isUpdating = false;
let retryCount = 0;
let chartsInitialized = false;
let currentCharts = {};

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

// Chart.js loading and initialization
function loadChartJS() {
    return new Promise((resolve, reject) => {
        if (window.Chart) {
            resolve();
            return;
        }
        
        const script = document.createElement('script');
        script.src = 'https://cdn.jsdelivr.net/npm/chart.js';
        script.onload = () => {
            console.log('Chart.js loaded successfully');
            resolve();
        };
        script.onerror = () => reject(new Error('Failed to load Chart.js'));
        document.head.appendChild(script);
    });
}

// Fetch visualization data for a terminal
async function fetchVisualizationData(terminalId) {
    try {
        const response = await fetch(`${CONFIG.CASH_FORECASTING_ENDPOINT}/visualization-data/${terminalId}`);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        const data = await response.json();
        console.log(`Visualization data loaded for terminal ${terminalId}:`, data);
        return data;
    } catch (error) {
        console.error(`Failed to fetch visualization data for terminal ${terminalId}:`, error);
        return null;
    }
}

// Create chart section HTML
function createChartSection(terminalId) {
    return `
        <div class="chart-section" id="charts-${terminalId}" style="margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 8px; background: #f9f9f9;">
            <h3 style="color: #333; margin-bottom: 15px;">Terminal ${terminalId} - Cash Forecasting Charts</h3>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px;">
                <div>
                    <h4>Historical Cash Levels (48h)</h4>
                    <canvas id="historical-chart-${terminalId}" width="400" height="200"></canvas>
                </div>
                <div>
                    <h4>Daily Trend (7 days)</h4>
                    <canvas id="trend-chart-${terminalId}" width="400" height="200"></canvas>
                </div>
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                <div>
                    <h4>Usage by Hour</h4>
                    <canvas id="usage-chart-${terminalId}" width="400" height="200"></canvas>
                </div>
                <div>
                    <h4>7-Day Predictions</h4>
                    <canvas id="predictions-chart-${terminalId}" width="400" height="200"></canvas>
                </div>
            </div>
        </div>
    `;
}

// Initialize charts for a terminal
async function initializeTerminalCharts(terminalId, data) {
    if (!data || !data.charts) {
        console.error(`No chart data available for terminal ${terminalId}`);
        return;
    }

    const charts = data.charts;
    
    // Historical Cash Levels Chart
    const histCtx = document.getElementById(`historical-chart-${terminalId}`);
    if (histCtx) {
        const histData = charts.historical_cash_levels || [];
        currentCharts[`historical-${terminalId}`] = new Chart(histCtx, {
            type: 'line',
            data: {
                labels: histData.map(d => new Date(d.timestamp).toLocaleDateString()),
                datasets: [{
                    label: 'Cash Level',
                    data: histData.map(d => d.cash_level),
                    borderColor: 'rgb(75, 192, 192)',
                    backgroundColor: 'rgba(75, 192, 192, 0.2)',
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: { display: true, text: 'Cash Amount' }
                    }
                }
            }
        });
    }

    // Daily Trend Chart
    const trendCtx = document.getElementById(`trend-chart-${terminalId}`);
    if (trendCtx) {
        const trendData = charts.daily_trend || [];
        currentCharts[`trend-${terminalId}`] = new Chart(trendCtx, {
            type: 'bar',
            data: {
                labels: trendData.map(d => d.date),
                datasets: [{
                    label: 'Average Cash',
                    data: trendData.map(d => d.average_cash),
                    backgroundColor: 'rgba(153, 102, 255, 0.6)',
                    borderColor: 'rgba(153, 102, 255, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: { display: true, text: 'Average Cash' }
                    }
                }
            }
        });
    }

    // Usage by Hour Chart
    const usageCtx = document.getElementById(`usage-chart-${terminalId}`);
    if (usageCtx) {
        const usageData = charts.usage_by_hour || {};
        const hours = Object.keys(usageData).sort((a, b) => parseInt(a) - parseInt(b));
        currentCharts[`usage-${terminalId}`] = new Chart(usageCtx, {
            type: 'bar',
            data: {
                labels: hours.map(h => `${h}:00`),
                datasets: [{
                    label: 'Average Dispensed',
                    data: hours.map(h => usageData[h]?.average_dispensed || 0),
                    backgroundColor: 'rgba(255, 99, 132, 0.6)',
                    borderColor: 'rgba(255, 99, 132, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: { display: true, text: 'Amount Dispensed' }
                    }
                }
            }
        });
    }

    // Predictions Chart
    const predCtx = document.getElementById(`predictions-chart-${terminalId}`);
    if (predCtx) {
        const predData = charts.predictions || [];
        currentCharts[`predictions-${terminalId}`] = new Chart(predCtx, {
            type: 'line',
            data: {
                labels: predData.map(d => d.date),
                datasets: [{
                    label: 'Predicted Cash',
                    data: predData.map(d => d.predicted_cash),
                    borderColor: 'rgb(255, 159, 64)',
                    backgroundColor: 'rgba(255, 159, 64, 0.2)',
                    tension: 0.1
                }, {
                    label: 'Confidence',
                    data: predData.map(d => d.confidence * 50000), // Scale confidence for visibility
                    borderColor: 'rgb(54, 162, 235)',
                    backgroundColor: 'rgba(54, 162, 235, 0.2)',
                    yAxisID: 'y1',
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: { display: true, text: 'Predicted Cash' }
                    },
                    y1: {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        title: { display: true, text: 'Confidence (scaled)' },
                        grid: {
                            drawOnChartArea: false,
                        },
                    }
                }
            }
        });
    }

    console.log(`Charts initialized for terminal ${terminalId}`);
}

// Add charts to the dashboard
async function addChartsToTerminals() {
    try {
        await loadChartJS();
        
        // Find all terminal sections
        const terminalElements = document.querySelectorAll('[data-terminal-id], .terminal-info');
        console.log(`Found ${terminalElements.length} terminal elements`);
        
        // Also try to find terminals by looking for ATM or Terminal text patterns
        const allElements = document.querySelectorAll('*');
        const terminalIds = new Set();
        
        allElements.forEach(el => {
            const text = el.textContent || '';
            const matches = text.match(/(?:ATM|Terminal)\s*(\d+)/gi);
            if (matches) {
                matches.forEach(match => {
                    const id = match.replace(/(?:ATM|Terminal)\s*/gi, '');
                    if (id && !isNaN(id)) {
                        terminalIds.add(id);
                    }
                });
            }
        });

        console.log(`Found terminal IDs: ${Array.from(terminalIds).join(', ')}`);
        
        // Create charts for each found terminal
        for (const terminalId of terminalIds) {
            const existingChart = document.getElementById(`charts-${terminalId}`);
            if (existingChart) {
                continue; // Already exists
            }
            
            // Find a good place to insert the charts (after terminal info)
            const terminalText = `ATM${terminalId}`;
            const terminalElements = Array.from(document.querySelectorAll('*')).filter(el => 
                el.textContent && el.textContent.includes(terminalText)
            );
            
            if (terminalElements.length > 0) {
                const insertPoint = terminalElements[0].closest('.terminal-section, .card, .panel, .row, div') || terminalElements[0].parentElement;
                if (insertPoint) {
                    insertPoint.insertAdjacentHTML('afterend', createChartSection(terminalId));
                    
                    // Fetch data and initialize charts
                    const data = await fetchVisualizationData(terminalId);
                    if (data) {
                        await initializeTerminalCharts(terminalId, data);
                    }
                }
            }
        }
        
        chartsInitialized = true;
        console.log('Charts initialization completed');
        
    } catch (error) {
        console.error('Failed to initialize charts:', error);
    }
}

// Start the permanent fix
initializeFix();

// Immediate chart injection - very aggressive approach
function immediateChartInjection() {
    console.log('Starting immediate chart injection...');
    
    // Inject charts at the very bottom of the page
    const chartHTML = `
        <div style="padding: 20px; background: #f5f5f5; margin: 20px 0; border-radius: 8px;">
            <h2>Cash Forecasting Visualizations</h2>
            <p>Real-time charts for ATM terminals</p>
            
            <div id="charts-416">
                <h3>Terminal 416 Charts</h3>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0;">
                    <div>
                        <h4>Historical Cash Levels</h4>
                        <canvas id="historical-chart-416" width="400" height="200"></canvas>
                    </div>
                    <div>
                        <h4>Daily Trends</h4>
                        <canvas id="trend-chart-416" width="400" height="200"></canvas>
                    </div>
                    <div>
                        <h4>Usage by Hour</h4>
                        <canvas id="usage-chart-416" width="400" height="200"></canvas>
                    </div>
                    <div>
                        <h4>7-Day Predictions</h4>
                        <canvas id="predictions-chart-416" width="400" height="200"></canvas>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    // Try multiple injection points
    const body = document.body;
    const container = document.querySelector('.container, .main-content, main') || body;
    
    container.insertAdjacentHTML('beforeend', chartHTML);
    console.log('Chart HTML injected');
    
    return true;
}

// Simple forced chart initialization for known terminals
async function forceChartCreation() {
    try {
        await loadChartJS();
        console.log('Chart.js loaded, creating charts for known terminals');
        
        // First inject the HTML
        immediateChartInjection();
        
        const knownTerminals = ['416'];
        
        for (const terminalId of knownTerminals) {
            console.log(`Creating charts for terminal ${terminalId}`);
            
            try {
                const data = await fetchVisualizationData(terminalId);
                if (data && data.charts) {
                    await initializeTerminalCharts(terminalId, data);
                    console.log(`Charts initialized for terminal ${terminalId}`);
                } else {
                    console.warn(`No data received for terminal ${terminalId}`);
                }
            } catch (error) {
                console.error(`Failed to create charts for terminal ${terminalId}:`, error);
            }
        }
        
        chartsInitialized = true;
        console.log('Forced chart creation completed');
        
    } catch (error) {
        console.error('Failed to force chart creation:', error);
    }
}

// Initialize charts after a delay to let the dashboard load
setTimeout(() => {
    console.log('Starting charts initialization...');
    addChartsToTerminals();
}, 3000);

// Force chart creation after a longer delay
setTimeout(() => {
    console.log('Starting forced chart creation...');
    forceChartCreation();
}, 5000);

// Immediate injection after very short delay
setTimeout(() => {
    console.log('Starting immediate chart injection...');
    immediateChartInjection();
    setTimeout(forceChartCreation, 2000);
}, 1000);

// Also try charts on page interaction
document.addEventListener('click', () => {
    if (!chartsInitialized) {
        setTimeout(() => {
            addChartsToTerminals();
            setTimeout(forceChartCreation, 1000);
        }, 1000);
    }
});

// Global access for debugging
window.dashboardFix = {
    fetchData: fetchCurrentData,
    updateDashboard: updateDashboard,
    forceUpdate: fixDashboardPermanent,
    initCharts: addChartsToTerminals,
    forceCharts: forceChartCreation,
    injectCharts: immediateChartInjection
};
