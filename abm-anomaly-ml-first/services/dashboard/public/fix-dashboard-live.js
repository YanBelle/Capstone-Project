console.log("Enhanced Dashboard Fix Script with Charts - Version 4.0");

// Configuration - Environment-aware API endpoints
const getApiBaseUrl = () => {
    // Check if we're in production (served through nginx) or development
    const isProduction = window.location.hostname !== 'localhost' || window.location.port === '80' || window.location.port === '';
    return isProduction ? '' : 'http://localhost:8000';
};

const CONFIG = {
    API_ENDPOINT: `${getApiBaseUrl()}/api/v1/dashboard/stats`,
    CASH_FORECASTING_ENDPOINT: `${getApiBaseUrl()}/api/cash-forecasting`,
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

// Enhanced dashboard update function with selective replacement (non-intrusive for cash forecasting)
function updateDashboard(apiData) {
    if (isUpdating) {
        console.log("Update already in progress, skipping...");
        return;
    }
    
    // Skip aggressive updates on cash forecasting pages to avoid text conflicts
    if (shouldInitializeCharts()) {
        console.log("Skipping aggressive text updates on cash forecasting page");
        addFixIndicator(apiData, 0);
        return;
    }
    
    isUpdating = true;
    console.log("🔄 SELECTIVE UPDATE - Updating dashboard with fresh data:", apiData);
    
    try {
        let updatesApplied = 0;
        
        // SELECTIVE MODE: Only update specific elements on non-cash-forecasting pages
        const allElements = document.querySelectorAll('*');
        console.log(`🔍 Scanning ${allElements.length} elements for cached data...`);
        
        allElements.forEach((element, index) => {
            // Skip script and style elements
            if (element.tagName === 'SCRIPT' || element.tagName === 'STYLE') return;
            
            const directText = element.childNodes.length === 1 && element.childNodes[0].nodeType === 3 ? element.textContent.trim() : null;
            
            // Only update if on overview/main dashboard page (not cash forecasting)
            if (directText && (directText === '1,250' || directText === '1250') && !element.dataset.fixed) {
                element.textContent = apiData.total_transactions.toString();
                element.dataset.fixed = 'transactions';
                element.style.color = '#10b981'; 
                console.log(`✅ [${index}] FIXED Total Transactions: ${apiData.total_transactions}`);
                updatesApplied++;
            }
            
            else if (directText && directText === '23' && !directText.includes('2023') && !element.dataset.fixed) {
                element.textContent = apiData.total_anomalies.toString();
                element.dataset.fixed = 'anomalies';
                element.style.color = '#10b981'; 
                console.log(`✅ [${index}] FIXED Total Anomalies: ${apiData.total_anomalies}`);
                updatesApplied++;
            }
            
            else if (directText && directText === '1.84%' && !element.dataset.fixed) {
                const newRate = (apiData.anomaly_rate * 100).toFixed(2) + '%';
                element.textContent = newRate;
                element.dataset.fixed = 'rate';
                element.style.color = '#10b981'; 
                console.log(`✅ [${index}] FIXED Anomaly Rate: ${newRate}`);
                updatesApplied++;
            }
            
            else if (directText && directText === '5' && !element.dataset.fixed) {
                const parentText = element.parentElement ? element.parentElement.textContent.toLowerCase() : '';
                if (parentText.includes('high') || parentText.includes('risk') || parentText.includes('alert')) {
                    element.textContent = apiData.high_risk_count.toString();
                    element.dataset.fixed = 'highrisk';
                    element.style.color = '#10b981'; 
                    console.log(`✅ [${index}] FIXED High Risk Count: ${apiData.high_risk_count}`);
                    updatesApplied++;
                }
            }
        });
        
        console.log(`🎯 SELECTIVE UPDATE COMPLETED - ${updatesApplied} values updated out of ${allElements.length} elements`);
        
        // Add visual indicator that fix is active
        addFixIndicator(apiData, updatesApplied);
        
    } catch (error) {
        console.error("❌ Error during dashboard update:", error);
    } finally {
        isUpdating = false;
    }
}

// Add a minimal visual indicator that the fix is active (non-intrusive)
function addFixIndicator(data, updatesCount = 0) {
    // Only show indicator on non-cash-forecasting pages or if explicitly needed
    const shouldShowIndicator = !shouldInitializeCharts() || window.location.search.includes('debug=true');
    
    if (!shouldShowIndicator) {
        // Remove any existing indicator on cash forecasting pages
        const existingIndicator = document.getElementById('fix-indicator');
        if (existingIndicator) {
            existingIndicator.remove();
        }
        return;
    }
    
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
            padding: 6px 10px;
            border-radius: 6px;
            font-size: 11px;
            z-index: 9999;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            max-width: 200px;
            opacity: 0.8;
            font-family: monospace;
        `;
        document.body.appendChild(indicator);
    }
    
    const lastUpdate = new Date().toLocaleTimeString();
    indicator.innerHTML = `
        ✅ Live: ${data.total_transactions}/${data.total_anomalies}<br>
        🔄 ${updatesCount} updates | ${lastUpdate}
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

// Create enhanced chart section HTML
function createChartSection(terminalId) {
    return `
        <div class="visualization-section" id="charts-${terminalId}">
            <div class="visualization-header">
                <h3 class="chart-title">
                    <span class="chart-icon">📊</span>
                    Terminal ${terminalId} Analytics Dashboard
                </h3>
                <div class="chart-status">
                    <span class="status-indicator active"></span>
                    Real-time data
                </div>
            </div>
            
            <div class="charts-container">
                <div class="chart-grid">
                    <div class="chart-card primary">
                        <div class="chart-header">
                            <h4>💰 Cash Level Trend (48 Hours)</h4>
                            <div class="chart-info">Historical cash depletion pattern</div>
                        </div>
                        <div class="chart-wrapper">
                            <canvas id="historical-chart-${terminalId}" width="400" height="250"></canvas>
                        </div>
                    </div>
                    
                    <div class="chart-card secondary">
                        <div class="chart-header">
                            <h4>📅 Daily Average (7 Days)</h4>
                            <div class="chart-info">Weekly cash level trends</div>
                        </div>
                        <div class="chart-wrapper">
                            <canvas id="trend-chart-${terminalId}" width="400" height="250"></canvas>
                        </div>
                    </div>
                    
                    <div class="chart-card tertiary">
                        <div class="chart-header">
                            <h4>⏰ Usage by Hour</h4>
                            <div class="chart-info">Transaction patterns throughout the day</div>
                        </div>
                        <div class="chart-wrapper">
                            <canvas id="usage-chart-${terminalId}" width="400" height="250"></canvas>
                        </div>
                    </div>
                    
                    <div class="chart-card quaternary">
                        <div class="chart-header">
                            <h4>🔮 7-Day Predictions</h4>
                            <div class="chart-info">AI-powered cash depletion forecast</div>
                        </div>
                        <div class="chart-wrapper">
                            <canvas id="predictions-chart-${terminalId}" width="400" height="250"></canvas>
                        </div>
                    </div>
                </div>
                
                <div class="chart-summary">
                    <div class="summary-item">
                        <span class="summary-label">Current Status:</span>
                        <span class="summary-value operational">Operational</span>
                    </div>
                    <div class="summary-item">
                        <span class="summary-label">Data Source:</span>
                        <span class="summary-value">Real-time API</span>
                    </div>
                    <div class="summary-item">
                        <span class="summary-label">Last Updated:</span>
                        <span class="summary-value">${new Date().toLocaleTimeString()}</span>
                    </div>
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

// Enhanced chart injection - integrated with React components
function immediateChartInjection() {
    console.log('Starting enhanced chart injection...');
    
    // Wait for React components to load
    setTimeout(() => {
        injectChartsIntoTerminalCards();
    }, 1000);
    
    // Also inject full charts section at the bottom
    const chartHTML = `
        <div class="visualization-dashboard" style="margin: 30px 0; padding: 25px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white;">
            <div class="viz-header" style="text-align: center; margin-bottom: 30px;">
                <h2 style="margin: 0; font-size: 2rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">📊 Advanced Analytics Dashboard</h2>
                <p style="margin: 10px 0 0 0; opacity: 0.9;">Real-time terminal monitoring and predictive analytics</p>
            </div>
            
            <div id="charts-416" style="background: rgba(255,255,255,0.95); border-radius: 12px; padding: 20px; color: #333;">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 20px; padding-bottom: 15px; border-bottom: 2px solid #667eea;">
                    <h3 style="margin: 0; color: #667eea; display: flex; align-items: center;">
                        <span style="margin-right: 10px;">🏛️</span>
                        Terminal 416 - Comprehensive Analytics
                    </h3>
                    <div class="chart-controls" style="display: flex; gap: 10px;">
                        <button onclick="refreshCharts()" style="padding: 8px 16px; background: #667eea; color: white; border: none; border-radius: 6px; cursor: pointer;">
                            🔄 Refresh
                        </button>
                        <button onclick="exportChartData()" style="padding: 8px 16px; background: #28a745; color: white; border: none; border-radius: 6px; cursor: pointer;">
                            📥 Export
                        </button>
                    </div>
                </div>
                
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(450px, 1fr)); gap: 25px; margin: 20px 0;">
                    <div class="chart-container" style="background: #f8f9fa; border-radius: 10px; padding: 20px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
                        <h4 style="margin: 0 0 15px 0; color: #495057; display: flex; align-items: center;">
                            <span style="margin-right: 8px;">💰</span>
                            Historical Cash Levels (48 Hours)
                        </h4>
                        <div style="position: relative;">
                            <canvas id="historical-chart-416" width="400" height="250"></canvas>
                        </div>
                    </div>
                    
                    <div class="chart-container" style="background: #f8f9fa; border-radius: 10px; padding: 20px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
                        <h4 style="margin: 0 0 15px 0; color: #495057; display: flex; align-items: center;">
                            <span style="margin-right: 8px;">📊</span>
                            Daily Trends (7 Days)
                        </h4>
                        <div style="position: relative;">
                            <canvas id="trend-chart-416" width="400" height="250"></canvas>
                        </div>
                    </div>
                    
                    <div class="chart-container" style="background: #f8f9fa; border-radius: 10px; padding: 20px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
                        <h4 style="margin: 0 0 15px 0; color: #495057; display: flex; align-items: center;">
                            <span style="margin-right: 8px;">⏰</span>
                            Usage by Hour
                        </h4>
                        <div style="position: relative;">
                            <canvas id="usage-chart-416" width="400" height="250"></canvas>
                        </div>
                    </div>
                    
                    <div class="chart-container" style="background: #f8f9fa; border-radius: 10px; padding: 20px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
                        <h4 style="margin: 0 0 15px 0; color: #495057; display: flex; align-items: center;">
                            <span style="margin-right: 8px;">🔮</span>
                            7-Day Predictions
                        </h4>
                        <div style="position: relative;">
                            <canvas id="predictions-chart-416" width="400" height="250"></canvas>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    // Inject at the bottom of the dashboard
    const body = document.body;
    const container = document.querySelector('.cash-forecasting-dashboard, .dashboard-container, .container, .main-content, main') || body;
    
    container.insertAdjacentHTML('beforeend', chartHTML);
    console.log('Enhanced chart HTML injected');
    
    return true;
}

// Inject charts into terminal cards
function injectChartsIntoTerminalCards() {
    try {
        console.log('Looking for terminal chart placeholders...');
        
        // Find all terminal chart placeholders
        const terminalPlaceholders = document.querySelectorAll('.terminal-chart-placeholder');
        console.log(`Found ${terminalPlaceholders.length} terminal chart placeholders`);
        
        if (terminalPlaceholders.length > 0) {
            // Inject mini charts into each terminal placeholder
            terminalPlaceholders.forEach((placeholder, index) => {
                const terminalId = placeholder.id.replace('terminal-chart-', '') || '416';
                console.log(`Injecting mini chart for terminal ${terminalId}`);
                
                // Create compact chart for terminal card
                const compactChart = `
                    <div class="terminal-chart-container" style="margin-top: 15px; padding: 12px; background: #f8f9fa; border-radius: 8px; border-left: 4px solid #667eea;">
                        <div class="mini-chart-header" style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                            <span class="chart-title" style="font-weight: 600; color: #495057; display: flex; align-items: center;">
                                <span style="margin-right: 6px;">📊</span>
                                Quick Analytics
                            </span>
                            <button class="expand-chart-btn" onclick="scrollToFullCharts()" style="padding: 4px 8px; background: #667eea; color: white; border: none; border-radius: 4px; font-size: 0.75rem; cursor: pointer;">
                                View Details
                            </button>
                        </div>
                        <div class="chart-status" style="font-size: 0.8rem; color: #6c757d; text-align: center;">
                            📈 Live data available in detailed view below
                        </div>
                    </div>
                `;
                
                placeholder.innerHTML = compactChart;
            });
        }
        
        return true;
    } catch (error) {
        console.error('Error injecting charts into terminal cards:', error);
        return false;
    }
}

// Helper functions for chart controls
window.refreshCharts = function() {
    console.log('Refreshing charts...');
    fetchVisualizationData('416').then(data => {
        if (data) initializeTerminalCharts('416', data);
    });
};

window.exportChartData = function() {
    window.open('/api/cash-forecasting/visualization-data/416', '_blank');
};

window.scrollToFullCharts = function() {
    const chartsSection = document.querySelector('#charts-416');
    if (chartsSection) {
        chartsSection.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
};

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

// Only initialize charts if we're on a cash-forecasting page
function shouldInitializeCharts() {
    const path = window.location.pathname;
    return path.includes('cash-forecasting') || path.includes('Cash-Forecasting');
}

// Initialize charts after a delay to let the dashboard load - but only on cash forecasting pages
setTimeout(() => {
    if (shouldInitializeCharts()) {
        console.log('Starting charts initialization on cash forecasting page...');
        addChartsToTerminals();
    } else {
        console.log('Skipping chart initialization - not on cash forecasting page');
    }
}, 3000);

// Force chart creation after a longer delay - but only on cash forecasting pages
setTimeout(() => {
    if (shouldInitializeCharts()) {
        console.log('Starting forced chart creation on cash forecasting page...');
        forceChartCreation();
    }
}, 5000);

// Immediate injection after very short delay - but only on cash forecasting pages
setTimeout(() => {
    if (shouldInitializeCharts()) {
        console.log('Starting immediate chart injection on cash forecasting page...');
        immediateChartInjection();
        setTimeout(forceChartCreation, 2000);
    }
}, 1000);

// Also try charts on page interaction - but only on cash forecasting pages
document.addEventListener('click', () => {
    if (!chartsInitialized && shouldInitializeCharts()) {
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
