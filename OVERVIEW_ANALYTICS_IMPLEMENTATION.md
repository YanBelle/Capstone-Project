# Overview and Analytics Pages Implementation

## Overview

This implementation provides comprehensive Overview and Analytics dashboard pages for the anomaly detection and cash forecasting system. The solution integrates anomaly detection ML pipeline with cash forecasting data to provide real-time insights.

## Architecture

### Backend API Endpoints

#### 1. Overview Stats Endpoint (`/api/v1/overview/stats`)
Provides real-time system overview including:
- **Session Statistics**: Total sessions, anomalies detected, anomaly rate
- **Alert Metrics**: High-risk alerts, critical alerts requiring immediate attention
- **Terminal Analytics**: Active terminals, terminals at risk, average cash levels
- **Cash Monitoring**: Total cash monitored, predicted depletions
- **System Health**: CPU/memory usage, database connectivity, uptime
- **Activity Timeline**: Recent anomaly detections and system events
- **Hourly Trends**: 24-hour activity patterns

#### 2. Analytics Data Endpoint (`/api/v1/analytics/data`)
Provides detailed analytics with optional timeframe filtering:
- **Model Performance**: Accuracy, precision, recall, F1 score, feature importance
- **Terminal Analytics**: Risk distribution, performance metrics, high-risk terminals
- **Pattern Analysis**: Detected patterns, confidence scores, recent pattern timeline
- **Cash Analytics**: Usage trends, forecast accuracy, low cash terminals
- **Risk Assessment**: Overall risk scores, mitigation recommendations, trend analysis

### Frontend Components

#### 1. OverviewPage Component (`OverviewPage.js`)
- **Real-time Metrics**: Key performance indicators with trend indicators
- **Interactive Charts**: 24-hour activity trends using recharts library
- **System Health Monitor**: Real-time status of all system components
- **Recent Activity Feed**: Live feed of anomaly detections and alerts
- **Terminal Status Summary**: Visual representation of terminal health
- **Auto-refresh**: Updates every 5 minutes with manual refresh option

#### 2. AnalyticsPage Component (`AnalyticsPage.js`)
- **Tabbed Interface**: Organized analytics across 5 main categories
- **Interactive Visualizations**: Charts, graphs, and scatter plots
- **Timeframe Selection**: 1h, 24h, 7d, 30d filtering options
- **Model Performance Tracking**: ML model accuracy and feature importance
- **Risk Assessment Dashboard**: Comprehensive risk analysis and recommendations
- **Auto-refresh**: Updates every 10 minutes with configurable timeframes

## Key Features

### 1. Data Models

#### OverviewStats Model
```python
class OverviewStats(BaseModel):
    total_sessions: int
    total_anomalies: int
    anomaly_rate: float
    high_risk_count: int
    critical_alerts: int
    recent_activity: List[ActivityItem]
    hourly_trend: List[HourlyTrend]
    terminal_summary: TerminalSummary
    system_health: SystemHealth
    cash_summary: CashSummary
```

#### AnalyticsData Model
```python
class AnalyticsData(BaseModel):
    anomaly_trends: List[AnomalyTrend]
    model_performance: ModelPerformance
    terminal_analytics: TerminalAnalytics
    pattern_analysis: PatternAnalysis
    cash_analytics: CashAnalytics
    risk_assessment: RiskAssessment
    operational_metrics: OperationalMetrics
```

### 2. Integration Points

- **Database Integration**: PostgreSQL queries for ml_sessions and cassette_counters tables
- **Redis Caching**: Real-time dashboard updates and session management
- **Cash Forecasting**: Seamless integration with existing cash forecasting system
- **Alert System**: Real-time anomaly detection alerts and notifications

### 3. Error Handling & Fallbacks

- **Graceful Degradation**: Sample data fallbacks when database is unavailable
- **Error Boundaries**: React error boundaries for component-level error handling
- **Retry Mechanisms**: Automatic retry with user-initiated refresh options
- **Loading States**: Comprehensive loading indicators and progress feedback

## Installation & Setup

### 1. Backend Setup
The API endpoints are automatically available in the existing FastAPI service at `/services/api/main.py`. No additional setup required.

### 2. Frontend Setup
```bash
cd /services/dashboard
npm install  # Dependencies already installed (recharts, lucide-react)
npm start    # Development server
```

### 3. Testing API Endpoints
```bash
./test_overview_analytics_api.sh
```

## Usage

### Accessing the Pages

#### Direct Routes:
- **Overview**: `http://localhost:3000/overview` or `http://localhost:3000/`
- **Analytics**: `http://localhost:3000/analytics`

#### Dashboard Tab Navigation:
- **Overview**: `http://localhost:3000/dashboard` (default tab)
- **Analytics**: `http://localhost:3000/dashboard/analytics`

### Navigation Features

1. **Tab-based Interface**: Seamless switching between Overview and Analytics
2. **Timeframe Selection**: Configurable data timeframes (1h, 24h, 7d, 30d)
3. **Auto-refresh**: Background data updates with manual refresh options
4. **Responsive Design**: Mobile-friendly interface with adaptive layouts

## Technical Implementation

### Data Flow

1. **Frontend Request** → React component mounts
2. **API Call** → Fetch data from FastAPI endpoints
3. **Database Query** → PostgreSQL data retrieval with Redis caching
4. **Data Processing** → Transform raw data into visualization-ready format
5. **Component Update** → Re-render with new data and charts
6. **Auto-refresh** → Periodic background updates

### Performance Optimizations

- **Lazy Loading**: React.lazy() for component code splitting
- **Data Caching**: Redis cache for frequently accessed data
- **Efficient Queries**: Optimized PostgreSQL queries with proper indexing
- **Chart Optimization**: Recharts with responsive containers
- **Memory Management**: Proper cleanup of intervals and subscriptions

## Configuration

### Environment Variables
```bash
REACT_APP_API_URL=http://localhost:8000  # Backend API URL
```

### API Configuration
- Overview refresh interval: 5 minutes (300,000ms)
- Analytics refresh interval: 10 minutes (600,000ms)
- Configurable in component state

## Monitoring & Debugging

### Health Checks
- Database connectivity status
- Redis cache status
- API endpoint availability
- System resource monitoring

### Error Logging
- Console error logging for debugging
- Error boundary components for graceful failures
- API error handling with user feedback

## Integration with Existing System

### Compatibility
- **Dashboard.js**: Updated to use new components with tab routing
- **App.js**: Enhanced routing with direct and dashboard-based navigation
- **LayoutFixed**: Maintains existing navigation and layout structure
- **API Structure**: Extends existing FastAPI service without breaking changes

### Migration Path
- Existing dashboard functionality preserved
- New components work alongside existing pages
- Backward compatibility maintained for all routes

## Future Enhancements

### Planned Features
1. **Real-time WebSocket Updates**: Live data streaming for instant updates
2. **Advanced Filtering**: Custom date ranges and advanced filter options
3. **Export Functionality**: PDF/Excel export of analytics reports
4. **Custom Dashboards**: User-configurable dashboard layouts
5. **Alert Configuration**: Customizable alert thresholds and notifications
6. **Historical Analysis**: Extended historical data analysis and trending

### Scalability Considerations
- **Database Optimization**: Query optimization for large datasets
- **Caching Strategy**: Advanced Redis caching patterns
- **Load Balancing**: Multi-instance deployment support
- **Data Archival**: Historical data management strategies

## Files Modified/Created

### New Files
- `/services/dashboard/src/OverviewPage.js` - Overview dashboard component
- `/services/dashboard/src/AnalyticsPage.js` - Analytics dashboard component
- `/test_overview_analytics_api.sh` - API testing script

### Modified Files
- `/services/api/main.py` - Added Overview and Analytics API endpoints
- `/services/dashboard/src/Dashboard.js` - Updated tab routing
- `/services/dashboard/src/App.js` - Enhanced route configuration

## API Documentation

### Overview Stats Endpoint
```
GET /api/v1/overview/stats
Response: OverviewStats model with comprehensive system metrics
```

### Analytics Data Endpoint
```
GET /api/v1/analytics/data?timeframe=24h
Parameters:
  - timeframe: 1h, 24h, 7d, 30d (optional, default: 24h)
Response: AnalyticsData model with detailed analytics
```

## Support & Maintenance

For issues or questions regarding the Overview and Analytics implementation:
1. Check API endpoint availability using the test script
2. Verify database connectivity and Redis cache status
3. Review browser console for frontend errors
4. Confirm environment variables are properly configured

The implementation is designed to be maintainable and extensible, with clear separation of concerns and comprehensive error handling.
