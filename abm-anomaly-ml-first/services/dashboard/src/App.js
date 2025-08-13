import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
// import Layout from './Layout';
// import LayoutSimple from './LayoutSimple';
import LayoutFixed from './LayoutFixed';

// Simple error boundary component
class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error('Error caught by boundary:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div style={{ padding: '20px', background: '#ffe6e6' }}>
          <h2>Something went wrong!</h2>
          <pre style={{ background: '#fff', padding: '10px' }}>
            {this.state.error?.toString()}
          </pre>
        </div>
      );
    }
    return this.props.children;
  }
}

// Lazy load components to catch import errors
const Dashboard = React.lazy(() => import('./Dashboard').catch(err => {
  console.error('Error loading Dashboard:', err);
  return { default: () => <div>Error loading Dashboard component</div> };
}));

const RealtimeMonitoringInterface = React.lazy(() => import('./RealtimeMonitoringInterface').catch(err => {
  console.error('Error loading RealtimeMonitoringInterface:', err);
  return { default: () => <div>Error loading RealtimeMonitoringInterface component</div> };
}));

const ExpertLabelingInterface = React.lazy(() => import('./ExpertLabelingInterface').catch(err => {
  console.error('Error loading ExpertLabelingInterface:', err);
  return { default: () => <div>Error loading ExpertLabelingInterface component</div> };
}));

const MultiAnomalyView = React.lazy(() => import('./MultiAnomalyView').catch(err => {
  console.error('Error loading MultiAnomalyView:', err);
  return { default: () => <div>Error loading MultiAnomalyView component</div> };
}));

const ContinuousLearningInterface = React.lazy(() => import('./ContinuousLearningInterface').catch(err => {
  console.error('Error loading ContinuousLearningInterface:', err);
  return { default: () => <div>Error loading ContinuousLearningInterface component</div> };
}));

const DataViewer = React.lazy(() => import('./DataViewer').catch(err => {
  console.error('Error loading DataViewer:', err);
  return { default: () => <div>Error loading DataViewer component</div> };
}));

const BertAnalysisInterface = React.lazy(() => import('./BertAnalysisInterface').catch(err => {
  console.error('Error loading BertAnalysisInterface:', err);
  return { default: () => <div>Error loading BertAnalysisInterface component</div> };
}));

const DeepLogDashboard = React.lazy(() => import('./DeepLogDashboard').catch(err => {
  console.error('Error loading DeepLogDashboard:', err);
  return { default: () => <div>Error loading DeepLogDashboard component</div> };
}));

const DBSCANVisualization = React.lazy(() => import('./DBSCANVisualization').catch(err => {
  console.error('Error loading DBSCANVisualization:', err);
  return { default: () => <div>Error loading DBSCANVisualization component</div> };
}));

const SessionReview = React.lazy(() => import('./SessionReview').catch(err => {
  console.error('Error loading SessionReview:', err);
  return { default: () => <div>Error loading SessionReview component</div> };
}));

const SessionEvaluation = React.lazy(() => import('./SessionEvaluation').catch(err => {
  console.error('Error loading SessionEvaluation:', err);
  return { default: () => <div>Error loading SessionEvaluation component</div> };
}));

const AnomaliesPage = React.lazy(() => import('./AnomaliesPage').catch(err => {
  console.error('Error loading AnomaliesPage:', err);
  return { default: () => <div>Error loading AnomaliesPage component</div> };
}));

const AlertsPage = React.lazy(() => import('./AlertsPage').catch(err => {
  console.error('Error loading AlertsPage:', err);
  return { default: () => <div>Error loading AlertsPage component</div> };
}));

const OverviewPage = React.lazy(() => import('./OverviewPage').catch(err => {
  console.error('Error loading OverviewPage:', err);
  return { default: () => <div>Error loading OverviewPage component</div> };
}));

const AnalyticsPage = React.lazy(() => import('./AnalyticsPage').catch(err => {
  console.error('Error loading AnalyticsPage:', err);
  return { default: () => <div>Error loading AnalyticsPage component</div> };
}));

const CashForecasting = React.lazy(() => import('./CashForecasting').catch(err => {
  console.error('Error loading CashForecasting:', err);
  return { default: () => <div>Error loading CashForecasting component</div> };
}));

function App() {
  console.log('🟢 App component loaded successfully!');
  
  return (
    <ErrorBoundary>
      <Router>
        <React.Suspense fallback={<div style={{ padding: '20px' }}>Loading...</div>}>
          <Routes>
            <Route path="/" element={<LayoutFixed><OverviewPage /></LayoutFixed>} />
            <Route path="/overview" element={<LayoutFixed><OverviewPage /></LayoutFixed>} />
            <Route path="/anomalies" element={<LayoutFixed><AnomaliesPage /></LayoutFixed>} />
            <Route path="/alerts" element={<LayoutFixed><AlertsPage /></LayoutFixed>} />
            <Route path="/analytics" element={<LayoutFixed><AnalyticsPage /></LayoutFixed>} />
            <Route path="/cash-forecasting" element={<LayoutFixed><CashForecasting /></LayoutFixed>} />
            <Route path="/multi-anomaly" element={<LayoutFixed><MultiAnomalyView /></LayoutFixed>} />
            <Route path="/expert-labeling" element={<LayoutFixed><ExpertLabelingInterface /></LayoutFixed>} />
            <Route path="/continuous-learning" element={<LayoutFixed><ContinuousLearningInterface /></LayoutFixed>} />
            <Route path="/session-review" element={<LayoutFixed><SessionReview /></LayoutFixed>} />
            <Route path="/session-evaluation" element={<LayoutFixed><SessionEvaluation /></LayoutFixed>} />
            <Route path="/realtime" element={<LayoutFixed><RealtimeMonitoringInterface /></LayoutFixed>} />
            <Route path="/data-viewer" element={<LayoutFixed><DataViewer /></LayoutFixed>} />
            <Route path="/bert-analysis" element={<LayoutFixed><BertAnalysisInterface /></LayoutFixed>} />
            <Route path="/deeplog" element={<LayoutFixed><DeepLogDashboard /></LayoutFixed>} />
            <Route path="/dbscan" element={<DBSCANVisualization />} />
            {/* Dashboard routes with tab-based navigation */}
            <Route path="/dashboard" element={<LayoutFixed><Dashboard /></LayoutFixed>} />
            <Route path="/dashboard/overview" element={<LayoutFixed><Dashboard /></LayoutFixed>} />
            <Route path="/dashboard/anomalies" element={<LayoutFixed><Dashboard /></LayoutFixed>} />
            <Route path="/dashboard/alerts" element={<LayoutFixed><Dashboard /></LayoutFixed>} />
            <Route path="/dashboard/analytics" element={<LayoutFixed><Dashboard /></LayoutFixed>} />
            <Route path="/dashboard/multi-anomaly" element={<LayoutFixed><Dashboard /></LayoutFixed>} />
            <Route path="/dashboard/expert-labeling" element={<LayoutFixed><Dashboard /></LayoutFixed>} />
            <Route path="/dashboard/continuous-learning" element={<LayoutFixed><Dashboard /></LayoutFixed>} />
            <Route path="/dashboard/session-review" element={<LayoutFixed><Dashboard /></LayoutFixed>} />
            <Route path="/dashboard/realtime" element={<LayoutFixed><Dashboard /></LayoutFixed>} />
            <Route path="/dashboard/svm-debug" element={<LayoutFixed><Dashboard /></LayoutFixed>} />
            {/* Case-insensitive routing - uppercase variants */}
            <Route path="/Dashboard" element={<LayoutFixed><Dashboard /></LayoutFixed>} />
            <Route path="/Overview" element={<LayoutFixed><OverviewPage /></LayoutFixed>} />
            <Route path="/Anomalies" element={<LayoutFixed><AnomaliesPage /></LayoutFixed>} />
            <Route path="/Alerts" element={<LayoutFixed><AlertsPage /></LayoutFixed>} />
            <Route path="/Analytics" element={<LayoutFixed><AnalyticsPage /></LayoutFixed>} />
            <Route path="/Cash-Forecasting" element={<LayoutFixed><CashForecasting /></LayoutFixed>} />
            <Route path="/Multi-anomaly" element={<LayoutFixed><MultiAnomalyView /></LayoutFixed>} />
            <Route path="/Multi-Anomaly" element={<LayoutFixed><MultiAnomalyView /></LayoutFixed>} />
            <Route path="/Expert-labeling" element={<LayoutFixed><ExpertLabelingInterface /></LayoutFixed>} />
            <Route path="/Expert-Labeling" element={<LayoutFixed><ExpertLabelingInterface /></LayoutFixed>} />
            <Route path="/Continuous-learning" element={<LayoutFixed><ContinuousLearningInterface /></LayoutFixed>} />
            <Route path="/Continuous-Learning" element={<LayoutFixed><ContinuousLearningInterface /></LayoutFixed>} />
            <Route path="/Session-review" element={<LayoutFixed><SessionReview /></LayoutFixed>} />
            <Route path="/Session-Review" element={<LayoutFixed><SessionReview /></LayoutFixed>} />
            <Route path="/Session-evaluation" element={<LayoutFixed><SessionEvaluation /></LayoutFixed>} />
            <Route path="/Session-Evaluation" element={<LayoutFixed><SessionEvaluation /></LayoutFixed>} />
            <Route path="/Realtime" element={<LayoutFixed><RealtimeMonitoringInterface /></LayoutFixed>} />
            <Route path="/Data-viewer" element={<LayoutFixed><DataViewer /></LayoutFixed>} />
            <Route path="/Data-Viewer" element={<LayoutFixed><DataViewer /></LayoutFixed>} />
            <Route path="/Bert-analysis" element={<LayoutFixed><BertAnalysisInterface /></LayoutFixed>} />
            <Route path="/Bert-Analysis" element={<LayoutFixed><BertAnalysisInterface /></LayoutFixed>} />
            <Route path="/BERT-Analysis" element={<LayoutFixed><BertAnalysisInterface /></LayoutFixed>} />
            <Route path="/Deeplog" element={<LayoutFixed><DeepLogDashboard /></LayoutFixed>} />
            <Route path="/DeepLog" element={<LayoutFixed><DeepLogDashboard /></LayoutFixed>} />
            <Route path="/DBSCAN" element={<DBSCANVisualization />} />
            {/* Legacy /dashboard/* routes for backwards compatibility */}
            <Route path="/dashboard" element={<LayoutFixed><Dashboard /></LayoutFixed>} />
            <Route path="/dashboard/" element={<LayoutFixed><Dashboard /></LayoutFixed>} />
            <Route path="/dashboard/anomalies" element={<LayoutFixed><AnomaliesPage /></LayoutFixed>} />
            <Route path="/dashboard/alerts" element={<LayoutFixed><AlertsPage /></LayoutFixed>} />
            <Route path="/dashboard/analytics" element={<LayoutFixed><Dashboard /></LayoutFixed>} />
            <Route path="/dashboard/cash-forecasting" element={<LayoutFixed><CashForecasting /></LayoutFixed>} />
            <Route path="/dashboard/multi-anomaly" element={<LayoutFixed><MultiAnomalyView /></LayoutFixed>} />
            <Route path="/dashboard/expert-labeling" element={<LayoutFixed><ExpertLabelingInterface /></LayoutFixed>} />
            <Route path="/dashboard/continuous-learning" element={<LayoutFixed><ContinuousLearningInterface /></LayoutFixed>} />
            <Route path="/dashboard/session-review" element={<LayoutFixed><SessionReview /></LayoutFixed>} />
            <Route path="/dashboard/session-evaluation" element={<LayoutFixed><SessionEvaluation /></LayoutFixed>} />
            <Route path="/dashboard/realtime" element={<LayoutFixed><RealtimeMonitoringInterface /></LayoutFixed>} />
            <Route path="/dashboard/data-viewer" element={<LayoutFixed><DataViewer /></LayoutFixed>} />
            <Route path="/dashboard/bert-analysis" element={<LayoutFixed><BertAnalysisInterface /></LayoutFixed>} />
            <Route path="/dashboard/deeplog" element={<LayoutFixed><DeepLogDashboard /></LayoutFixed>} />
            <Route path="/dashboard/dbscan" element={<DBSCANVisualization />} />
            {/* Case-insensitive legacy routes - uppercase variants */}
            <Route path="/Dashboard/" element={<LayoutFixed><Dashboard /></LayoutFixed>} />
            <Route path="/Dashboard/anomalies" element={<LayoutFixed><Dashboard /></LayoutFixed>} />
            <Route path="/Dashboard/alerts" element={<LayoutFixed><Dashboard /></LayoutFixed>} />
            <Route path="/Dashboard/analytics" element={<LayoutFixed><Dashboard /></LayoutFixed>} />
            <Route path="/Dashboard/cash-forecasting" element={<LayoutFixed><CashForecasting /></LayoutFixed>} />
            <Route path="/Dashboard/multi-anomaly" element={<LayoutFixed><MultiAnomalyView /></LayoutFixed>} />
            <Route path="/Dashboard/expert-labeling" element={<LayoutFixed><ExpertLabelingInterface /></LayoutFixed>} />
            <Route path="/Dashboard/continuous-learning" element={<LayoutFixed><ContinuousLearningInterface /></LayoutFixed>} />
            <Route path="/Dashboard/session-review" element={<LayoutFixed><SessionReview /></LayoutFixed>} />
            <Route path="/Dashboard/session-evaluation" element={<LayoutFixed><SessionEvaluation /></LayoutFixed>} />
            <Route path="/Dashboard/realtime" element={<LayoutFixed><RealtimeMonitoringInterface /></LayoutFixed>} />
            <Route path="/Dashboard/data-viewer" element={<LayoutFixed><DataViewer /></LayoutFixed>} />
            <Route path="/Dashboard/bert-analysis" element={<LayoutFixed><BertAnalysisInterface /></LayoutFixed>} />
            <Route path="/Dashboard/deeplog" element={<LayoutFixed><DeepLogDashboard /></LayoutFixed>} />
            <Route path="/Dashboard/dbscan" element={<DBSCANVisualization />} />
          </Routes>
        </React.Suspense>
      </Router>
    </ErrorBoundary>
  );
}

export default App;
