import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';

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
// Lazy load components to catch import errors
const Dashboard = React.lazy(() => import('./Dashboard').catch(err => {
  console.error('Error loading Dashboard:', err);
  return { default: () => <div>Error loading Dashboard component</div> };
}));

const SimpleDashboard = React.lazy(() => import('./SimpleDashboard').catch(err => {
  console.error('Error loading SimpleDashboard:', err);
  return { default: () => <div>Error loading SimpleDashboard component</div> };
}));

const Layout = React.lazy(() => import('./Layout').catch(err => {
  console.error('Error loading Layout:', err);
  return { default: () => <div>Error loading Layout component</div> };
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

function App() {
  console.log('🟢 App component loaded successfully!');
  
  return (
    <ErrorBoundary>
      <Router>
        <React.Suspense fallback={<div style={{ padding: '20px' }}>Loading...</div>}>
          <Routes>
            <Route path="/dashboard" element={<Layout><Dashboard /></Layout>} />
            <Route path="/dashboard/" element={<Layout><Dashboard /></Layout>} />
            <Route path="/dashboard/anomalies" element={<Layout><Dashboard /></Layout>} />
            <Route path="/dashboard/alerts" element={<Layout><Dashboard /></Layout>} />
            <Route path="/dashboard/analytics" element={<Layout><Dashboard /></Layout>} />
            <Route path="/dashboard/multi-anomaly" element={<Layout><MultiAnomalyView /></Layout>} />
            <Route path="/dashboard/expert-labeling" element={<Layout><ExpertLabelingInterface /></Layout>} />
            <Route path="/dashboard/continuous-learning" element={<Layout><ContinuousLearningInterface /></Layout>} />
            <Route path="/dashboard/session-review" element={<Layout><SessionReview /></Layout>} />
            <Route path="/dashboard/realtime" element={<Layout><RealtimeMonitoringInterface /></Layout>} />
            <Route path="/dashboard/data-viewer" element={<Layout><DataViewer /></Layout>} />
            <Route path="/dashboard/bert-analysis" element={<Layout><BertAnalysisInterface /></Layout>} />
            <Route path="/dashboard/deeplog" element={<Layout><DeepLogDashboard /></Layout>} />
            <Route path="/dashboard/dbscan" element={<DBSCANVisualization />} />
            <Route path="/" element={<Dashboard />} />
          </Routes>
        </React.Suspense>
      </Router>
    </ErrorBoundary>
  );
}

export default App;
