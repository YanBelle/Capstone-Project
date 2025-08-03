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
const Dashboard = React.lazy(() => import('./SimplifiedDashboard').catch(err => {
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

function App() {
  console.log('🟢 App component loaded successfully!');
  
  return (
    <ErrorBoundary>
      <Router>
        <React.Suspense fallback={<div style={{ padding: '20px' }}>Loading...</div>}>
          <Routes>
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/dashboard/" element={<Dashboard />} />
            <Route path="/dashboard/anomalies" element={<Dashboard />} />
            <Route path="/dashboard/alerts" element={<Dashboard />} />
            <Route path="/dashboard/analytics" element={<Dashboard />} />
            <Route path="/dashboard/realtime" element={<RealtimeMonitoringInterface />} />
            <Route path="/dashboard/multi-anomaly" element={<MultiAnomalyView />} />
            <Route path="/dashboard/expert-labeling" element={<ExpertLabelingInterface />} />
            <Route path="/dashboard/continuous-learning" element={<ContinuousLearningInterface />} />
            <Route path="/dashboard/data-viewer" element={<DataViewer />} />
            <Route path="/dashboard/bert-analysis" element={<BertAnalysisInterface />} />
            <Route path="/dashboard/deeplog" element={<DeepLogDashboard />} />
            <Route path="/" element={<Dashboard />} />
          </Routes>
        </React.Suspense>
      </Router>
    </ErrorBoundary>
  );
}

export default App;
