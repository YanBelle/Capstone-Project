import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Dashboard from './Dashboard';
import RealtimeMonitoringInterface from './RealtimeMonitoringInterface';
import ExpertLabelingInterface from './ExpertLabelingInterface';
import MultiAnomalyView from './MultiAnomalyView';
import ContinuousLearningInterface from './ContinuousLearningInterface';
import DataViewer from './DataViewer';

function App() {
  return (
    <Router basename="/dashboard">
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/realtime" element={<RealtimeMonitoringInterface />} />
        <Route path="/multi-anomaly" element={<MultiAnomalyView />} />
        <Route path="/expert-labeling" element={<ExpertLabelingInterface />} />
        <Route path="/continuous-learning" element={<ContinuousLearningInterface />} />
        <Route path="/data-viewer" element={<DataViewer />} />
      </Routes>
    </Router>
  );
}

export default App;
