import React, { useState, useEffect } from 'react';
import { Tab, Tabs, TabList, TabPanel } from 'react-tabs';
import 'react-tabs/style/react-tabs.css';
import './App.css';
import Overview from './components/Overview';
import Training from './components/Training';
import Prediction from './components/Prediction';
import DBSCANVisualization from './components/DBSCANVisualization';

function App() {
  const [modelInfo, setModelInfo] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Check model status on startup
    fetchModelInfo();
  }, []);

  const fetchModelInfo = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${process.env.REACT_APP_API_URL || 'http://localhost:8001'}/api/model_info`);
      const data = await response.json();
      setModelInfo(data.model_info);
    } catch (error) {
      console.error('Failed to load model info:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="app-header">
        <h1>🎯 Ensemble Anomaly Detection Dashboard</h1>
        <p>Advanced EJ Session Anomaly Detection using Ensemble Machine Learning</p>
        <div className="model-status">
          {loading ? (
            <span className="status loading">Loading...</span>
          ) : modelInfo?.is_trained ? (
            <span className="status trained">✅ Model Trained & Ready</span>
          ) : (
            <span className="status untrained">⚠️ Model Not Trained</span>
          )}
        </div>
      </header>

      <main className="dashboard-main">
        <Tabs>
          <TabList>
            <Tab>📊 Overview</Tab>
            <Tab>🚀 Training</Tab>
            <Tab>🔍 Prediction</Tab>
            <Tab>🔬 DBSCAN Analysis</Tab>
          </TabList>

          <TabPanel>
            <Overview modelInfo={modelInfo} onRefresh={fetchModelInfo} />
          </TabPanel>

          <TabPanel>
            <Training onTrainingComplete={fetchModelInfo} />
          </TabPanel>

          <TabPanel>
            <Prediction modelInfo={modelInfo} />
          </TabPanel>

          <TabPanel>
            <DBSCANVisualization modelInfo={modelInfo} />
          </TabPanel>
        </Tabs>
      </main>
    </div>
  );
}

export default App;
