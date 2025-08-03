import React, { useState } from 'react';
import Dashboard from './components/Dashboard';
import './styles/Dashboard.css';

function App() {
  const [modelInfo, setModelInfo] = useState(null);

  React.useEffect(() => {
    // Check if model is loaded
    fetch(`${process.env.REACT_APP_API_URL}/api/model_info`)
      .then(res => res.json())
      .then(data => setModelInfo(data))
      .catch(err => console.error('Failed to load model info:', err));
  }, []);

  return (
    <div className="App">
      <header className="App-header">
        <h1>BERT EJ Log Debugging Dashboard</h1>
        {modelInfo && (
          <div className="model-info">
            <span>Model: Loaded on {modelInfo.device}</span>
            <span>Classes: {modelInfo.num_labels}</span>
          </div>
        )}
      </header>
      <Dashboard />
    </div>
  );
}

export default App;
