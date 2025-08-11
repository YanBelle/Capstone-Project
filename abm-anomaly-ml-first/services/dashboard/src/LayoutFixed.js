import React from 'react';
import { Link, useLocation } from 'react-router-dom';

const LayoutFixed = ({ children }) => {
  const location = useLocation();

  const getCurrentTab = () => {
    const path = location.pathname;
    if (path === '/' || path === '/dashboard' || path === '/dashboard/') return 'overview';
    if (path.includes('/anomalies')) return 'anomalies';
    if (path.includes('/multi-anomaly')) return 'multi-anomaly';
    if (path.includes('/alerts')) return 'alerts';
    if (path.includes('/cash-forecasting')) return 'cash-forecasting';
    if (path.includes('/expert-labeling')) return 'expert-labeling';
    if (path.includes('/continuous-learning')) return 'continuous-learning';
    if (path.includes('/session-review')) return 'session-review';
    if (path.includes('/session-evaluation')) return 'session-evaluation';
    if (path.includes('/realtime')) return 'monitoring';
    if (path.includes('/analytics')) return 'analytics';
    if (path.includes('/bert-analysis')) return 'bert-analysis';
    if (path.includes('/deeplog')) return 'deeplog';
    return 'overview';
  };

  const [liveDataActive, setLiveDataActive] = React.useState(true);

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('/api/v1/upload', {
        method: 'POST',
        body: formData
      });
      
      if (response.ok) {
        alert('File uploaded successfully. Processing will begin shortly.');
      }
    } catch (error) {
      console.error('Upload error:', error);
      alert('Failed to upload file');
    }
  };

  const handleClearAllData = async () => {
    const firstConfirm = window.confirm('⚠️ Are you sure you want to clear ALL data?');
    if (!firstConfirm) return;

    const secondConfirm = window.prompt('🔴 FINAL CONFIRMATION: Type "CLEAR ALL" to confirm:');
    if (secondConfirm !== 'CLEAR ALL') {
      alert('❌ Operation cancelled.');
      return;
    }

    try {
      const response = await fetch('/api/v1/data/clear-all?confirm=true', {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' }
      });

      if (response.ok) {
        alert('✅ All data cleared successfully.');
        window.location.reload();
      } else {
        alert(`❌ Failed to clear data`);
      }
    } catch (error) {
      console.error('Clear data error:', error);
      alert('❌ Failed to clear data.');
    }
  };

  const handleForceProcessInput = async () => {
    const confirm = window.confirm('🔄 Force process any EJ files in the input directory?');
    if (!confirm) return;

    try {
      const response = await fetch('/api/v1/process/force-input', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });

      if (response.ok) {
        const result = await response.json();
        if (result.files_processed > 0) {
          alert(`✅ Successfully processed ${result.files_processed} files.`);
        } else {
          alert(`ℹ️ ${result.message}`);
        }
        window.location.reload();
      } else {
        alert(`❌ Failed to process input directory`);
      }
    } catch (error) {
      console.error('Force process error:', error);
      alert('❌ Failed to process input directory.');
    }
  };

  const headerStyle = {
    backgroundColor: '#6366f1',
    color: 'white',
    padding: '16px',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center'
  };

  const titleStyle = {
    fontSize: '24px',
    fontWeight: 'bold',
    margin: 0
  };

  const buttonStyle = {
    backgroundColor: '#3b82f6',
    color: 'white',
    border: 'none',
    padding: '8px 16px',
    borderRadius: '4px',
    marginLeft: '8px',
    cursor: 'pointer'
  };

  const navStyle = {
    backgroundColor: 'white',
    borderBottom: '1px solid #e5e7eb',
    padding: '0 16px',
    overflowX: 'auto',
    whiteSpace: 'nowrap',
    display: 'flex',
    minHeight: '52px'
  };

  const tabStyle = {
    display: 'inline-block',
    padding: '12px 16px',
    textDecoration: 'none',
    color: '#6b7280',
    borderBottom: '2px solid transparent',
    flexShrink: 0,
    whiteSpace: 'nowrap'
  };

  const activeTabStyle = {
    ...tabStyle,
    color: '#6366f1',
    borderBottomColor: '#6366f1'
  };

  const currentTab = getCurrentTab();

  return (
    <div style={{ minHeight: '100vh', backgroundColor: '#f3f4f6' }}>
      {/* Header */}
      <div style={headerStyle}>
        <div style={{ display: 'flex', alignItems: 'center' }}>
          <h1 style={titleStyle}>🧠 ML-First ABM Anomaly Detection</h1>
        </div>
        <div style={{ display: 'flex', alignItems: 'center' }}>
          <input
            type="file"
            id="file-upload"
            style={{ display: 'none' }}
            accept=".txt,.log"
            onChange={handleFileUpload}
          />
          <label htmlFor="file-upload" style={buttonStyle}>
            Upload EJournal
          </label>
          <button onClick={handleForceProcessInput} style={buttonStyle}>
            🔄 Process Input
          </button>
          <button 
            onClick={handleClearAllData} 
            style={{...buttonStyle, backgroundColor: '#ef4444'}}
          >
            🗑️ Clear All Data
          </button>
          <div style={{ marginLeft: '16px', fontSize: '14px' }}>
            🕒 Last updated: {new Date().toLocaleTimeString()}
          </div>
        </div>
      </div>

      {/* Live Data Notification */}
      {liveDataActive && (
        <div style={{
          backgroundColor: '#dcfce7',
          borderLeft: '4px solid #22c55e',
          padding: '16px',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center'
        }}>
          <div>
            <div style={{ color: '#15803d', fontWeight: 'bold', fontSize: '14px' }}>
              🟢 Live Data Active
            </div>
            <div style={{ color: '#16a34a', fontSize: '12px' }}>
              Real-time monitoring: 84 new transactions, 5 anomalies detected.
            </div>
          </div>
          <button
            onClick={() => setLiveDataActive(false)}
            style={{
              background: 'none',
              border: 'none',
              color: '#22c55e',
              cursor: 'pointer',
              fontSize: '18px'
            }}
          >
            ×
          </button>
        </div>
      )}

      {/* Navigation Tabs */}
      <div style={navStyle}>
        {[
          { key: 'overview', label: 'Overview', path: '/' },
          { key: 'anomalies', label: 'Anomalies', path: '/anomalies' },
          { key: 'multi-anomaly', label: 'Multi-Anomaly', path: '/multi-anomaly' },
          { key: 'alerts', label: 'Alerts', path: '/alerts' },
          { key: 'cash-forecasting', label: '💰 CASH FORECASTING', path: '/cash-forecasting', highlight: true },
          { key: 'expert-labeling', label: 'Expert Review', path: '/expert-labeling' },
          { key: 'continuous-learning', label: 'ML Training', path: '/continuous-learning' },
          { key: 'session-review', label: 'Session Review', path: '/session-review' },
          { key: 'session-evaluation', label: 'Evaluate Session', path: '/session-evaluation' },
          { key: 'bert-analysis', label: 'BERT Analysis', path: '/bert-analysis' },
          { key: 'deeplog', label: 'DeepLog', path: '/deeplog' },
          { key: 'monitoring', label: 'Real-time Monitor', path: '/realtime' },
          { key: 'analytics', label: 'Analytics', path: '/analytics' }
        ].map((tab) => (
          <Link
            key={tab.key}
            to={tab.path}
            style={{
              ...(currentTab === tab.key ? activeTabStyle : tabStyle),
              ...(tab.highlight && { 
                backgroundColor: '#fbbf24', 
                color: '#000',
                fontWeight: 'bold',
                border: '2px solid #f59e0b'
              })
            }}
          >
            {tab.label}
          </Link>
        ))}
      </div>

      {/* Main Content */}
      <div style={{ padding: '32px 16px' }}>
        {children}
      </div>
    </div>
  );
};

export default LayoutFixed;
