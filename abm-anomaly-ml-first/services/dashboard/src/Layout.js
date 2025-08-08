import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Clock, Brain } from 'lucide-react';

const Layout = ({ children }) => {
  const location = useLocation();

  const getCurrentTab = () => {
    const path = location.pathname;
    if (path === '/dashboard' || path === '/dashboard/') return 'overview';
    if (path.includes('/dashboard/anomalies')) return 'anomalies';
    if (path.includes('/dashboard/multi-anomaly')) return 'multi-anomaly';
    if (path.includes('/dashboard/alerts')) return 'alerts';
    if (path.includes('/dashboard/expert-labeling')) return 'expert-labeling';
    if (path.includes('/dashboard/continuous-learning')) return 'continuous-learning';
    if (path.includes('/dashboard/realtime')) return 'monitoring';
    if (path.includes('/dashboard/analytics')) return 'analytics';
    if (path.includes('/dashboard/bert-analysis')) return 'bert-analysis';
    if (path.includes('/dashboard/deeplog')) return 'deeplog';
    return 'overview';
  };

  const [liveDataActive, setLiveDataActive] = React.useState(true);
  const [isProcessing, setIsProcessing] = React.useState(false);

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

  const handleProcessInput = async () => {
    try {
      setIsProcessing(true);
      const response = await fetch('/api/v1/process-input', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      
      if (response.ok) {
        alert('Input processing started successfully.');
      } else {
        alert('Failed to start processing.');
      }
    } catch (error) {
      console.error('Process error:', error);
      alert('Failed to start processing.');
    } finally {
      setIsProcessing(false);
    }
  };

  const handleClearAllData = async () => {
    const firstConfirm = window.confirm('Are you sure you want to clear all data? This action cannot be undone.');
    if (!firstConfirm) {
      return;
    }
    
    const secondConfirm = window.confirm('FINAL WARNING: This will permanently delete ALL data, sessions, files, and cache. Type "CONFIRM" in the next dialog to proceed.');
    if (!secondConfirm) {
      return;
    }
    
    const finalConfirm = window.prompt('Type "CONFIRM" to permanently clear all data:');
    if (finalConfirm !== 'CONFIRM') {
      alert('❌ Clear data cancelled. Data was not cleared.');
      return;
    }

    try {
      const response = await fetch('/api/v1/clear-data?confirm=true', {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' }
      });
      
      if (response.ok) {
        const result = await response.json();
        let message = '✅ All data cleared successfully!\n\n';
        
        if (result.database_tables_cleared && result.database_tables_cleared.length > 0) {
          message += `📊 Database tables cleared: ${result.total_tables}\n`;
          message += `   ${result.database_tables_cleared.join(', ')}\n\n`;
        }
        
        if (result.files_cleared && result.files_cleared.length > 0) {
          message += `📁 File groups cleared: ${result.total_file_groups}\n`;
          message += `   ${result.files_cleared.join(', ')}\n\n`;
        }
        
        if (result.redis_cleared) {
          message += `🔄 Redis cache cleared: Yes\n\n`;
        }
        
        if (result.method_used) {
          message += `🔧 Method used: ${result.method_used}\n`;
        }
        
        if (result.errors && result.errors.length > 0) {
          message += `\n⚠️ Some warnings occurred:\n${result.errors.join('\n')}`;
        }
        
        alert(message);
        window.location.reload();
      } else {
        const errorText = await response.text();
        alert(`❌ Failed to clear data: ${errorText}`);
      }
    } catch (error) {
      console.error('Clear data error:', error);
      alert(`❌ Failed to clear data: ${error.message}`);
    }
  };

  return (
    <div className="min-h-screen bg-gray-100">
      {/* Header */}
      <div className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center">
              <Brain className="w-8 h-8 text-purple-600 mr-3" />
              <h1 className="text-2xl font-bold text-gray-900">ML-First ABM Anomaly Detection</h1>
            </div>
            <div className="flex items-center space-x-4">
              <input
                type="file"
                id="file-upload"
                className="hidden"
                accept=".txt,.log"
                onChange={handleFileUpload}
              />
              <label
                htmlFor="file-upload"
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 cursor-pointer"
              >
                Upload EJournal
              </label>
              <button
                onClick={handleProcessInput}
                disabled={isProcessing}
                className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:bg-purple-400 disabled:cursor-not-allowed"
              >
                {isProcessing ? 'Processing...' : 'Process Input'}
              </button>
              <button
                onClick={handleClearAllData}
                className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700"
              >
                Clear All Data
              </button>
              <div className="flex items-center text-sm text-gray-500">
                <Clock className="w-4 h-4 mr-1" />
                Last updated: {new Date().toLocaleTimeString()}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Live Data Active Notification */}
      {liveDataActive && (
        <div className="bg-green-100 border-l-4 border-green-500 p-4">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex items-center justify-between">
              <div className="flex items-center">
                <div className="flex-shrink-0">
                  <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse mr-2"></div>
                </div>
                <div className="ml-3">
                  <p className="text-sm text-green-700 font-medium">
                    🟢 Live Data Active
                  </p>
                  <p className="text-xs text-green-600">
                    Real-time monitoring: 84 new transactions, 5g anomalies detected. Data updated.
                  </p>
                </div>
              </div>
              <button
                onClick={() => setLiveDataActive(false)}
                className="text-green-500 hover:text-green-700"
              >
                ×
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Navigation Tabs */}
      <div className="bg-white border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex space-x-8">
            {[
              { key: 'overview', label: 'Overview', path: '/dashboard' },
              { key: 'anomalies', label: 'Anomalies', path: '/dashboard/anomalies' },
              { key: 'multi-anomaly', label: 'Multi-Anomaly', path: '/dashboard/multi-anomaly' },
              { key: 'alerts', label: 'Alerts', path: '/dashboard/alerts' },
              { key: 'expert-labeling', label: 'Expert Review', path: '/dashboard/expert-labeling' },
              { key: 'continuous-learning', label: 'ML Training', path: '/dashboard/continuous-learning' },
              { key: 'bert-analysis', label: 'BERT Analysis', path: '/dashboard/bert-analysis' },
              { key: 'deeplog', label: 'DeepLog', path: '/dashboard/deeplog' },
              { key: 'monitoring', label: 'Real-time Monitor', path: '/dashboard/realtime' },
              { key: 'analytics', label: 'Analytics', path: '/dashboard/analytics' }
            ].map((tab) => (
              <Link
                key={tab.key}
                to={tab.path}
                className={`py-3 px-1 border-b-2 font-medium text-sm ${
                  getCurrentTab() === tab.key
                    ? 'border-purple-600 text-purple-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700'
                }`}
              >
                {tab.label}
              </Link>
            ))}
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {children}
      </div>
    </div>
  );
};

export default Layout;
