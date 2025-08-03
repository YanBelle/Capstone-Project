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
              <div className="flex items-center text-sm text-gray-500">
                <Clock className="w-4 h-4 mr-1" />
                Last updated: {new Date().toLocaleTimeString()}
              </div>
            </div>
          </div>
        </div>
      </div>

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
