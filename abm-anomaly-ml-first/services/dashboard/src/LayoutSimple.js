import React from 'react';
import { Link, useLocation } from 'react-router-dom';

const LayoutSimple = ({ children }) => {
  console.log('🟢 LayoutSimple rendering...');
  
  const location = useLocation();
  
  return (
    <div style={{ minHeight: '100vh', backgroundColor: '#f5f5f5' }}>
      {/* DEBUG Header */}
      <div style={{
        backgroundColor: '#2563eb',
        color: 'white',
        padding: '15px 20px',
        borderBottom: '2px solid #1d4ed8'
      }}>
        <h1 style={{ margin: 0, fontSize: '24px' }}>
          🧠 ML-First ABM Anomaly Detection - SIMPLE LAYOUT
        </h1>
        <p style={{ margin: '5px 0 0 0', fontSize: '14px', opacity: 0.9 }}>
          Current Path: {location.pathname}
        </p>
      </div>

      {/* Navigation */}
      <div style={{
        backgroundColor: 'white',
        borderBottom: '1px solid #e5e7eb',
        padding: '0 20px'
      }}>
        <div style={{ display: 'flex', gap: '30px', paddingTop: '10px' }}>
          {[
            { label: 'Overview', path: '/' },
            { label: 'Anomalies', path: '/anomalies' },
            { label: 'Expert Review', path: '/expert-labeling' },
            { label: 'Session Review', path: '/session-review' },
            { label: 'Session Evaluation', path: '/session-evaluation' },
            { label: 'Real-time Monitor', path: '/realtime' },
          ].map((tab) => (
            <Link
              key={tab.path}
              to={tab.path}
              style={{
                textDecoration: 'none',
                padding: '10px 0',
                borderBottom: location.pathname === tab.path ? '2px solid #2563eb' : '2px solid transparent',
                color: location.pathname === tab.path ? '#2563eb' : '#6b7280',
                fontWeight: location.pathname === tab.path ? 'bold' : 'normal'
              }}
            >
              {tab.label}
            </Link>
          ))}
        </div>
      </div>

      {/* Content */}
      <div style={{ padding: '20px' }}>
        {children}
      </div>
    </div>
  );
};

export default LayoutSimple;
