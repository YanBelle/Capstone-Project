import React, { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import { AlertTriangle, Shield, Activity, TrendingUp, Database } from 'lucide-react';
import apiConfig from './config/api';

const SEVERITY_COLORS = {
  'critical': '#dc2626',
  'high': '#ea580c',
  'medium': '#d97706',
  'low': '#65a30d',
  'normal': '#16a34a'
};

const MultiAnomalyView = ({ anomalies: anomaliesProp }) => {
  const [anomalies, setAnomalies] = useState(anomaliesProp || []);
  const [loading, setLoading] = useState(false);
  const [stats, setStats] = useState({
    multiAnomalyCount: 0,
    singleAnomalyCount: 0,
    totalAnomalySessions: 0,
    averageAnomaliesPerSession: 0,
    severityDistribution: [],
    typeDistribution: [],
    methodDistribution: [],
    combinationPatterns: []
  });

  // Separate effect for fetching data - only runs once
  useEffect(() => {
    // Only fetch if no anomalies prop was provided and we haven't already loaded data
    if (!anomaliesProp && anomalies.length === 0 && !loading) {
      const fetchAnomalies = async () => {
        setLoading(true);
        try {
          console.log('Fetching anomalies data from API...');
          const response = await fetch(apiConfig.endpoint('/api/v1/anomalies?unlimited=true'));
          if (response.ok) {
            const data = await response.json();
            console.log('API response:', data);
            setAnomalies(data.anomalies || []);
          } else {
            console.warn('API request failed, using empty data');
            setAnomalies([]);
          }
        } catch (error) {
          console.error('Error fetching anomalies:', error);
          setAnomalies([]);
        } finally {
          setLoading(false);
        }
      };
      
      fetchAnomalies();
    }
  }, [anomaliesProp]); // Only depend on anomaliesProp, not anomalies

  // Separate effect for calculating stats - only runs when anomalies data changes
  useEffect(() => {
    console.log('MultiAnomalyView calculating stats for anomalies:', anomalies?.length || 0);
    
    const calculateMultiAnomalyStats = () => {
      if (!anomalies || !Array.isArray(anomalies)) {
        console.log('No anomalies data available for multi-anomaly stats');
        return;
      }
      
      console.log('Calculating multi-anomaly stats from', anomalies.length, 'sessions');
      
      const anomalySessions = anomalies.filter(session => session && session.is_anomaly);
      console.log('Found', anomalySessions.length, 'anomaly sessions');
      
      // Helper function to safely parse JSON strings or return arrays
      const parseArrayField = (field, fallback = []) => {
        if (Array.isArray(field)) return field;
        if (typeof field === 'string') {
          try {
            const parsed = JSON.parse(field);
            return Array.isArray(parsed) ? parsed : fallback;
          } catch (e) {
            return field ? [field] : fallback;
          }
        }
        return fallback;
      };
      
      // Process sessions and ensure they have multi-anomaly data
      const processedSessions = anomalySessions.map(session => {
        // Use existing anomaly_count or default to 1 for legacy data
        const anomalyCount = session.anomaly_count || (session.is_anomaly ? 1 : 0);
        
        // Parse anomaly_types (could be JSON string, array, or single value)
        let anomalyTypes = parseArrayField(session.anomaly_types);
        if (anomalyTypes.length === 0 && session.anomaly_type) {
          anomalyTypes = [session.anomaly_type];
        }
        
        // Parse detection_methods
        let detectionMethods = parseArrayField(session.detection_methods);
        if (detectionMethods.length === 0) {
          detectionMethods = ['isolation_forest']; // default for legacy data
        }
        
        return {
          ...session,
          anomaly_count: anomalyCount,
          anomaly_types: anomalyTypes,
          detection_methods: detectionMethods,
          max_severity: session.max_severity || 'medium'
        };
      });
      
      const multiAnomalySessions = processedSessions.filter(session => session.anomaly_count > 1);
      const singleAnomalySessions = processedSessions.filter(session => session.anomaly_count === 1);
      
      console.log('Multi-anomaly sessions:', multiAnomalySessions.length);
      console.log('Single anomaly sessions:', singleAnomalySessions.length);
      
      // Calculate severity distribution
      const severityCount = {};
      const typeCount = {};
      const methodCount = {};
      const combinationPatterns = {};
      
      let totalAnomalies = 0;
      
      processedSessions.forEach(session => {
        totalAnomalies += session.anomaly_count || 0;
        
        // Track severity
        const severity = session.max_severity || 'medium';
        severityCount[severity] = (severityCount[severity] || 0) + 1;
        
        // Track anomaly types
        (session.anomaly_types || []).forEach(type => {
          if (type) {
            typeCount[type] = (typeCount[type] || 0) + 1;
          }
        });
        
        // Track combination patterns for multi-anomaly sessions
        if ((session.anomaly_types || []).length > 1) {
          const pattern = session.anomaly_types.sort().join(' + ');
          combinationPatterns[pattern] = (combinationPatterns[pattern] || 0) + 1;
        }
        
        // Track detection methods
        (session.detection_methods || []).forEach(method => {
          if (method) {
            methodCount[method] = (methodCount[method] || 0) + 1;
          }
        });
      });
      
      setStats({
        multiAnomalyCount: multiAnomalySessions.length,
        singleAnomalyCount: singleAnomalySessions.length,
        totalAnomalySessions: processedSessions.length,
        averageAnomaliesPerSession: processedSessions.length > 0 ? (totalAnomalies / processedSessions.length).toFixed(2) : 0,
        severityDistribution: Object.entries(severityCount).map(([severity, count]) => ({
          name: severity,
          value: count,
          color: SEVERITY_COLORS[severity]
        })),
        typeDistribution: Object.entries(typeCount).map(([type, count]) => ({
          name: type.replace(/_/g, ' '),
          value: count
        })),
        methodDistribution: Object.entries(methodCount).map(([method, count]) => ({
          name: method.replace(/_/g, ' '),
          value: count
        })),
        combinationPatterns: Object.entries(combinationPatterns).map(([pattern, count]) => ({
          name: pattern.replace(/_/g, ' '),
          value: count
        })).sort((a, b) => b.value - a.value).slice(0, 10) // Top 10 patterns
      });
      
      console.log('Stats calculated:', {
        multiAnomalyCount: multiAnomalySessions.length,
        singleAnomalyCount: singleAnomalySessions.length,
        totalAnomalySessions: processedSessions.length,
        typeDistribution: Object.keys(typeCount),
        severityDistribution: Object.keys(severityCount)
      });
    };
    
    if (anomalies && Array.isArray(anomalies) && anomalies.length > 0) {
      try {
        calculateMultiAnomalyStats();
      } catch (error) {
        console.error('Error calculating multi-anomaly stats:', error);
        // Set default stats in case of error
        setStats({
          multiAnomalyCount: 0,
          singleAnomalyCount: 0,
          totalAnomalySessions: 0,
          averageAnomaliesPerSession: 0,
          severityDistribution: [],
          typeDistribution: [],
          methodDistribution: [],
          combinationPatterns: []
        });
      }
    } else {
      console.log('No anomaly data available');
    }
  }, [anomalies]); // Only depend on anomalies data, not anomaliesProp

  const getMultiAnomalySessions = () => {
    if (!anomalies || !Array.isArray(anomalies)) {
      return [];
    }
    
    try {
      return anomalies
        .filter(session => {
          // Extra safety checks for each session object
          if (!session || typeof session !== 'object') {
            return false;
          }
          // Check if anomaly_count exists and is a number greater than 1
          const count = session.anomaly_count;
          return typeof count === 'number' && count > 1;
        })
        .slice(0, 20); // Show top 20
    } catch (error) {
      console.error('Error in getMultiAnomalySessions:', error);
      return [];
    }
  };

  const renderSeverityIcon = (severity) => {
    const iconProps = { size: 16 };
    
    switch (severity) {
      case 'critical':
        return <AlertTriangle {...iconProps} className="text-red-600" />;
      case 'high':
        return <Shield {...iconProps} className="text-orange-500" />;
      case 'medium':
        return <Activity {...iconProps} className="text-yellow-500" />;
      default:
        return <TrendingUp {...iconProps} className="text-green-500" />;
    }
  };

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Multi-Anomaly Analysis</h1>
        <p className="text-gray-600 mt-1">Advanced analysis of sessions with multiple concurrent anomalies</p>
      </div>

        {/* Debug Information when no data */}
        {loading && (
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-6">
            <div className="flex">
              <div className="flex-shrink-0">
                <Activity className="h-5 w-5 text-blue-400 animate-spin" />
              </div>
              <div className="ml-3">
                <h3 className="text-sm font-medium text-blue-800">
                  Loading Multi-Anomaly Data...
                </h3>
                <p className="mt-1 text-sm text-blue-700">
                  Fetching anomaly data from the API...
                </p>
              </div>
            </div>
          </div>
        )}

        {(!loading && (!anomalies || anomalies.length === 0 || stats.totalAnomalySessions === 0)) && (
          <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-6">
            <div className="flex">
              <div className="flex-shrink-0">
                <Database className="h-5 w-5 text-yellow-400" />
              </div>
              <div className="ml-3">
                <h3 className="text-sm font-medium text-yellow-800">
                  Multi-Anomaly Data Status
                </h3>
                <div className="mt-2 text-sm text-yellow-700">
                  <p>No multi-anomaly data available. This could mean:</p>
                  <ul className="mt-2 list-disc list-inside space-y-1">
                    <li>No EJ logs have been processed yet</li>
                    <li>Database migration for multi-anomaly fields is pending</li>
                    <li>All detected anomalies are single-anomaly cases</li>
                  </ul>
                  <div className="mt-4 text-xs">
                    <p><strong>Debug Info:</strong></p>
                    <p>• Anomalies prop: {anomalies ? `Array(${anomalies.length})` : 'null/undefined'}</p>
                    <p>• Total anomaly sessions: {stats.totalAnomalySessions}</p>
                    {anomalies && anomalies.length > 0 && (
                      <p>• Sample session keys: {Object.keys(anomalies[0] || {}).join(', ')}</p>
                    )}
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Summary Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <AlertTriangle className="h-8 w-8 text-red-600" />
              </div>
              <div className="ml-5 w-0 flex-1">
                <dl>
                  <dt className="text-sm font-medium text-gray-500 truncate">Multi-Anomaly Sessions</dt>
                  <dd className="text-lg font-medium text-gray-900">{stats.multiAnomalyCount}</dd>
                </dl>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <Shield className="h-8 w-8 text-orange-500" />
              </div>
              <div className="ml-5 w-0 flex-1">
                <dl>
                  <dt className="text-sm font-medium text-gray-500 truncate">Single-Anomaly Sessions</dt>
                  <dd className="text-lg font-medium text-gray-900">{stats.singleAnomalyCount}</dd>
                </dl>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <Activity className="h-8 w-8 text-blue-600" />
              </div>
              <div className="ml-5 w-0 flex-1">
                <dl>
                  <dt className="text-sm font-medium text-gray-500 truncate">Total Anomaly Sessions</dt>
                  <dd className="text-lg font-medium text-gray-900">{stats.totalAnomalySessions}</dd>
                </dl>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <TrendingUp className="h-8 w-8 text-green-600" />
              </div>
              <div className="ml-5 w-0 flex-1">
                <dl>
                  <dt className="text-sm font-medium text-gray-500 truncate">Avg Anomalies/Session</dt>
                  <dd className="text-lg font-medium text-gray-900">{stats.averageAnomaliesPerSession}</dd>
                </dl>
              </div>
            </div>
          </div>
        </div>

        {/* Charts */}
        {stats.totalAnomalySessions > 0 && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Severity Distribution */}
            <div className="bg-white rounded-lg shadow p-6">
              <h3 className="text-lg font-medium text-gray-900 mb-4">Severity Distribution</h3>
              <ResponsiveContainer width="100%" height={200}>
                <PieChart>
                  <Pie 
                    data={stats.severityDistribution} 
                    cx="50%" 
                    cy="50%" 
                    outerRadius={80} 
                    fill="#8884d8"
                    dataKey="value"
                    label={({ name, value }) => `${name}: ${value}`}
                  >
                    {stats.severityDistribution.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>

            {/* Type Distribution */}
            <div className="bg-white rounded-lg shadow p-6">
              <h3 className="text-lg font-medium text-gray-900 mb-4">Anomaly Type Distribution</h3>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={stats.typeDistribution}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="value" fill="#3B82F6" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {/* Multi-Anomaly Sessions Table */}
        {stats.multiAnomalyCount > 0 && (
          <div className="bg-white shadow rounded-lg">
            <div className="px-4 py-5 sm:p-6">
              <h3 className="text-lg leading-6 font-medium text-gray-900 mb-4">
                Multi-Anomaly Sessions Details
              </h3>
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Session ID</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Anomaly Count</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Max Severity</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Anomaly Types</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Critical</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">High Severity</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Overall Score</th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {getMultiAnomalySessions().map((session, index) => {
                      // Extra safety check for each session in render
                      if (!session || typeof session !== 'object') {
                        return null;
                      }
                      
                      const sessionId = session.session_id || `session-${index}`;
                      const anomalyCount = session.anomaly_count || 0;
                      const anomalyTypes = session.anomaly_types || [];
                      
                      return (
                        <tr key={sessionId} className="hover:bg-gray-50">
                          <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                            {sessionId}
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap">
                            <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-red-100 text-red-800">
                              {anomalyCount}
                            </span>
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap">
                            <div className="flex items-center">
                              {renderSeverityIcon(session.max_severity)}
                              <span className="ml-2 text-sm text-gray-900 capitalize">{session.max_severity || 'medium'}</span>
                            </div>
                          </td>
                          <td className="px-6 py-4 text-sm text-gray-900">
                            <div className="flex flex-wrap gap-1">
                              {anomalyTypes.map((type, typeIndex) => (
                                <span key={typeIndex} className="inline-flex items-center px-2 py-1 rounded text-xs font-medium bg-blue-100 text-blue-800">
                                  {type.replace(/_/g, ' ')}
                                </span>
                              ))}
                            </div>
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                            {session.critical_anomalies_count || 0}
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                            {session.high_severity_anomalies_count || 0}
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                            <div className="flex items-center">
                              <div className="flex-shrink-0 w-10 h-10">
                                <div className="w-full bg-gray-200 rounded-full h-2">
                                  <div 
                                    className="bg-red-600 h-2 rounded-full" 
                                    style={{ width: `${Math.min(100, (session.overall_anomaly_score || 0) * 100)}%` }}
                                  ></div>
                                </div>
                              </div>
                              <span className="ml-3 text-sm font-medium text-gray-900">
                                {((session.overall_anomaly_score || 0) * 100).toFixed(1)}%
                              </span>
                            </div>
                          </td>
                        </tr>
                      );
                    }).filter(Boolean)}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}
      </div>
  );
};

export default MultiAnomalyView;
