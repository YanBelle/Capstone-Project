import React, { useState, useEffect } from 'react';
import { 
  LineChart, Line, BarChart, Bar, PieChart, Pie, Cell, XAxis, YAxis, 
  CartesianGrid, Tooltip, ResponsiveContainer 
} from 'recharts';
import { 
  AlertTriangle, AlertCircle, Activity, Clock, Filter, Search, Download, 
  Eye, FileText 
} from 'lucide-react';
import apiConfig from './config/api';

const ANOMALY_TYPE_COLORS = {
  'dispense_failure': '#ef4444',
  'timeout_error': '#f97316',
  'network_error': '#eab308',
  'hardware_malfunction': '#84cc16',
  'authentication_failure': '#06b6d4',
  'transaction_anomaly': '#8b5cf6',
  'cash_shortage': '#ec4899',
  'sensor_error': '#64748b'
};

const AnomaliesPage = () => {
  const [anomalies, setAnomalies] = useState([]);
  const [filteredAnomalies, setFilteredAnomalies] = useState([]);
  const [loading, setLoading] = useState(true);
  const [stats, setStats] = useState({
    total_anomalies: 0,
    critical_count: 0,
    high_count: 0,
    medium_count: 0,
    low_count: 0,
    resolved_count: 0,
    pending_count: 0,
    severity_trend: [],
    type_distribution: [],
    detection_methods: [],
    hourly_pattern: []
  });

  // Filter and search states
  const [filters, setFilters] = useState({
    severity: 'all',
    status: 'all',
    type: 'all',
    dateRange: 'today',
    searchTerm: ''
  });

  const [selectedAnomaly, setSelectedAnomaly] = useState(null);
  const [showDetailModal, setShowDetailModal] = useState(false);

  useEffect(() => {
    fetchAnomalies();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    applyFilters();
  }, [anomalies, filters]); // eslint-disable-line react-hooks/exhaustive-deps

  const fetchAnomalies = async () => {
    try {
      setLoading(true);
      console.log('Fetching anomalies from:', apiConfig.endpoint('/api/v1/anomalies'));
      
      // Try to fetch real data from API
      try {
        const response = await fetch(apiConfig.endpoint('/api/v1/anomalies'));
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        setAnomalies(data.anomalies || []);
        calculateStats(data.anomalies || []);
        console.log('Successfully loaded anomalies from API:', data.anomalies?.length || 0);
      } catch (apiError) {
        console.log('API not available, showing empty state:', apiError.message);
        // Show empty data instead of mock data
        setAnomalies([]);
        calculateStats([]);
      }
    } catch (error) {
      console.error('Error fetching anomalies:', error);
      // Show empty data on error instead of misleading mock data
      setAnomalies([]);
      calculateStats([]);
    } finally {
      setLoading(false);
    }
  };

  const generateMockAnomalies = () => {
    const types = ['dispense_failure', 'timeout_error', 'network_error', 'hardware_malfunction', 'authentication_failure', 'transaction_anomaly', 'cash_shortage', 'sensor_error'];
    const severities = ['critical', 'high', 'medium', 'low'];
    const statuses = ['active', 'investigating', 'resolved'];
    const methods = ['isolation_forest', 'one_class_svm', 'autoencoder', 'bert_embedding'];
    
    const anomalies = [];
    for (let i = 1; i <= 50; i++) {
      const timestamp = new Date(Date.now() - Math.random() * 7 * 24 * 60 * 60 * 1000);
      anomalies.push({
        id: `anomaly_${i.toString().padStart(3, '0')}`,
        session_id: `ABM250_${timestamp.toISOString().split('T')[0].replace(/-/g, '')}_SESSION_${i.toString().padStart(3, '0')}`,
        type: types[Math.floor(Math.random() * types.length)],
        severity: severities[Math.floor(Math.random() * severities.length)],
        status: statuses[Math.floor(Math.random() * statuses.length)],
        anomaly_score: Math.random() * 0.4 + 0.6, // 0.6 to 1.0
        detected_at: timestamp.toISOString(),
        detection_method: methods[Math.floor(Math.random() * methods.length)],
        terminal_id: `ATM_${Math.floor(Math.random() * 20) + 1}`,
        location: `Branch ${Math.floor(Math.random() * 10) + 1}`,
        description: generateAnomalyDescription(),
        impact_score: Math.random() * 10,
        resolution_time: Math.random() * 120, // minutes
        false_positive_probability: Math.random() * 0.3
      });
    }
    return anomalies.sort((a, b) => new Date(b.detected_at) - new Date(a.detected_at));
  };

  const generateAnomalyDescription = () => {
    const descriptions = [
      'Unusual dispense pattern detected with multiple failed attempts',
      'Network timeout exceeding normal thresholds during transaction',
      'Hardware sensor reading outside expected parameters',
      'Authentication failure sequence anomaly detected',
      'Transaction amount pattern deviates from learned behavior',
      'Cash counting discrepancy identified during audit',
      'Sensor calibration drift detected in multiple readings',
      'Network latency spike correlating with transaction failures'
    ];
    return descriptions[Math.floor(Math.random() * descriptions.length)];
  };

  const calculateStats = (anomaliesData) => {
    const stats = {
      total_anomalies: anomaliesData.length,
      critical_count: anomaliesData.filter(a => a.severity === 'critical').length,
      high_count: anomaliesData.filter(a => a.severity === 'high').length,
      medium_count: anomaliesData.filter(a => a.severity === 'medium').length,
      low_count: anomaliesData.filter(a => a.severity === 'low').length,
      resolved_count: anomaliesData.filter(a => a.status === 'resolved').length,
      pending_count: anomaliesData.filter(a => a.status !== 'resolved').length,
      severity_trend: [],
      type_distribution: [],
      detection_methods: [],
      hourly_pattern: []
    };

    // Calculate type distribution
    const typeMap = {};
    anomaliesData.forEach(a => {
      typeMap[a.type] = (typeMap[a.type] || 0) + 1;
    });
    stats.type_distribution = Object.entries(typeMap).map(([type, count]) => ({
      name: type.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
      value: count,
      color: ANOMALY_TYPE_COLORS[type] || '#64748b'
    }));

    // Calculate detection methods
    const methodMap = {};
    anomaliesData.forEach(a => {
      methodMap[a.detection_method] = (methodMap[a.detection_method] || 0) + 1;
    });
    stats.detection_methods = Object.entries(methodMap).map(([method, count]) => ({
      method: method.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
      count,
      percentage: ((count / anomaliesData.length) * 100).toFixed(1)
    }));

    // Calculate hourly pattern
    const hourlyMap = {};
    anomaliesData.forEach(a => {
      const hour = new Date(a.detected_at).getHours();
      hourlyMap[hour] = (hourlyMap[hour] || 0) + 1;
    });
    stats.hourly_pattern = Array.from({ length: 24 }, (_, hour) => ({
      hour: `${hour.toString().padStart(2, '0')}:00`,
      anomalies: hourlyMap[hour] || 0
    }));

    // Calculate severity trend (last 7 days)
    const today = new Date();
    stats.severity_trend = Array.from({ length: 7 }, (_, i) => {
      const date = new Date(today.getTime() - i * 24 * 60 * 60 * 1000);
      const dayAnomalies = anomaliesData.filter(a => {
        const anomalyDate = new Date(a.detected_at);
        return anomalyDate.toDateString() === date.toDateString();
      });
      
      return {
        date: date.toISOString().split('T')[0],
        critical: dayAnomalies.filter(a => a.severity === 'critical').length,
        high: dayAnomalies.filter(a => a.severity === 'high').length,
        medium: dayAnomalies.filter(a => a.severity === 'medium').length,
        low: dayAnomalies.filter(a => a.severity === 'low').length
      };
    }).reverse();

    setStats(stats);
  };

  const applyFilters = () => {
    let filtered = [...anomalies];

    // Apply severity filter
    if (filters.severity !== 'all') {
      filtered = filtered.filter(a => a.severity === filters.severity);
    }

    // Apply status filter
    if (filters.status !== 'all') {
      filtered = filtered.filter(a => a.status === filters.status);
    }

    // Apply type filter
    if (filters.type !== 'all') {
      filtered = filtered.filter(a => a.type === filters.type);
    }

    // Apply date range filter
    const now = new Date();
    switch (filters.dateRange) {
      case 'today':
        filtered = filtered.filter(a => {
          const anomalyDate = new Date(a.detected_at);
          return anomalyDate.toDateString() === now.toDateString();
        });
        break;
      case 'week':
        const weekAgo = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
        filtered = filtered.filter(a => new Date(a.detected_at) >= weekAgo);
        break;
      case 'month':
        const monthAgo = new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000);
        filtered = filtered.filter(a => new Date(a.detected_at) >= monthAgo);
        break;
      default:
        // 'all' - no filtering needed
        break;
    }

    // Apply search term
    if (filters.searchTerm) {
      const searchLower = filters.searchTerm.toLowerCase();
      filtered = filtered.filter(a => 
        a.session_id.toLowerCase().includes(searchLower) ||
        a.type.toLowerCase().includes(searchLower) ||
        a.terminal_id.toLowerCase().includes(searchLower) ||
        a.location.toLowerCase().includes(searchLower) ||
        a.description.toLowerCase().includes(searchLower)
      );
    }

    setFilteredAnomalies(filtered);
  };

  const getSeverityBadgeColor = (severity) => {
    const colors = {
      critical: 'bg-red-100 text-red-800',
      high: 'bg-orange-100 text-orange-800',
      medium: 'bg-yellow-100 text-yellow-800',
      low: 'bg-green-100 text-green-800'
    };
    return colors[severity] || 'bg-gray-100 text-gray-800';
  };

  const getStatusBadgeColor = (status) => {
    const colors = {
      active: 'bg-red-100 text-red-800',
      investigating: 'bg-blue-100 text-blue-800',
      resolved: 'bg-green-100 text-green-800'
    };
    return colors[status] || 'bg-gray-100 text-gray-800';
  };

  const exportAnomalies = () => {
    const csv = [
      ['ID', 'Session ID', 'Type', 'Severity', 'Status', 'Score', 'Detected At', 'Terminal', 'Location'].join(','),
      ...filteredAnomalies.map(a => [
        a.id,
        a.session_id,
        a.type,
        a.severity,
        a.status,
        a.anomaly_score.toFixed(3),
        a.detected_at,
        a.terminal_id,
        a.location
      ].join(','))
    ].join('\n');

    const blob = new Blob([csv], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `anomalies_${new Date().toISOString().split('T')[0]}.csv`;
    a.click();
    window.URL.revokeObjectURL(url);
  };

  const StatCard = ({ title, value, icon: Icon, color, subtitle }) => (
    <div className="bg-white rounded-lg shadow-md p-6">
      <div className="flex items-center">
        <div className="flex-shrink-0">
          <div className={`${color} rounded-md p-3`}>
            <Icon className="h-6 w-6 text-white" />
          </div>
        </div>
        <div className="ml-5 w-0 flex-1">
          <dl>
            <dt className="text-sm font-medium text-gray-500 truncate">{title}</dt>
            <dd className="text-lg font-medium text-gray-900">{value}</dd>
            {subtitle && <dd className="text-sm text-gray-600">{subtitle}</dd>}
          </dl>
        </div>
      </div>
    </div>
  );

  const FilterSection = () => (
    <div className="bg-white rounded-lg shadow-md p-6 mb-6">
      <div className="flex flex-wrap gap-4 items-center">
        <div className="flex items-center space-x-2">
          <Filter className="h-4 w-4 text-gray-500" />
          <span className="text-sm font-medium text-gray-700">Filters:</span>
        </div>
        
        <select
          value={filters.severity}
          onChange={(e) => setFilters({...filters, severity: e.target.value})}
          className="border border-gray-300 rounded-md px-3 py-1 text-sm"
        >
          <option value="all">All Severities</option>
          <option value="critical">Critical</option>
          <option value="high">High</option>
          <option value="medium">Medium</option>
          <option value="low">Low</option>
        </select>

        <select
          value={filters.status}
          onChange={(e) => setFilters({...filters, status: e.target.value})}
          className="border border-gray-300 rounded-md px-3 py-1 text-sm"
        >
          <option value="all">All Statuses</option>
          <option value="active">Active</option>
          <option value="investigating">Investigating</option>
          <option value="resolved">Resolved</option>
        </select>

        <select
          value={filters.dateRange}
          onChange={(e) => setFilters({...filters, dateRange: e.target.value})}
          className="border border-gray-300 rounded-md px-3 py-1 text-sm"
        >
          <option value="today">Today</option>
          <option value="week">Last Week</option>
          <option value="month">Last Month</option>
          <option value="all">All Time</option>
        </select>

        <div className="flex items-center space-x-2">
          <Search className="h-4 w-4 text-gray-500" />
          <input
            type="text"
            placeholder="Search anomalies..."
            value={filters.searchTerm}
            onChange={(e) => setFilters({...filters, searchTerm: e.target.value})}
            className="border border-gray-300 rounded-md px-3 py-1 text-sm w-64"
          />
        </div>

        <button
          onClick={exportAnomalies}
          className="flex items-center space-x-1 bg-blue-600 text-white px-3 py-1 rounded-md text-sm hover:bg-blue-700"
        >
          <Download className="h-4 w-4" />
          <span>Export</span>
        </button>
      </div>
      
      <div className="mt-4 text-sm text-gray-600">
        Showing {filteredAnomalies.length} of {anomalies.length} anomalies
      </div>
    </div>
  );

  const AnomalyDetailModal = () => {
    if (!showDetailModal || !selectedAnomaly) return null;

    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
        <div className="bg-white rounded-lg shadow-xl max-w-4xl w-full mx-4 max-h-[90vh] overflow-auto">
          <div className="p-6">
            <div className="flex justify-between items-start mb-4">
              <h2 className="text-xl font-bold text-gray-900">Anomaly Details</h2>
              <button
                onClick={() => setShowDetailModal(false)}
                className="text-gray-400 hover:text-gray-600"
              >
                ✕
              </button>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-4">
                <div>
                  <label className="text-sm font-medium text-gray-500">Session ID</label>
                  <p className="font-mono text-sm">{selectedAnomaly.session_id}</p>
                </div>
                <div>
                  <label className="text-sm font-medium text-gray-500">Anomaly Type</label>
                  <p className="text-sm capitalize">{selectedAnomaly.type.replace(/_/g, ' ')}</p>
                </div>
                <div>
                  <label className="text-sm font-medium text-gray-500">Severity</label>
                  <span className={`inline-block px-2 py-1 rounded-full text-xs font-medium ${getSeverityBadgeColor(selectedAnomaly.severity)}`}>
                    {selectedAnomaly.severity.toUpperCase()}
                  </span>
                </div>
                <div>
                  <label className="text-sm font-medium text-gray-500">Detection Method</label>
                  <p className="text-sm capitalize">{selectedAnomaly.detection_method.replace(/_/g, ' ')}</p>
                </div>
                <div>
                  <label className="text-sm font-medium text-gray-500">Anomaly Score</label>
                  <p className="text-sm">{selectedAnomaly.anomaly_score.toFixed(3)}</p>
                </div>
              </div>
              
              <div className="space-y-4">
                <div>
                  <label className="text-sm font-medium text-gray-500">Terminal ID</label>
                  <p className="text-sm">{selectedAnomaly.terminal_id}</p>
                </div>
                <div>
                  <label className="text-sm font-medium text-gray-500">Location</label>
                  <p className="text-sm">{selectedAnomaly.location}</p>
                </div>
                <div>
                  <label className="text-sm font-medium text-gray-500">Detected At</label>
                  <p className="text-sm">{new Date(selectedAnomaly.detected_at).toLocaleString()}</p>
                </div>
                <div>
                  <label className="text-sm font-medium text-gray-500">Status</label>
                  <span className={`inline-block px-2 py-1 rounded-full text-xs font-medium ${getStatusBadgeColor(selectedAnomaly.status)}`}>
                    {selectedAnomaly.status.toUpperCase()}
                  </span>
                </div>
                <div>
                  <label className="text-sm font-medium text-gray-500">Impact Score</label>
                  <p className="text-sm">{selectedAnomaly.impact_score.toFixed(2)}</p>
                </div>
              </div>
            </div>
            
            <div className="mt-6">
              <label className="text-sm font-medium text-gray-500">Description</label>
              <p className="text-sm mt-1 p-3 bg-gray-50 rounded-md">{selectedAnomaly.description}</p>
            </div>
          </div>
        </div>
      </div>
    );
  };

  if (loading) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
        <span className="ml-2 text-gray-600">Loading anomalies...</span>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-start">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Anomaly Detection Dashboard</h1>
          <p className="text-gray-600 mt-1">Monitor and analyze detected anomalies across all ATM systems</p>
        </div>
        <button
          onClick={fetchAnomalies}
          className="flex items-center space-x-2 bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700"
        >
          <Activity className="h-4 w-4" />
          <span>Refresh</span>
        </button>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatCard
          title="Total Anomalies"
          value={stats.total_anomalies}
          icon={AlertTriangle}
          color="bg-red-600"
          subtitle="All detected anomalies"
        />
        <StatCard
          title="Critical & High"
          value={stats.critical_count + stats.high_count}
          icon={AlertCircle}
          color="bg-orange-600"
          subtitle="Requires immediate attention"
        />
        <StatCard
          title="Pending Resolution"
          value={stats.pending_count}
          icon={Clock}
          color="bg-yellow-600"
          subtitle="Active investigations"
        />
        <StatCard
          title="Resolved"
          value={stats.resolved_count}
          icon={Activity}
          color="bg-green-600"
          subtitle="Successfully resolved"
        />
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Severity Trend */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-lg font-semibold mb-4">7-Day Severity Trend</h3>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={stats.severity_trend}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" tickFormatter={(value) => new Date(value).toLocaleDateString()} />
              <YAxis />
              <Tooltip />
              <Bar dataKey="critical" stackId="a" fill="#dc2626" name="Critical" />
              <Bar dataKey="high" stackId="a" fill="#ea580c" name="High" />
              <Bar dataKey="medium" stackId="a" fill="#d97706" name="Medium" />
              <Bar dataKey="low" stackId="a" fill="#65a30d" name="Low" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Type Distribution */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-lg font-semibold mb-4">Anomaly Types</h3>
          <ResponsiveContainer width="100%" height={250}>
            <PieChart>
              <Pie
                data={stats.type_distribution}
                cx="50%"
                cy="50%"
                outerRadius={80}
                dataKey="value"
                label={({name, value}) => `${name}: ${value}`}
              >
                {stats.type_distribution.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Hourly Pattern */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-lg font-semibold mb-4">Hourly Detection Pattern</h3>
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={stats.hourly_pattern}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="hour" />
              <YAxis />
              <Tooltip />
              <Line type="monotone" dataKey="anomalies" stroke="#8884d8" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Filters */}
      <FilterSection />

      {/* Anomalies Table */}
      <div className="bg-white rounded-lg shadow-md overflow-hidden">
        <div className="px-6 py-4 border-b border-gray-200">
          <h3 className="text-lg font-semibold">Detected Anomalies</h3>
        </div>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Session ID
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Type
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Severity
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Score
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Terminal
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Detected At
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Status
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Actions
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {filteredAnomalies.map((anomaly) => (
                <tr key={anomaly.id} className="hover:bg-gray-50">
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-mono text-gray-900">
                    {anomaly.session_id}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 capitalize">
                    {anomaly.type.replace(/_/g, ' ')}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${getSeverityBadgeColor(anomaly.severity)}`}>
                      {anomaly.severity.toUpperCase()}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    {anomaly.anomaly_score.toFixed(3)}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    {anomaly.terminal_id}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    {new Date(anomaly.detected_at).toLocaleString()}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${getStatusBadgeColor(anomaly.status)}`}>
                      {anomaly.status.toUpperCase()}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                    <button
                      onClick={() => {
                        setSelectedAnomaly(anomaly);
                        setShowDetailModal(true);
                      }}
                      className="text-blue-600 hover:text-blue-900 mr-3"
                    >
                      <Eye className="h-4 w-4 inline" />
                    </button>
                    <button className="text-gray-600 hover:text-gray-900">
                      <FileText className="h-4 w-4 inline" />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          
          {filteredAnomalies.length === 0 && (
            <div className="text-center py-8 text-gray-500">
              No anomalies found matching current filters
            </div>
          )}
        </div>
      </div>

      {/* Detail Modal */}
      <AnomalyDetailModal />
    </div>
  );
};

export default AnomaliesPage;
