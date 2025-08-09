import React, { useState, useEffect } from 'react';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, 
  Tooltip, ResponsiveContainer, PieChart, Pie, Cell 
} from 'recharts';
import { 
  AlertTriangle, Bell, Clock, CheckCircle, Eye, 
  Filter, Search, Download, RefreshCw, Settings, Users 
} from 'lucide-react';
import apiConfig from './config/api';

const ALERT_PRIORITY_COLORS = {
  'critical': '#dc2626',
  'high': '#ea580c',
  'medium': '#d97706',
  'low': '#65a30d'
};

const AlertsPage = () => {
  const [alerts, setAlerts] = useState([]);
  const [filteredAlerts, setFilteredAlerts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [stats, setStats] = useState({
    total_alerts: 0,
    active_alerts: 0,
    critical_alerts: 0,
    high_priority_alerts: 0,
    acknowledged_alerts: 0,
    resolved_alerts: 0,
    false_positive_alerts: 0,
    avg_resolution_time: 0,
    alert_trend: [],
    priority_distribution: [],
    source_distribution: [],
    response_times: []
  });

  // Filter states
  const [filters, setFilters] = useState({
    priority: 'all',
    status: 'all',
    source: 'all',
    dateRange: 'today',
    searchTerm: '',
    assignee: 'all'
  });

  const [selectedAlert, setSelectedAlert] = useState(null);
  const [showDetailModal, setShowDetailModal] = useState(false);

  useEffect(() => {
    fetchAlerts();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    applyFilters();
  }, [alerts, filters]); // eslint-disable-line react-hooks/exhaustive-deps

  const fetchAlerts = async () => {
    try {
      setLoading(true);
      console.log('Fetching alerts from:', apiConfig.endpoint('/api/v1/alerts'));
      
      // Try to fetch real data from API
      try {
        const response = await fetch(apiConfig.endpoint('/api/v1/alerts'));
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        setAlerts(data.alerts || []);
        calculateStats(data.alerts || []);
        console.log('Successfully loaded alerts from API:', data.alerts?.length || 0);
      } catch (apiError) {
        console.log('API not available, showing empty state:', apiError.message);
        // Show empty data instead of mock data
        setAlerts([]);
        calculateStats([]);
      }
    } catch (error) {
      console.error('Error fetching alerts:', error);
      // Show empty data on error instead of misleading mock data
      setAlerts([]);
      calculateStats([]);
    } finally {
      setLoading(false);
    }
  };

  const generateMockAlerts = () => {
    const priorities = ['critical', 'high', 'medium', 'low'];
    const statuses = ['active', 'acknowledged', 'investigating', 'resolved', 'false_positive'];
    const sources = ['anomaly_detection', 'threshold_breach', 'network_monitoring', 'hardware_sensor', 'user_report'];
    const assignees = ['John Doe', 'Jane Smith', 'Mike Johnson', 'Sarah Wilson', 'Unassigned'];
    
    const alerts = [];
    for (let i = 1; i <= 100; i++) {
      const createdAt = new Date(Date.now() - Math.random() * 7 * 24 * 60 * 60 * 1000);
      const resolvedAt = Math.random() > 0.6 ? new Date(createdAt.getTime() + Math.random() * 24 * 60 * 60 * 1000) : null;
      
      alerts.push({
        id: `alert_${i.toString().padStart(3, '0')}`,
        title: generateAlertTitle(),
        description: generateAlertDescription(),
        priority: priorities[Math.floor(Math.random() * priorities.length)],
        status: resolvedAt ? 'resolved' : statuses[Math.floor(Math.random() * (statuses.length - 1))],
        source: sources[Math.floor(Math.random() * sources.length)],
        assignee: assignees[Math.floor(Math.random() * assignees.length)],
        created_at: createdAt.toISOString(),
        acknowledged_at: Math.random() > 0.7 ? new Date(createdAt.getTime() + Math.random() * 60 * 60 * 1000).toISOString() : null,
        resolved_at: resolvedAt ? resolvedAt.toISOString() : null,
        terminal_id: `ATM_${Math.floor(Math.random() * 20) + 1}`,
        location: `Branch ${Math.floor(Math.random() * 10) + 1}`,
        affected_sessions: Math.floor(Math.random() * 10) + 1,
        related_anomaly_id: Math.random() > 0.5 ? `anomaly_${Math.floor(Math.random() * 50) + 1}` : null,
        escalation_level: Math.floor(Math.random() * 3) + 1,
        notes: [],
        attachments: []
      });
    }
    return alerts.sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
  };

  const generateAlertTitle = () => {
    const titles = [
      'High Transaction Failure Rate Detected',
      'ATM Network Connectivity Issues',
      'Cash Dispenser Malfunction Alert',
      'Unusual Transaction Pattern Detected',
      'Security System Anomaly',
      'Temperature Sensor Out of Range',
      'Network Latency Threshold Exceeded',
      'Multiple Authentication Failures',
      'Cash Level Below Minimum Threshold',
      'Hardware Diagnostic Error'
    ];
    return titles[Math.floor(Math.random() * titles.length)];
  };

  const generateAlertDescription = () => {
    const descriptions = [
      'Automated monitoring detected unusual patterns requiring immediate attention',
      'System sensors indicate potential hardware issues affecting service availability',
      'Network monitoring alerts indicate connectivity problems with multiple terminals',
      'Security systems have flagged potential unauthorized access attempts',
      'Environmental monitoring systems report conditions outside normal parameters',
      'Transaction processing systems show degraded performance metrics',
      'User authentication systems experiencing higher than normal failure rates',
      'Cash management systems report inventory levels requiring immediate action'
    ];
    return descriptions[Math.floor(Math.random() * descriptions.length)];
  };

  const calculateStats = (alertsData) => {
    const stats = {
      total_alerts: alertsData.length,
      active_alerts: alertsData.filter(a => a.status === 'active').length,
      critical_alerts: alertsData.filter(a => a.priority === 'critical').length,
      high_priority_alerts: alertsData.filter(a => a.priority === 'high').length,
      acknowledged_alerts: alertsData.filter(a => a.status === 'acknowledged').length,
      resolved_alerts: alertsData.filter(a => a.status === 'resolved').length,
      false_positive_alerts: alertsData.filter(a => a.status === 'false_positive').length,
      avg_resolution_time: 0,
      alert_trend: [],
      priority_distribution: [],
      source_distribution: [],
      response_times: []
    };

    // Calculate average resolution time
    const resolvedAlerts = alertsData.filter(a => a.resolved_at);
    if (resolvedAlerts.length > 0) {
      const totalResolutionTime = resolvedAlerts.reduce((sum, alert) => {
        const created = new Date(alert.created_at);
        const resolved = new Date(alert.resolved_at);
        return sum + (resolved - created);
      }, 0);
      stats.avg_resolution_time = totalResolutionTime / resolvedAlerts.length / (1000 * 60 * 60); // hours
    }

    // Calculate priority distribution
    const priorityMap = {};
    alertsData.forEach(a => {
      priorityMap[a.priority] = (priorityMap[a.priority] || 0) + 1;
    });
    stats.priority_distribution = Object.entries(priorityMap).map(([priority, count]) => ({
      name: priority.charAt(0).toUpperCase() + priority.slice(1),
      value: count,
      color: ALERT_PRIORITY_COLORS[priority] || '#64748b'
    }));

    // Calculate source distribution
    const sourceMap = {};
    alertsData.forEach(a => {
      sourceMap[a.source] = (sourceMap[a.source] || 0) + 1;
    });
    stats.source_distribution = Object.entries(sourceMap).map(([source, count]) => ({
      source: source.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
      count,
      percentage: ((count / alertsData.length) * 100).toFixed(1)
    }));

    // Calculate 7-day trend
    const today = new Date();
    stats.alert_trend = Array.from({ length: 7 }, (_, i) => {
      const date = new Date(today.getTime() - i * 24 * 60 * 60 * 1000);
      const dayAlerts = alertsData.filter(a => {
        const alertDate = new Date(a.created_at);
        return alertDate.toDateString() === date.toDateString();
      });
      
      return {
        date: date.toISOString().split('T')[0],
        critical: dayAlerts.filter(a => a.priority === 'critical').length,
        high: dayAlerts.filter(a => a.priority === 'high').length,
        medium: dayAlerts.filter(a => a.priority === 'medium').length,
        low: dayAlerts.filter(a => a.priority === 'low').length,
        total: dayAlerts.length
      };
    }).reverse();

    setStats(stats);
  };

  const applyFilters = () => {
    let filtered = [...alerts];

    // Apply priority filter
    if (filters.priority !== 'all') {
      filtered = filtered.filter(a => a.priority === filters.priority);
    }

    // Apply status filter
    if (filters.status !== 'all') {
      filtered = filtered.filter(a => a.status === filters.status);
    }

    // Apply source filter
    if (filters.source !== 'all') {
      filtered = filtered.filter(a => a.source === filters.source);
    }

    // Apply assignee filter
    if (filters.assignee !== 'all') {
      filtered = filtered.filter(a => a.assignee === filters.assignee);
    }

    // Apply date range filter
    const now = new Date();
    switch (filters.dateRange) {
      case 'today':
        filtered = filtered.filter(a => {
          const alertDate = new Date(a.created_at);
          return alertDate.toDateString() === now.toDateString();
        });
        break;
      case 'week':
        const weekAgo = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
        filtered = filtered.filter(a => new Date(a.created_at) >= weekAgo);
        break;
      case 'month':
        const monthAgo = new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000);
        filtered = filtered.filter(a => new Date(a.created_at) >= monthAgo);
        break;
      default:
        // 'all' - no filtering needed
        break;
    }

    // Apply search term
    if (filters.searchTerm) {
      const searchLower = filters.searchTerm.toLowerCase();
      filtered = filtered.filter(a => 
        a.title.toLowerCase().includes(searchLower) ||
        a.description.toLowerCase().includes(searchLower) ||
        a.terminal_id.toLowerCase().includes(searchLower) ||
        a.location.toLowerCase().includes(searchLower) ||
        a.assignee.toLowerCase().includes(searchLower)
      );
    }

    setFilteredAlerts(filtered);
  };

  const getPriorityBadgeColor = (priority) => {
    const colors = {
      critical: 'bg-red-100 text-red-800',
      high: 'bg-orange-100 text-orange-800',
      medium: 'bg-yellow-100 text-yellow-800',
      low: 'bg-green-100 text-green-800'
    };
    return colors[priority] || 'bg-gray-100 text-gray-800';
  };

  const getStatusBadgeColor = (status) => {
    const colors = {
      active: 'bg-red-100 text-red-800',
      acknowledged: 'bg-yellow-100 text-yellow-800',
      investigating: 'bg-blue-100 text-blue-800',
      resolved: 'bg-green-100 text-green-800',
      false_positive: 'bg-gray-100 text-gray-800'
    };
    return colors[status] || 'bg-gray-100 text-gray-800';
  };

  const updateAlertStatus = async (alertId, newStatus) => {
    try {
      // Update local state immediately for better UX
      setAlerts(prev => prev.map(alert => 
        alert.id === alertId 
          ? { ...alert, status: newStatus, acknowledged_at: newStatus === 'acknowledged' ? new Date().toISOString() : alert.acknowledged_at }
          : alert
      ));

      // In a real implementation, make API call here
      console.log(`Updating alert ${alertId} status to ${newStatus}`);
      
    } catch (error) {
      console.error('Error updating alert status:', error);
    }
  };

  const exportAlerts = () => {
    const csv = [
      ['ID', 'Title', 'Priority', 'Status', 'Source', 'Assignee', 'Created At', 'Terminal', 'Location'].join(','),
      ...filteredAlerts.map(a => [
        a.id,
        `"${a.title}"`,
        a.priority,
        a.status,
        a.source,
        a.assignee,
        a.created_at,
        a.terminal_id,
        a.location
      ].join(','))
    ].join('\n');

    const blob = new Blob([csv], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `alerts_${new Date().toISOString().split('T')[0]}.csv`;
    link.click();
    window.URL.revokeObjectURL(url);
  };

  const StatCard = ({ title, value, icon: Icon, color, subtitle, trend }) => (
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
            {trend && <dd className="text-xs text-gray-500">{trend}</dd>}
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
          value={filters.priority}
          onChange={(e) => setFilters({...filters, priority: e.target.value})}
          className="border border-gray-300 rounded-md px-3 py-1 text-sm"
        >
          <option value="all">All Priorities</option>
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
          <option value="acknowledged">Acknowledged</option>
          <option value="investigating">Investigating</option>
          <option value="resolved">Resolved</option>
          <option value="false_positive">False Positive</option>
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
            placeholder="Search alerts..."
            value={filters.searchTerm}
            onChange={(e) => setFilters({...filters, searchTerm: e.target.value})}
            className="border border-gray-300 rounded-md px-3 py-1 text-sm w-64"
          />
        </div>

        <button
          onClick={exportAlerts}
          className="flex items-center space-x-1 bg-blue-600 text-white px-3 py-1 rounded-md text-sm hover:bg-blue-700"
        >
          <Download className="h-4 w-4" />
          <span>Export</span>
        </button>

        <button
          onClick={() => console.log('Settings modal not implemented yet')}
          className="flex items-center space-x-1 bg-gray-600 text-white px-3 py-1 rounded-md text-sm hover:bg-gray-700"
        >
          <Settings className="h-4 w-4" />
          <span>Settings</span>
        </button>
      </div>
      
      <div className="mt-4 text-sm text-gray-600">
        Showing {filteredAlerts.length} of {alerts.length} alerts
      </div>
    </div>
  );

  const AlertDetailModal = () => {
    if (!showDetailModal || !selectedAlert) return null;

    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
        <div className="bg-white rounded-lg shadow-xl max-w-4xl w-full mx-4 max-h-[90vh] overflow-auto">
          <div className="p-6">
            <div className="flex justify-between items-start mb-4">
              <h2 className="text-xl font-bold text-gray-900">Alert Details</h2>
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
                  <label className="text-sm font-medium text-gray-500">Alert ID</label>
                  <p className="font-mono text-sm">{selectedAlert.id}</p>
                </div>
                <div>
                  <label className="text-sm font-medium text-gray-500">Title</label>
                  <p className="text-sm font-medium">{selectedAlert.title}</p>
                </div>
                <div>
                  <label className="text-sm font-medium text-gray-500">Priority</label>
                  <span className={`inline-block px-2 py-1 rounded-full text-xs font-medium ${getPriorityBadgeColor(selectedAlert.priority)}`}>
                    {selectedAlert.priority.toUpperCase()}
                  </span>
                </div>
                <div>
                  <label className="text-sm font-medium text-gray-500">Source</label>
                  <p className="text-sm capitalize">{selectedAlert.source.replace(/_/g, ' ')}</p>
                </div>
                <div>
                  <label className="text-sm font-medium text-gray-500">Assignee</label>
                  <p className="text-sm">{selectedAlert.assignee}</p>
                </div>
              </div>
              
              <div className="space-y-4">
                <div>
                  <label className="text-sm font-medium text-gray-500">Status</label>
                  <div className="flex items-center space-x-2">
                    <span className={`inline-block px-2 py-1 rounded-full text-xs font-medium ${getStatusBadgeColor(selectedAlert.status)}`}>
                      {selectedAlert.status.toUpperCase()}
                    </span>
                    <select
                      value={selectedAlert.status}
                      onChange={(e) => updateAlertStatus(selectedAlert.id, e.target.value)}
                      className="border border-gray-300 rounded px-2 py-1 text-xs"
                    >
                      <option value="active">Active</option>
                      <option value="acknowledged">Acknowledged</option>
                      <option value="investigating">Investigating</option>
                      <option value="resolved">Resolved</option>
                      <option value="false_positive">False Positive</option>
                    </select>
                  </div>
                </div>
                <div>
                  <label className="text-sm font-medium text-gray-500">Terminal ID</label>
                  <p className="text-sm">{selectedAlert.terminal_id}</p>
                </div>
                <div>
                  <label className="text-sm font-medium text-gray-500">Location</label>
                  <p className="text-sm">{selectedAlert.location}</p>
                </div>
                <div>
                  <label className="text-sm font-medium text-gray-500">Created At</label>
                  <p className="text-sm">{new Date(selectedAlert.created_at).toLocaleString()}</p>
                </div>
                <div>
                  <label className="text-sm font-medium text-gray-500">Affected Sessions</label>
                  <p className="text-sm">{selectedAlert.affected_sessions}</p>
                </div>
              </div>
            </div>
            
            <div className="mt-6">
              <label className="text-sm font-medium text-gray-500">Description</label>
              <p className="text-sm mt-1 p-3 bg-gray-50 rounded-md">{selectedAlert.description}</p>
            </div>

            {selectedAlert.related_anomaly_id && (
              <div className="mt-4">
                <label className="text-sm font-medium text-gray-500">Related Anomaly</label>
                <p className="text-sm mt-1 font-mono">{selectedAlert.related_anomaly_id}</p>
              </div>
            )}
          </div>
        </div>
      </div>
    );
  };

  if (loading) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
        <span className="ml-2 text-gray-600">Loading alerts...</span>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-start">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Alert Management Dashboard</h1>
          <p className="text-gray-600 mt-1">Monitor and manage system alerts and notifications</p>
        </div>
        <button
          onClick={fetchAlerts}
          className="flex items-center space-x-2 bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700"
        >
          <RefreshCw className="h-4 w-4" />
          <span>Refresh</span>
        </button>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatCard
          title="Active Alerts"
          value={stats.active_alerts}
          icon={Bell}
          color="bg-red-600"
          subtitle="Requiring attention"
        />
        <StatCard
          title="Critical Priority"
          value={stats.critical_alerts}
          icon={AlertTriangle}
          color="bg-red-700"
          subtitle="Immediate action needed"
        />
        <StatCard
          title="Acknowledged"
          value={stats.acknowledged_alerts}
          icon={Eye}
          color="bg-yellow-600"
          subtitle="Under review"
        />
        <StatCard
          title="Avg Resolution Time"
          value={`${stats.avg_resolution_time.toFixed(1)}h`}
          icon={Clock}
          color="bg-blue-600"
          subtitle="Hours to resolve"
        />
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Alert Trend */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-lg font-semibold mb-4">7-Day Alert Trend</h3>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={stats.alert_trend}>
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

        {/* Priority Distribution */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-lg font-semibold mb-4">Priority Distribution</h3>
          <ResponsiveContainer width="100%" height={250}>
            <PieChart>
              <Pie
                data={stats.priority_distribution}
                cx="50%"
                cy="50%"
                outerRadius={80}
                dataKey="value"
                label={({name, value}) => `${name}: ${value}`}
              >
                {stats.priority_distribution.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Alert Sources */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-lg font-semibold mb-4">Alert Sources</h3>
          <div className="space-y-3">
            {stats.source_distribution.map((source, index) => (
              <div key={index} className="flex justify-between items-center">
                <span className="text-sm text-gray-600">{source.source}</span>
                <div className="flex items-center space-x-2">
                  <span className="text-sm font-medium">{source.count}</span>
                  <span className="text-xs text-gray-500">({source.percentage}%)</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Filters */}
      <FilterSection />

      {/* Alerts Table */}
      <div className="bg-white rounded-lg shadow-md overflow-hidden">
        <div className="px-6 py-4 border-b border-gray-200">
          <h3 className="text-lg font-semibold">System Alerts</h3>
        </div>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Alert
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Priority
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Status
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Source
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Assignee
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Created
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Actions
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {filteredAlerts.map((alert) => (
                <tr key={alert.id} className="hover:bg-gray-50">
                  <td className="px-6 py-4">
                    <div>
                      <div className="text-sm font-medium text-gray-900">{alert.title}</div>
                      <div className="text-sm text-gray-500 truncate max-w-xs">{alert.description}</div>
                      <div className="text-xs text-gray-400">ID: {alert.id}</div>
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${getPriorityBadgeColor(alert.priority)}`}>
                      {alert.priority.toUpperCase()}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${getStatusBadgeColor(alert.status)}`}>
                      {alert.status.replace('_', ' ').toUpperCase()}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 capitalize">
                    {alert.source.replace(/_/g, ' ')}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex items-center">
                      <Users className="h-4 w-4 text-gray-400 mr-1" />
                      <span className="text-sm text-gray-900">{alert.assignee}</span>
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    {new Date(alert.created_at).toLocaleString()}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium space-x-2">
                    <button
                      onClick={() => {
                        setSelectedAlert(alert);
                        setShowDetailModal(true);
                      }}
                      className="text-blue-600 hover:text-blue-900"
                    >
                      <Eye className="h-4 w-4 inline" />
                    </button>
                    {alert.status === 'active' && (
                      <button
                        onClick={() => updateAlertStatus(alert.id, 'acknowledged')}
                        className="text-yellow-600 hover:text-yellow-900"
                      >
                        <CheckCircle className="h-4 w-4 inline" />
                      </button>
                    )}
                    {alert.status !== 'resolved' && (
                      <button
                        onClick={() => updateAlertStatus(alert.id, 'resolved')}
                        className="text-green-600 hover:text-green-900"
                      >
                        <CheckCircle className="h-4 w-4 inline" />
                      </button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          
          {filteredAlerts.length === 0 && (
            <div className="text-center py-8 text-gray-500">
              No alerts found matching current filters
            </div>
          )}
        </div>
      </div>

      {/* Detail Modal */}
      <AlertDetailModal />
    </div>
  );
};

export default AlertsPage;
