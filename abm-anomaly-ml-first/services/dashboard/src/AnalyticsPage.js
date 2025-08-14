import React, { useState, useEffect } from 'react';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ScatterChart,
  Scatter
} from 'recharts';
import {
  TrendingUp,
  Brain,
  AlertTriangle,
  Target,
  Clock,
  DollarSign,
  Activity,
  Shield,
  Zap,
  Database,
  BarChart3,
  Settings
} from 'lucide-react';
import apiConfig from './config/api';

const AnalyticsPage = () => {
  const [analyticsData, setAnalyticsData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedTimeRange, setSelectedTimeRange] = useState('24h');
  const [activeTab, setActiveTab] = useState('performance');

  const fetchAnalyticsData = async () => {
    try {
      setLoading(true);
      const response = await fetch(apiConfig.endpoint(`/api/v1/analytics/data?timeframe=${selectedTimeRange}`));
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      setAnalyticsData(data);
      setError(null);
    } catch (err) {
      console.error('Error fetching analytics data:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchAnalyticsData();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedTimeRange]);

  useEffect(() => {
    // Refresh every 10 minutes
    const interval = setInterval(fetchAnalyticsData, 600000);
    return () => clearInterval(interval);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedTimeRange]);

  const MetricCard = ({ title, value, subtitle, icon: Icon, color, change }) => (
    <div className="bg-white rounded-lg shadow-md p-6">
      <div className="flex items-center">
        <div className={`flex-shrink-0 ${color} rounded-md p-3`}>
          <Icon className="h-6 w-6 text-white" />
        </div>
        <div className="ml-5 w-0 flex-1">
          <dl>
            <dt className="text-sm font-medium text-gray-500 truncate">{title}</dt>
            <dd className="flex items-baseline">
              <div className="text-2xl font-semibold text-gray-900">{value}</div>
              {change && (
                <div className={`ml-2 flex items-baseline text-sm font-semibold ${
                  change > 0 ? 'text-green-600' : change < 0 ? 'text-red-600' : 'text-gray-500'
                }`}>
                  {change > 0 ? '+' : ''}{change}%
                </div>
              )}
            </dd>
            {subtitle && <div className="text-sm text-gray-500 mt-1">{subtitle}</div>}
          </dl>
        </div>
      </div>
    </div>
  );

  const TabButton = ({ id, label, icon: Icon, isActive, onClick }) => (
    <button
      onClick={() => onClick(id)}
      className={`flex items-center px-4 py-2 text-sm font-medium rounded-md ${
        isActive
          ? 'bg-blue-100 text-blue-700 border-blue-300'
          : 'text-gray-500 hover:text-gray-700 hover:bg-gray-100'
      }`}
    >
      <Icon className="w-4 h-4 mr-2" />
      {label}
    </button>
  );

  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8'];

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-100 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading analytics dashboard...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gray-100 flex items-center justify-center">
        <div className="text-center">
          <AlertTriangle className="h-12 w-12 text-red-600 mx-auto mb-4" />
          <p className="text-red-600 mb-4">Error loading analytics: {error}</p>
          <button 
            onClick={fetchAnalyticsData}
            className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-100">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="mb-8">
          <div className="flex justify-between items-start">
            <div>
              <h1 className="text-3xl font-bold text-gray-900">Analytics Dashboard</h1>
              <p className="mt-2 text-gray-600">
                Advanced analytics for anomaly detection and cash forecasting systems
              </p>
            </div>
            <div className="flex items-center space-x-4">
              <select
                value={selectedTimeRange}
                onChange={(e) => setSelectedTimeRange(e.target.value)}
                className="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="1h">Last Hour</option>
                <option value="24h">Last 24 Hours</option>
                <option value="7d">Last 7 Days</option>
                <option value="30d">Last 30 Days</option>
              </select>
              <div className="text-sm text-gray-500">
                Updated: {new Date().toLocaleString()}
              </div>
            </div>
          </div>
        </div>

        {/* Tab Navigation */}
        <div className="mb-8">
          <div className="flex space-x-2 border-b border-gray-200">
            <TabButton
              id="performance"
              label="Model Performance"
              icon={Brain}
              isActive={activeTab === 'performance'}
              onClick={setActiveTab}
            />
            <TabButton
              id="terminals"
              label="Terminal Analytics"
              icon={Activity}
              isActive={activeTab === 'terminals'}
              onClick={setActiveTab}
            />
            <TabButton
              id="patterns"
              label="Pattern Analysis"
              icon={BarChart3}
              isActive={activeTab === 'patterns'}
              onClick={setActiveTab}
            />
            <TabButton
              id="cash"
              label="Cash Analytics"
              icon={DollarSign}
              isActive={activeTab === 'cash'}
              onClick={setActiveTab}
            />
            <TabButton
              id="risk"
              label="Risk Assessment"
              icon={Shield}
              isActive={activeTab === 'risk'}
              onClick={setActiveTab}
            />
          </div>
        </div>

        {/* Model Performance Tab */}
        {activeTab === 'performance' && (
          <div className="space-y-6">
            {/* Performance Metrics */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              <MetricCard
                title="Model Accuracy"
                value={`${((analyticsData?.model_performance?.ensemble_model?.accuracy || 0) * 100).toFixed(2)}%`}
                icon={Target}
                color="bg-green-600"
                subtitle="Overall accuracy"
              />
              <MetricCard
                title="Precision"
                value={`${((analyticsData?.model_performance?.ensemble_model?.precision || 0) * 100).toFixed(2)}%`}
                icon={Zap}
                color="bg-blue-600"
                subtitle="Anomaly detection precision"
              />
              <MetricCard
                title="Recall"
                value={`${((analyticsData?.model_performance?.ensemble_model?.recall || 0) * 100).toFixed(2)}%`}
                icon={Database}
                color="bg-purple-600"
                subtitle="Anomaly detection recall"
              />
              <MetricCard
                title="F1 Score"
                value={(analyticsData?.model_performance?.ensemble_model?.f1_score || 0).toFixed(3)}
                icon={BarChart3}
                color="bg-orange-600"
                subtitle="Harmonic mean"
              />
            </div>

            {/* Performance Charts */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Anomaly Trends */}
              <div className="bg-white rounded-lg shadow-md p-6">
                <h3 className="text-lg font-semibold mb-4">Anomaly Detection Trends</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={analyticsData.anomaly_trends}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="timestamp" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Line 
                      type="monotone" 
                      dataKey="anomaly_count" 
                      stroke="#EF4444" 
                      strokeWidth={2}
                      name="Anomalies"
                    />
                    <Line 
                      type="monotone" 
                      dataKey="accuracy" 
                      stroke="#10B981" 
                      strokeWidth={2}
                      name="Accuracy"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              {/* Feature Importance */}
              <div className="bg-white rounded-lg shadow-md p-6">
                <h3 className="text-lg font-semibold mb-4">Feature Importance</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={analyticsData.model_performance.feature_importance}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="feature" angle={-45} textAnchor="end" height={100} />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="importance" fill="#3B82F6" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        )}

        {/* Terminal Analytics Tab */}
        {activeTab === 'terminals' && (
          <div className="space-y-6">
            {/* Terminal Metrics */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              <MetricCard
                title="High Risk Terminals"
                value={analyticsData?.terminal_analytics?.filter(t => t.risk_level === 'HIGH')?.length || 0}
                icon={AlertTriangle}
                color="bg-red-600"
                subtitle="Immediate attention needed"
              />
              <MetricCard
                title="Avg Risk Score"
                value={analyticsData?.terminal_analytics?.length > 0 ? 
                  (analyticsData.terminal_analytics.reduce((sum, t) => sum + (t.avg_anomaly_score || 0), 0) / analyticsData.terminal_analytics.length).toFixed(2) : 
                  '0.00'}
                icon={Shield}
                color="bg-yellow-600"
                subtitle="Across all terminals"
              />
              <MetricCard
                title="Active Terminals"
                value={analyticsData?.terminal_analytics?.length || 0}
                icon={Activity}
                color="bg-green-600"
                subtitle="Currently monitored"
              />
              <MetricCard
                title="Performance Score"
                value={analyticsData?.terminal_analytics?.length > 0 ? 
                  `${(100 - (analyticsData.terminal_analytics.reduce((sum, t) => sum + (t.anomaly_rate || 0), 0) / analyticsData.terminal_analytics.length)).toFixed(1)}%` :
                  '0.0%'}
                icon={TrendingUp}
                color="bg-blue-600"
                subtitle="Overall performance"
              />
            </div>

            {/* Terminal Charts */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Risk Distribution */}
              <div className="bg-white rounded-lg shadow-md p-6">
                <h3 className="text-lg font-semibold mb-4">Terminal Risk Distribution</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie
                      data={analyticsData?.terminal_analytics ? [
                        {name: 'HIGH', count: analyticsData.terminal_analytics.filter(t => t.risk_level === 'HIGH').length},
                        {name: 'MEDIUM', count: analyticsData.terminal_analytics.filter(t => t.risk_level === 'MEDIUM').length},
                        {name: 'LOW', count: analyticsData.terminal_analytics.filter(t => t.risk_level === 'LOW').length}
                      ].filter(item => item.count > 0) : []}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      label={({ name, value }) => `${name}: ${value}`}
                      outerRadius={80}
                      fill="#8884d8"
                      dataKey="count"
                    >
                      {(analyticsData?.terminal_analytics ? [
                        {name: 'HIGH', count: analyticsData.terminal_analytics.filter(t => t.risk_level === 'HIGH').length},
                        {name: 'MEDIUM', count: analyticsData.terminal_analytics.filter(t => t.risk_level === 'MEDIUM').length},
                        {name: 'LOW', count: analyticsData.terminal_analytics.filter(t => t.risk_level === 'LOW').length}
                      ].filter(item => item.count > 0) : []).map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </div>

              {/* Terminal Performance */}
              <div className="bg-white rounded-lg shadow-md p-6">
                <h3 className="text-lg font-semibold mb-4">Terminal Performance Metrics</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <ScatterChart data={analyticsData?.terminal_analytics?.map(t => ({
                    risk_score: t.avg_anomaly_score || 0,
                    anomaly_count: t.anomaly_count || 0,
                    terminal_id: t.terminal_id
                  })) || []}>
                    <CartesianGrid />
                    <XAxis dataKey="risk_score" name="Risk Score" />
                    <YAxis dataKey="anomaly_count" name="Anomaly Count" />
                    <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                    <Scatter name="Terminals" data={analyticsData?.terminal_analytics?.map(t => ({
                      risk_score: t.avg_anomaly_score || 0,
                      anomaly_count: t.anomaly_count || 0,
                      terminal_id: t.terminal_id
                    })) || []} fill="#8884d8" />
                  </ScatterChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        )}

        {/* Pattern Analysis Tab */}
        {activeTab === 'patterns' && (
          <div className="space-y-6">
            {/* Pattern Metrics */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              <MetricCard
                title="Unique Patterns"
                value={analyticsData?.pattern_analysis?.most_common_patterns?.length || 0}
                icon={BarChart3}
                color="bg-indigo-600"
                subtitle="Detected patterns"
              />
                            <MetricCard
                title="Top Pattern"
                value={analyticsData?.pattern_analysis?.most_common_patterns?.[0]?.pattern || 'No data'}
                icon={TrendingUp}
                color="bg-green-600"
                subtitle={`${analyticsData?.pattern_analysis?.most_common_patterns?.[0]?.count || 0} occurrences`}
              />
              <MetricCard
                title="Pattern Confidence"
                value={`${(analyticsData?.pattern_analysis?.most_common_patterns?.[0]?.percentage || 0).toFixed(1)}%`}
                icon={Target}
                color="bg-blue-600"
                subtitle="Pattern reliability"
              />
              <MetricCard
                title="Recent Patterns"
                value={analyticsData?.pattern_analysis?.most_common_patterns?.length || 0}
                icon={Clock}
                color="bg-purple-600"
                subtitle="Pattern types"
              />
            </div>

            {/* Pattern Charts */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Pattern Types Distribution */}
              <div className="bg-white rounded-lg shadow-md p-6">
                <h3 className="text-lg font-semibold mb-4">Pattern Types Distribution</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={analyticsData?.pattern_analysis?.most_common_patterns?.map(p => ({
                    pattern_type: p.pattern,
                    count: p.count
                  })) || []}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="pattern_type" angle={-45} textAnchor="end" height={100} />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="count" fill="#6366F1" />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              {/* Recent Patterns Timeline */}
              <div className="bg-white rounded-lg shadow-md p-6">
                <h3 className="text-lg font-semibold mb-4">Recent Pattern Timeline</h3>
                <div className="space-y-3 max-h-80 overflow-y-auto">
                  {(analyticsData?.pattern_analysis?.most_common_patterns || []).map((pattern, index) => (
                    <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-md">
                      <div>
                        <span className="font-medium text-gray-900">{pattern?.pattern || 'Unknown'}</span>
                        <div className="text-sm text-gray-500">
                          Count: {pattern?.count || 0} ({(pattern?.percentage || 0).toFixed(1)}%)
                        </div>
                      </div>
                      <div className="text-sm text-gray-400">
                        Common Pattern
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Cash Analytics Tab */}
        {activeTab === 'cash' && (
          <div className="space-y-6">
            {/* Cash Metrics */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                            <MetricCard
                title="Cash Monitored"
                value={`$${((analyticsData.cash_analytics?.total_monitored_cash || 0) / 1000000).toFixed(1)}M`}
                icon={DollarSign}
                color="bg-green-600"
                subtitle="Total system cash"
              />
              <MetricCard
                title="Avg Daily Usage"
                value={`$${(analyticsData.cash_analytics?.daily_dispensing_trend?.[0]?.amount || 0).toLocaleString()}`}
                icon={TrendingUp}
                color="bg-blue-600"
                subtitle="Per terminal"
              />
              <MetricCard
                title="Forecast Accuracy"
                value={`${(analyticsData.cash_analytics?.forecasting_accuracy?.last_30_days ? (analyticsData.cash_analytics.forecasting_accuracy.last_30_days * 100).toFixed(1) : '0.0')}%`}
                icon={Target}
                color="bg-purple-600"
                subtitle="24-hour predictions"
              />
              <MetricCard
                title="Predicted Depletions"
                value={analyticsData.cash_analytics?.terminal_cash_levels?.filter(t => t.risk === 'HIGH')?.length || 0}
                icon={AlertTriangle}
                color="bg-red-600"
                subtitle="Next 24 hours"
              />
            </div>

            {/* Cash Charts */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Cash Usage Trends */}
              <div className="bg-white rounded-lg shadow-md p-6">
                <h3 className="text-lg font-semibold mb-4">Cash Usage Trends</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <AreaChart data={analyticsData.cash_analytics?.daily_dispensing_trend || []}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Area 
                      type="monotone" 
                      dataKey="amount" 
                      stroke="#10B981" 
                      fill="#10B981" 
                      fillOpacity={0.6}
                      name="Daily Amount"
                    />
                    <Area 
                      type="monotone" 
                      dataKey="transactions" 
                      stroke="#3B82F6" 
                      fill="#3B82F6" 
                      fillOpacity={0.4}
                      name="Transactions"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>

              {/* Low Cash Terminals */}
              <div className="bg-white rounded-lg shadow-md p-6">
                <h3 className="text-lg font-semibold mb-4">Cash Levels by Terminal</h3>
                <div className="space-y-3 max-h-80 overflow-y-auto">
                  {(analyticsData.cash_analytics?.terminal_cash_levels || []).map((terminal, index) => (
                    <div key={index} className={`flex items-center justify-between p-3 rounded-md ${
                      terminal.risk === 'HIGH' ? 'bg-red-50' :
                      terminal.risk === 'MEDIUM' ? 'bg-yellow-50' : 'bg-green-50'
                    }`}>
                      <div>
                        <span className={`font-medium ${
                          terminal.risk === 'HIGH' ? 'text-red-900' :
                          terminal.risk === 'MEDIUM' ? 'text-yellow-900' : 'text-green-900'
                        }`}>{terminal.terminal_id}</span>
                        <div className={`text-sm ${
                          terminal.risk === 'HIGH' ? 'text-red-600' :
                          terminal.risk === 'MEDIUM' ? 'text-yellow-600' : 'text-green-600'
                        }`}>
                          Current: ${terminal.cash_level.toLocaleString()}
                        </div>
                      </div>
                      <div className="text-right">
                        <div className={`text-sm font-medium ${
                          terminal.risk === 'HIGH' ? 'text-red-700' :
                          terminal.risk === 'MEDIUM' ? 'text-yellow-700' : 'text-green-700'
                        }`}>
                          {terminal.risk} Risk
                        </div>
                        <div className={`text-xs ${
                          terminal?.risk === 'HIGH' ? 'text-red-500' :
                          terminal?.risk === 'MEDIUM' ? 'text-yellow-500' : 'text-green-500'
                        }`}>
                          {(terminal?.percentage || 0).toFixed(1)}% capacity
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Risk Assessment Tab */}
        {activeTab === 'risk' && (
          <div className="space-y-6">
            {/* Risk Metrics */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              <MetricCard
                title="Overall Risk Score"
                value={(analyticsData?.risk_assessment?.overall_risk_score || 0).toFixed(2)}
                icon={Shield}
                color="bg-orange-600"
                subtitle="System-wide risk"
              />
              <MetricCard
                title="High Risk Events"
                value={analyticsData?.risk_assessment?.risk_distribution?.high || 0}
                icon={AlertTriangle}
                color="bg-red-600"
                subtitle="Immediate attention"
              />
              <MetricCard
                title="Risk Trend"
                value={(analyticsData?.risk_assessment?.risk_factors?.[0]?.trend || "stable")}
                icon={TrendingUp}
                color={(analyticsData?.risk_assessment?.risk_factors?.[0]?.trend || "").includes("increas") ? "bg-red-600" : "bg-green-600"}
                subtitle={`${(analyticsData?.risk_assessment?.risk_factors?.[0]?.impact || 0).toFixed(1)} impact score`}
              />
              <MetricCard
                title="Critical Risks"
                value={analyticsData?.risk_assessment?.risk_distribution?.critical || 0}
                icon={Settings}
                color="bg-blue-600"
                subtitle="High priority items"
              />
            </div>

            {/* Risk Assessment Details */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Risk Categories */}
              <div className="bg-white rounded-lg shadow-md p-6">
                <h3 className="text-lg font-semibold mb-4">Risk Factors Impact</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={(analyticsData?.risk_assessment?.risk_factors || []).map(factor => ({
                    category: factor.factor,
                    score: factor.impact
                  }))}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="category" angle={-45} textAnchor="end" height={100} />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="score" fill="#F59E0B" />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              {/* Risk Factors Details */}
              <div className="bg-white rounded-lg shadow-md p-6">
                <h3 className="text-lg font-semibold mb-4">Risk Factors Analysis</h3>
                <div className="space-y-3 max-h-80 overflow-y-auto">
                  {(analyticsData?.risk_assessment?.risk_factors || []).map((factor, index) => (
                    <div key={index} className="p-3 bg-blue-50 rounded-md">
                      <div className="font-medium text-blue-900">{factor.factor}</div>
                      <div className="text-sm text-blue-600 mt-1">
                        Impact: {factor.impact} • Trend: {factor.trend}
                      </div>
                      <div className="text-xs text-blue-500 mt-1">
                        Risk factor trending {factor.trend}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default AnalyticsPage;
