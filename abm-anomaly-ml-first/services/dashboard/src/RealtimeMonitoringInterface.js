import React, { useState, useEffect, useRef } from 'react';
import { 
  Activity, 
  Database, 
  Brain, 
  FileText, 
  Cpu, 
  Clock, 
  AlertCircle, 
  CheckCircle, 
  XCircle, 
  RefreshCw,
  Monitor,
  BarChart3,
  Zap,
  Settings,
  Play,
  Pause,
  Square
} from 'lucide-react';
import { LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const RealtimeMonitoringInterface = () => {
  const [isMonitoring, setIsMonitoring] = useState(false);
  const [monitoringData, setMonitoringData] = useState({
    parsing: {
      status: 'idle',
      files_processed: 0,
      current_file: null,
      lines_processed: 0,
      processing_rate: 0,
      errors: 0,
      last_error: null
    },
    sessionization: {
      status: 'idle',
      sessions_created: 0,
      current_session: null,
      avg_session_length: 0,
      sessionization_rate: 0,
      errors: 0,
      last_error: null
    },
    ml_training: {
      status: 'idle',
      model_status: 'not_loaded',
      embeddings_generated: 0,
      training_progress: 0,
      current_batch: 0,
      total_batches: 0,
      accuracy: 0,
      loss: 0,
      errors: 0,
      last_error: null
    },
    system: {
      cpu_usage: 0,
      memory_usage: 0,
      disk_usage: 0,
      active_processes: 0,
      uptime: 0
    }
  });

  const [performanceHistory, setPerformanceHistory] = useState([]);
  const [logs, setLogs] = useState([]);
  const [filters, setFilters] = useState({
    logLevel: 'all',
    component: 'all',
    autoScroll: true
  });

  const logsEndRef = useRef(null);
  const websocketRef = useRef(null);
  const intervalRef = useRef(null);

  // Auto-scroll logs to bottom
  const scrollToBottom = () => {
    if (filters.autoScroll) {
      logsEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  };

  useEffect(() => {
    scrollToBottom();
  }, [logs]);

  // WebSocket connection for real-time updates
  useEffect(() => {
    if (isMonitoring) {
      connectWebSocket();
      startPolling();
    } else {
      disconnectWebSocket();
      stopPolling();
    }

    return () => {
      disconnectWebSocket();
      stopPolling();
    };
  }, [isMonitoring]);

  const connectWebSocket = () => {
    try {
      const wsUrl = API_URL.replace('http://', 'ws://').replace('https://', 'wss://');
      websocketRef.current = new WebSocket(`${wsUrl}/ws/monitoring`);

      websocketRef.current.onopen = () => {
        addLog('WebSocket connected', 'info', 'system');
      };

      websocketRef.current.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleRealtimeUpdate(data);
      };

      websocketRef.current.onerror = (error) => {
        addLog(`WebSocket error: ${error.message}`, 'error', 'system');
      };

      websocketRef.current.onclose = () => {
        addLog('WebSocket disconnected', 'warning', 'system');
      };
    } catch (error) {
      addLog(`Failed to connect WebSocket: ${error.message}`, 'error', 'system');
    }
  };

  const disconnectWebSocket = () => {
    if (websocketRef.current) {
      websocketRef.current.close();
      websocketRef.current = null;
    }
  };

  const startPolling = () => {
    intervalRef.current = setInterval(fetchMonitoringData, 2000);
  };

  const stopPolling = () => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  };

  const fetchMonitoringData = async () => {
    try {
      const response = await fetch(`${API_URL}/api/v1/monitoring/status`);
      if (response.ok) {
        const data = await response.json();
        
        // Map the API response to the expected component structure
        const mappedData = {
          parsing: {
            status: data.parsing?.status || 'idle',
            files_processed: data.parsing?.processed || 0,
            current_file: null,
            lines_processed: data.parsing?.processed || 0,
            processing_rate: data.parsing?.rate || 0,
            errors: data.parsing?.errors || 0,
            last_error: null
          },
          sessionization: {
            status: data.sessionization?.status || 'idle',
            sessions_created: data.sessionization?.sessions_created || 0,
            current_session: null,
            avg_session_length: 0,
            sessionization_rate: data.sessionization?.rate || 0,
            errors: 0,
            last_error: null
          },
          ml_training: {
            status: data.ml_training?.status || 'idle',
            model_status: 'loaded',
            embeddings_generated: 0,
            training_progress: data.ml_training?.accuracy || 0,
            current_accuracy: data.ml_training?.accuracy || 0,
            training_time: data.ml_training?.training_time || 0,
            models_trained: data.ml_training?.models_trained || 0,
            errors: 0,
            last_error: null
          },
          system: {
            cpu_usage: data.system?.cpu || 0,
            memory_usage: data.system?.memory || 0,
            disk_usage: data.system?.disk || 0,
            uptime: data.system?.uptime || 0,
            active_connections: 0,
            service_status: 'healthy'
          }
        };
        
        setMonitoringData(mappedData);
        
        // Add to performance history
        const timestamp = new Date().toLocaleTimeString();
        setPerformanceHistory(prev => {
          const newEntry = {
            timestamp,
            cpu: mappedData.system.cpu_usage,
            memory: mappedData.system.memory_usage,
            parsing_rate: mappedData.parsing.processing_rate,
            sessionization_rate: mappedData.sessionization.sessionization_rate
          };
          return [...prev.slice(-29), newEntry]; // Keep last 30 entries
        });
      }
    } catch (error) {
      addLog(`Failed to fetch monitoring data: ${error.message}`, 'error', 'system');
    }
  };

  const handleRealtimeUpdate = (data) => {
    // Handle the actual monitoring status format from WebSocket
    if (data.parsing && data.sessionization && data.ml_training && data.system) {
      // This is the full monitoring status update
      const mappedData = {
        parsing: {
          status: data.parsing?.status || 'idle',
          files_processed: data.parsing?.processed || 0,
          current_file: null,
          lines_processed: data.parsing?.processed || 0,
          processing_rate: data.parsing?.rate || 0,
          errors: data.parsing?.errors || 0,
          last_error: null
        },
        sessionization: {
          status: data.sessionization?.status || 'idle',
          sessions_created: data.sessionization?.sessions_created || 0,
          current_session: null,
          avg_session_length: 0,
          sessionization_rate: data.sessionization?.rate || 0,
          errors: 0,
          last_error: null
        },
        ml_training: {
          status: data.ml_training?.status || 'idle',
          model_status: 'loaded',
          embeddings_generated: 0,
          training_progress: data.ml_training?.accuracy || 0,
          current_accuracy: data.ml_training?.accuracy || 0,
          training_time: data.ml_training?.training_time || 0,
          models_trained: data.ml_training?.models_trained || 0,
          errors: 0,
          last_error: null
        },
        system: {
          cpu_usage: data.system?.cpu || 0,
          memory_usage: data.system?.memory || 0,
          disk_usage: data.system?.disk || 0,
          uptime: data.system?.uptime || 0,
          active_connections: 0,
          service_status: 'healthy'
        }
      };
      
      setMonitoringData(mappedData);
      addLog('Real-time monitoring data updated', 'info', 'system');
      return;
    }

    // Handle legacy message format (if any)
    switch (data.type) {
      case 'parsing_update':
        setMonitoringData(prev => ({
          ...prev,
          parsing: { ...prev.parsing, ...data.payload }
        }));
        addLog(`Parsing: ${data.message}`, 'info', 'parsing');
        break;
      
      case 'sessionization_update':
        setMonitoringData(prev => ({
          ...prev,
          sessionization: { ...prev.sessionization, ...data.payload }
        }));
        addLog(`Sessionization: ${data.message}`, 'info', 'sessionization');
        break;
      
      case 'ml_training_update':
        setMonitoringData(prev => ({
          ...prev,
          ml_training: { ...prev.ml_training, ...data.payload }
        }));
        addLog(`ML Training: ${data.message}`, 'info', 'ml_training');
        break;
      
      case 'error':
        addLog(data.message, 'error', data.component);
        break;
      
      default:
        addLog(data.message || 'Unknown update received', 'info', 'system');
    }
  };

  const addLog = (message, level, component) => {
    const timestamp = new Date().toLocaleTimeString();
    const newLog = {
      id: Date.now() + Math.random(),
      timestamp,
      level,
      component,
      message
    };

    setLogs(prev => [...prev.slice(-199), newLog]); // Keep last 200 logs
  };

  const toggleMonitoring = () => {
    setIsMonitoring(!isMonitoring);
    if (!isMonitoring) {
      addLog('Real-time monitoring started', 'info', 'system');
    } else {
      addLog('Real-time monitoring stopped', 'info', 'system');
    }
  };

  const clearLogs = () => {
    setLogs([]);
    addLog('Logs cleared', 'info', 'system');
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'running':
      case 'active':
        return <CheckCircle className="w-5 h-5 text-green-500" />;
      case 'error':
      case 'failed':
        return <XCircle className="w-5 h-5 text-red-500" />;
      case 'loading':
      case 'processing':
        return <RefreshCw className="w-5 h-5 text-blue-500 animate-spin" />;
      default:
        return <Clock className="w-5 h-5 text-gray-500" />;
    }
  };

  const getLogIcon = (level) => {
    switch (level) {
      case 'error':
        return <XCircle className="w-4 h-4 text-red-500" />;
      case 'warning':
        return <AlertCircle className="w-4 h-4 text-yellow-500" />;
      case 'info':
        return <CheckCircle className="w-4 h-4 text-blue-500" />;
      default:
        return <Clock className="w-4 h-4 text-gray-500" />;
    }
  };

  const filteredLogs = logs.filter(log => {
    const levelMatch = filters.logLevel === 'all' || log.level === filters.logLevel;
    const componentMatch = filters.component === 'all' || log.component === filters.component;
    return levelMatch && componentMatch;
  });

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="flex justify-between items-center">
          <div className="flex items-center">
            <Monitor className="w-8 h-8 text-purple-600 mr-3" />
            <div>
              <h1 className="text-2xl font-bold text-gray-900">Real-time System Monitoring</h1>
              <p className="text-gray-600 mt-1">Monitor parsing, sessionization, and ML training processes</p>
            </div>
          </div>
          
          <div className="flex items-center space-x-4">
            <button
              onClick={toggleMonitoring}
              className={`flex items-center px-4 py-2 rounded-lg font-medium ${
                isMonitoring 
                  ? 'bg-red-600 text-white hover:bg-red-700' 
                  : 'bg-green-600 text-white hover:bg-green-700'
              }`}
            >
              {isMonitoring ? (
                <>
                  <Square className="w-4 h-4 mr-2" />
                  Stop Monitoring
                </>
              ) : (
                <>
                  <Play className="w-4 h-4 mr-2" />
                  Start Monitoring
                </>
              )}
            </button>
            
            <button
              onClick={clearLogs}
              className="flex items-center px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50"
            >
              <RefreshCw className="w-4 h-4 mr-2" />
              Clear Logs
            </button>
          </div>
        </div>
      </div>

      {/* Status Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {/* Parsing Status */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center">
              <FileText className="w-6 h-6 text-blue-600 mr-2" />
              <h3 className="text-lg font-semibold">Parsing</h3>
            </div>
            {getStatusIcon(monitoringData.parsing.status)}
          </div>
          
          <div className="space-y-2">
            <div className="flex justify-between">
              <span className="text-sm text-gray-600">Files Processed:</span>
              <span className="text-sm font-medium">{monitoringData.parsing.files_processed || 0}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-sm text-gray-600">Lines/sec:</span>
              <span className="text-sm font-medium">{(monitoringData.parsing.processing_rate || 0).toFixed(1)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-sm text-gray-600">Errors:</span>
              <span className={`text-sm font-medium ${(monitoringData.parsing.errors || 0) > 0 ? 'text-red-600' : 'text-green-600'}`}>
                {monitoringData.parsing.errors || 0}
              </span>
            </div>
            {monitoringData.parsing.current_file && (
              <div className="mt-2 p-2 bg-gray-50 rounded text-xs">
                Current: {monitoringData.parsing.current_file}
              </div>
            )}
          </div>
        </div>

        {/* Sessionization Status */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center">
              <Database className="w-6 h-6 text-green-600 mr-2" />
              <h3 className="text-lg font-semibold">Sessionization</h3>
            </div>
            {getStatusIcon(monitoringData.sessionization.status)}
          </div>
          
          <div className="space-y-2">
            <div className="flex justify-between">
              <span className="text-sm text-gray-600">Sessions Created:</span>
              <span className="text-sm font-medium">{monitoringData.sessionization.sessions_created || 0}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-sm text-gray-600">Sessions/sec:</span>
              <span className="text-sm font-medium">{(monitoringData.sessionization.sessionization_rate || 0).toFixed(1)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-sm text-gray-600">Avg Length:</span>
              <span className="text-sm font-medium">{(monitoringData.sessionization.avg_session_length || 0).toFixed(0)} chars</span>
            </div>
            <div className="flex justify-between">
              <span className="text-sm text-gray-600">Errors:</span>
              <span className={`text-sm font-medium ${(monitoringData.sessionization.errors || 0) > 0 ? 'text-red-600' : 'text-green-600'}`}>
                {monitoringData.sessionization.errors || 0}
              </span>
            </div>
          </div>
        </div>

        {/* ML Training Status */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center">
              <Brain className="w-6 h-6 text-purple-600 mr-2" />
              <h3 className="text-lg font-semibold">ML Training</h3>
            </div>
            {getStatusIcon(monitoringData.ml_training.status)}
          </div>
          
          <div className="space-y-2">
            <div className="flex justify-between">
              <span className="text-sm text-gray-600">Model Status:</span>
              <span className="text-sm font-medium capitalize">{monitoringData.ml_training.model_status.replace('_', ' ')}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-sm text-gray-600">Embeddings:</span>
              <span className="text-sm font-medium">{monitoringData.ml_training.embeddings_generated || 0}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-sm text-gray-600">Progress:</span>
              <span className="text-sm font-medium">{(monitoringData.ml_training.training_progress || 0).toFixed(1)}%</span>
            </div>
            {(monitoringData.ml_training.training_progress || 0) > 0 && (
              <div className="w-full bg-gray-200 rounded-full h-2 mt-2">
                <div 
                  className="bg-purple-600 h-2 rounded-full" 
                  style={{ width: `${monitoringData.ml_training.training_progress || 0}%` }}
                ></div>
              </div>
            )}
          </div>
        </div>

        {/* System Resources */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center">
              <Cpu className="w-6 h-6 text-orange-600 mr-2" />
              <h3 className="text-lg font-semibold">System Resources</h3>
            </div>
            <Activity className="w-5 h-5 text-gray-500" />
          </div>
          
          <div className="space-y-2">
            <div className="flex justify-between">
              <span className="text-sm text-gray-600">CPU Usage:</span>
              <span className="text-sm font-medium">{(monitoringData.system.cpu_usage || 0).toFixed(1)}%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-sm text-gray-600">Memory:</span>
              <span className="text-sm font-medium">{(monitoringData.system.memory_usage || 0).toFixed(1)}%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-sm text-gray-600">Processes:</span>
              <span className="text-sm font-medium">{monitoringData.system.active_processes}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-sm text-gray-600">Uptime:</span>
              <span className="text-sm font-medium">{Math.floor(monitoringData.system.uptime / 3600)}h</span>
            </div>
          </div>
        </div>
      </div>

      {/* Performance Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* System Performance */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-lg font-semibold mb-4 flex items-center">
            <BarChart3 className="w-5 h-5 mr-2" />
            System Performance
          </h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={performanceHistory}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="timestamp" />
                <YAxis />
                <Tooltip />
                <Line type="monotone" dataKey="cpu" stroke="#8884d8" name="CPU %" />
                <Line type="monotone" dataKey="memory" stroke="#82ca9d" name="Memory %" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Processing Rates */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-lg font-semibold mb-4 flex items-center">
            <Zap className="w-5 h-5 mr-2" />
            Processing Rates
          </h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={performanceHistory}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="timestamp" />
                <YAxis />
                <Tooltip />
                <Area type="monotone" dataKey="parsing_rate" stackId="1" stroke="#8884d8" fill="#8884d8" name="Parsing Rate" />
                <Area type="monotone" dataKey="sessionization_rate" stackId="1" stroke="#82ca9d" fill="#82ca9d" name="Sessionization Rate" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Real-time Logs */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-lg font-semibold flex items-center">
            <FileText className="w-5 h-5 mr-2" />
            Real-time Logs
          </h3>
          
          <div className="flex items-center space-x-4">
            <select
              value={filters.logLevel}
              onChange={(e) => setFilters({ ...filters, logLevel: e.target.value })}
              className="px-3 py-1 border border-gray-300 rounded text-sm"
            >
              <option value="all">All Levels</option>
              <option value="info">Info</option>
              <option value="warning">Warning</option>
              <option value="error">Error</option>
            </select>
            
            <select
              value={filters.component}
              onChange={(e) => setFilters({ ...filters, component: e.target.value })}
              className="px-3 py-1 border border-gray-300 rounded text-sm"
            >
              <option value="all">All Components</option>
              <option value="parsing">Parsing</option>
              <option value="sessionization">Sessionization</option>
              <option value="ml_training">ML Training</option>
              <option value="system">System</option>
            </select>
            
            <label className="flex items-center text-sm">
              <input
                type="checkbox"
                checked={filters.autoScroll}
                onChange={(e) => setFilters({ ...filters, autoScroll: e.target.checked })}
                className="mr-2"
              />
              Auto-scroll
            </label>
          </div>
        </div>
        
        <div className="bg-gray-900 text-white p-4 rounded-lg h-96 overflow-y-auto font-mono text-sm">
          {filteredLogs.length === 0 ? (
            <div className="text-gray-400 text-center py-8">No logs to display</div>
          ) : (
            filteredLogs.map(log => (
              <div key={log.id} className="flex items-start space-x-2 mb-1">
                <span className="text-gray-400 text-xs whitespace-nowrap">{log.timestamp}</span>
                {getLogIcon(log.level)}
                <span className="text-xs text-gray-300 uppercase min-w-0 flex-shrink-0">[{log.component}]</span>
                <span className="text-sm break-words">{log.message}</span>
              </div>
            ))
          )}
          <div ref={logsEndRef} />
        </div>
      </div>
    </div>
  );
};

export default RealtimeMonitoringInterface;
