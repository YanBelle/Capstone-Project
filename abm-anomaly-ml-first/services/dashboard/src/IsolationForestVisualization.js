import React, { useState, useEffect } from 'react';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BoxPlot, ComposedChart, Bar, Line, ReferenceLine } from 'recharts';
import { AlertCircle, Activity, TrendingUp, FileText, BarChart3, BarChart2, Box } from 'lucide-react';
import apiConfig from './config/api';

const IsolationForestVisualization = () => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [analysisData, setAnalysisData] = useState(null);
  const [activeView, setActiveView] = useState('overview');
  const [modelInfo, setModelInfo] = useState(null);
  const [apiStatus, setApiStatus] = useState('connecting'); // 'connecting', 'connected', 'fallback'
  const [dataSource, setDataSource] = useState('unknown'); // 'api', 'mock'

  // Fetch Isolation Forest analysis data
  const fetchAnalysisData = async () => {
    try {
      setLoading(true);
      setError(null);
      setApiStatus('connecting');

      // Get model info first
      try {
        const modelResponse = await fetch(apiConfig.endpoint('/api/model_info'));
        if (modelResponse.ok) {
          const modelData = await modelResponse.json();
          setModelInfo(modelData);
        }
      } catch (err) {
        console.log('Model info not available, continuing with analysis');
      }

      // Try multiple endpoint paths for Isolation Forest analysis
      const endpointCandidates = [
        '/api/v1/isolation-forest/analysis',
        '/api/v1/isolation',               // Simpler working endpoint
        '/api/v1/isolation/analysis', 
        '/api/v1/dashboard/isolation-forest',
        '/api/v1/test-isolation'           // Test endpoint
      ];

      let dataFetched = false;
      
      for (const endpoint of endpointCandidates) {
        try {
          console.log(`🔍 Trying endpoint: ${endpoint}`);
          const response = await fetch(apiConfig.endpoint(endpoint), {
            method: 'GET',
            headers: {
              'Content-Type': 'application/json',
            }
          });

          if (response.ok) {
            const data = await response.json();
            console.log(`✅ Success with endpoint: ${endpoint}`, data);
            
            // Check if this is our actual data or just a test response
            if (data.total_sessions || data.scatter_data) {
              setAnalysisData(data);
              setDataSource('api');
              setApiStatus('connected');
              dataFetched = true;
              break;
            } else if (data.message && endpoint.includes('test')) {
              console.log('🧪 Test endpoint working, but main endpoint still needs fixing');
              setApiStatus('partial');
            }
          } else {
            console.log(`❌ Failed endpoint: ${endpoint} (${response.status})`);
          }
        } catch (err) {
          console.log(`❌ Error with endpoint: ${endpoint}`, err.message);
        }
      }

      if (!dataFetched) {
        console.log('📊 All API endpoints failed, using comprehensive mock data');
        setApiStatus('fallback');
        setDataSource('mock');
        // Use enhanced mock data that matches API structure exactly
        const mockData = {
          total_sessions: 1500,
          normal_sessions: 1350,
          anomalous_sessions: 150,
          scatter_data: Array.from({ length: 200 }, (_, i) => ({
            features: [
              Math.random() * 10 - 5 + (Math.random() > 0.9 ? 3 : 0),
              Math.random() * 10 - 5 + (Math.random() > 0.9 ? 3 : 0)
            ],
            anomaly_score: Math.random() > 0.9 ? Math.random() * 0.4 + 0.6 : Math.random() * 0.5,
            is_anomaly: Math.random() > 0.9,
            session_id: `session_${i + 1}`
          })),
          feature_distributions: [
            {
              name: 'Session Length',
              normal_values: Array.from({ length: 100 }, () => Math.random() * 50 + 10),
              anomaly_values: Array.from({ length: 15 }, () => Math.random() * 30 + 70),
              normal_q1: 15, normal_median: 25, normal_q3: 35,
              anomaly_q1: 75, anomaly_median: 85, anomaly_q3: 95
            },
            {
              name: 'Unique Events Count',
              normal_values: Array.from({ length: 100 }, () => Math.random() * 20 + 5),
              anomaly_values: Array.from({ length: 15 }, () => Math.random() * 15 + 30),
              normal_q1: 8, normal_median: 15, normal_q3: 22,
              anomaly_q1: 32, anomaly_median: 38, anomaly_q3: 42
            },
            {
              name: 'Event Frequency',
              normal_values: Array.from({ length: 100 }, () => Math.random() * 5 + 1),
              anomaly_values: Array.from({ length: 15 }, () => Math.random() * 3 + 8),
              normal_q1: 1.5, normal_median: 3, normal_q3: 4.5,
              anomaly_q1: 8.2, anomaly_median: 9.1, anomaly_q3: 10.5
            }
          ],
          score_distribution: Array.from({ length: 10 }, (_, i) => ({
            min: i * 0.1,
            max: (i + 1) * 0.1,
            normal_count: Math.floor(Math.random() * 50 + 10) * (i < 5 ? 3 : 1),
            anomaly_count: Math.floor(Math.random() * 20) * (i > 5 ? 3 : 1)
          })),
          threshold_info: {
            threshold: 0.5,
            description: 'Sessions above this score are classified as anomalies'
          },
          feature_importance: [
            { name: 'Session Length', importance: 0.25 },
            { name: 'Unique Events Count', importance: 0.20 },
            { name: 'Event Frequency', importance: 0.18 },
            { name: 'Pattern Complexity', importance: 0.15 },
            { name: 'Temporal Distribution', importance: 0.12 },
            { name: 'Event Diversity', importance: 0.10 }
          ],
          performance_metrics: {
            precision: 0.87,
            recall: 0.82,
            f1_score: 0.84,
            auc: 0.89,
            confusion_matrix: {
              true_positive: 123,
              false_positive: 18,
              true_negative: 1285,
              false_negative: 27
            }
          }
        };
        setAnalysisData(mockData);
      }
    } catch (err) {
      console.error('Error fetching Isolation Forest analysis:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchAnalysisData();
  }, []);

  // Transform data for scatter plot visualization
  const prepareScatterData = () => {
    if (!analysisData || !analysisData.scatter_data) return [];
    
    return analysisData.scatter_data.map((point, index) => ({
      x: point.features[0],
      y: point.features[1],
      anomaly_score: point.anomaly_score,
      is_anomaly: point.is_anomaly,
      session_id: point.session_id || `session_${index}`,
      color: point.is_anomaly ? '#ef4444' : '#10b981'
    }));
  };

  // Transform data for box plots
  const prepareBoxPlotData = () => {
    if (!analysisData || !analysisData.feature_distributions) return [];

    return analysisData.feature_distributions.map((feature, index) => ({
      feature_name: feature.name,
      normal_data: feature.normal_values,
      anomaly_data: feature.anomaly_values,
      q1_normal: feature.normal_q1,
      median_normal: feature.normal_median,
      q3_normal: feature.normal_q3,
      q1_anomaly: feature.anomaly_q1,
      median_anomaly: feature.anomaly_median,
      q3_anomaly: feature.anomaly_q3,
      outliers_normal: feature.normal_outliers || [],
      outliers_anomaly: feature.anomaly_outliers || []
    }));
  };

  // Custom scatter plot component
  const IsolationForestScatterPlot = ({ data }) => {
    if (!data || data.length === 0) {
      return <div className="text-gray-500">No scatter plot data available</div>;
    }

    return (
      <div className="bg-white p-6 rounded-lg shadow-lg">
        <h3 className="text-lg font-semibold mb-4">Isolation Forest 2D Feature Space</h3>
        <ResponsiveContainer width="100%" height={400}>
          <ScatterChart
            data={data}
            margin={{ top: 20, right: 20, bottom: 20, left: 20 }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
              type="number" 
              dataKey="x" 
              name="Feature 1"
              tick={{ fontSize: 12 }}
            />
            <YAxis 
              type="number" 
              dataKey="y" 
              name="Feature 2"
              tick={{ fontSize: 12 }}
            />
            <Tooltip
              cursor={{ strokeDasharray: '3 3' }}
              content={({ active, payload }) => {
                if (active && payload && payload.length) {
                  const data = payload[0].payload;
                  return (
                    <div className="bg-white p-3 border rounded shadow-lg">
                      <p className="font-semibold">{data.session_id}</p>
                      <p>Feature 1: {data.x.toFixed(3)}</p>
                      <p>Feature 2: {data.y.toFixed(3)}</p>
                      <p>Anomaly Score: {data.anomaly_score.toFixed(3)}</p>
                      <p className={`font-semibold ${data.is_anomaly ? 'text-red-600' : 'text-green-600'}`}>
                        {data.is_anomaly ? 'Anomaly' : 'Normal'}
                      </p>
                    </div>
                  );
                }
                return null;
              }}
            />
            <Scatter
              name="Normal Sessions"
              data={data.filter(d => !d.is_anomaly)}
              fill="#10b981"
              r={4}
            />
            <Scatter
              name="Anomalous Sessions"
              data={data.filter(d => d.is_anomaly)}
              fill="#ef4444"
              r={6}
            />
            <Legend />
          </ScatterChart>
        </ResponsiveContainer>
        <div className="mt-4 text-sm text-gray-600">
          <p>• Green dots: Normal sessions identified by Isolation Forest</p>
          <p>• Red dots: Anomalous sessions detected by Isolation Forest</p>
          <p>• Larger red dots indicate higher anomaly confidence</p>
        </div>
      </div>
    );
  };

  // Custom box plot component using bars to simulate box plots
  const IsolationForestBoxPlots = ({ data }) => {
    if (!data || data.length === 0) {
      return <div className="text-gray-500">No box plot data available</div>;
    }

    return (
      <div className="bg-white p-6 rounded-lg shadow-lg">
        <h3 className="text-lg font-semibold mb-4">Feature Distribution Analysis</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {data.slice(0, 6).map((feature, index) => (
            <div key={index} className="border rounded-lg p-4">
              <h4 className="font-medium text-sm mb-3">{feature.feature_name}</h4>
              <div className="space-y-3">
                {/* Normal distribution */}
                <div>
                  <label className="text-xs text-green-600 font-medium">Normal Sessions</label>
                  <div className="relative bg-green-50 rounded p-2">
                    <div className="flex justify-between text-xs">
                      <span>Q1: {feature.q1_normal?.toFixed(2)}</span>
                      <span>Median: {feature.median_normal?.toFixed(2)}</span>
                      <span>Q3: {feature.q3_normal?.toFixed(2)}</span>
                    </div>
                    <div className="mt-1 bg-green-200 h-4 rounded relative">
                      <div 
                        className="absolute bg-green-500 h-full rounded"
                        style={{
                          left: '25%',
                          width: '50%'
                        }}
                      />
                      <div 
                        className="absolute bg-green-700 w-0.5 h-full"
                        style={{ left: '50%' }}
                      />
                    </div>
                  </div>
                </div>

                {/* Anomaly distribution */}
                <div>
                  <label className="text-xs text-red-600 font-medium">Anomalous Sessions</label>
                  <div className="relative bg-red-50 rounded p-2">
                    <div className="flex justify-between text-xs">
                      <span>Q1: {feature.q1_anomaly?.toFixed(2) || 'N/A'}</span>
                      <span>Median: {feature.median_anomaly?.toFixed(2) || 'N/A'}</span>
                      <span>Q3: {feature.q3_anomaly?.toFixed(2) || 'N/A'}</span>
                    </div>
                    <div className="mt-1 bg-red-200 h-4 rounded relative">
                      <div 
                        className="absolute bg-red-500 h-full rounded"
                        style={{
                          left: '25%',
                          width: '50%'
                        }}
                      />
                      <div 
                        className="absolute bg-red-700 w-0.5 h-full"
                        style={{ left: '50%' }}
                      />
                    </div>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
        <div className="mt-4 text-sm text-gray-600">
          <p>• Box plots show the distribution of feature values for normal vs anomalous sessions</p>
          <p>• Significant differences in distributions indicate discriminative features</p>
        </div>
      </div>
    );
  };

  // Anomaly Score Distribution
  const AnomalyScoreDistribution = () => {
    if (!analysisData || !analysisData.score_distribution) {
      return <div className="text-gray-500">No score distribution data available</div>;
    }

    const distributionData = analysisData.score_distribution.map((bin, index) => ({
      bin: `${bin.min.toFixed(2)}-${bin.max.toFixed(2)}`,
      normal_count: bin.normal_count,
      anomaly_count: bin.anomaly_count,
      total: bin.normal_count + bin.anomaly_count
    }));

    return (
      <div className="bg-white p-6 rounded-lg shadow-lg">
        <h3 className="text-lg font-semibold mb-4">Anomaly Score Distribution</h3>
        <ResponsiveContainer width="100%" height={300}>
          <ComposedChart data={distributionData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
              dataKey="bin" 
              tick={{ fontSize: 11 }}
              angle={-45}
              textAnchor="end"
              height={80}
            />
            <YAxis />
            <Tooltip />
            <Legend />
            <Bar dataKey="normal_count" stackId="a" fill="#10b981" name="Normal Sessions" />
            <Bar dataKey="anomaly_count" stackId="a" fill="#ef4444" name="Anomalous Sessions" />
            <ReferenceLine 
              x={analysisData.threshold_info?.threshold || 0.5} 
              stroke="#ff7300" 
              strokeDasharray="5 5"
              label="Threshold"
            />
          </ComposedChart>
        </ResponsiveContainer>
        <div className="mt-4 text-sm text-gray-600">
          <p>• Distribution of anomaly scores across all sessions</p>
          <p>• Orange dashed line shows the anomaly detection threshold</p>
          <p>• Sessions to the right of the threshold are classified as anomalies</p>
        </div>
      </div>
    );
  };

  // Model Performance Metrics
  const ModelPerformanceMetrics = () => {
    if (!analysisData || !analysisData.performance_metrics) {
      return <div className="text-gray-500">No performance metrics available</div>;
    }

    const metrics = analysisData.performance_metrics;

    return (
      <div className="bg-white p-6 rounded-lg shadow-lg">
        <h3 className="text-lg font-semibold mb-4">Isolation Forest Performance</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="text-center p-4 bg-blue-50 rounded-lg">
            <div className="text-2xl font-bold text-blue-600">{(metrics.precision * 100).toFixed(1)}%</div>
            <div className="text-sm text-blue-700">Precision</div>
          </div>
          <div className="text-center p-4 bg-green-50 rounded-lg">
            <div className="text-2xl font-bold text-green-600">{(metrics.recall * 100).toFixed(1)}%</div>
            <div className="text-sm text-green-700">Recall</div>
          </div>
          <div className="text-center p-4 bg-purple-50 rounded-lg">
            <div className="text-2xl font-bold text-purple-600">{(metrics.f1_score * 100).toFixed(1)}%</div>
            <div className="text-sm text-purple-700">F1 Score</div>
          </div>
          <div className="text-center p-4 bg-orange-50 rounded-lg">
            <div className="text-2xl font-bold text-orange-600">{(metrics.auc * 100).toFixed(1)}%</div>
            <div className="text-sm text-orange-700">AUC</div>
          </div>
        </div>
        
        {metrics.confusion_matrix && (
          <div className="mt-6">
            <h4 className="font-medium mb-3">Confusion Matrix</h4>
            <div className="grid grid-cols-2 gap-2 w-64">
              <div className="text-center p-3 bg-green-100 rounded">
                <div className="text-lg font-bold">{metrics.confusion_matrix.true_negative}</div>
                <div className="text-xs">True Negative</div>
              </div>
              <div className="text-center p-3 bg-red-100 rounded">
                <div className="text-lg font-bold">{metrics.confusion_matrix.false_positive}</div>
                <div className="text-xs">False Positive</div>
              </div>
              <div className="text-center p-3 bg-red-100 rounded">
                <div className="text-lg font-bold">{metrics.confusion_matrix.false_negative}</div>
                <div className="text-xs">False Negative</div>
              </div>
              <div className="text-center p-3 bg-green-100 rounded">
                <div className="text-lg font-bold">{metrics.confusion_matrix.true_positive}</div>
                <div className="text-xs">True Positive</div>
              </div>
            </div>
          </div>
        )}
      </div>
    );
  };

  // Feature Importance Analysis
  const FeatureImportanceAnalysis = () => {
    if (!analysisData || !analysisData.feature_importance) {
      return <div className="text-gray-500">No feature importance data available</div>;
    }

    const importance = analysisData.feature_importance.slice(0, 10); // Top 10 features

    return (
      <div className="bg-white p-6 rounded-lg shadow-lg">
        <h3 className="text-lg font-semibold mb-4">Top 10 Most Important Features</h3>
        <div className="space-y-3">
          {importance.map((feature, index) => (
            <div key={index} className="flex items-center space-x-3">
              <div className="w-32 text-sm font-medium truncate" title={feature.name}>
                {feature.name}
              </div>
              <div className="flex-1 bg-gray-200 rounded-full h-4 relative">
                <div 
                  className="bg-blue-500 h-4 rounded-full"
                  style={{ width: `${(feature.importance * 100)}%` }}
                />
                <span className="absolute inset-0 flex items-center justify-center text-xs text-white font-medium">
                  {(feature.importance * 100).toFixed(1)}%
                </span>
              </div>
            </div>
          ))}
        </div>
        <div className="mt-4 text-sm text-gray-600">
          <p>• Features with higher importance contribute more to anomaly detection</p>
          <p>• Based on the variance reduction in Isolation Forest decision trees</p>
        </div>
      </div>
    );
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-100 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading Isolation Forest analysis...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gray-100 flex items-center justify-center">
        <div className="text-center">
          <AlertCircle className="h-12 w-12 text-red-500 mx-auto mb-4" />
          <h3 className="text-lg font-semibold text-red-800 mb-2">Error Loading Analysis</h3>
          <p className="text-red-700 mb-4">{error}</p>
          <button
            onClick={fetchAnalysisData}
            className="bg-red-600 text-white px-4 py-2 rounded hover:bg-red-700 transition-colors"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  const scatterData = prepareScatterData();
  const boxPlotData = prepareBoxPlotData();

  return (
    <div className="min-h-screen bg-gray-100 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">Isolation Forest Analysis</h1>
          <p className="text-gray-600">
            Comprehensive analysis of the Isolation Forest anomaly detection algorithm
          </p>
          {modelInfo && (
            <div className="mt-4 flex items-center space-x-4 text-sm text-gray-500">
              <span>Model Status: <span className="text-green-600 font-medium">Trained</span></span>
              <span>Training Time: {new Date(modelInfo.training_timestamp).toLocaleString()}</span>
              <span>Contamination: {(modelInfo.contamination * 100).toFixed(1)}%</span>
            </div>
          )}
        </div>

        {/* API Status Indicator */}
        <div className="mb-6">
          <div className={`inline-flex items-center px-3 py-1 rounded-full text-sm ${
            apiStatus === 'connected' ? 'bg-green-100 text-green-800' :
            apiStatus === 'connecting' ? 'bg-yellow-100 text-yellow-800' :
            apiStatus === 'partial' ? 'bg-blue-100 text-blue-800' :
            'bg-orange-100 text-orange-800'
          }`}>
            <div className={`w-2 h-2 rounded-full mr-2 ${
              apiStatus === 'connected' ? 'bg-green-500' :
              apiStatus === 'connecting' ? 'bg-yellow-500 animate-pulse' :
              apiStatus === 'partial' ? 'bg-blue-500' :
              'bg-orange-500'
            }`}></div>
            {apiStatus === 'connected' ? 'API Connected' :
             apiStatus === 'connecting' ? 'Connecting to API...' :
             apiStatus === 'partial' ? 'API Partially Available' :
             'Using Mock Data'}
            <span className="ml-2 text-xs opacity-75">
              ({dataSource === 'api' ? 'Live Data' : 'Demo Data'})
            </span>
          </div>
        </div>

        {/* Navigation Tabs */}
        <div className="mb-6">
          <div className="border-b border-gray-200">
            <nav className="-mb-px flex space-x-8">
              {[
                { id: 'overview', label: 'Overview', icon: Activity },
                { id: 'scatter', label: '2D Scatter Plot', icon: BarChart2 },
                { id: 'boxplots', label: 'Box Plots', icon: Box },
                { id: 'distribution', label: 'Score Distribution', icon: BarChart3 },
                { id: 'performance', label: 'Performance', icon: TrendingUp },
                { id: 'features', label: 'Feature Importance', icon: FileText }
              ].map(({ id, label, icon: Icon }) => (
                <button
                  key={id}
                  onClick={() => setActiveView(id)}
                  className={`py-2 px-1 border-b-2 font-medium text-sm flex items-center space-x-2 ${
                    activeView === id
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
                >
                  <Icon className="h-4 w-4" />
                  <span>{label}</span>
                </button>
              ))}
            </nav>
          </div>
        </div>

        {/* Content */}
        <div className="space-y-6">
          {activeView === 'overview' && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <ModelPerformanceMetrics />
              <AnomalyScoreDistribution />
            </div>
          )}

          {activeView === 'scatter' && (
            <IsolationForestScatterPlot data={scatterData} />
          )}

          {activeView === 'boxplots' && (
            <IsolationForestBoxPlots data={boxPlotData} />
          )}

          {activeView === 'distribution' && (
            <AnomalyScoreDistribution />
          )}

          {activeView === 'performance' && (
            <div className="grid grid-cols-1 gap-6">
              <ModelPerformanceMetrics />
              <div className="bg-white p-6 rounded-lg shadow-lg">
                <h3 className="text-lg font-semibold mb-4">Model Configuration</h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                  <div>
                    <span className="font-medium">Algorithm:</span>
                    <div>Isolation Forest</div>
                  </div>
                  <div>
                    <span className="font-medium">Contamination:</span>
                    <div>{analysisData?.model_info?.contamination ? (analysisData.model_info.contamination * 100).toFixed(1) + '%' : (modelInfo?.contamination ? (modelInfo.contamination * 100).toFixed(1) + '%' : '10%')}</div>
                  </div>
                  <div>
                    <span className="font-medium">Estimators:</span>
                    <div>{analysisData?.model_info?.n_estimators || modelInfo?.n_estimators || 100}</div>
                  </div>
                  <div>
                    <span className="font-medium">Features:</span>
                    <div>{analysisData?.model_info?.feature_count || 'N/A'}</div>
                  </div>
                  <div>
                    <span className="font-medium">Training Status:</span>
                    <div className={`inline-flex px-2 py-1 rounded text-xs ${
                      analysisData?.model_info?.is_trained ? 'bg-green-100 text-green-800' : 'bg-yellow-100 text-yellow-800'
                    }`}>
                      {analysisData?.model_info?.is_trained ? 'Trained' : 'Not Trained'}
                    </div>
                  </div>
                  <div>
                    <span className="font-medium">Data Source:</span>
                    <div className={`inline-flex px-2 py-1 rounded text-xs ${
                      dataSource === 'api' ? 'bg-blue-100 text-blue-800' : 'bg-orange-100 text-orange-800'
                    }`}>
                      {dataSource === 'api' ? 'Live Model' : 'Demo Data'}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeView === 'features' && (
            <FeatureImportanceAnalysis />
          )}
        </div>

        {/* Refresh Button */}
        <div className="mt-8 text-center">
          <button
            onClick={fetchAnalysisData}
            className="bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 transition-colors"
            disabled={loading}
          >
            {loading ? 'Refreshing...' : 'Refresh Analysis'}
          </button>
        </div>
      </div>
    </div>
  );
};

export default IsolationForestVisualization;
