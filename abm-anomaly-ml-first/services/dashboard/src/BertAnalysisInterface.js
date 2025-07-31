import React, { useState } from 'react';
import Layout from './Layout';
import apiConfig from './config/api';

const BertAnalysisInterface = () => {
  const [textInput, setTextInput] = useState(`2025-01-06 14:30:15 ERROR Transaction failed: Card read timeout after 30 seconds
2025-01-06 14:30:16 INFO Attempting card read retry
2025-01-06 14:30:17 ERROR Card read failed again: Hardware malfunction detected
2025-01-06 14:30:18 WARN Dispensing mechanism blocked
2025-01-06 14:30:19 ERROR Critical system failure: Unable to complete transaction`);
  const [analysisType, setAnalysisType] = useState('full');
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const analyzeText = async () => {
    if (!textInput.trim()) {
      setError('Please enter some text to analyze.');
      return;
    }

    setLoading(true);
    setError(null);
    setResults(null);

    try {
      const response = await fetch(apiConfig.endpoint('/api/v1/bert/analyze'), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text: textInput,
          analysis_type: analysisType
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      setResults(data);
    } catch (error) {
      setError(`Analysis failed: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const createVisualization = async () => {
    if (!textInput.trim()) {
      setError('Please enter some text to visualize.');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await fetch(apiConfig.endpoint('/api/v1/bert/visualize'), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text: textInput
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || 'Visualization failed');
      }

      setResults(data);
    } catch (error) {
      setError(`Visualization failed: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const getPatterns = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(apiConfig.endpoint('/api/v1/bert/patterns'));
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const data = await response.json();

      setResults(data);
    } catch (error) {
      setError(`Pattern detection failed: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const optimizeBert = async () => {
    if (!textInput.trim()) {
      setError('Please enter some text to optimize.');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await fetch(apiConfig.endpoint('/api/v1/bert/optimize'), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text: textInput
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();

      setResults(data);
    } catch (error) {
      setError(`Optimization analysis failed: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const renderTokenImportance = (tokenImportance) => {
    if (!tokenImportance) return null;

    return (
      <div className="bg-white rounded-lg p-6 shadow-sm border">
        <h3 className="text-lg font-semibold mb-4 flex items-center">
          🎯 Token Importance
        </h3>
        <div className="flex flex-wrap gap-2 mb-4">
          {tokenImportance.map((token, index) => {
            const intensity = Math.min(255, Math.max(0, token.importance * 255));
            const bgColor = `rgb(${255-intensity}, 255, ${255-intensity})`;
            return (
              <span
                key={index}
                className="px-2 py-1 rounded text-sm font-mono border"
                style={{ backgroundColor: bgColor }}
                title={`Importance: ${token.importance.toFixed(3)}`}
              >
                {token.token} ({token.importance.toFixed(3)})
              </span>
            );
          })}
        </div>
      </div>
    );
  };

  const renderPatterns = (patterns) => {
    if (!patterns || patterns.length === 0) return null;

    return (
      <div className="bg-white rounded-lg p-6 shadow-sm border">
        <h3 className="text-lg font-semibold mb-4 flex items-center">
          🔍 Detected Patterns
        </h3>
        <div className="space-y-3">
          {patterns.map((pattern, index) => (
            <div key={index} className="border rounded p-3 bg-gray-50">
              <div className="flex justify-between items-start">
                <span className="font-semibold text-green-600">{pattern.type}</span>
                <span className="bg-green-100 text-green-800 text-xs px-2 py-1 rounded">
                  {(pattern.confidence * 100).toFixed(1)}%
                </span>
              </div>
              {pattern.description && (
                <p className="text-sm text-gray-600 mt-1">{pattern.description}</p>
              )}
            </div>
          ))}
        </div>
      </div>
    );
  };

  const renderOptimizationSuggestions = (suggestions) => {
    if (!suggestions || suggestions.length === 0) {
      return (
        <div className="bg-white rounded-lg p-6 shadow-sm border">
          <h3 className="text-lg font-semibold mb-4 flex items-center">
            ✅ BERT Performance
          </h3>
          <p className="text-gray-600">No optimization issues detected. Your BERT model appears to be performing well for this text!</p>
        </div>
      );
    }

    return (
      <div className="bg-white rounded-lg p-6 shadow-sm border">
        <h3 className="text-lg font-semibold mb-4 flex items-center">
          💡 Optimization Suggestions
        </h3>
        <div className="space-y-4">
          {suggestions.map((suggestion, index) => (
            <div key={index} className="bg-orange-50 border-l-4 border-orange-400 p-4 rounded-r">
              <div className="flex justify-between items-start mb-2">
                <span className="font-semibold text-orange-800">{suggestion.type}</span>
                <span className="bg-orange-100 text-orange-800 text-xs px-2 py-1 rounded">
                  {(suggestion.confidence * 100).toFixed(1)}%
                </span>
              </div>
              <p className="text-sm text-gray-700 mb-2">
                <strong>Issue:</strong> {suggestion.issue}
              </p>
              <p className="text-sm text-gray-700">
                <strong>Suggestion:</strong> {suggestion.suggestion}
              </p>
            </div>
          ))}
        </div>
      </div>
    );
  };

  const renderVisualizations = (visualizations) => {
    if (!visualizations) return null;

    return (
      <div className="space-y-6">
        {Object.entries(visualizations).map(([type, visualization]) => {
          if (visualization.base64_image) {
            return (
              <div key={type} className="bg-white rounded-lg p-6 shadow-sm border">
                <h3 className="text-lg font-semibold mb-4">
                  {type.replace('_', ' ').toUpperCase()}
                </h3>
                <div className="text-center">
                  <img 
                    src={`data:image/png;base64,${visualization.base64_image}`}
                    alt={`${type} visualization`}
                    className="max-w-full h-auto rounded border"
                  />
                </div>
                {visualization.description && (
                  <p className="text-sm text-gray-600 mt-3">{visualization.description}</p>
                )}
              </div>
            );
          }
          return null;
        })}
      </div>
    );
  };

  return (
    <Layout>
      <div className="space-y-6">
        {/* Header */}
        <div className="bg-gradient-to-r from-purple-600 to-blue-600 rounded-lg p-6 text-white">
          <h1 className="text-3xl font-bold mb-2">🧠 BERT Attention Analysis</h1>
          <p className="text-xl opacity-90">Understand how BERT processes ABM transaction logs for anomaly detection</p>
        </div>

        {/* Input Section */}
        <div className="bg-white rounded-lg p-6 shadow-sm border">
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                ABM Transaction Log Text:
              </label>
              <textarea
                value={textInput}
                onChange={(e) => setTextInput(e.target.value)}
                className="w-full h-32 p-3 border border-gray-300 rounded-md resize-vertical focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                placeholder="Enter ABM transaction log text to analyze..."
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Analysis Type:
              </label>
              <select
                value={analysisType}
                onChange={(e) => setAnalysisType(e.target.value)}
                className="w-full p-3 border border-gray-300 rounded-md focus:ring-2 focus:ring-purple-500 focus:border-transparent"
              >
                <option value="full">Full Analysis (All Components)</option>
                <option value="attention">Attention Patterns Only</option>
                <option value="importance">Token Importance Only</option>
                <option value="patterns">Pattern Detection Only</option>
              </select>
            </div>

            <div className="flex flex-wrap gap-3">
              <button
                onClick={analyzeText}
                disabled={loading}
                className="px-6 py-3 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center"
              >
                🔍 Analyze Attention
              </button>
              <button
                onClick={createVisualization}
                disabled={loading}
                className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center"
              >
                📊 Create Visualizations
              </button>
              <button
                onClick={getPatterns}
                disabled={loading}
                className="px-6 py-3 bg-orange-600 text-white rounded-lg hover:bg-orange-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center"
              >
                🔎 Detect Patterns
              </button>
              <button
                onClick={optimizeBert}
                disabled={loading}
                className="px-6 py-3 bg-purple-800 text-white rounded-lg hover:bg-purple-900 disabled:opacity-50 disabled:cursor-not-allowed flex items-center"
              >
                ⚡ Optimize BERT
              </button>
            </div>
          </div>
        </div>

        {/* Loading State */}
        {loading && (
          <div className="bg-white rounded-lg p-6 shadow-sm border text-center">
            <div className="inline-flex items-center">
              <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-purple-600 mr-3"></div>
              Processing...
            </div>
          </div>
        )}

        {/* Error State */}
        {error && (
          <div className="bg-red-50 border-l-4 border-red-400 p-4 rounded-r">
            <h3 className="text-lg font-semibold text-red-800 mb-2">❌ Error</h3>
            <p className="text-red-700">{error}</p>
          </div>
        )}

        {/* Results */}
        {results && !loading && (
          <div className="space-y-6">
            {/* Success Header */}
            <div className="bg-green-50 border-l-4 border-green-400 p-4 rounded-r">
              <h3 className="text-lg font-semibold text-green-800 mb-2">
                ✅ {results.status === 'success' ? 'Analysis Complete' : 'Results'}
              </h3>
              {results.text && (
                <p className="text-green-700">
                  <strong>Text analyzed:</strong> "{results.text.substring(0, 100)}{results.text.length > 100 ? '...' : ''}"
                </p>
              )}
              {results.analysis_type && (
                <p className="text-green-700">
                  <strong>Analysis type:</strong> {results.analysis_type}
                </p>
              )}
            </div>

            {/* Token Importance */}
            {results.results?.token_importance && renderTokenImportance(results.results.token_importance)}

            {/* Detected Patterns */}
            {results.results?.detected_patterns && renderPatterns(results.results.detected_patterns)}
            {results.patterns && renderPatterns(results.patterns)}

            {/* Attention Analysis */}
            {results.results?.attention_analysis && (
              <div className="bg-white rounded-lg p-6 shadow-sm border">
                <h3 className="text-lg font-semibold mb-4">🧠 Attention Analysis</h3>
                <div className="space-y-2 text-sm">
                  <p><strong>Dominant layers:</strong> {JSON.stringify(results.results.attention_analysis.dominant_layers || [])}</p>
                  <p><strong>Key attention heads:</strong> {JSON.stringify(results.results.attention_analysis.key_heads || [])}</p>
                  <p><strong>Attention distribution:</strong> {results.results.attention_analysis.attention_distribution || 'Not available'}</p>
                </div>
              </div>
            )}

            {/* Visualizations */}
            {results.visualizations && renderVisualizations(results.visualizations)}

            {/* Optimization Suggestions */}
            {results.optimization_suggestions && renderOptimizationSuggestions(results.optimization_suggestions)}

            {/* Pattern Summary */}
            {results.pattern_summary && (
              <div className="bg-white rounded-lg p-6 shadow-sm border">
                <h3 className="text-lg font-semibold mb-4">📈 Pattern Summary</h3>
                <p className="text-sm text-gray-600 mb-4">
                  <strong>Sample count:</strong> {results.sample_count}
                </p>
                <div className="space-y-2">
                  {Object.entries(results.pattern_summary).map(([type, summary]) => (
                    <div key={type} className="bg-gray-50 p-3 rounded">
                      <strong>{type}:</strong> {summary.count} occurrences 
                      (avg confidence: {(summary.avg_confidence * 100).toFixed(1)}%)
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </Layout>
  );
};

export default BertAnalysisInterface;
