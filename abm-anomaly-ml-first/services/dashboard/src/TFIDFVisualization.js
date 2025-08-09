import React, { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import { AlertCircle, TrendingUp, FileText, Search, Download } from 'lucide-react';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const TFIDFVisualization = () => {
  const [analysisData, setAnalysisData] = useState(null);
  const [selectedSession, setSelectedSession] = useState('');
  const [sessionText, setSessionText] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [vocabulary, setVocabulary] = useState(null);
  const [activeCategory, setActiveCategory] = useState('all');

  // Sample anomalous sessions for testing
  const sampleSessions = {
    'power_reset_anomaly': `[020t15706/18/202513:39
TRANSACTION START
[020t CARD INSERTED
POWER-UP/RESET OCCURRED
HARDWARE ERROR DETECTED
RECOVERY FAILED
[020t 13:39:56 CARD TAKEN
[020t 13:39:56 TRANSACTION END`,
    
    'incomplete_transaction': `[020t*209*06/18/2025*14:23*
TRANSACTION START
[020t CARD INSERTED
14:23:03 ATR RECEIVED T=0
[020t 14:23:06 OPCODE = FI
PIN ENTERED
DEVICE MALFUNCTION
[020t CARD TAKEN
[020t TRANSACTION END`,
    
    'normal_transaction': `[020t*209*06/18/2025*14:23*
TRANSACTION START
[020t CARD INSERTED
14:23:03 ATR RECEIVED T=0
[020t 14:23:06 OPCODE = FI
PAN 0004263********6687
PIN ENTERED
[020t 14:23:36 OPCODE = BC
CASH DISPENSED SUCCESSFULLY
[020t 14:24:28 CARD TAKEN
[020t 14:24:29 TRANSACTION END`
  };

  useEffect(() => {
    fetchVocabulary();
  }, []);

  const fetchVocabulary = async () => {
    try {
      const response = await fetch(`${API_URL}/api/v1/svm-tfidf/vocabulary`);
      if (!response.ok) {
        console.log('TF-IDF vocabulary not available - model may not be trained');
        return;
      }
      const data = await response.json();
      setVocabulary(data);
    } catch (err) {
      console.log('Could not fetch vocabulary:', err.message);
    }
  };

  const analyzeTFIDF = async () => {
    let textToAnalyze = sessionText.trim();
    let sessionIdToUse = selectedSession;

    // If no session text but we have a selected session, try to fetch it
    if (!textToAnalyze && selectedSession && !sampleSessions[selectedSession]) {
      setLoading(true);
      setError(null);
      
      try {
        const fetchedText = await fetchSessionText(selectedSession);
        if (fetchedText) {
          textToAnalyze = fetchedText;
          setSessionText(fetchedText); // Update the text area with fetched text
        } else {
          setError('Failed to fetch session text');
          setLoading(false);
          return;
        }
      } catch (error) {
        setError(`Error fetching session: ${error.message}`);
        setLoading(false);
        return;
      }
    }

    if (!textToAnalyze) {
      setError('Please enter session text or select a sample');
      setLoading(false);
      return;
    }

    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`${API_URL}/api/v1/svm-tfidf/analyze-session`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          session_id: sessionIdToUse || `session_${Date.now()}`,
          raw_text: textToAnalyze
        })
      });
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const data = await response.json();
      console.log('TF-IDF Analysis Response:', data);
      console.log('TF-IDF Analysis Array:', data.tfidf_analysis);
      console.log('Word Categories:', data.word_categories);
      console.log('Top TF-IDF Words:', data.top_tfidf_words);
      console.log('Categorized Words:', data.categorized_words);
      
      // Transform API response to match expected frontend structure
      const transformedData = {
        ...data,
        tfidf_analysis: data.top_tfidf_words || [],
        word_categories: data.categorized_words || {},
        prediction_result: {
          is_anomaly: data.anomaly_prediction === 'anomaly',
          ensemble_score: data.anomaly_score || 0
        }
      };
      
      console.log('Transformed Data:', transformedData);
      setAnalysisData(transformedData);
    } catch (err) {
      console.error('Error analyzing TF-IDF:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const fetchSessionText = async (sessionId) => {
    try {
      const response = await fetch(`${API_URL}/api/v1/sessions/${sessionId}/raw-text`);
      if (!response.ok) {
        throw new Error(`Failed to fetch session: ${response.statusText}`);
      }
      const data = await response.json();
      return data.raw_text;
    } catch (error) {
      console.error('Error fetching session text:', error);
      setError(`Failed to fetch session: ${error.message}`);
      return null;
    }
  };

  const loadSampleSession = (sessionKey) => {
    setSelectedSession(sessionKey);
    setSessionText(sampleSessions[sessionKey]);
    setAnalysisData(null); // Clear previous analysis
  };

  const getBarChartData = () => {
    if (!analysisData?.tfidf_analysis) {
      console.log('No tfidf_analysis data:', analysisData);
      return [];
    }
    
    // Handle different data structures
    let words = analysisData.tfidf_analysis;
    
    // If it's an object with words as keys, convert to array
    if (typeof words === 'object' && !Array.isArray(words)) {
      words = Object.entries(words).map(([word, score]) => ({
        word: word,
        tfidf_score: score,
        importance: score // Use score as importance for now
      }));
    }
    
    // If it's already an array, ensure it has the right structure
    if (Array.isArray(words)) {
      words = words.map(item => {
        if (typeof item === 'object' && item.word && item.tfidf_score !== undefined) {
          return {
            word: item.word,
            tfidf_score: item.tfidf_score,
            importance: (item.importance || item.tfidf_score) * 100
          };
        } else if (typeof item === 'object' && Object.keys(item).length === 1) {
          // Handle {word: score} format
          const [word, score] = Object.entries(item)[0];
          return {
            word: word,
            tfidf_score: score,
            importance: score * 100
          };
        }
        return item;
      });
    }
    
    console.log('Bar Chart Data:', words);
    return words || [];
  };

  const getCategoryData = () => {
    if (!analysisData?.word_categories) return [];
    
    const categories = analysisData.word_categories;
    console.log('Categories for pie chart:', categories);
    
    return Object.keys(categories).map(category => ({
      name: category.replace('_', ' ').toUpperCase(),
      value: Array.isArray(categories[category]) ? categories[category].length : Object.keys(categories[category]).length,
      words: categories[category]
    }));
  };

  const getFilteredWords = () => {
    if (!analysisData?.word_categories || activeCategory === 'all') {
      const result = getBarChartData(); // Use the processed bar chart data
      console.log('Filtered Words (all):', result);
      return result;
    }
    
    const categoryData = analysisData.word_categories[activeCategory];
    console.log('Category data for', activeCategory, ':', categoryData);
    
    if (!categoryData) return [];
    
    // Handle different category data formats
    let result = [];
    
    if (Array.isArray(categoryData)) {
      // If it's an array of word objects
      result = categoryData.map(item => {
        if (typeof item === 'string') {
          return { word: item, tfidf_score: 0, importance: 0 };
        }
        return item;
      });
    } else if (typeof categoryData === 'object') {
      // If it's an object with word: score pairs
      result = Object.entries(categoryData).map(([word, score]) => ({
        word: word,
        tfidf_score: score,
        importance: score * 100
      }));
    }
    
    console.log('Filtered Words (category):', activeCategory, result);
    return result;
  };

  const COLORS = ['#8884d8', '#82ca9d', '#ffc658', '#ff7300', '#8dd1e1'];

  return (
    <div className="p-6 bg-gray-50 min-h-screen">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="bg-white rounded-lg shadow-sm p-6">
          <div className="flex items-center gap-3 mb-4">
            <FileText className="h-6 w-6 text-blue-600" />
            <h1 className="text-2xl font-bold text-gray-900">
              One-Class SVM TF-IDF Analysis
            </h1>
          </div>
          <p className="text-gray-600">
            Analyze which words contribute most to anomaly detection decisions in ABM transaction logs.
          </p>
        </div>

        {/* Input Section */}
        <div className="bg-white rounded-lg shadow-sm p-6">
          <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Search className="h-5 w-5" />
            Session Analysis
          </h2>
          
          {/* Sample Sessions */}
          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Quick Test Samples:
            </label>
            <div className="flex flex-wrap gap-2">
              {Object.keys(sampleSessions).map(key => (
                <button
                  key={key}
                  onClick={() => loadSampleSession(key)}
                  className={`px-3 py-1 text-sm rounded-md border ${
                    selectedSession === key
                      ? 'bg-blue-500 text-white border-blue-500'
                      : 'bg-gray-100 text-gray-700 border-gray-300 hover:bg-gray-200'
                  }`}
                >
                  {key.replace('_', ' ').toUpperCase()}
                </button>
              ))}
            </div>
          </div>

          {/* Session ID Input */}
          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Or Enter Session ID:
            </label>
            <input
              type="text"
              value={selectedSession}
              onChange={(e) => setSelectedSession(e.target.value)}
              placeholder="e.g., ABM250_20250618_SESSION_1_aeee1806_20250723_042505"
              className="w-full p-3 border border-gray-300 rounded-md text-sm"
            />
            <p className="text-xs text-gray-500 mt-1">
              Enter a session ID to automatically fetch and analyze the session data
            </p>
          </div>

          {/* Text Input */}
          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Session Text:
            </label>
            <textarea
              value={sessionText}
              onChange={(e) => setSessionText(e.target.value)}
              placeholder="Enter ABM transaction session text..."
              rows={8}
              className="w-full p-3 border border-gray-300 rounded-md font-mono text-sm"
            />
          </div>

          <button
            onClick={analyzeTFIDF}
            disabled={loading}
            className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50"
          >
            {loading ? (
              <>
                <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent"></div>
                Analyzing...
              </>
            ) : (
              <>
                <TrendingUp className="h-4 w-4" />
                Analyze TF-IDF Features
              </>
            )}
          </button>

          {error && (
            <div className="mt-4 p-4 bg-red-100 border-l-4 border-red-500 text-red-700">
              <div className="flex items-center gap-2">
                <AlertCircle className="h-5 w-5" />
                <span>{error}</span>
              </div>
            </div>
          )}
        </div>

        {/* Results Section */}
        {analysisData && (
          <div className="space-y-6">
            {/* Analysis Summary */}
            <div className="bg-white rounded-lg shadow-sm p-6">
              <h2 className="text-lg font-semibold mb-4">Analysis Summary</h2>
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <div className="text-center p-4 bg-blue-50 rounded-lg">
                  <div className="text-2xl font-bold text-blue-600">
                    {analysisData.prediction_result?.is_anomaly ? 'ANOMALY' : 'NORMAL'}
                  </div>
                  <div className="text-sm text-gray-600">Prediction</div>
                </div>
                <div className="text-center p-4 bg-green-50 rounded-lg">
                  <div className="text-2xl font-bold text-green-600">
                    {analysisData.prediction_result?.ensemble_score?.toFixed(3)}
                  </div>
                  <div className="text-sm text-gray-600">Ensemble Score</div>
                </div>
                <div className="text-center p-4 bg-yellow-50 rounded-lg">
                  <div className="text-2xl font-bold text-yellow-600">
                    {analysisData.tfidf_analysis?.length || 0}
                  </div>
                  <div className="text-sm text-gray-600">Key Features</div>
                </div>
                <div className="text-center p-4 bg-purple-50 rounded-lg">
                  <div className="text-2xl font-bold text-purple-600">
                    {Object.keys(analysisData.word_categories || {}).length}
                  </div>
                  <div className="text-sm text-gray-600">Word Categories</div>
                </div>
              </div>
            </div>

            {/* Word Category Distribution */}
            {analysisData.word_categories && (
              <div className="bg-white rounded-lg shadow-sm p-6">
                <h2 className="text-lg font-semibold mb-4">Word Category Distribution</h2>
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <div>
                    <ResponsiveContainer width="100%" height={250}>
                      <PieChart>
                        <Pie
                          data={getCategoryData()}
                          cx="50%"
                          cy="50%"
                          outerRadius={80}
                          fill="#8884d8"
                          dataKey="value"
                          label={({ name, value }) => `${name}: ${value}`}
                        >
                          {getCategoryData().map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                          ))}
                        </Pie>
                        <Tooltip />
                      </PieChart>
                    </ResponsiveContainer>
                  </div>
                  <div>
                    <div className="space-y-2">
                      <button
                        onClick={() => setActiveCategory('all')}
                        className={`px-3 py-1 text-sm rounded-md border ${
                          activeCategory === 'all'
                            ? 'bg-blue-500 text-white border-blue-500'
                            : 'bg-gray-100 text-gray-700 border-gray-300 hover:bg-gray-200'
                        }`}
                      >
                        All Categories
                      </button>
                      {Object.keys(analysisData.word_categories).map(category => (
                        <button
                          key={category}
                          onClick={() => setActiveCategory(category)}
                          className={`ml-2 px-3 py-1 text-sm rounded-md border ${
                            activeCategory === category
                              ? 'bg-blue-500 text-white border-blue-500'
                              : 'bg-gray-100 text-gray-700 border-gray-300 hover:bg-gray-200'
                          }`}
                        >
                          {category.replace('_', ' ').toUpperCase()}
                        </button>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* TF-IDF Bar Chart */}
            <div className="bg-white rounded-lg shadow-sm p-6">
              <h2 className="text-lg font-semibold mb-4">
                Top TF-IDF Words Contributing to Decision
                {activeCategory !== 'all' && ` (${activeCategory.replace('_', ' ').toUpperCase()})`}
              </h2>
              <ResponsiveContainer width="100%" height={400}>
                <BarChart
                  data={getFilteredWords().slice(0, 15)}
                  margin={{ top: 20, right: 30, left: 20, bottom: 60 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="word" 
                    angle={-45}
                    textAnchor="end"
                    height={80}
                    fontSize={12}
                  />
                  <YAxis />
                  <Tooltip 
                    formatter={(value, name) => [
                      name === 'tfidf_score' ? value.toFixed(4) : `${value.toFixed(1)}%`,
                      name === 'tfidf_score' ? 'TF-IDF Score' : 'Importance'
                    ]}
                  />
                  <Legend />
                  <Bar 
                    dataKey="tfidf_score" 
                    fill="#8884d8" 
                    name="TF-IDF Score"
                  />
                  <Bar 
                    dataKey="importance" 
                    fill="#82ca9d" 
                    name="Relative Importance (%)"
                  />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Detailed Word Analysis */}
            <div className="bg-white rounded-lg shadow-sm p-6">
              <h2 className="text-lg font-semibold mb-4">Detailed Word Analysis</h2>
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Word
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        TF-IDF Score
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Importance %
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Category
                      </th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {getFilteredWords().slice(0, 20).map((word, index) => (
                      <tr key={index} className={index % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                          {word.word}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          {word.tfidf_score?.toFixed(4)}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          {(word.importance * 100)?.toFixed(1)}%
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          {/* Find category for this word */}
                          {activeCategory !== 'all' ? activeCategory.replace('_', ' ').toUpperCase() : 'Mixed'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* Model Vocabulary Info */}
        {vocabulary && (
          <div className="bg-white rounded-lg shadow-sm p-6">
            <h2 className="text-lg font-semibold mb-4">Model Vocabulary Information</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <h3 className="font-medium mb-2">Configuration</h3>
                <div className="space-y-1 text-sm text-gray-600">
                  <div>Vocabulary Size: {vocabulary.vocabulary_size}</div>
                  <div>Max Features: {vocabulary.feature_extraction_config.max_features}</div>
                  <div>N-gram Range: {JSON.stringify(vocabulary.feature_extraction_config.ngram_range)}</div>
                </div>
              </div>
              <div>
                <h3 className="font-medium mb-2">Sample Words</h3>
                <div className="text-sm text-gray-600 max-h-32 overflow-y-auto">
                  {vocabulary.top_100_words.slice(0, 20).join(', ')}...
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default TFIDFVisualization;
