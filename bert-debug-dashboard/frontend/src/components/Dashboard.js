import React, { useState } from 'react';
import { Tab, Tabs, TabList, TabPanel } from 'react-tabs';
import 'react-tabs/style/react-tabs.css';
import AttentionVisualizer from './AttentionVisualizer';
import PredictionPanel from './PredictionPanel';
import SaliencyMap from './SaliencyMap';
import EmbeddingAnalysis from './EmbeddingAnalysis';
import MetricsPanel from './MetricsPanel';
import MisclassificationExplorer from './MisclassificationExplorer';
import TokenBiasInsights from './TokenBiasInsights';
import { TextField, Button, CircularProgress, Paper } from '@mui/material';
import axios from 'axios';

const Dashboard = () => {
  const [inputText, setInputText] = useState('');
  const [analysisResult, setAnalysisResult] = useState(null);
  const [batchResults, setBatchResults] = useState(null);
  const [loading, setLoading] = useState(false);

  const analyzeText = async () => {
    if (!inputText) return;
    
    setLoading(true);
    try {
      const formData = new FormData();
      formData.append('text', inputText);
      
      const response = await axios.post(
        `${process.env.REACT_APP_API_URL}/api/analyze`,
        formData
      );
      
      setAnalysisResult(response.data);
    } catch (error) {
      console.error('Analysis failed:', error);
      alert('Analysis failed. Please check the console for details.');
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = async (file) => {
    setLoading(true);
    try {
      const formData = new FormData();
      formData.append('file', file);
      
      const response = await axios.post(
        `${process.env.REACT_APP_API_URL}/api/batch_analyze`,
        formData
      );
      
      setBatchResults(response.data);
    } catch (error) {
      console.error('Batch analysis failed:', error);
      alert('Batch analysis failed. Please check the console for details.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="dashboard">
      <Paper className="input-section" elevation={3}>
        <h2>Input EJ Log</h2>
        <TextField
          multiline
          rows={8}
          fullWidth
          variant="outlined"
          placeholder="Enter EJ log text here..."
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
        />
        <Button 
          variant="contained" 
          color="primary" 
          onClick={analyzeText}
          disabled={loading}
          style={{ marginTop: 10 }}
        >
          {loading ? <CircularProgress size={24} /> : 'Analyze'}
        </Button>
      </Paper>

      {analysisResult && (
        <Tabs>
          <TabList>
            <Tab>Prediction</Tab>
            <Tab>Attention</Tab>
            <Tab>Saliency</Tab>
            <Tab>Embeddings</Tab>
            <Tab>Metrics</Tab>
            <Tab>Misclassifications</Tab>
            <Tab>Token Bias</Tab>
          </TabList>

          <TabPanel>
            <PredictionPanel result={analysisResult} />
          </TabPanel>

          <TabPanel>
            <AttentionVisualizer 
              tokens={analysisResult.tokens}
              attentionWeights={analysisResult.attention_weights}
            />
          </TabPanel>

          <TabPanel>
            <SaliencyMap 
              tokens={analysisResult.tokens}
              importance={analysisResult.token_importance}
            />
          </TabPanel>

          <TabPanel>
            <EmbeddingAnalysis 
              hiddenStates={analysisResult.hidden_states}
              onFileUpload={handleFileUpload}
            />
          </TabPanel>

          <TabPanel>
            <MetricsPanel 
              metrics={batchResults?.metrics}
              onFileUpload={handleFileUpload}
            />
          </TabPanel>

          <TabPanel>
            <MisclassificationExplorer 
              results={batchResults?.results}
              onFileUpload={handleFileUpload}
            />
          </TabPanel>

          <TabPanel>
            <TokenBiasInsights 
              onFileUpload={handleFileUpload}
            />
          </TabPanel>
        </Tabs>
      )}
    </div>
  );
};

export default Dashboard;
