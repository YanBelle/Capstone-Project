import React from 'react';
import { Paper, Typography, LinearProgress } from '@mui/material';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';

const PredictionPanel = ({ result }) => {
  if (!result) return null;

  const classNames = ['Normal', 'Anomaly', 'Fault', 'Warning']; // Adjust based on your model
  
  const predictionData = result.probabilities.map((prob, idx) => ({
    class: classNames[idx] || `Class ${idx}`,
    probability: (prob * 100).toFixed(2)
  }));

  const predictedClassName = classNames[result.predicted_class] || `Class ${result.predicted_class}`;
  const confidence = Math.max(...result.probabilities) * 100;

  return (
    <Paper className="prediction-panel" elevation={2} style={{ padding: 20 }}>
      <Typography variant="h5" gutterBottom>
        Prediction Results
      </Typography>
      
      <div className="prediction-summary">
        <Typography variant="h6">
          Predicted: <strong>{predictedClassName}</strong>
        </Typography>
        <Typography variant="body1">
          Confidence: {confidence.toFixed(2)}%
        </Typography>
        <LinearProgress 
          variant="determinate" 
          value={confidence} 
          style={{ marginTop: 10, marginBottom: 20 }}
        />
      </div>

      <Typography variant="h6" gutterBottom>
        Class Probabilities
      </Typography>
      <BarChart width={600} height={300} data={predictionData}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="class" />
        <YAxis />
        <Tooltip />
        <Legend />
        <Bar dataKey="probability" fill="#8884d8" />
      </BarChart>
    </Paper>
  );
};

export default PredictionPanel;
