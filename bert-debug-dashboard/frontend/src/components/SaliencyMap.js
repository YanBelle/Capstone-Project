import React from 'react';
import { Paper, Typography } from '@mui/material';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const SaliencyMap = ({ tokens, importance }) => {
  if (!tokens || !importance) return null;

  // Prepare data for bar chart
  const data = tokens.map((token, idx) => ({
    token: token,
    importance: importance[idx] * 100
  })).filter(item => item.token !== '[PAD]');

  // Color tokens based on importance
  const getImportanceClass = (value) => {
    if (value < 0.2) return 'token-very-low';
    if (value < 0.4) return 'token-low';
    if (value < 0.6) return 'token-medium';
    if (value < 0.8) return 'token-high';
    return 'token-very-high';
  };

  return (
    <Paper className="saliency-map" elevation={2} style={{ padding: 20 }}>
      <Typography variant="h5" gutterBottom>
        Token Importance Analysis
      </Typography>
      
      <div className="token-importance">
        {tokens.map((token, idx) => (
          token !== '[PAD]' && (
            <span 
              key={idx} 
              className={`token ${getImportanceClass(importance[idx])}`}
              title={`Importance: ${(importance[idx] * 100).toFixed(2)}%`}
            >
              {token}
            </span>
          )
        ))}
      </div>

      <Typography variant="h6" style={{ marginTop: 30 }}>
        Importance Scores
      </Typography>
      <ResponsiveContainer width="100%" height={400}>
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="token" angle={-45} textAnchor="end" height={100} />
          <YAxis />
          <Tooltip />
          <Bar dataKey="importance" fill="#82ca9d" />
        </BarChart>
      </ResponsiveContainer>
    </Paper>
  );
};

export default SaliencyMap;
