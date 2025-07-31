import React, { useState } from 'react';
import { Paper, Typography, CircularProgress } from '@mui/material';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';

const TokenBiasInsights = ({ onFileUpload }) => {
  const [tokenData, setTokenData] = useState(null);
  const [loading, setLoading] = useState(false);

  const onDrop = async (acceptedFiles) => {
    if (acceptedFiles.length === 0) return;
    
    setLoading(true);
    const file = acceptedFiles[0];
    onFileUpload(file);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await axios.post(
        `${process.env.REACT_APP_API_URL}/api/token_bias`,
        formData
      );

      setTokenData(response.data);
    } catch (error) {
      console.error('Token bias analysis failed:', error);
    } finally {
      setLoading(false);
    }
  };

  const { getRootProps, getInputProps } = useDropzone({ 
    onDrop,
    accept: {
      'text/csv': ['.csv']
    }
  });

  return (
    <Paper elevation={2} style={{ padding: 20 }}>
      <Typography variant="h5" gutterBottom>
        Token Bias Insights
      </Typography>

      <div {...getRootProps()} className="upload-zone">
        <input {...getInputProps()} />
        <p>Upload a CSV file with 'text' and 'label' columns to analyze token bias</p>
      </div>

      {loading && <CircularProgress />}

      {tokenData && (
        <div className="token-bias-container">
          {Object.entries(tokenData.token_frequencies).map(([label, tokens]) => (
            <div key={label} className="class-tokens">
              <Typography variant="h6">
                Class {label} - Top Tokens
              </Typography>
              <div className="token-cloud">
                {tokens.slice(0, 20).map(([token, count], idx) => (
                  <span 
                    key={idx} 
                    style={{ 
                      fontSize: Math.min(20, 12 + Math.log(count)),
                      opacity: 0.7 + (idx === 0 ? 0.3 : 0)
                    }}
                  >
                    {token} ({count})
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </Paper>
  );
};

export default TokenBiasInsights;
