import React, { useState } from 'react';
import { Paper, Typography, TextField } from '@mui/material';
import { useDropzone } from 'react-dropzone';

const MisclassificationExplorer = ({ results, onFileUpload }) => {
  const [filter, setFilter] = useState('');
  
  const { getRootProps, getInputProps } = useDropzone({ 
    onDrop: (files) => files.length > 0 && onFileUpload(files[0]),
    accept: {
      'text/csv': ['.csv']
    }
  });

  if (!results || results.length === 0) {
    return (
      <Paper elevation={2} style={{ padding: 20 }}>
        <Typography variant="h5" gutterBottom>
          Misclassification Explorer
        </Typography>
        <div {...getRootProps()} className="upload-zone">
          <input {...getInputProps()} />
          <p>Upload a CSV file with 'text' and 'label' columns to explore misclassifications</p>
        </div>
      </Paper>
    );
  }

  const misclassified = results.filter(r => 
    r.true_label !== -1 && r.true_label !== r.predicted_class
  );

  const filtered = filter 
    ? misclassified.filter(r => r.text.toLowerCase().includes(filter.toLowerCase()))
    : misclassified;

  return (
    <Paper elevation={2} style={{ padding: 20 }}>
      <Typography variant="h5" gutterBottom>
        Misclassification Explorer
      </Typography>
      
      <Typography variant="body1" gutterBottom>
        Found {misclassified.length} misclassifications out of {results.length} samples
      </Typography>

      <TextField
        fullWidth
        variant="outlined"
        placeholder="Filter by text content..."
        value={filter}
        onChange={(e) => setFilter(e.target.value)}
        style={{ marginBottom: 20 }}
      />

      <div className="misclassification-list">
        {filtered.map((item, idx) => (
          <div key={idx} className="misclassification-item">
            <Typography variant="body2" style={{ fontFamily: 'monospace' }}>
              {item.text}
            </Typography>
            <div style={{ marginTop: 10 }}>
              <span style={{ marginRight: 20 }}>
                True: <strong>{item.true_label}</strong>
              </span>
              <span style={{ marginRight: 20 }}>
                Predicted: <strong>{item.predicted_class}</strong>
              </span>
              <span>
                Confidence: <strong>{(Math.max(...item.probabilities) * 100).toFixed(2)}%</strong>
              </span>
            </div>
          </div>
        ))}
      </div>
    </Paper>
  );
};

export default MisclassificationExplorer;
