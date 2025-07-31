import React, { useState } from 'react';
import { Paper, Typography, FormControl, InputLabel, Select, MenuItem } from '@mui/material';
import { useDropzone } from 'react-dropzone';
import Plot from 'react-plotly.js';
import axios from 'axios';

const EmbeddingAnalysis = ({ onFileUpload }) => {
  const [embeddings, setEmbeddings] = useState(null);
  const [method, setMethod] = useState('tsne');
  const [layer, setLayer] = useState(-1);
  const [loading, setLoading] = useState(false);

  const onDrop = async (acceptedFiles) => {
    if (acceptedFiles.length === 0) return;
    
    setLoading(true);
    const file = acceptedFiles[0];
    onFileUpload(file);

    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('method', method);
      formData.append('layer', layer);

      const response = await axios.post(
        `${process.env.REACT_APP_API_URL}/api/embeddings`,
        formData
      );

      setEmbeddings(response.data);
    } catch (error) {
      console.error('Embedding analysis failed:', error);
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

  const plotData = embeddings ? [{
    x: embeddings.embeddings.map(e => e[0]),
    y: embeddings.embeddings.map(e => e[1]),
    mode: 'markers',
    type: 'scatter',
    marker: {
      color: embeddings.labels,
      colorscale: 'Viridis',
      size: 8
    },
    text: embeddings.labels.map(l => `Class: ${l}`)
  }] : [];

  const layout = {
    title: `${method.toUpperCase()} Visualization - Layer ${layer}`,
    xaxis: { title: 'Component 1' },
    yaxis: { title: 'Component 2' },
    width: 800,
    height: 600
  };

  return (
    <Paper className="embedding-analysis" elevation={2} style={{ padding: 20 }}>
      <Typography variant="h5" gutterBottom>
        Embedding Analysis
      </Typography>

      <div style={{ marginBottom: 20 }}>
        <FormControl style={{ marginRight: 20 }}>
          <InputLabel>Method</InputLabel>
          <Select value={method} onChange={(e) => setMethod(e.target.value)}>
            <MenuItem value="tsne">t-SNE</MenuItem>
            <MenuItem value="umap">UMAP</MenuItem>
          </Select>
        </FormControl>

        <FormControl>
          <InputLabel>Layer</InputLabel>
          <Select value={layer} onChange={(e) => setLayer(e.target.value)}>
            <MenuItem value={-1}>Last Layer</MenuItem>
            <MenuItem value={0}>Layer 0</MenuItem>
            <MenuItem value={6}>Layer 6</MenuItem>
            <MenuItem value={11}>Layer 11</MenuItem>
          </Select>
        </FormControl>
      </div>

      <div {...getRootProps()} className="upload-zone">
        <input {...getInputProps()} />
        <p>Drop a CSV file here or click to upload</p>
        <p>File should contain 'text' column</p>
      </div>

      {loading && <Typography>Processing embeddings...</Typography>}

      {embeddings && (
        <Plot data={plotData} layout={layout} />
      )}
    </Paper>
  );
};

export default EmbeddingAnalysis;
