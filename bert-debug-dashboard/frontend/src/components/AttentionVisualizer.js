import React, { useState } from 'react';
import Plot from 'react-plotly.js';
import { FormControl, InputLabel, Select, MenuItem } from '@mui/material';

const AttentionVisualizer = ({ tokens, attentionWeights }) => {
  const [selectedLayer, setSelectedLayer] = useState(0);
  const [selectedHead, setSelectedHead] = useState(0);

  if (!attentionWeights || attentionWeights.length === 0) {
    return <div>No attention data available</div>;
  }

  const currentAttention = attentionWeights[selectedLayer].heads[selectedHead].attention;

  // Create heatmap data
  const heatmapData = [{
    z: currentAttention,
    x: tokens,
    y: tokens,
    type: 'heatmap',
    colorscale: 'Viridis',
    showscale: true
  }];

  const layout = {
    title: `Attention Weights - Layer ${selectedLayer}, Head ${selectedHead}`,
    xaxis: { 
      title: 'Keys',
      tickangle: -45
    },
    yaxis: { 
      title: 'Queries',
      autorange: 'reversed'
    },
    width: 800,
    height: 600
  };

  return (
    <div className="attention-visualizer">
      <div className="controls">
        <FormControl style={{ marginRight: 20 }}>
          <InputLabel>Layer</InputLabel>
          <Select
            value={selectedLayer}
            onChange={(e) => setSelectedLayer(e.target.value)}
          >
            {attentionWeights.map((_, idx) => (
              <MenuItem key={idx} value={idx}>Layer {idx}</MenuItem>
            ))}
          </Select>
        </FormControl>

        <FormControl>
          <InputLabel>Head</InputLabel>
          <Select
            value={selectedHead}
            onChange={(e) => setSelectedHead(e.target.value)}
          >
            {attentionWeights[selectedLayer].heads.map((_, idx) => (
              <MenuItem key={idx} value={idx}>Head {idx}</MenuItem>
            ))}
          </Select>
        </FormControl>
      </div>

      <Plot data={heatmapData} layout={layout} />
    </div>
  );
};

export default AttentionVisualizer;
