import React from 'react';
import { Paper, Typography } from '@mui/material';
import { useDropzone } from 'react-dropzone';
import Plot from 'react-plotly.js';

const MetricsPanel = ({ metrics, onFileUpload }) => {
  const { getRootProps, getInputProps } = useDropzone({ 
    onDrop: (files) => files.length > 0 && onFileUpload(files[0]),
    accept: {
      'text/csv': ['.csv']
    }
  });

  if (!metrics || Object.keys(metrics).length === 0) {
    return (
      <Paper elevation={2} style={{ padding: 20 }}>
        <Typography variant="h5" gutterBottom>
          Performance Metrics
        </Typography>
        <div {...getRootProps()} className="upload-zone">
          <input {...getInputProps()} />
          <p>Upload a CSV file with 'text' and 'label' columns to see metrics</p>
        </div>
      </Paper>
    );
  }

  // Confusion Matrix Plot
  const confusionData = [{
    z: metrics.confusion_matrix,
    type: 'heatmap',
    colorscale: 'Blues',
    showscale: true
  }];

  const confusionLayout = {
    title: 'Confusion Matrix',
    xaxis: { title: 'Predicted' },
    yaxis: { title: 'True', autorange: 'reversed' },
    width: 500,
    height: 500
  };

  return (
    <Paper elevation={2} style={{ padding: 20 }}>
      <Typography variant="h5" gutterBottom>
        Performance Metrics
      </Typography>

      <div className="metrics-grid">
        <div className="metric-card">
          <h3>Accuracy</h3>
          <div className="value">{(metrics.accuracy * 100).toFixed(2)}%</div>
        </div>
        <div className="metric-card">
          <h3>Precision</h3>
          <div className="value">{(metrics.precision * 100).toFixed(2)}%</div>
        </div>
        <div className="metric-card">
          <h3>Recall</h3>
          <div className="value">{(metrics.recall * 100).toFixed(2)}%</div>
        </div>
        <div className="metric-card">
          <h3>F1 Score</h3>
          <div className="value">{(metrics.f1_score * 100).toFixed(2)}%</div>
        </div>
      </div>

      <div className="confusion-matrix">
        <Plot data={confusionData} layout={confusionLayout} />
      </div>

      {metrics.roc_curve && metrics.roc_curve.fpr && (
        <Plot
          data={[{
            x: metrics.roc_curve.fpr,
            y: metrics.roc_curve.tpr,
            type: 'scatter',
            mode: 'lines',
            name: `ROC (AUC = ${metrics.roc_curve.auc.toFixed(3)})`
          }]}
          layout={{
            title: 'ROC Curve',
            xaxis: { title: 'False Positive Rate' },
            yaxis: { title: 'True Positive Rate' },
            width: 600,
            height: 500
          }}
        />
      )}
    </Paper>
  );
};

export default MetricsPanel;
