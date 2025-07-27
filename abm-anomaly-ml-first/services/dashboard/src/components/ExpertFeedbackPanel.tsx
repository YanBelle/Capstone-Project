import React, { useState, useEffect } from 'react';
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  Button,
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
  Textarea,
  Badge,
  Alert,
  AlertDescription,
  Progress,
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from '@/components/ui';
import { CheckCircle, XCircle, AlertTriangle, Brain, TrendingUp, Users } from 'lucide-react';

interface ExpertFeedbackProps {
  sessionId: string;
  sessionData: any;
  currentPrediction: {
    isAnomaly: boolean;
    confidence: number;
    detectionMethods: string[];
    anomalyTypes: string[];
  };
  onFeedbackSubmitted: () => void;
}

interface FeedbackStats {
  total_feedback_count: number;
  feedback_by_type: Record<string, number>;
  model_accuracy_by_method: Record<string, any>;
  recent_training_sessions: any[];
  pending_feedback_count: number;
}

interface ModelPerformance {
  overall_accuracy: number;
  overall_precision: number;
  overall_recall: number;
  total_feedback_samples: number;
  active_detection_methods: number;
  dynamic_thresholds: Record<string, number>;
  ensemble_weights: Record<string, number>;
}

export const ExpertFeedbackPanel: React.FC<ExpertFeedbackProps> = ({
  sessionId,
  sessionData,
  currentPrediction,
  onFeedbackSubmitted
}) => {
  const [expertLabel, setExpertLabel] = useState<string>('');
  const [expertConfidence, setExpertConfidence] = useState<number>(0.8);
  const [feedbackType, setFeedbackType] = useState<string>('');
  const [explanation, setExplanation] = useState<string>('');
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false);
  const [feedback, setFeedback] = useState<any>(null);
  const [stats, setStats] = useState<FeedbackStats | null>(null);
  const [performance, setPerformance] = useState<ModelPerformance | null>(null);

  // Load feedback stats and model performance
  useEffect(() => {
    loadFeedbackStats();
    loadModelPerformance();
  }, []);

  const loadFeedbackStats = async () => {
    try {
      const response = await fetch('/api/v1/expert-feedback/stats');
      if (response.ok) {
        const data = await response.json();
        setStats(data);
      }
    } catch (error) {
      console.error('Failed to load feedback stats:', error);
    }
  };

  const loadModelPerformance = async () => {
    try {
      const response = await fetch('/api/v1/expert-feedback/model-performance');
      if (response.ok) {
        const data = await response.json();
        setPerformance(data);
      }
    } catch (error) {
      console.error('Failed to load model performance:', error);
    }
  };

  const submitFeedback = async () => {
    if (!expertLabel || !feedbackType) {
      setFeedback({
        success: false,
        message: 'Please select both expert label and feedback type'
      });
      return;
    }

    setIsSubmitting(true);
    
    try {
      const response = await fetch('/api/v1/expert-feedback/submit', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          session_id: sessionId,
          expert_label: expertLabel,
          expert_confidence: expertConfidence,
          feedback_type: feedbackType,
          expert_explanation: explanation,
          expert_name: 'Security Expert' // In real app, get from auth
        }),
      });

      const result = await response.json();
      setFeedback(result);
      
      if (result.success) {
        onFeedbackSubmitted();
        // Reload stats after successful submission
        loadFeedbackStats();
        loadModelPerformance();
      }
    } catch (error) {
      setFeedback({
        success: false,
        message: `Failed to submit feedback: ${error.message}`
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  const triggerManualTraining = async () => {
    try {
      const response = await fetch('/api/v1/expert-feedback/trigger-training', {
        method: 'POST'
      });
      const result = await response.json();
      setFeedback(result);
      
      if (result.triggered) {
        loadFeedbackStats();
        loadModelPerformance();
      }
    } catch (error) {
      setFeedback({
        success: false,
        message: `Failed to trigger training: ${error.message}`
      });
    }
  };

  // Determine feedback type based on ML prediction vs expert assessment
  const determineFeedbackType = (expertLabel: string) => {
    const expertIsAnomaly = expertLabel !== 'normal';
    if (expertIsAnomaly === currentPrediction.isAnomaly) {
      return 'confirmation';
    } else {
      return 'correction';
    }
  };

  const handleExpertLabelChange = (value: string) => {
    setExpertLabel(value);
    setFeedbackType(determineFeedbackType(value));
  };

  return (
    <div className="space-y-6">
      <Tabs defaultValue="feedback" className="w-full">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="feedback">Expert Feedback</TabsTrigger>
          <TabsTrigger value="performance">Model Performance</TabsTrigger>
          <TabsTrigger value="training">Training Control</TabsTrigger>
        </TabsList>

        <TabsContent value="feedback" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Users className="h-5 w-5" />
                Expert Assessment for Session {sessionId}
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Current ML Prediction Display */}
              <div className="bg-gray-50 p-4 rounded-lg">
                <h4 className="font-medium mb-2">Current ML Prediction:</h4>
                <div className="flex items-center gap-2 mb-2">
                  {currentPrediction.isAnomaly ? (
                    <Badge variant="destructive">Anomaly Detected</Badge>
                  ) : (
                    <Badge variant="success">Normal Transaction</Badge>
                  )}
                  <span className="text-sm text-gray-600">
                    Confidence: {(currentPrediction.confidence * 100).toFixed(1)}%
                  </span>
                </div>
                {currentPrediction.anomalyTypes.length > 0 && (
                  <div className="text-sm text-gray-600">
                    Types: {currentPrediction.anomalyTypes.join(', ')}
                  </div>
                )}
                <div className="text-sm text-gray-600">
                  Methods: {currentPrediction.detectionMethods.join(', ')}
                </div>
              </div>

              {/* Expert Assessment Form */}
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium mb-2">
                    Expert Assessment
                  </label>
                  <Select value={expertLabel} onValueChange={handleExpertLabelChange}>
                    <SelectTrigger>
                      <SelectValue placeholder="Select your assessment" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="normal">Normal Transaction</SelectItem>
                      <SelectItem value="anomaly">General Anomaly</SelectItem>
                      <SelectItem value="hardware_error">Hardware Error</SelectItem>
                      <SelectItem value="incomplete_transaction">Incomplete Transaction</SelectItem>
                      <SelectItem value="security_concern">Security Concern</SelectItem>
                      <SelectItem value="fraud_indicator">Fraud Indicator</SelectItem>
                      <SelectItem value="network_issue">Network Issue</SelectItem>
                      <SelectItem value="user_error">User Error</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div>
                  <label className="block text-sm font-medium mb-2">
                    Confidence Level: {(expertConfidence * 100).toFixed(0)}%
                  </label>
                  <input
                    type="range"
                    min="0.1"
                    max="1.0"
                    step="0.1"
                    value={expertConfidence}
                    onChange={(e) => setExpertConfidence(parseFloat(e.target.value))}
                    className="w-full"
                  />
                </div>
              </div>

              {feedbackType && (
                <div>
                  <Badge variant={feedbackType === 'confirmation' ? 'success' : 'warning'}>
                    {feedbackType === 'confirmation' ? 'Confirms ML Prediction' : 'Corrects ML Prediction'}
                  </Badge>
                </div>
              )}

              <div>
                <label className="block text-sm font-medium mb-2">
                  Expert Explanation (Optional)
                </label>
                <Textarea
                  value={explanation}
                  onChange={(e) => setExplanation(e.target.value)}
                  placeholder="Provide reasoning for your assessment..."
                  rows={3}
                />
              </div>

              <Button
                onClick={submitFeedback}
                disabled={isSubmitting || !expertLabel || !feedbackType}
                className="w-full"
              >
                {isSubmitting ? 'Submitting...' : 'Submit Expert Feedback'}
              </Button>

              {feedback && (
                <Alert className={feedback.success ? 'border-green-200' : 'border-red-200'}>
                  <AlertDescription>
                    {feedback.message}
                    {feedback.training_triggered && (
                      <div className="mt-2">
                        <Badge variant="success">Model Retraining Triggered!</Badge>
                      </div>
                    )}
                  </AlertDescription>
                </Alert>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="performance" className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <TrendingUp className="h-5 w-5" />
                  Overall Performance
                </CardTitle>
              </CardHeader>
              <CardContent>
                {performance ? (
                  <div className="space-y-3">
                    <div>
                      <div className="flex justify-between text-sm">
                        <span>Accuracy</span>
                        <span>{(performance.overall_accuracy * 100).toFixed(1)}%</span>
                      </div>
                      <Progress value={performance.overall_accuracy * 100} className="mt-1" />
                    </div>
                    <div>
                      <div className="flex justify-between text-sm">
                        <span>Precision</span>
                        <span>{(performance.overall_precision * 100).toFixed(1)}%</span>
                      </div>
                      <Progress value={performance.overall_precision * 100} className="mt-1" />
                    </div>
                    <div>
                      <div className="flex justify-between text-sm">
                        <span>Recall</span>
                        <span>{(performance.overall_recall * 100).toFixed(1)}%</span>
                      </div>
                      <Progress value={performance.overall_recall * 100} className="mt-1" />
                    </div>
                    <div className="text-sm text-gray-600">
                      Based on {performance.total_feedback_samples} expert feedback samples
                    </div>
                  </div>
                ) : (
                  <div>Loading performance data...</div>
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Detection Method Accuracy</CardTitle>
              </CardHeader>
              <CardContent>
                {stats ? (
                  <div className="space-y-2">
                    {Object.entries(stats.model_accuracy_by_method).map(([method, metrics]: [string, any]) => (
                      <div key={method} className="flex justify-between items-center">
                        <span className="text-sm capitalize">{method.replace('_', ' ')}</span>
                        <Badge variant="outline">
                          {(metrics.accuracy * 100).toFixed(1)}%
                        </Badge>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div>Loading method accuracy...</div>
                )}
              </CardContent>
            </Card>
          </div>

          {performance && (
            <Card>
              <CardHeader>
                <CardTitle>Dynamic Thresholds & Weights</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <h4 className="font-medium mb-2">Detection Thresholds</h4>
                    {Object.entries(performance.dynamic_thresholds).map(([threshold, value]) => (
                      <div key={threshold} className="flex justify-between text-sm">
                        <span className="capitalize">{threshold.replace('_', ' ')}</span>
                        <span>{value.toFixed(3)}</span>
                      </div>
                    ))}
                  </div>
                  <div>
                    <h4 className="font-medium mb-2">Ensemble Weights</h4>
                    {Object.entries(performance.ensemble_weights).map(([method, weight]) => (
                      <div key={method} className="flex justify-between text-sm">
                        <span className="capitalize">{method.replace('_', ' ')}</span>
                        <span>{(weight * 100).toFixed(1)}%</span>
                      </div>
                    ))}
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="training" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Brain className="h-5 w-5" />
                Training Control
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {stats && (
                <div className="bg-blue-50 p-4 rounded-lg">
                  <h4 className="font-medium mb-2">Feedback Status</h4>
                  <div className="grid grid-cols-3 gap-4 text-sm">
                    <div>
                      <div className="font-medium">Total Feedback</div>
                      <div className="text-2xl">{stats.total_feedback_count}</div>
                    </div>
                    <div>
                      <div className="font-medium">Pending Training</div>
                      <div className="text-2xl">{stats.pending_feedback_count}</div>
                    </div>
                    <div>
                      <div className="font-medium">Recent Sessions</div>
                      <div className="text-2xl">{stats.recent_training_sessions.length}</div>
                    </div>
                  </div>
                </div>
              )}

              <div className="flex gap-2">
                <Button
                  onClick={triggerManualTraining}
                  variant="outline"
                  disabled={!stats || stats.pending_feedback_count === 0}
                >
                  Trigger Manual Training
                </Button>
                <Button
                  onClick={loadModelPerformance}
                  variant="outline"
                >
                  Refresh Performance
                </Button>
              </div>

              {stats && stats.feedback_by_type && (
                <div>
                  <h4 className="font-medium mb-2">Feedback Distribution</h4>
                  <div className="space-y-2">
                    {Object.entries(stats.feedback_by_type).map(([type, count]) => (
                      <div key={type} className="flex justify-between items-center">
                        <span className="capitalize">{type}</span>
                        <Badge variant="outline">{count}</Badge>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default ExpertFeedbackPanel;
