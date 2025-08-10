import React, { useState, useEffect } from 'react';
import { RefreshCw, CheckCircle, XCircle, AlertTriangle, TrendingUp, Database, Play, Pause } from 'lucide-react';

const ContinuousLearningInterface = () => {
  const [learningStatus, setLearningStatus] = useState({
    is_active: false,
    current_accuracy: 0.85,
    training_cycles: 12,
    last_updated: new Date().toISOString()
  });

  return (
    <>
      <div className="p-6 space-y-6">
        {/* Header */}
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Continuous Learning</h1>
          <p className="text-gray-600 mt-1">Monitor and control the ML model's continuous learning process</p>
        </div>

        {/* Status Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center">
              <div className={`p-3 rounded-full ${learningStatus.is_active ? 'bg-green-100' : 'bg-gray-100'}`}>
                {learningStatus.is_active ? (
                  <Play className="w-6 h-6 text-green-600" />
                ) : (
                  <Pause className="w-6 h-6 text-gray-600" />
                )}
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">Learning Status</p>
                <p className="text-lg font-bold">
                  {learningStatus.is_active ? 'Active' : 'Paused'}
                </p>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center">
              <div className="p-3 rounded-full bg-blue-100">
                <TrendingUp className="w-6 h-6 text-blue-600" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">Current Accuracy</p>
                <p className="text-lg font-bold">{(learningStatus.current_accuracy * 100).toFixed(1)}%</p>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center">
              <div className="p-3 rounded-full bg-purple-100">
                <Database className="w-6 h-6 text-purple-600" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">Training Cycles</p>
                <p className="text-lg font-bold">{learningStatus.training_cycles}</p>
              </div>
            </div>
          </div>
        </div>

        {/* Controls */}
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-xl font-bold mb-4">Learning Controls</h2>
          <div className="flex space-x-4">
            <button
              className={`px-4 py-2 rounded-lg font-medium ${
                learningStatus.is_active
                  ? 'bg-red-600 text-white hover:bg-red-700'
                  : 'bg-green-600 text-white hover:bg-green-700'
              }`}
            >
              {learningStatus.is_active ? (
                <>
                  <Pause className="w-4 h-4 inline mr-2" />
                  Pause Learning
                </>
              ) : (
                <>
                  <Play className="w-4 h-4 inline mr-2" />
                  Start Learning
                </>
              )}
            </button>
            
            <button className="px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 font-medium">
              <RefreshCw className="w-4 h-4 inline mr-2" />
              Reset Model
            </button>
          </div>
        </div>

        {/* Recent Activity */}
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-xl font-bold mb-4">Recent Activity</h2>
          <div className="space-y-3">
            <div className="flex items-center p-3 bg-green-50 rounded-lg">
              <CheckCircle className="w-5 h-5 text-green-600 mr-3" />
              <div>
                <p className="font-medium">Model updated successfully</p>
                <p className="text-sm text-gray-600">Accuracy improved by 0.3%</p>
              </div>
              <span className="ml-auto text-sm text-gray-500">2 minutes ago</span>
            </div>
            
            <div className="flex items-center p-3 bg-blue-50 rounded-lg">
              <Database className="w-5 h-5 text-blue-600 mr-3" />
              <div>
                <p className="font-medium">New training batch processed</p>
                <p className="text-sm text-gray-600">24 new labeled samples</p>
              </div>
              <span className="ml-auto text-sm text-gray-500">5 minutes ago</span>
            </div>
          </div>
        </div>
      </div>
    </>
  );
};

export default ContinuousLearningInterface;
