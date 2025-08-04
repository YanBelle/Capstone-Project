#!/bin/bash

# Fix the tensor dimension error in all bert_deeplog_model.py files
echo "Fixing tensor dimension error in bert_deeplog_model.py files..."

# Fix the main API file
sed -i '' 's/sequence_targets = batch_X\[:, 1:, :\]/sequence_targets = self.model.bert_projection(batch_X[:, 1:, :])/g' services/api/bert_deeplog_model.py

# Fix the anomaly detector file  
sed -i '' 's/sequence_targets = batch_X\[:, 1:, :\]/sequence_targets = self.model.bert_projection(batch_X[:, 1:, :])/g' services/anomaly-detector/bert_deeplog_model.py

# Fix the root level file
sed -i '' 's/sequence_targets = batch_X\[:, 1:, :\]/sequence_targets = self.model.bert_projection(batch_X[:, 1:, :])/g' bert_deeplog_model.py

echo "Fixed tensor dimension error in all files"

# Verify the fixes
echo "Verifying fixes:"
grep -n "sequence_targets.*bert_projection" services/api/bert_deeplog_model.py
grep -n "sequence_targets.*bert_projection" services/anomaly-detector/bert_deeplog_model.py  
grep -n "sequence_targets.*bert_projection" bert_deeplog_model.py

echo "Fix applied successfully!"
