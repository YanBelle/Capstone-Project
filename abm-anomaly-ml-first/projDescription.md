# ML-First ABM EJ Log Anomaly Detection Solution

## Overview

This solution provides an end-to-end, continuously learning anomaly detection system for Automated Banking Machine (ABM) Electronic Journal (EJ) logs. It leverages multiple machine learning (ML) models and expert feedback to improve detection accuracy and adapt to new fraud or failure patterns over time.

---

## Machine Learning Models Utilized

The system uses a hybrid ML approach, combining unsupervised, supervised, and rule-based models:

### 1. **Isolation Forest**
- **Type:** Unsupervised anomaly detection
- **Purpose:** Detects outliers in transaction patterns without prior labels.
- **Usage:** Flags rare or unusual transaction sequences.

### 2. **One-Class SVM**
- **Type:** Unsupervised anomaly detection
- **Purpose:** Learns the "normal" transaction boundary and flags deviations.
- **Usage:** Identifies subtle anomalies not caught by Isolation Forest.

### 3. **Supervised Classifier**
- **Type:** Supervised learning (e.g., RandomForest, Logistic Regression)
- **Purpose:** Learns from labeled feedback (anomaly/normal) provided by experts.
- **Usage:** Improves with each retraining cycle as more expert feedback is collected.

### 4. **Expert Rules Engine**
- **Type:** Rule-based system
- **Purpose:** Encodes domain knowledge and business logic (e.g., "PIN entered but no cash dispensed").
- **Usage:** Catches known patterns and edge cases, and can override ML predictions.

---

## Data Flow / Pipeline

Below is the typical data flow from EJ log ingestion to anomaly detection and model retraining:

```mermaid
graph TD
    A[EJ Log Upload] --> B[Preprocessing & Parsing]
    B --> C[Feature Extraction]
    C --> D[Anomaly Detection (ML Models)]
    D --> E[Expert Review & Feedback]
    E --> F[Feedback Buffer]
    F --> G[Model Retraining]
    G --> D
    D --> H[Alerts & Reports]