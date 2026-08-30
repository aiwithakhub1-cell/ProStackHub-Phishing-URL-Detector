# Architecture — Phishing URL Detector

## 1. System Overview

This project uses supervised machine learning to classify URLs as potentially phishing or legitimate.

Random Forest and XGBoost models are trained and evaluated using labeled phishing URL data. The selected model is saved and deployed through a Flask application.

## 2. Architecture

```text
Labeled Dataset
      |
      v
Feature Preparation
      |
      +----------------------+
      |                      |
      v                      v
Random Forest           XGBoost
      |                      |
      +----------+-----------+
                 |
                 v
       Model Evaluation
                 |
                 v
 Accuracy / Precision
 Recall / F1 / Matrix
                 |
                 v
          Best Model
                 |
                 v
        Saved Model File
                 |
                 v
          Flask API
                 |
                 v
          User URL Input
                 |
                 v
        URL Feature Extraction
                 |
                 v
          ML Prediction
                 |
        +--------+--------+
        |                 |
        v                 v
 Prediction          Explainability
        |                 |
        +--------+--------+
                 |
                 v
        PhishTank Validation