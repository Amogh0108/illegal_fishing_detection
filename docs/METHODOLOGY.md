# IUU Fishing Detection Methodology

## Overview
This document describes the methodology for detecting Illegal, Unreported, and Unregulated (IUU) fishing activities using machine learning and AIS data.

## Problem Statement
Monitoring the Indian Exclusive Economic Zone (EEZ) manually is inefficient due to its vast area (~200 nautical miles maritime boundary). Conventional rule-based systems fail to capture unseen vessel behaviors and deceptive tactics such as:
- AIS blackout/spoofing
- Spoofed MMSI identifiers
- Slow-speed fishing patterns
- Loitering in restricted areas

## Data Sources

### AIS Data
Automatic Identification System (AIS) provides real-time vessel tracking data:
- **MMSI**: Maritime Mobile Service Identity (vessel ID)
- **Position**: Latitude, Longitude
- **Timestamp**: UTC time
- **SOG**: Speed Over Ground (knots)
- **COG**: Course Over Ground (degrees)
- **Heading**: Vessel heading (degrees)

### EEZ Boundaries
GeoJSON polygons defining the Indian Exclusive Economic Zone maritime boundaries.

## Feature Engineering

### Behavioral Features
1. **Speed Analysis**
   - Average speed, speed variance
   - Speed changes and acceleration
   - Fishing speed patterns (1-5 knots)

2. **Course Analysis**
   - Turn rate and course changes
   - Heading deviation from course
   - Erratic movement patterns

3. **Loitering Detection**
   - Time spent within small radius
   - Slow movement in restricted areas
   - Repeated visits to same location

### Transmission Features
1. **AIS Gaps**
   - Transmission dropouts > 60 minutes
   - Sudden disappearance/reappearance
   - Irregular transmission patterns

2. **Position Anomalies**
   - Unrealistic position jumps
   - Speed-distance inconsistencies
   - Potential MMSI spoofing

## Machine Learning Models

### 1. Supervised Learning
**Random Forest Classifier**
- Handles non-linear relationships
- Feature importance analysis
- Robust to outliers

**Support Vector Machine (SVM)**
- Effective for high-dimensional data
- Kernel trick for complex boundaries
- Probability estimates for ensemble

### 2. Unsupervised Learning
**Isolation Forest**
- Detects unknown anomaly patterns
- No labeled data required
- Efficient for large datasets

**Local Outlier Factor (LOF)**
- Density-based anomaly detection
- Identifies local anomalies
- Complements global methods

### 3. Sequential Learning
**LSTM Neural Network**
- Captures temporal dependencies
- Trajectory pattern analysis
- Long-term behavior modeling

## Ensemble Approach
Combines predictions from all models using weighted averaging:
- Supervised: 40%
- Unsupervised: 30%
- Sequential: 30%

Final anomaly score: `score = 0.4*supervised + 0.3*unsupervised + 0.3*lstm`

## Evaluation Metrics
- **Accuracy**: Overall correctness
- **Precision**: True anomalies / Detected anomalies
- **Recall**: Detected anomalies / True anomalies
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve

## Baseline Comparison
Rule-based system using fixed thresholds:
- Speed > 30 knots
- AIS gap > 120 minutes
- Fishing speed pattern (1-5 knots)
- Position jumps
- Loitering detection

## Real-time Detection Pipeline
1. Ingest AIS data stream
2. Filter within EEZ boundaries
3. Extract features in real-time
4. Apply ensemble model
5. Generate alerts for high-risk vessels
6. Visualize on dashboard

## Expected Outcomes
- Higher detection accuracy vs rule-based systems
- Discovery of novel IUU patterns
- Real-time alerting capability
- Scalable to other maritime zones
- Integration with satellite imagery
