# IUU Fishing Anomaly Detection System

## Overview
Machine learning system to detect Illegal, Unreported, and Unregulated (IUU) fishing within the Indian Exclusive Economic Zone (EEZ) using AIS data.

## Features
- Real-time AIS data processing and cleaning
- Spatio-temporal trajectory analysis
- Behavioral and transmission feature extraction
- Hybrid ML pipeline (supervised + unsupervised)
- Interactive dashboard with anomaly alerts
- Rule-based baseline comparison

## Project Structure
```
iuu-fishing-detection/
├── data/                    # AIS data, EEZ boundaries
├── src/
│   ├── preprocessing/       # Data cleaning, EEZ filtering
│   ├── features/           # Feature engineering
│   ├── models/             # ML models
│   ├── evaluation/         # Metrics and validation
│   └── dashboard/          # Visualization
├── notebooks/              # Analysis notebooks
├── config/                 # Configuration files
└── outputs/                # Results, models, reports
```

## Installation
```bash
pip install -r requirements.txt
```

## Quick Start
```bash
# 1. Preprocess AIS data
python src/preprocessing/clean_ais.py

# 2. Extract features
python src/features/extract_features.py

# 3. Train models
python src/models/train.py

# 4. Launch dashboard
python src/dashboard/app.py
```

## Models
- **Supervised**: Random Forest, SVM
- **Sequential**: LSTM/GRU for trajectory analysis
- **Unsupervised**: Isolation Forest, Local Outlier Factor

## Objectives
✓ Detect abnormal vessel behavior using ML
✓ Spatio-temporal analytics
✓ AIS blackout and spoofing detection
✓ Real-time alerting system
✓ Improved accuracy vs rule-based systems
