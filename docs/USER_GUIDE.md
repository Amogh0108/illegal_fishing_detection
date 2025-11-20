# User Guide

## Quick Start

### 1. Generate Sample Data
```bash
python scripts/generate_sample_data.py
```

### 2. Run Complete Pipeline
```bash
python scripts/run_pipeline.py
```

### 3. Launch Dashboard
```bash
python src/dashboard/app.py
```

## Running Individual Components

### Data Preprocessing
```bash
# Clean AIS data
python src/preprocessing/clean_ais.py

# Filter within EEZ
python src/preprocessing/eez_filter.py
```

### Feature Extraction
```bash
# Extract all features
python src/features/extract_features.py

# Or extract separately
python src/features/behavior_features.py
python src/features/transmission_features.py
```

### Model Training
```bash
# Train all models
python src/models/train.py

# Or train individually
python -c "from src.models.supervised_models import *; ..."
```

### Evaluation
```bash
# Run baseline
python src/evaluation/baseline.py

# Run ensemble
python src/models/ensemble.py

# Evaluate
python src/evaluation/metrics.py
```

## Dashboard Usage

### Main Features
1. **Interactive Map**: View vessel trajectories and anomalies
2. **Statistics Cards**: Real-time metrics
3. **Timeline Plot**: Anomaly scores over time
4. **Model Comparison**: Compare different model outputs

### Controls
- **Anomaly Threshold Slider**: Adjust sensitivity (0.0 - 1.0)
- **Vessel Dropdown**: Filter by specific vessel (MMSI)
- **Refresh Button**: Reload latest data
- **Auto-refresh**: Updates every 5 minutes

### Interpreting Results
- **Red points**: Anomalies detected
- **Blue points**: Normal behavior
- **Score > 0.7**: High-risk vessel (default threshold)
- **Score 0.5-0.7**: Medium risk
- **Score < 0.5**: Low risk

## Configuration

### Edit config/config.yaml
```yaml
# Adjust feature parameters
features:
  behavior:
    loitering_radius_km: 5
    fishing_speed_min: 1
    fishing_speed_max: 5

# Adjust model parameters
models:
  random_forest:
    n_estimators: 200
    max_depth: 20

# Adjust anomaly detection
anomaly:
  threshold: 0.7
  ensemble_weights:
    supervised: 0.4
    unsupervised: 0.3
    sequential: 0.3
```

## Using Your Own Data

### AIS Data Format
CSV file with columns:
- `MMSI`: Vessel identifier (integer)
- `timestamp`: ISO format (YYYY-MM-DD HH:MM:SS)
- `lat`: Latitude (-90 to 90)
- `lon`: Longitude (-180 to 180)
- `SOG`: Speed over ground (knots)
- `COG`: Course over ground (0-360 degrees)
- `heading`: Vessel heading (0-360 degrees, optional)

### EEZ Boundary Format
GeoJSON file with Polygon geometry:
```json
{
  "type": "FeatureCollection",
  "features": [{
    "type": "Feature",
    "geometry": {
      "type": "Polygon",
      "coordinates": [[[lon, lat], ...]]
    }
  }]
}
```

### Update Paths
Edit `config/config.yaml`:
```yaml
data:
  ais_data: "path/to/your/ais_data.csv"
  eez_boundary: "path/to/your/eez.geojson"
```

## Labeling Anomalies

### For Supervised Training
Add `anomaly` column to your AIS data:
- `0`: Normal behavior
- `1`: Anomalous behavior

### Labeling Guidelines
Mark as anomaly if:
- Vessel enters restricted zone
- AIS transmission gap > 2 hours
- Unrealistic speed or position jump
- Known IUU vessel from registry
- Suspicious loitering pattern

## Performance Optimization

### For Large Datasets
1. **Reduce data**: Sample or filter by time/region
2. **Batch processing**: Process in chunks
3. **Parallel processing**: Use multiprocessing
4. **GPU acceleration**: Enable CUDA for LSTM

### Memory Management
```python
# Process in chunks
chunk_size = 10000
for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
    process_chunk(chunk)
```

## Troubleshooting

### No Anomalies Detected
- Lower threshold in config or dashboard
- Check if features are being extracted correctly
- Verify data quality and coverage

### Poor Model Performance
- Increase training data
- Adjust feature parameters
- Retrain with better labels
- Check for data imbalance

### Dashboard Not Loading
- Check if predictions file exists: `outputs/anomaly_predictions.csv`
- Verify port 8050 is available
- Check logs: `logs/dashboard.log`

## API Usage

### Programmatic Access
```python
from src.models.ensemble import EnsembleAnomalyDetector
from src.utils.config_loader import load_config

# Load config and model
config = load_config()
detector = EnsembleAnomalyDetector(config)
detector.load_models("outputs/models")

# Predict on new data
import pandas as pd
df = pd.read_csv("new_ais_data.csv")
results = detector.predict_with_details(df)
```

## Support
For issues or questions:
1. Check logs in `logs/` directory
2. Review documentation in `docs/`
3. Open an issue on GitHub
