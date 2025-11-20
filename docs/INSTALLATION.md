# Installation Guide

## Prerequisites
- Python 3.8 or higher
- pip package manager
- 8GB RAM minimum (16GB recommended)
- GPU optional (for LSTM training)

## Step 1: Clone Repository
```bash
git clone <repository-url>
cd iuu-fishing-detection
```

## Step 2: Create Virtual Environment
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n iuu-detection python=3.9
conda activate iuu-detection
```

## Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Key Dependencies
- **pandas, numpy**: Data manipulation
- **scikit-learn**: ML models
- **torch**: Deep learning (LSTM)
- **geopandas**: Geospatial operations
- **dash, plotly**: Interactive dashboard
- **folium**: Map visualization

## Step 4: Verify Installation
```bash
python -c "import pandas, sklearn, torch, geopandas, dash; print('All dependencies installed successfully')"
```

## Step 5: Generate Sample Data
```bash
python scripts/generate_sample_data.py
```

This creates:
- `data/raw/ais_data.csv`: Sample AIS data
- `data/raw/indian_eez.geojson`: EEZ boundary
- `data/raw/vessel_registry.csv`: Vessel information

## Step 6: Run Pipeline
```bash
python scripts/run_pipeline.py
```

This executes:
1. Data cleaning
2. EEZ filtering
3. Feature extraction
4. Model training
5. Baseline comparison
6. Ensemble prediction
7. Evaluation

## Step 7: Launch Dashboard
```bash
python src/dashboard/app.py
```

Access at: http://localhost:8050

## Troubleshooting

### GDAL/Geopandas Issues
If geopandas installation fails:
```bash
# On Ubuntu/Debian
sudo apt-get install gdal-bin libgdal-dev

# On macOS
brew install gdal

# Then install geopandas
pip install geopandas
```

### PyTorch GPU Support
For CUDA support:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Memory Issues
For large datasets, increase batch size or reduce data:
```yaml
# In config/config.yaml
models:
  lstm:
    batch_size: 16  # Reduce from 32
```

## Directory Structure After Installation
```
iuu-fishing-detection/
├── data/
│   ├── raw/              # Original data
│   └── processed/        # Processed data
├── outputs/
│   ├── models/           # Trained models
│   └── evaluation/       # Evaluation results
├── logs/                 # Application logs
└── src/                  # Source code
```

## Next Steps
1. Review `docs/METHODOLOGY.md` for technical details
2. Explore `docs/USER_GUIDE.md` for usage instructions
3. Check `docs/API.md` for programmatic access
