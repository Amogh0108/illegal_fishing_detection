# IUU Fishing Detection Dashboard

## 🎨 Modern, Professional UI

A beautiful, responsive dashboard for real-time monitoring of IUU fishing activities in the Indian EEZ.

## ✨ Features

### 📊 Real-Time Monitoring
- Live vessel tracking on interactive map
- Anomaly score timeline visualization
- Model comparison charts
- Automatic data refresh every 5 minutes

### 🎯 Interactive Controls
- Adjustable anomaly threshold slider
- Vessel filtering by MMSI
- Manual refresh button
- Responsive design for all screen sizes

### 📈 Visualizations
1. **Interactive Map**
   - Vessel trajectories with color-coded anomalies
   - Hover tooltips with detailed information
   - Zoom and pan controls
   - Size-based anomaly score representation

2. **Timeline Chart**
   - Anomaly scores over time
   - Threshold indicator line
   - Smooth animations

3. **Model Comparison**
   - Supervised vs Unsupervised vs Ensemble scores
   - Area fill for better visualization
   - Interactive legend

4. **Anomaly Table**
   - Top 10 recent anomalies
   - Sortable columns
   - Color-coded high-risk alerts
   - Pagination support

### 📱 Statistics Cards
- Total Vessels
- Anomalies Detected
- Anomaly Rate
- Average Anomaly Score

## 🚀 Quick Start

### Run the Dashboard

```bash
# From project root
python src/dashboard/app.py
```

Or use the convenience script:

```bash
python -m src.dashboard.app
```

### Access the Dashboard

Open your browser and navigate to:
```
http://localhost:8050
```

Or if running on a server:
```
http://your-server-ip:8050
```

## ⚙️ Configuration

Edit `config/config.yaml` to customize:

```yaml
dashboard:
  host: "0.0.0.0"  # Listen on all interfaces
  port: 8050       # Dashboard port
  update_interval: 300  # Auto-refresh interval (seconds)
```

## 🎨 Design Features

### Color Scheme
- **Primary**: Deep Blue (#1e3a8a)
- **Secondary**: Bright Blue (#3b82f6)
- **Success**: Green (#10b981)
- **Warning**: Orange (#f59e0b)
- **Danger**: Red (#ef4444)

### Typography
- Font Family: Inter (Google Fonts)
- Modern, clean, and highly readable

### UI Components
- Rounded corners (12px border-radius)
- Subtle shadows for depth
- Smooth transitions and animations
- Hover effects on interactive elements
- Responsive grid layout

## 📊 Data Requirements

The dashboard expects data in the following format:

```csv
MMSI,timestamp,lat,lon,ensemble_score,supervised_score,unsupervised_score
123456789,2024-01-01 00:00:00,10.5,75.2,0.85,0.82,0.88
```

Required columns:
- `MMSI`: Vessel identifier
- `timestamp`: ISO format datetime
- `lat`, `lon`: Coordinates
- `ensemble_score`: Combined anomaly score (0-1)
- `supervised_score`: Supervised model score
- `unsupervised_score`: Unsupervised model score

## 🔧 Customization

### Change Colors

Edit `src/dashboard/app.py` and modify the `COLORS` dictionary:

```python
COLORS = {
    'primary': '#your-color',
    'secondary': '#your-color',
    # ... etc
}
```

### Add Custom CSS

Create or edit `src/dashboard/assets/custom.css` to add your own styles.

### Modify Layout

The dashboard layout is defined in `app.layout`. You can:
- Add new cards
- Rearrange components
- Add new visualizations
- Customize existing charts

## 📱 Responsive Design

The dashboard automatically adapts to different screen sizes:
- **Desktop**: Full multi-column layout
- **Tablet**: Stacked layout with optimized spacing
- **Mobile**: Single column, touch-friendly controls

## 🎯 Usage Tips

### Adjusting Sensitivity
- Move the threshold slider left for more sensitive detection
- Move right to reduce false positives

### Vessel Tracking
- Select a specific vessel from the dropdown to focus on its trajectory
- Clear selection to view all vessels

### Interpreting Scores
- **0.0 - 0.5**: Low risk (normal behavior)
- **0.5 - 0.7**: Medium risk (monitor)
- **0.7 - 1.0**: High risk (investigate)

### Map Navigation
- Click and drag to pan
- Scroll to zoom
- Click markers for details

## 🐛 Troubleshooting

### Dashboard won't start
```bash
# Check if port is already in use
netstat -ano | findstr :8050

# Try a different port
# Edit config/config.yaml and change port number
```

### No data showing
```bash
# Ensure predictions file exists
ls outputs/anomaly_predictions.csv

# Run the pipeline first
python scripts/run_pipeline.py
```

### Slow performance
```bash
# Reduce data size by filtering
# Or increase update interval in config
```

## 🔄 Auto-Refresh

The dashboard automatically refreshes data every 5 minutes. You can:
- Click "Refresh Data" for manual update
- Modify interval in `config/config.yaml`
- Disable by removing the `dcc.Interval` component

## 📸 Screenshots

The dashboard features:
- Clean, modern interface
- Professional color scheme
- Smooth animations
- Interactive visualizations
- Real-time updates

## 🚀 Deployment

### Local Development
```bash
python src/dashboard/app.py
```

### Production Deployment

Using Gunicorn:
```bash
pip install gunicorn
gunicorn src.dashboard.app:server -b 0.0.0.0:8050
```

Using Docker:
```dockerfile
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "src/dashboard/app.py"]
```

## 📚 Dependencies

- dash >= 2.14.0
- plotly >= 5.18.0
- pandas >= 2.0.0
- numpy >= 1.24.0

## 🤝 Contributing

To improve the dashboard:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

Part of the IUU Fishing Detection System project.

---

**Built with ❤️ using Dash and Plotly**
