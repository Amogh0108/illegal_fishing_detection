"""Generate project summary"""
import pandas as pd
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

print("=" * 70)
print("IUU FISHING DETECTION PROJECT - EXECUTION SUMMARY")
print("=" * 70)

# Data Summary
print("\n[1] DATA PROCESSING")
print("-" * 70)

cleaned_df = pd.read_csv("data/processed/ais_cleaned.csv")
print(f"✓ Cleaned AIS Data: {len(cleaned_df):,} records")

eez_df = pd.read_csv("data/processed/ais_eez_filtered.csv")
print(f"✓ EEZ Filtered Data: {len(eez_df):,} records ({len(eez_df)/len(cleaned_df)*100:.1f}% within EEZ)")

features_df = pd.read_csv("data/processed/ais_all_features.csv")
print(f"✓ Feature Extraction: {len(features_df):,} records, {len(features_df.columns)} features")
print(f"✓ Unique Vessels: {features_df['MMSI'].nunique()}")

# Feature Summary
print("\n[2] EXTRACTED FEATURES")
print("-" * 70)
print("Behavioral Features:")
behavior_features = ['speed_mean', 'speed_std', 'speed_variance', 'speed_max', 'speed_min',
                     'course_change', 'turn_rate', 'heading_deviation', 'loitering',
                     'fishing_speed', 'fishing_speed_pct']
for feat in behavior_features:
    if feat in features_df.columns:
        print(f"  • {feat}")

print("\nTransmission Features:")
transmission_features = ['time_gap', 'ais_gap', 'gap_count', 'avg_gap_duration',
                         'disappeared', 'position_jump', 'gap_std', 'transmission_freq']
for feat in transmission_features:
    if feat in features_df.columns:
        print(f"  • {feat}")

# Model Summary
print("\n[3] TRAINED MODELS")
print("-" * 70)

models_dir = Path("outputs/models")
if models_dir.exists():
    models = list(models_dir.glob("*.pkl"))
    print(f"✓ Total Models Saved: {len(models)}")
    for model in models:
        size_kb = model.stat().st_size / 1024
        print(f"  • {model.name}: {size_kb:.1f} KB")

# Anomaly Statistics
print("\n[4] ANOMALY DETECTION RESULTS")
print("-" * 70)
if 'anomaly' in features_df.columns:
    anomaly_count = features_df['anomaly'].sum()
    anomaly_pct = anomaly_count / len(features_df) * 100
    print(f"✓ Synthetic Anomalies Created: {anomaly_count:,} ({anomaly_pct:.2f}%)")
    print(f"✓ Normal Behavior: {len(features_df) - anomaly_count:,} ({100-anomaly_pct:.2f}%)")

print("\n[5] PROJECT STATUS")
print("-" * 70)
print("✓ Data Preprocessing: COMPLETE")
print("✓ EEZ Filtering: COMPLETE")
print("✓ Feature Engineering: COMPLETE")
print("✓ Supervised Models (RF, SVM): COMPLETE")
print("✓ Unsupervised Models (Isolation Forest, LOF): COMPLETE")
print("⏳ LSTM Model: IN PROGRESS (training takes ~10-15 minutes)")
print("⏳ Baseline Comparison: PENDING")
print("⏳ Ensemble Prediction: PENDING")
print("⏳ Model Evaluation: PENDING")
print("⏳ Dashboard: PENDING")

print("\n" + "=" * 70)
print("NEXT STEPS:")
print("=" * 70)
print("1. Wait for LSTM training to complete (or skip with Ctrl+C)")
print("2. Run baseline comparison: python src/evaluation/baseline.py")
print("3. Run ensemble prediction: python src/models/ensemble.py")
print("4. Evaluate models: python src/evaluation/metrics.py")
print("5. Launch dashboard: python src/dashboard/app.py")
print("=" * 70)
