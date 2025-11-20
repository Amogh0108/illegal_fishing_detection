"""Unsupervised ML models for anomaly detection"""
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.config_loader import load_config
from src.utils.logger import setup_logger

logger = setup_logger(__name__, "logs/models.log")

class UnsupervisedAnomalyDetector:
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()
        self.isolation_forest = None
        self.lof = None
        self.feature_columns = None
        
    def prepare_data(self, df):
        """Prepare data for unsupervised learning"""
        logger.info("Preparing data for unsupervised learning...")
        
        # Select feature columns
        exclude_cols = ['MMSI', 'timestamp', 'lat', 'lon', 'anomaly',
                       'lat_diff', 'lon_diff', 'geometry']
        self.feature_columns = [col for col in df.columns if col not in exclude_cols]
        
        X = df[self.feature_columns].copy()
        
        # Handle missing and infinite values
        X = X.fillna(X.mean())
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.mean())
        
        logger.info(f"Features: {len(self.feature_columns)}, Samples: {len(X)}")
        return X
    
    def train_isolation_forest(self, X):
        """Train Isolation Forest"""
        logger.info("Training Isolation Forest...")
        
        contamination = self.config.get('models', 'isolation_forest', 'contamination', default=0.1)
        random_state = self.config.get('models', 'isolation_forest', 'random_state', default=42)
        
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1
        )
        
        self.isolation_forest.fit(X)
        logger.info("Isolation Forest training complete")
        
        return self.isolation_forest
    
    def train_lof(self, X):
        """Train Local Outlier Factor"""
        logger.info("Training Local Outlier Factor...")
        
        contamination = self.config.get('models', 'isolation_forest', 'contamination', default=0.1)
        
        self.lof = LocalOutlierFactor(
            contamination=contamination,
            novelty=True,
            n_jobs=-1
        )
        
        self.lof.fit(X)
        logger.info("LOF training complete")
        
        return self.lof
    
    def train(self, df):
        """Train all unsupervised models"""
        logger.info("=" * 50)
        logger.info("UNSUPERVISED MODEL TRAINING")
        logger.info("=" * 50)
        
        # Prepare data
        X = self.prepare_data(df)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train models
        self.train_isolation_forest(X_scaled)
        self.train_lof(X_scaled)
        
        # Get anomaly scores
        if_scores = self.isolation_forest.score_samples(X_scaled)
        lof_scores = self.lof.score_samples(X_scaled)
        
        logger.info(f"Isolation Forest anomaly score range: [{if_scores.min():.4f}, {if_scores.max():.4f}]")
        logger.info(f"LOF anomaly score range: [{lof_scores.min():.4f}, {lof_scores.max():.4f}]")
        
        return X_scaled
    
    def predict(self, X):
        """Predict anomaly scores"""
        X_scaled = self.scaler.transform(X)
        
        # Get anomaly scores (lower = more anomalous)
        if_scores = self.isolation_forest.score_samples(X_scaled)
        lof_scores = self.lof.score_samples(X_scaled)
        
        # Normalize to [0, 1] where 1 = anomaly
        if_scores_norm = 1 - (if_scores - if_scores.min()) / (if_scores.max() - if_scores.min())
        lof_scores_norm = 1 - (lof_scores - lof_scores.min()) / (lof_scores.max() - lof_scores.min())
        
        # Ensemble (average)
        ensemble_scores = (if_scores_norm + lof_scores_norm) / 2
        
        return ensemble_scores
    
    def save_models(self, output_dir):
        """Save trained models"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.isolation_forest, output_dir / "isolation_forest.pkl")
        joblib.dump(self.lof, output_dir / "lof.pkl")
        joblib.dump(self.scaler, output_dir / "unsupervised_scaler.pkl")
        joblib.dump(self.feature_columns, output_dir / "unsupervised_feature_columns.pkl")
        
        logger.info(f"Unsupervised models saved to {output_dir}")
    
    def load_models(self, model_dir):
        """Load trained models"""
        model_dir = Path(model_dir)
        
        self.isolation_forest = joblib.load(model_dir / "isolation_forest.pkl")
        self.lof = joblib.load(model_dir / "lof.pkl")
        self.scaler = joblib.load(model_dir / "unsupervised_scaler.pkl")
        self.feature_columns = joblib.load(model_dir / "unsupervised_feature_columns.pkl")
        
        logger.info(f"Unsupervised models loaded from {model_dir}")
