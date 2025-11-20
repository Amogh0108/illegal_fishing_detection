"""Supervised ML models for anomaly detection"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.config_loader import load_config
from src.utils.logger import setup_logger

logger = setup_logger(__name__, "logs/models.log")

class SupervisedAnomalyDetector:
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()
        self.rf_model = None
        self.svm_model = None
        self.feature_columns = None
        
    def prepare_data(self, df, label_column='anomaly'):
        """Prepare data for training"""
        logger.info("Preparing data for supervised learning...")
        
        # Select feature columns (exclude metadata)
        exclude_cols = ['MMSI', 'timestamp', 'lat', 'lon', label_column, 
                       'lat_diff', 'lon_diff', 'geometry']
        self.feature_columns = [col for col in df.columns if col not in exclude_cols]
        
        X = df[self.feature_columns].copy()
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Handle infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.mean())
        
        if label_column in df.columns:
            y = df[label_column]
            logger.info(f"Features: {len(self.feature_columns)}, Samples: {len(X)}")
            logger.info(f"Anomaly distribution: {y.value_counts().to_dict()}")
            return X, y
        else:
            logger.info(f"Features: {len(self.feature_columns)}, Samples: {len(X)}")
            return X, None
    
    def train_random_forest(self, X_train, y_train):
        """Train Random Forest classifier"""
        logger.info("Training Random Forest...")
        
        n_estimators = self.config.get('models', 'random_forest', 'n_estimators', default=200)
        max_depth = self.config.get('models', 'random_forest', 'max_depth', default=20)
        random_state = self.config.get('models', 'random_forest', 'random_state', default=42)
        
        self.rf_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        self.rf_model.fit(X_train, y_train)
        logger.info("Random Forest training complete")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info(f"Top 10 important features:\n{feature_importance.head(10)}")
        
        return self.rf_model
    
    def train_svm(self, X_train, y_train):
        """Train SVM classifier"""
        logger.info("Training SVM...")
        
        kernel = self.config.get('models', 'svm', 'kernel', default='rbf')
        C = self.config.get('models', 'svm', 'C', default=1.0)
        
        self.svm_model = SVC(
            kernel=kernel,
            C=C,
            probability=True,
            class_weight='balanced'
        )
        
        self.svm_model.fit(X_train, y_train)
        logger.info("SVM training complete")
        
        return self.svm_model
    
    def train(self, df, label_column='anomaly'):
        """Train all supervised models"""
        logger.info("=" * 50)
        logger.info("SUPERVISED MODEL TRAINING")
        logger.info("=" * 50)
        
        # Prepare data
        X, y = self.prepare_data(df, label_column)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train models
        self.train_random_forest(X_train_scaled, y_train)
        self.train_svm(X_train_scaled, y_train)
        
        # Evaluate
        self.evaluate(X_test_scaled, y_test)
        
        return X_test_scaled, y_test
    
    def evaluate(self, X_test, y_test):
        """Evaluate models"""
        logger.info("=" * 50)
        logger.info("MODEL EVALUATION")
        logger.info("=" * 50)
        
        # Random Forest
        logger.info("\nRandom Forest:")
        rf_pred = self.rf_model.predict(X_test)
        rf_proba = self.rf_model.predict_proba(X_test)[:, 1]
        logger.info(f"\n{classification_report(y_test, rf_pred)}")
        logger.info(f"ROC-AUC: {roc_auc_score(y_test, rf_proba):.4f}")
        
        # SVM
        logger.info("\nSVM:")
        svm_pred = self.svm_model.predict(X_test)
        svm_proba = self.svm_model.predict_proba(X_test)[:, 1]
        logger.info(f"\n{classification_report(y_test, svm_pred)}")
        logger.info(f"ROC-AUC: {roc_auc_score(y_test, svm_proba):.4f}")
    
    def predict(self, X):
        """Predict anomalies"""
        X_scaled = self.scaler.transform(X)
        
        rf_proba = self.rf_model.predict_proba(X_scaled)[:, 1]
        svm_proba = self.svm_model.predict_proba(X_scaled)[:, 1]
        
        # Ensemble prediction (average)
        ensemble_proba = (rf_proba + svm_proba) / 2
        
        return ensemble_proba
    
    def save_models(self, output_dir):
        """Save trained models"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.rf_model, output_dir / "random_forest.pkl")
        joblib.dump(self.svm_model, output_dir / "svm.pkl")
        joblib.dump(self.scaler, output_dir / "scaler.pkl")
        joblib.dump(self.feature_columns, output_dir / "feature_columns.pkl")
        
        logger.info(f"Models saved to {output_dir}")
    
    def load_models(self, model_dir):
        """Load trained models"""
        model_dir = Path(model_dir)
        
        self.rf_model = joblib.load(model_dir / "random_forest.pkl")
        self.svm_model = joblib.load(model_dir / "svm.pkl")
        self.scaler = joblib.load(model_dir / "scaler.pkl")
        self.feature_columns = joblib.load(model_dir / "feature_columns.pkl")
        
        logger.info(f"Models loaded from {model_dir}")
