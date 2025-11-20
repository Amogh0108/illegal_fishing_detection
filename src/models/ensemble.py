"""Ensemble model combining all detectors"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.config_loader import load_config
from src.utils.logger import setup_logger
from src.models.supervised_models import SupervisedAnomalyDetector
from src.models.unsupervised_models import UnsupervisedAnomalyDetector
from src.models.lstm_model import LSTMTrainer

logger = setup_logger(__name__, "logs/models.log")

class EnsembleAnomalyDetector:
    def __init__(self, config):
        self.config = config
        self.supervised = SupervisedAnomalyDetector(config)
        self.unsupervised = UnsupervisedAnomalyDetector(config)
        self.lstm = LSTMTrainer(config)
        
        # Ensemble weights
        self.weights = {
            'supervised': config.get('anomaly', 'ensemble_weights', 'supervised', default=0.4),
            'unsupervised': config.get('anomaly', 'ensemble_weights', 'unsupervised', default=0.3),
            'sequential': config.get('anomaly', 'ensemble_weights', 'sequential', default=0.3)
        }
        
        self.threshold = config.get('anomaly', 'threshold', default=0.7)
    
    def load_models(self, model_dir):
        """Load all trained models"""
        logger.info(f"Loading models from {model_dir}")
        
        self.supervised.load_models(model_dir)
        self.unsupervised.load_models(model_dir)
        
        # Load LSTM (need to know input size)
        # This is a simplified version - in production, save model config
        logger.info("LSTM model loading skipped in ensemble (requires sequence data)")
    
    def predict(self, df, use_lstm=False):
        """Predict anomaly scores using ensemble"""
        logger.info("Running ensemble prediction...")
        
        # Prepare data for supervised/unsupervised
        X_supervised, _ = self.supervised.prepare_data(df.copy())
        X_unsupervised = self.unsupervised.prepare_data(df.copy())
        
        # Get predictions
        supervised_scores = self.supervised.predict(X_supervised)
        unsupervised_scores = self.unsupervised.predict(X_unsupervised)
        
        # Combine scores
        if use_lstm:
            # LSTM requires sequence data - simplified here
            ensemble_scores = (
                self.weights['supervised'] * supervised_scores +
                self.weights['unsupervised'] * unsupervised_scores
            ) / (self.weights['supervised'] + self.weights['unsupervised'])
        else:
            ensemble_scores = (
                self.weights['supervised'] * supervised_scores +
                self.weights['unsupervised'] * unsupervised_scores
            ) / (self.weights['supervised'] + self.weights['unsupervised'])
        
        # Apply threshold
        anomaly_predictions = (ensemble_scores >= self.threshold).astype(int)
        
        logger.info(f"Detected {anomaly_predictions.sum()} anomalies ({anomaly_predictions.sum()/len(anomaly_predictions)*100:.2f}%)")
        
        return ensemble_scores, anomaly_predictions
    
    def predict_with_details(self, df):
        """Predict with detailed scores from each model"""
        logger.info("Running detailed ensemble prediction...")
        
        # Prepare data
        X_supervised, _ = self.supervised.prepare_data(df.copy())
        X_unsupervised = self.unsupervised.prepare_data(df.copy())
        
        # Get predictions
        supervised_scores = self.supervised.predict(X_supervised)
        unsupervised_scores = self.unsupervised.predict(X_unsupervised)
        
        # Ensemble
        ensemble_scores = (
            self.weights['supervised'] * supervised_scores +
            self.weights['unsupervised'] * unsupervised_scores
        ) / (self.weights['supervised'] + self.weights['unsupervised'])
        
        # Create results dataframe
        results = df[['MMSI', 'timestamp', 'lat', 'lon']].copy()
        results['supervised_score'] = supervised_scores
        results['unsupervised_score'] = unsupervised_scores
        results['ensemble_score'] = ensemble_scores
        results['anomaly'] = (ensemble_scores >= self.threshold).astype(int)
        
        return results

def main():
    """Test ensemble prediction"""
    config = load_config()
    
    # Load test data
    input_path = Path(config.get('data', 'output_dir')) / "ais_all_features.csv"
    df = pd.read_csv(input_path, parse_dates=['timestamp'])
    
    # Initialize ensemble
    ensemble = EnsembleAnomalyDetector(config)
    ensemble.load_models("outputs/models")
    
    # Predict
    results = ensemble.predict_with_details(df)
    
    # Save results
    output_path = Path("outputs") / "anomaly_predictions.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    logger.info(f"Saved predictions to {output_path}")
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("PREDICTION SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Total records: {len(results)}")
    logger.info(f"Anomalies detected: {results['anomaly'].sum()}")
    logger.info(f"Anomaly rate: {results['anomaly'].sum()/len(results)*100:.2f}%")
    logger.info(f"Average ensemble score: {results['ensemble_score'].mean():.4f}")

if __name__ == "__main__":
    main()
