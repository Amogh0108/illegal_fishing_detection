"""Main training pipeline for all models"""
import pandas as pd
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.config_loader import load_config
from src.utils.logger import setup_logger
from src.models.supervised_models import SupervisedAnomalyDetector
from src.models.unsupervised_models import UnsupervisedAnomalyDetector
from src.models.lstm_model import LSTMTrainer

logger = setup_logger(__name__, "logs/models.log")

def create_synthetic_labels(df):
    """Create synthetic anomaly labels for demonstration"""
    logger.info("Creating synthetic anomaly labels...")
    
    # Mark as anomaly if multiple suspicious behaviors detected
    df['anomaly'] = 0
    
    # High speed variance
    if 'speed_variance' in df.columns:
        df.loc[df['speed_variance'] > df['speed_variance'].quantile(0.9), 'anomaly'] = 1
    
    # AIS gaps
    if 'ais_gap' in df.columns:
        df.loc[df['ais_gap'] == 1, 'anomaly'] = 1
    
    # Position jumps
    if 'position_jump' in df.columns:
        df.loc[df['position_jump'] == 1, 'anomaly'] = 1
    
    # Loitering + fishing speed
    if 'loitering' in df.columns and 'fishing_speed' in df.columns:
        df.loc[(df['loitering'] == 1) & (df['fishing_speed'] == 1), 'anomaly'] = 1
    
    anomaly_count = df['anomaly'].sum()
    logger.info(f"Created {anomaly_count} anomaly labels ({anomaly_count/len(df)*100:.2f}%)")
    
    return df

def main():
    """Run complete training pipeline"""
    config = load_config()
    
    # Load features
    input_path = Path(config.get('data', 'output_dir')) / "ais_all_features.csv"
    logger.info(f"Loading features from {input_path}")
    df = pd.read_csv(input_path, parse_dates=['timestamp'])
    
    # Create synthetic labels (replace with real labels if available)
    df = create_synthetic_labels(df)
    
    # Train supervised models
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING SUPERVISED MODELS")
    logger.info("=" * 70)
    supervised_detector = SupervisedAnomalyDetector(config)
    supervised_detector.train(df, label_column='anomaly')
    supervised_detector.save_models("outputs/models")
    
    # Train unsupervised models
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING UNSUPERVISED MODELS")
    logger.info("=" * 70)
    unsupervised_detector = UnsupervisedAnomalyDetector(config)
    unsupervised_detector.train(df)
    unsupervised_detector.save_models("outputs/models")
    
    # Train LSTM model
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING LSTM MODEL")
    logger.info("=" * 70)
    lstm_trainer = LSTMTrainer(config)
    lstm_trainer.train(df, sequence_length=50)
    lstm_trainer.save_model("outputs/models")
    
    logger.info("\n" + "=" * 70)
    logger.info("ALL MODELS TRAINED SUCCESSFULLY")
    logger.info("=" * 70)

if __name__ == "__main__":
    main()
