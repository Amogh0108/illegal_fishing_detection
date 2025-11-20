"""Main feature extraction pipeline"""
import pandas as pd
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.config_loader import load_config
from src.utils.logger import setup_logger
from src.features.behavior_features import BehaviorFeatureExtractor
from src.features.transmission_features import TransmissionFeatureExtractor

logger = setup_logger(__name__, "logs/features.log")

def main():
    """Run complete feature extraction pipeline"""
    config = load_config()
    
    # Load EEZ-filtered data
    input_path = Path(config.get('data', 'output_dir')) / "ais_eez_filtered.csv"
    logger.info(f"Loading data from {input_path}")
    df = pd.read_csv(input_path, parse_dates=['timestamp'])
    
    # Extract behavior features
    logger.info("=" * 50)
    logger.info("BEHAVIOR FEATURES")
    logger.info("=" * 50)
    behavior_extractor = BehaviorFeatureExtractor(config)
    df = behavior_extractor.extract_features(df)
    
    # Extract transmission features
    logger.info("=" * 50)
    logger.info("TRANSMISSION FEATURES")
    logger.info("=" * 50)
    transmission_extractor = TransmissionFeatureExtractor(config)
    df = transmission_extractor.extract_features(df)
    
    # Save final features
    output_path = Path(config.get('data', 'output_dir')) / "ais_all_features.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Saved all features to {output_path}")
    
    # Print feature summary
    logger.info("=" * 50)
    logger.info("FEATURE SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Total records: {len(df)}")
    logger.info(f"Total vessels: {df['MMSI'].nunique()}")
    logger.info(f"Total features: {len(df.columns)}")
    logger.info(f"Feature columns: {list(df.columns)}")

if __name__ == "__main__":
    main()
