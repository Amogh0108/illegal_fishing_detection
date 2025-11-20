"""Rule-based baseline for comparison"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.config_loader import load_config
from src.utils.logger import setup_logger

logger = setup_logger(__name__, "logs/evaluation.log")

class RuleBasedDetector:
    """Traditional rule-based anomaly detection"""
    
    def __init__(self, config):
        self.config = config
        
        # Rule thresholds
        self.speed_threshold = 30  # knots
        self.gap_threshold = 120  # minutes
        self.fishing_speed_min = 1
        self.fishing_speed_max = 5
        self.loitering_threshold = 3  # hours
    
    def detect_anomalies(self, df):
        """Apply rule-based detection"""
        logger.info("Running rule-based anomaly detection...")
        
        df['rule_anomaly'] = 0
        
        # Rule 1: Excessive speed
        if 'SOG' in df.columns:
            df.loc[df['SOG'] > self.speed_threshold, 'rule_anomaly'] = 1
        
        # Rule 2: AIS transmission gaps
        if 'time_gap' in df.columns:
            df.loc[df['time_gap'] > self.gap_threshold, 'rule_anomaly'] = 1
        
        # Rule 3: Fishing speed pattern
        if 'fishing_speed' in df.columns:
            df.loc[df['fishing_speed'] == 1, 'rule_anomaly'] = 1
        
        # Rule 4: Position jumps
        if 'position_jump' in df.columns:
            df.loc[df['position_jump'] == 1, 'rule_anomaly'] = 1
        
        # Rule 5: Loitering
        if 'loitering' in df.columns:
            df.loc[df['loitering'] == 1, 'rule_anomaly'] = 1
        
        anomaly_count = df['rule_anomaly'].sum()
        logger.info(f"Rule-based detection found {anomaly_count} anomalies ({anomaly_count/len(df)*100:.2f}%)")
        
        return df

def main():
    """Run rule-based baseline"""
    config = load_config()
    detector = RuleBasedDetector(config)
    
    # Load features
    input_path = Path(config.get('data', 'output_dir')) / "ais_all_features.csv"
    df = pd.read_csv(input_path, parse_dates=['timestamp'])
    
    # Detect anomalies
    df = detector.detect_anomalies(df)
    
    # Save results
    output_path = Path("outputs") / "rule_based_predictions.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df[['MMSI', 'timestamp', 'lat', 'lon', 'rule_anomaly']].to_csv(output_path, index=False)
    logger.info(f"Saved rule-based predictions to {output_path}")

if __name__ == "__main__":
    main()
