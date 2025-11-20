import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.config_loader import load_config
from src.utils.logger import setup_logger

logger = setup_logger(__name__, "logs/features.log")

class TransmissionFeatureExtractor:
    def __init__(self, config):
        self.config = config
        self.max_gap_minutes = config.get('features', 'transmission', 'max_gap_minutes', default=60)
        self.mmsi_change_threshold = config.get('features', 'transmission', 'mmsi_change_threshold', default=3)
    
    def detect_ais_gaps(self, group):
        """Detect AIS transmission gaps"""
        group = group.sort_values('timestamp').copy()
        
        # Calculate time gaps between consecutive transmissions
        group['time_gap'] = group['timestamp'].diff().dt.total_seconds() / 60  # minutes
        
        # Flag significant gaps
        group['ais_gap'] = (group['time_gap'] > self.max_gap_minutes).astype(int)
        
        # Count gaps in rolling window
        group['gap_count'] = group['ais_gap'].rolling(window=20, min_periods=1).sum()
        
        # Calculate average gap duration
        group['avg_gap_duration'] = group['time_gap'].rolling(window=20, min_periods=1).mean()
        
        return group
    
    def detect_sudden_disappearance(self, group):
        """Detect sudden disappearance and reappearance"""
        group['disappeared'] = 0
        
        for i in range(1, len(group)):
            # Check if vessel disappeared for extended period
            if group['time_gap'].iloc[i] > self.max_gap_minutes * 2:
                group.loc[group.index[i], 'disappeared'] = 1
        
        return group
    
    def detect_position_jumps(self, group):
        """Detect unrealistic position jumps (potential spoofing)"""
        group = group.sort_values('timestamp').copy()
        
        # Calculate distance between consecutive points
        group['lat_diff'] = group['lat'].diff()
        group['lon_diff'] = group['lon'].diff()
        
        # Calculate distance using Haversine
        R = 6371  # Earth radius in km
        lat1 = np.radians(group['lat'].shift(1))
        lat2 = np.radians(group['lat'])
        lon1 = np.radians(group['lon'].shift(1))
        lon2 = np.radians(group['lon'])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        distance = R * c
        
        # Calculate expected max distance based on time gap and max speed
        max_speed_kmh = 50 * 1.852  # 50 knots to km/h
        time_hours = group['time_gap'] / 60
        expected_max_distance = max_speed_kmh * time_hours
        
        # Flag unrealistic jumps
        group['position_jump'] = (distance > expected_max_distance * 1.5).astype(int)
        
        return group
    
    def calculate_transmission_regularity(self, group):
        """Calculate transmission regularity metrics"""
        group = group.sort_values('timestamp').copy()
        
        # Standard deviation of time gaps
        group['gap_std'] = group['time_gap'].rolling(window=20, min_periods=1).std()
        
        # Transmission frequency (messages per hour)
        group['transmission_freq'] = 60 / group['time_gap'].rolling(window=20, min_periods=1).mean()
        
        return group
    
    def extract_features(self, df):
        """Extract all transmission features"""
        logger.info("Extracting transmission features...")
        
        # Group by vessel (MMSI)
        df_features = df.groupby('MMSI', group_keys=False).apply(
            lambda g: self._extract_vessel_features(g)
        )
        
        logger.info(f"Extracted transmission features for {df_features['MMSI'].nunique()} vessels")
        return df_features
    
    def _extract_vessel_features(self, group):
        """Extract features for a single vessel"""
        group = self.detect_ais_gaps(group)
        group = self.detect_sudden_disappearance(group)
        group = self.detect_position_jumps(group)
        group = self.calculate_transmission_regularity(group)
        return group

def main():
    config = load_config()
    extractor = TransmissionFeatureExtractor(config)
    
    # Load behavior features data
    input_path = Path(config.get('data', 'output_dir')) / "ais_behavior_features.csv"
    df = pd.read_csv(input_path, parse_dates=['timestamp'])
    
    # Extract transmission features
    df_features = extractor.extract_features(df)
    
    # Save features
    output_path = Path(config.get('data', 'output_dir')) / "ais_all_features.csv"
    df_features.to_csv(output_path, index=False)
    logger.info(f"Saved all features to {output_path}")

if __name__ == "__main__":
    main()
