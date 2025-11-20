import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.config_loader import load_config
from src.utils.logger import setup_logger

logger = setup_logger(__name__, "logs/features.log")

class BehaviorFeatureExtractor:
    def __init__(self, config):
        self.config = config
        self.speed_window = config.get('features', 'behavior', 'speed_window', default=10)
        self.loitering_radius = config.get('features', 'behavior', 'loitering_radius_km', default=5)
        self.loitering_time = config.get('features', 'behavior', 'loitering_time_hours', default=2)
        self.fishing_speed_min = config.get('features', 'behavior', 'fishing_speed_min', default=1)
        self.fishing_speed_max = config.get('features', 'behavior', 'fishing_speed_max', default=5)
    
    def calculate_speed_features(self, group):
        """Calculate speed-based features"""
        group['speed_mean'] = group['SOG'].rolling(window=self.speed_window, min_periods=1).mean()
        group['speed_std'] = group['SOG'].rolling(window=self.speed_window, min_periods=1).std()
        group['speed_variance'] = group['speed_std'] ** 2
        group['speed_max'] = group['SOG'].rolling(window=self.speed_window, min_periods=1).max()
        group['speed_min'] = group['SOG'].rolling(window=self.speed_window, min_periods=1).min()
        return group
    
    def calculate_course_features(self, group):
        """Calculate course and heading features"""
        # Course change (turn rate)
        group['course_change'] = group['COG'].diff().abs()
        group['course_change'] = group['course_change'].apply(
            lambda x: min(x, 360 - x) if pd.notna(x) else x
        )
        group['turn_rate'] = group['course_change'].rolling(window=self.speed_window, min_periods=1).mean()
        
        # Heading deviation
        if 'heading' in group.columns:
            group['heading_deviation'] = (group['heading'] - group['COG']).abs()
            group['heading_deviation'] = group['heading_deviation'].apply(
                lambda x: min(x, 360 - x) if pd.notna(x) else x
            )
        
        return group

    def calculate_distance(self, lat1, lon1, lat2, lon2):
        """Calculate distance between two points using Haversine formula (km)"""
        R = 6371  # Earth radius in km
        
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
    
    def detect_loitering(self, group):
        """Detect loitering behavior"""
        loitering_flags = []
        
        for i in range(len(group)):
            if i < self.speed_window:
                loitering_flags.append(0)
                continue
            
            # Check if vessel stayed within radius for extended time
            window = group.iloc[max(0, i-self.speed_window):i+1]
            
            # Calculate distances from current position
            distances = self.calculate_distance(
                window['lat'].iloc[-1], window['lon'].iloc[-1],
                window['lat'].values, window['lon'].values
            )
            
            # Check if all points within radius
            within_radius = (distances <= self.loitering_radius).sum()
            time_span = (window['timestamp'].iloc[-1] - window['timestamp'].iloc[0]).total_seconds() / 3600
            
            if within_radius >= len(window) * 0.8 and time_span >= self.loitering_time:
                loitering_flags.append(1)
            else:
                loitering_flags.append(0)
        
        group['loitering'] = loitering_flags
        return group
    
    def detect_fishing_speed(self, group):
        """Detect fishing speed patterns (1-5 knots)"""
        group['fishing_speed'] = (
            (group['SOG'] >= self.fishing_speed_min) & 
            (group['SOG'] <= self.fishing_speed_max)
        ).astype(int)
        
        # Calculate percentage of time in fishing speed
        group['fishing_speed_pct'] = group['fishing_speed'].rolling(
            window=self.speed_window, min_periods=1
        ).mean()
        
        return group
    
    def extract_features(self, df):
        """Extract all behavior features"""
        logger.info("Extracting behavior features...")
        
        # Group by vessel (MMSI)
        df_features = df.groupby('MMSI', group_keys=False).apply(
            lambda g: self._extract_vessel_features(g)
        )
        
        logger.info(f"Extracted behavior features for {df_features['MMSI'].nunique()} vessels")
        return df_features
    
    def _extract_vessel_features(self, group):
        """Extract features for a single vessel"""
        group = group.sort_values('timestamp').copy()
        group = self.calculate_speed_features(group)
        group = self.calculate_course_features(group)
        group = self.detect_loitering(group)
        group = self.detect_fishing_speed(group)
        return group

def main():
    config = load_config()
    extractor = BehaviorFeatureExtractor(config)
    
    # Load EEZ-filtered data
    input_path = Path(config.get('data', 'output_dir')) / "ais_eez_filtered.csv"
    df = pd.read_csv(input_path, parse_dates=['timestamp'])
    
    # Extract features
    df_features = extractor.extract_features(df)
    
    # Save features
    output_path = Path(config.get('data', 'output_dir')) / "ais_behavior_features.csv"
    df_features.to_csv(output_path, index=False)
    logger.info(f"Saved behavior features to {output_path}")

if __name__ == "__main__":
    main()
