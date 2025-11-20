import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.config_loader import load_config
from src.utils.logger import setup_logger

logger = setup_logger(__name__, "logs/preprocessing.log")

class AISCleaner:
    def __init__(self, config):
        self.config = config
        
    def load_data(self, filepath):
        """Load AIS data from CSV"""
        logger.info(f"Loading AIS data from {filepath}")
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df)} records")
        return df
    
    def clean_coordinates(self, df):
        """Remove invalid coordinates"""
        logger.info("Cleaning coordinates...")
        initial_count = len(df)
        
        # Valid lat/lon ranges
        df = df[
            (df['lat'].between(-90, 90)) &
            (df['lon'].between(-180, 180))
        ]
        
        # Remove zeros
        df = df[(df['lat'] != 0) | (df['lon'] != 0)]
        
        removed = initial_count - len(df)
        logger.info(f"Removed {removed} invalid coordinate records")
        return df
    
    def clean_timestamps(self, df):
        """Parse and validate timestamps"""
        logger.info("Cleaning timestamps...")
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp'])
        df = df.sort_values(['MMSI', 'timestamp'])
        return df
    
    def clean_speed_course(self, df):
        """Clean speed and course values"""
        logger.info("Cleaning speed and course...")
        
        # Speed over ground (SOG) should be 0-50 knots for fishing vessels
        df = df[df['SOG'].between(0, 50)]
        
        # Course over ground (COG) should be 0-360
        df = df[df['COG'].between(0, 360)]
        
        return df

    def remove_duplicates(self, df):
        """Remove duplicate records"""
        logger.info("Removing duplicates...")
        initial_count = len(df)
        df = df.drop_duplicates(subset=['MMSI', 'timestamp', 'lat', 'lon'])
        removed = initial_count - len(df)
        logger.info(f"Removed {removed} duplicate records")
        return df
    
    def clean(self, df):
        """Run full cleaning pipeline"""
        logger.info("Starting AIS data cleaning pipeline...")
        df = self.clean_coordinates(df)
        df = self.clean_timestamps(df)
        df = self.clean_speed_course(df)
        df = self.remove_duplicates(df)
        logger.info(f"Cleaning complete. Final records: {len(df)}")
        return df

def main():
    config = load_config()
    cleaner = AISCleaner(config)
    
    # Load and clean data
    input_path = config.get('data', 'ais_data')
    df = cleaner.load_data(input_path)
    df_clean = cleaner.clean(df)
    
    # Save cleaned data
    output_dir = Path(config.get('data', 'output_dir'))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "ais_cleaned.csv"
    df_clean.to_csv(output_path, index=False)
    logger.info(f"Saved cleaned data to {output_path}")

if __name__ == "__main__":
    main()
