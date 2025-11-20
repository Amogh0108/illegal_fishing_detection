import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.config_loader import load_config
from src.utils.logger import setup_logger

logger = setup_logger(__name__, "logs/preprocessing.log")

class EEZFilter:
    def __init__(self, config):
        self.config = config
        self.eez_boundary = None
        
    def load_eez_boundary(self, filepath):
        """Load EEZ boundary from GeoJSON"""
        logger.info(f"Loading EEZ boundary from {filepath}")
        self.eez_boundary = gpd.read_file(filepath)
        logger.info(f"Loaded EEZ boundary with {len(self.eez_boundary)} features")
        return self.eez_boundary
    
    def create_geodataframe(self, df):
        """Convert DataFrame to GeoDataFrame"""
        logger.info("Creating GeoDataFrame from AIS data...")
        geometry = [Point(xy) for xy in zip(df['lon'], df['lat'])]
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
        return gdf
    
    def filter_within_eez(self, gdf):
        """Filter trajectories within EEZ"""
        logger.info("Filtering trajectories within EEZ...")
        
        if self.eez_boundary is None:
            raise ValueError("EEZ boundary not loaded")
        
        # Ensure same CRS
        if gdf.crs != self.eez_boundary.crs:
            gdf = gdf.to_crs(self.eez_boundary.crs)
        
        # Get original columns
        original_columns = gdf.columns.tolist()
        
        # Spatial join
        initial_count = len(gdf)
        gdf_filtered = gpd.sjoin(gdf, self.eez_boundary, how='inner', predicate='within')
        
        # Keep only original columns (remove spatial join columns)
        columns_to_keep = [col for col in original_columns if col in gdf_filtered.columns]
        gdf_filtered = gdf_filtered[columns_to_keep]
        
        # Remove duplicates that may have been created by spatial join
        gdf_filtered = gdf_filtered.drop_duplicates()
        
        filtered_count = len(gdf_filtered)
        logger.info(f"Filtered to {filtered_count} records within EEZ ({filtered_count/initial_count*100:.2f}%)")
        
        return gdf_filtered
    
    def filter(self, df):
        """Run full EEZ filtering pipeline"""
        logger.info("Starting EEZ filtering pipeline...")
        
        # Load EEZ boundary
        eez_path = self.config.get('data', 'eez_boundary')
        self.load_eez_boundary(eez_path)
        
        # Create GeoDataFrame and filter
        gdf = self.create_geodataframe(df)
        gdf_filtered = self.filter_within_eez(gdf)
        
        # Convert back to DataFrame
        df_filtered = pd.DataFrame(gdf_filtered.drop(columns='geometry'))
        
        logger.info(f"EEZ filtering complete. Final records: {len(df_filtered)}")
        return df_filtered

def main():
    config = load_config()
    eez_filter = EEZFilter(config)
    
    # Load cleaned data
    input_path = Path(config.get('data', 'output_dir')) / "ais_cleaned.csv"
    df = pd.read_csv(input_path)
    
    # Filter within EEZ
    df_eez = eez_filter.filter(df)
    
    # Save filtered data
    output_path = Path(config.get('data', 'output_dir')) / "ais_eez_filtered.csv"
    df_eez.to_csv(output_path, index=False)
    logger.info(f"Saved EEZ-filtered data to {output_path}")

if __name__ == "__main__":
    main()
