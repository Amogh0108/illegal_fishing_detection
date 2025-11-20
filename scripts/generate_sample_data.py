"""Generate sample AIS data for testing"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json

def generate_sample_ais_data(n_vessels=50, n_points_per_vessel=200):
    """Generate synthetic AIS data"""
    print("Generating sample AIS data...")
    
    # Indian EEZ approximate bounds
    lat_min, lat_max = 6.0, 22.0
    lon_min, lon_max = 68.0, 88.0
    
    data = []
    start_time = datetime(2024, 1, 1)
    
    for vessel_id in range(1, n_vessels + 1):
        mmsi = 400000000 + vessel_id  # Indian MMSI range
        
        # Random starting position
        lat = np.random.uniform(lat_min, lat_max)
        lon = np.random.uniform(lon_min, lon_max)
        
        # Random vessel behavior
        is_anomalous = np.random.random() < 0.15  # 15% anomalous vessels
        
        timestamp = start_time
        
        for point in range(n_points_per_vessel):
            # Normal behavior
            if not is_anomalous or np.random.random() > 0.3:
                # Normal fishing vessel
                sog = np.random.uniform(2, 8)  # Speed over ground (knots)
                cog = np.random.uniform(0, 360)  # Course over ground
                heading = cog + np.random.uniform(-10, 10)
                
                # Small movements
                lat += np.random.uniform(-0.01, 0.01)
                lon += np.random.uniform(-0.01, 0.01)
                
                # Regular transmission
                time_gap = np.random.uniform(5, 15)  # minutes
            else:
                # Anomalous behavior
                behavior_type = np.random.choice(['speed', 'gap', 'jump'])
                
                if behavior_type == 'speed':
                    # Excessive speed
                    sog = np.random.uniform(25, 40)
                    cog = np.random.uniform(0, 360)
                    heading = cog + np.random.uniform(-5, 5)
                    lat += np.random.uniform(-0.05, 0.05)
                    lon += np.random.uniform(-0.05, 0.05)
                    time_gap = np.random.uniform(5, 15)
                    
                elif behavior_type == 'gap':
                    # AIS blackout
                    sog = np.random.uniform(2, 8)
                    cog = np.random.uniform(0, 360)
                    heading = cog + np.random.uniform(-10, 10)
                    lat += np.random.uniform(-0.02, 0.02)
                    lon += np.random.uniform(-0.02, 0.02)
                    time_gap = np.random.uniform(120, 300)  # Long gap
                    
                else:  # jump
                    # Position jump (spoofing)
                    sog = np.random.uniform(2, 8)
                    cog = np.random.uniform(0, 360)
                    heading = cog + np.random.uniform(-10, 10)
                    lat += np.random.uniform(-0.5, 0.5)  # Large jump
                    lon += np.random.uniform(-0.5, 0.5)
                    time_gap = np.random.uniform(5, 15)
            
            # Keep within bounds
            lat = np.clip(lat, lat_min, lat_max)
            lon = np.clip(lon, lon_min, lon_max)
            heading = heading % 360
            cog = cog % 360
            
            timestamp += timedelta(minutes=time_gap)
            
            data.append({
                'MMSI': mmsi,
                'timestamp': timestamp,
                'lat': lat,
                'lon': lon,
                'SOG': sog,
                'COG': cog,
                'heading': heading
            })
    
    df = pd.DataFrame(data)
    print(f"Generated {len(df)} AIS records for {n_vessels} vessels")
    
    return df

def generate_eez_boundary():
    """Generate simplified Indian EEZ boundary"""
    print("Generating EEZ boundary...")
    
    # Simplified polygon for Indian EEZ
    coordinates = [
        [68.0, 6.0],
        [88.0, 6.0],
        [88.0, 22.0],
        [68.0, 22.0],
        [68.0, 6.0]
    ]
    
    geojson = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {
                "name": "Indian EEZ",
                "country": "India"
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [coordinates]
            }
        }]
    }
    
    return geojson

def main():
    """Generate all sample data"""
    # Create data directory
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate AIS data
    ais_df = generate_sample_ais_data(n_vessels=50, n_points_per_vessel=200)
    ais_path = data_dir / "ais_data.csv"
    ais_df.to_csv(ais_path, index=False)
    print(f"Saved AIS data to {ais_path}")
    
    # Generate EEZ boundary
    eez_geojson = generate_eez_boundary()
    eez_path = data_dir / "indian_eez.geojson"
    with open(eez_path, 'w') as f:
        json.dump(eez_geojson, f, indent=2)
    print(f"Saved EEZ boundary to {eez_path}")
    
    # Generate empty vessel registry
    vessel_registry = pd.DataFrame({
        'MMSI': ais_df['MMSI'].unique(),
        'vessel_name': [f'Vessel_{i}' for i in range(len(ais_df['MMSI'].unique()))],
        'vessel_type': ['Fishing'] * len(ais_df['MMSI'].unique())
    })
    registry_path = data_dir / "vessel_registry.csv"
    vessel_registry.to_csv(registry_path, index=False)
    print(f"Saved vessel registry to {registry_path}")
    
    print("\nSample data generation complete!")
    print(f"Total AIS records: {len(ais_df)}")
    print(f"Total vessels: {ais_df['MMSI'].nunique()}")
    print(f"Time range: {ais_df['timestamp'].min()} to {ais_df['timestamp'].max()}")

if __name__ == "__main__":
    main()
