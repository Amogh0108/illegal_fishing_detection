"""Quick launcher for IUU Fishing Detection Dashboard"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.dashboard.app import app, main

if __name__ == '__main__':
    print("=" * 70)
    print("🚀 Launching IUU Fishing Detection Dashboard")
    print("=" * 70)
    print()
    print("📊 Dashboard Features:")
    print("   • Real-time vessel tracking")
    print("   • Interactive anomaly detection")
    print("   • Model performance comparison")
    print("   • Detailed anomaly reports")
    print()
    print("🌐 Access the dashboard at:")
    print("   http://localhost:8050")
    print()
    print("💡 Tips:")
    print("   • Adjust threshold slider for sensitivity")
    print("   • Select specific vessels for detailed view")
    print("   • Click refresh to update data")
    print()
    print("=" * 70)
    print("Starting server...")
    print("=" * 70)
    print()
    
    main()
