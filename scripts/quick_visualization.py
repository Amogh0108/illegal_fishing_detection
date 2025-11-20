"""Quick visualization of results"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

# Load data
df = pd.read_csv("data/processed/ais_all_features.csv")

# Create figure with subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('IUU Fishing Detection - Data Analysis', fontsize=16, fontweight='bold')

# 1. Vessel Distribution
ax1 = axes[0, 0]
vessel_counts = df.groupby('MMSI').size().sort_values(ascending=False).head(10)
vessel_counts.plot(kind='bar', ax=ax1, color='steelblue')
ax1.set_title('Top 10 Vessels by Record Count')
ax1.set_xlabel('MMSI')
ax1.set_ylabel('Number of Records')
ax1.tick_params(axis='x', rotation=45)

# 2. Speed Distribution
ax2 = axes[0, 1]
df['SOG'].hist(bins=50, ax=ax2, color='coral', edgecolor='black')
ax2.set_title('Speed Over Ground Distribution')
ax2.set_xlabel('Speed (knots)')
ax2.set_ylabel('Frequency')
ax2.axvline(df['SOG'].mean(), color='red', linestyle='--', label=f'Mean: {df["SOG"].mean():.2f}')
ax2.legend()

# 3. Fishing Speed Detection
ax3 = axes[0, 2]
fishing_speed_counts = df['fishing_speed'].value_counts()
ax3.pie(fishing_speed_counts, labels=['Normal Speed', 'Fishing Speed'], autopct='%1.1f%%',
        colors=['lightblue', 'orange'], startangle=90)
ax3.set_title('Fishing Speed Pattern Detection')

# 4. AIS Gap Detection
ax4 = axes[1, 0]
ais_gap_counts = df['ais_gap'].value_counts()
ax4.pie(ais_gap_counts, labels=['Normal', 'AIS Gap'], autopct='%1.1f%%',
        colors=['lightgreen', 'red'], startangle=90)
ax4.set_title('AIS Transmission Gaps')

# 5. Loitering Detection
ax5 = axes[1, 1]
loitering_counts = df['loitering'].value_counts()
ax5.pie(loitering_counts, labels=['Normal', 'Loitering'], autopct='%1.1f%%',
        colors=['skyblue', 'purple'], startangle=90)
ax5.set_title('Loitering Behavior Detection')

# 6. Speed Variance
ax6 = axes[1, 2]
df['speed_variance'].hist(bins=50, ax=ax6, color='green', edgecolor='black', alpha=0.7)
ax6.set_title('Speed Variance Distribution')
ax6.set_xlabel('Speed Variance')
ax6.set_ylabel('Frequency')
ax6.axvline(df['speed_variance'].quantile(0.9), color='red', linestyle='--', 
            label=f'90th percentile: {df["speed_variance"].quantile(0.9):.2f}')
ax6.legend()

plt.tight_layout()
plt.savefig('outputs/data_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Visualization saved to: outputs/data_analysis.png")

# Feature Correlation Heatmap
plt.figure(figsize=(14, 10))
feature_cols = ['speed_mean', 'speed_variance', 'turn_rate', 'loitering', 
                'fishing_speed', 'ais_gap', 'gap_count', 'position_jump']
corr_matrix = df[feature_cols].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('outputs/feature_correlation.png', dpi=300, bbox_inches='tight')
print("✓ Correlation matrix saved to: outputs/feature_correlation.png")

# Summary Statistics
print("\n" + "="*70)
print("DATASET STATISTICS")
print("="*70)
print(f"Total Records: {len(df):,}")
print(f"Unique Vessels: {df['MMSI'].nunique()}")
print(f"Date Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"\nSpeed Statistics:")
print(f"  Mean: {df['SOG'].mean():.2f} knots")
print(f"  Median: {df['SOG'].median():.2f} knots")
print(f"  Max: {df['SOG'].max():.2f} knots")
print(f"\nAnomaly Indicators:")
print(f"  Fishing Speed Pattern: {df['fishing_speed'].sum():,} records ({df['fishing_speed'].sum()/len(df)*100:.1f}%)")
print(f"  AIS Gaps: {df['ais_gap'].sum():,} records ({df['ais_gap'].sum()/len(df)*100:.1f}%)")
print(f"  Loitering: {df['loitering'].sum():,} records ({df['loitering'].sum()/len(df)*100:.1f}%)")
print(f"  Position Jumps: {df['position_jump'].sum():,} records ({df['position_jump'].sum()/len(df)*100:.1f}%)")
print("="*70)
