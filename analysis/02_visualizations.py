import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import sys
import os
import logging
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_ingestion import DataIngestion

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

plt.style.use('dark_background')
sns.set_palette("husl")

def setup_plots():
    """Setup plot configurations."""
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10

def create_distribution_plots(df):
    """Create distribution plots for key variables."""
    try:
        logger.info("Creating distribution plots...")

        numeric_cols = ['energy_output', 'temperature', 'humidity', 'wind_speed',
                       'solar_irradiance', 'panel_temp', 'voltage', 'current', 'dust_level']

        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle('Distribution of Key Variables', fontsize=16, y=0.98)

        for i, col in enumerate(numeric_cols):
            row = i // 3
            col_idx = i % 3

            if col in df.columns:
                sns.histplot(data=df, x=col, kde=True, ax=axes[row, col_idx])
                axes[row, col_idx].set_title(f'Distribution of {col}')
                axes[row, col_idx].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('analysis/distribution_plots.png', dpi=300, bbox_inches='tight')
        plt.show()

        logger.info("Distribution plots created successfully")

    except Exception as e:
        logger.error(f"Error creating distribution plots: {e}")

def create_time_series_plots(df):
    """Create time series visualizations."""
    try:
        logger.info("Creating time series plots...")

        df_sorted = df.sort_values('timestamp')

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Time Series Analysis', fontsize=16, y=0.98)

        axes[0, 0].plot(df_sorted['timestamp'], df_sorted['energy_output'],
                       color='gold', alpha=0.7, linewidth=0.5)
        axes[0, 0].set_title('Energy Output Over Time')
        axes[0, 0].set_ylabel('Energy Output (kW)')
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(df_sorted['timestamp'], df_sorted['solar_irradiance'],
                       color='orange', alpha=0.7, linewidth=0.5)
        axes[0, 1].set_title('Solar Irradiance Over Time')
        axes[0, 1].set_ylabel('Solar Irradiance (W/m²)')
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].plot(df_sorted['timestamp'], df_sorted['temperature'],
                       label='Ambient Temp', color='blue', alpha=0.7, linewidth=0.5)
        axes[1, 0].plot(df_sorted['timestamp'], df_sorted['panel_temp'],
                       label='Panel Temp', color='red', alpha=0.7, linewidth=0.5)
        axes[1, 0].set_title('Temperature Comparison')
        axes[1, 0].set_ylabel('Temperature (°C)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        maintenance_events = df_sorted[df_sorted['maintenance_needed'] == 1]
        axes[1, 1].scatter(maintenance_events['timestamp'],
                          maintenance_events['energy_output'],
                          color='red', alpha=0.8, s=50, label='Maintenance Events')
        axes[1, 1].plot(df_sorted['timestamp'], df_sorted['energy_output'],
                       color='gold', alpha=0.3, linewidth=0.5)
        axes[1, 1].set_title('Maintenance Events vs Energy Output')
        axes[1, 1].set_ylabel('Energy Output (kW)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('analysis/time_series_plots.png', dpi=300, bbox_inches='tight')
        plt.show()

        logger.info("Time series plots created successfully")

    except Exception as e:
        logger.error(f"Error creating time series plots: {e}")

def create_correlation_heatmap(df):
    """Create correlation heatmap."""
    try:
        logger.info("Creating correlation heatmap...")

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlation_matrix = df[numeric_cols].corr()

        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

        sns.heatmap(correlation_matrix,
                   mask=mask,
                   annot=True,
                   cmap='RdBu_r',
                   center=0,
                   square=True,
                   fmt='.2f',
                   cbar_kws={"shrink": .8})

        plt.title('Feature Correlation Matrix', fontsize=16, pad=20)
        plt.tight_layout()
        plt.savefig('analysis/correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()

        logger.info("Correlation heatmap created successfully")

    except Exception as e:
        logger.error(f"Error creating correlation heatmap: {e}")

def create_seasonal_patterns(df):
    """Create seasonal pattern visualizations."""
    try:
        logger.info("Creating seasonal pattern plots...")

        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Seasonal and Temporal Patterns', fontsize=16, y=0.98)

        hourly_energy = df.groupby('hour')['energy_output'].mean()
        axes[0, 0].plot(hourly_energy.index, hourly_energy.values,
                       marker='o', color='gold', linewidth=2, markersize=6)
        axes[0, 0].set_title('Average Energy Output by Hour')
        axes[0, 0].set_xlabel('Hour of Day')
        axes[0, 0].set_ylabel('Energy Output (kW)')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_xticks(range(0, 24, 2))

        monthly_energy = df.groupby('month')['energy_output'].mean()
        axes[0, 1].bar(monthly_energy.index, monthly_energy.values,
                      color='orange', alpha=0.8)
        axes[0, 1].set_title('Average Energy Output by Month')
        axes[0, 1].set_xlabel('Month')
        axes[0, 1].set_ylabel('Energy Output (kW)')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_xticks(range(1, 13))

        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        daily_energy = df.groupby('day_of_week')['energy_output'].mean()
        axes[1, 0].bar(range(7), daily_energy.values, color='lightblue', alpha=0.8)
        axes[1, 0].set_title('Average Energy Output by Day of Week')
        axes[1, 0].set_xlabel('Day of Week')
        axes[1, 0].set_ylabel('Energy Output (kW)')
        axes[1, 0].set_xticks(range(7))
        axes[1, 0].set_xticklabels(day_names)
        axes[1, 0].grid(True, alpha=0.3)

        pivot_data = df.pivot_table(values='energy_output',
                                   index='hour',
                                   columns='month',
                                   aggfunc='mean')

        sns.heatmap(pivot_data,
                   cmap='YlOrRd',
                   cbar_kws={'label': 'Energy Output (kW)'},
                   ax=axes[1, 1])
        axes[1, 1].set_title('Energy Output Heatmap: Hour vs Month')
        axes[1, 1].set_xlabel('Month')
        axes[1, 1].set_ylabel('Hour of Day')

        plt.tight_layout()
        plt.savefig('analysis/seasonal_patterns.png', dpi=300, bbox_inches='tight')
        plt.show()

        logger.info("Seasonal pattern plots created successfully")

    except Exception as e:
        logger.error(f"Error creating seasonal patterns: {e}")

def create_performance_analysis(df):
    """Create performance analysis visualizations."""
    try:
        logger.info("Creating performance analysis plots...")

        df['efficiency'] = np.where(df['solar_irradiance'] > 0,
                                   df['energy_output'] / (df['solar_irradiance'] / 1000),
                                   0)

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Performance Analysis', fontsize=16, y=0.98)

        axes[0, 0].scatter(df['solar_irradiance'], df['energy_output'],
                          alpha=0.6, s=20, color='gold')
        axes[0, 0].set_title('Energy Output vs Solar Irradiance')
        axes[0, 0].set_xlabel('Solar Irradiance (W/m²)')
        axes[0, 0].set_ylabel('Energy Output (kW)')
        axes[0, 0].grid(True, alpha=0.3)

        sns.histplot(data=df[df['efficiency'] > 0], x='efficiency',
                    kde=True, ax=axes[0, 1], color='lightgreen')
        axes[0, 1].set_title('Efficiency Distribution')
        axes[0, 1].set_xlabel('Efficiency')
        axes[0, 1].grid(True, alpha=0.3)

        temp_bins = pd.cut(df['panel_temp'], bins=5)
        efficiency_by_temp = df.groupby(temp_bins)['efficiency'].mean()

        axes[1, 0].bar(range(len(efficiency_by_temp)), efficiency_by_temp.values,
                      color='red', alpha=0.7)
        axes[1, 0].set_title('Efficiency vs Panel Temperature')
        axes[1, 0].set_xlabel('Panel Temperature Range')
        axes[1, 0].set_ylabel('Average Efficiency')
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].scatter(df['dust_level'], df['efficiency'],
                          alpha=0.6, s=20, color='brown')
        axes[1, 1].set_title('Efficiency vs Dust Level')
        axes[1, 1].set_xlabel('Dust Level')
        axes[1, 1].set_ylabel('Efficiency')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('analysis/performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

        logger.info("Performance analysis plots created successfully")

    except Exception as e:
        logger.error(f"Error creating performance analysis: {e}")

def create_maintenance_analysis(df):
    """Create maintenance-related visualizations."""
    try:
        logger.info("Creating maintenance analysis plots...")

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Maintenance Analysis', fontsize=16, y=0.98)

        maintenance_events = df[df['maintenance_needed'] == 1]

        if not maintenance_events.empty:
            monthly_maintenance = maintenance_events.groupby(
                maintenance_events['timestamp'].dt.month
            ).size()

            axes[0, 0].bar(monthly_maintenance.index, monthly_maintenance.values,
                          color='red', alpha=0.7)
            axes[0, 0].set_title('Maintenance Events by Month')
            axes[0, 0].set_xlabel('Month')
            axes[0, 0].set_ylabel('Number of Events')
            axes[0, 0].grid(True, alpha=0.3)

        maintenance_comparison = []
        for col in ['energy_output', 'panel_temp', 'dust_level', 'efficiency']:
            if col in df.columns:
                maintenance_comparison.append(col)

        if maintenance_comparison:
            df_melted = df[maintenance_comparison + ['maintenance_needed']].melt(
                id_vars=['maintenance_needed'],
                var_name='metric',
                value_name='value'
            )

            sns.boxplot(data=df_melted, x='metric', y='value',
                       hue='maintenance_needed', ax=axes[0, 1])
            axes[0, 1].set_title('Metrics: Maintenance vs Normal Operation')
            axes[0, 1].tick_params(axis='x', rotation=45)

        if not maintenance_events.empty:
            axes[1, 0].hist([df[df['maintenance_needed'] == 0]['dust_level'],
                           df[df['maintenance_needed'] == 1]['dust_level']],
                          bins=20, alpha=0.7, label=['Normal', 'Maintenance'],
                          color=['green', 'red'])
            axes[1, 0].set_title('Dust Level Distribution')
            axes[1, 0].set_xlabel('Dust Level')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

        if not maintenance_events.empty:
            axes[1, 1].hist([df[df['maintenance_needed'] == 0]['panel_temp'],
                           df[df['maintenance_needed'] == 1]['panel_temp']],
                          bins=20, alpha=0.7, label=['Normal', 'Maintenance'],
                          color=['blue', 'red'])
            axes[1, 1].set_title('Panel Temperature Distribution')
            axes[1, 1].set_xlabel('Panel Temperature (°C)')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('analysis/maintenance_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

        logger.info("Maintenance analysis plots created successfully")

    except Exception as e:
        logger.error(f"Error creating maintenance analysis: {e}")

def main():
    """Main function to run all visualization analyses."""
    print("Solar Panel Data Visualization Analysis")
    print("=" * 50)

    try:
        setup_plots()

        os.makedirs('analysis', exist_ok=True)

        ingestion = DataIngestion()
        df = ingestion.load_historical_data()

        if df is None or df.empty:
            logger.warning("No data available, generating sample data")
            df = ingestion._generate_sample_data(days=365)

        logger.info(f"Loaded data with shape: {df.shape}")

        create_distribution_plots(df)
        create_time_series_plots(df)
        create_correlation_heatmap(df)
        create_seasonal_patterns(df)
        create_performance_analysis(df)
        create_maintenance_analysis(df)

        print("\n" + "=" * 50)
        print("Visualization analysis completed successfully!")
        print("Check 'analysis/' directory for generated plots")

    except Exception as e:
        logger.error(f"Error in visualization analysis: {e}")
        print("Analysis failed. Check logs for details.")

if __name__ == "__main__":
    main()
