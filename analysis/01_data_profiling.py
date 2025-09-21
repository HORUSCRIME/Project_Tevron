import pandas as pd
import numpy as np
from ydata_profiling import ProfileReport
import sys
import os
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_ingestion import DataIngestion

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_data_profile():
    """Generate comprehensive data profiling report."""
    try:
        logger.info("Starting data profiling analysis...")

        ingestion = DataIngestion()
        df = ingestion.load_historical_data()

        if df is None or df.empty:
            logger.warning("No data available, generating sample data")
            df = ingestion._generate_sample_data(days=365)

        logger.info(f"Loaded data with shape: {df.shape}")

        print("Dataset Overview:")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

        logger.info("Generating profiling report...")

        profile = ProfileReport(
            df,
            title="Solar Panel Data Profiling Report",
            explorative=True,
            config_file=None
        )

        profile.config.html.minify_html = False
        profile.config.html.use_local_assets = True

        output_file = "data_profiling_report.html"
        profile.to_file(output_file)

        logger.info(f"Profiling report saved as: {output_file}")

        print("\nSummary Statistics:")
        print(df.describe())

        print("\nData Quality Assessment:")

        missing_values = df.isnull().sum()
        missing_percent = (missing_values / len(df)) * 100

        print("\nMissing Values:")
        for col in df.columns:
            if missing_values[col] > 0:
                print(f"  {col}: {missing_values[col]} ({missing_percent[col]:.2f}%)")

        duplicates = df.duplicated().sum()
        print(f"\nDuplicate rows: {duplicates} ({duplicates/len(df)*100:.2f}%)")

        print("\nData Types:")
        for col, dtype in df.dtypes.items():
            print(f"  {col}: {dtype}")

        print("\nOutlier Detection (IQR method):")
        numeric_columns = df.select_dtypes(include=[np.number]).columns

        for col in numeric_columns:
            if col != 'maintenance_needed': 
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                outlier_count = len(outliers)
                outlier_percent = (outlier_count / len(df)) * 100

                if outlier_count > 0:
                    print(f"  {col}: {outlier_count} outliers ({outlier_percent:.2f}%)")

        print("\nHigh Correlations (|r| > 0.7):")
        corr_matrix = df[numeric_columns].corr()

        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    high_corr_pairs.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_val
                    ))

        for col1, col2, corr_val in high_corr_pairs:
            print(f"  {col1} - {col2}: {corr_val:.3f}")

        print("\nTime Series Characteristics:")
        df_sorted = df.sort_values('timestamp')
        time_diff = df_sorted['timestamp'].diff().dropna()

        print(f"  Time frequency: {time_diff.mode().iloc[0] if not time_diff.empty else 'Unknown'}")
        print(f"  Regular intervals: {time_diff.nunique() == 1}")
        print(f"  Total time span: {df['timestamp'].max() - df['timestamp'].min()}")

        print("\nEnergy Production Patterns:")
        df['hour'] = df['timestamp'].dt.hour
        hourly_energy = df.groupby('hour')['energy_output'].mean()

        peak_hour = hourly_energy.idxmax()
        peak_energy = hourly_energy.max()

        print(f"  Peak production hour: {peak_hour}:00")
        print(f"  Peak production: {peak_energy:.2f} kW")
        print(f"  Daily total (avg): {df.groupby(df['timestamp'].dt.date)['energy_output'].sum().mean():.2f} kWh")

        maintenance_rate = df['maintenance_needed'].mean()
        print(f"\nMaintenance Patterns:")
        print(f"  Overall maintenance rate: {maintenance_rate:.3f} ({maintenance_rate*100:.1f}%)")

        if maintenance_rate > 0:
            maintenance_events = df[df['maintenance_needed'] == 1]['timestamp']
            if len(maintenance_events) > 1:
                time_between = maintenance_events.diff().dropna()
                avg_time_between = time_between.mean()
                print(f"  Average time between maintenance: {avg_time_between}")

        logger.info("Data profiling analysis completed successfully")
        return True

    except Exception as e:
        logger.error(f"Error in data profiling: {e}")
        return False

def analyze_seasonal_patterns():
    """Analyze seasonal and temporal patterns in the data."""
    try:
        logger.info("Analyzing seasonal patterns...")

        ingestion = DataIngestion()
        df = ingestion.load_historical_data()

        if df is None or df.empty:
            df = ingestion._generate_sample_data(days=365)

        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['day_of_year'] = df['timestamp'].dt.dayofyear

        print("Seasonal Pattern Analysis:")
        print("=" * 50)

        monthly_stats = df.groupby('month').agg({
            'energy_output': ['mean', 'sum', 'std'],
            'solar_irradiance': 'mean',
            'temperature': 'mean',
            'maintenance_needed': 'sum'
        }).round(2)

        print("\nMonthly Patterns:")
        print(monthly_stats)

        daily_stats = df.groupby('day_of_week').agg({
            'energy_output': ['mean', 'sum'],
            'maintenance_needed': 'sum'
        }).round(2)

        print("\nDaily Patterns (0=Monday, 6=Sunday):")
        print(daily_stats)

        hourly_stats = df.groupby('hour').agg({
            'energy_output': 'mean',
            'solar_irradiance': 'mean',
            'panel_temp': 'mean'
        }).round(2)

        print("\nHourly Patterns:")
        print(hourly_stats)

        logger.info("Seasonal pattern analysis completed")

    except Exception as e:
        logger.error(f"Error in seasonal analysis: {e}")

def main():
    """Main function to run all profiling analyses."""
    print("Solar Panel Data Profiling Analysis")
    print("=" * 50)

    success = generate_data_profile()

    if success:
        print("\n" + "=" * 50)

        analyze_seasonal_patterns()

        print("\n" + "=" * 50)
        print("Analysis completed successfully!")
        print("Check 'data_profiling_report.html' for detailed report")
    else:
        print("Analysis failed. Check logs for details.")

if __name__ == "__main__":
    main()

