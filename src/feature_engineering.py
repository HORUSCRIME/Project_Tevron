import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self):
        self.feature_columns = [
            'temperature', 'humidity', 'wind_speed', 'solar_irradiance',
            'panel_temp', 'voltage', 'current', 'power_factor', 'dust_level'
        ]
        self.target_columns = ['energy_output', 'maintenance_needed']

    def create_temporal_features(self, df):
        try:
            df = df.copy()
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['month'] = df['timestamp'].dt.month
            df['day_of_year'] = df['timestamp'].dt.dayofyear
            df['quarter'] = df['timestamp'].dt.quarter

            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

            df['solar_elevation'] = self._calculate_solar_elevation(df)

            df['is_daylight'] = ((df['hour'] >= 6) & (df['hour'] <= 18)).astype(int)
            df['is_peak_hours'] = ((df['hour'] >= 10) & (df['hour'] <= 14)).astype(int)
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

            logger.info("Temporal features created successfully")
            return df

        except Exception as e:
            logger.error(f"Error creating temporal features: {e}")
            return df

    def _calculate_solar_elevation(self, df):
        try:
            day_of_year = df['day_of_year']
            hour = df['hour']

            declination = 23.45 * np.sin(np.radians(360 * (284 + day_of_year) / 365))

            hour_angle = 15 * (hour - 12)

            latitude = 11.0168

            elevation = np.arcsin(
                np.sin(np.radians(declination)) * np.sin(np.radians(latitude)) +
                np.cos(np.radians(declination)) * np.cos(np.radians(latitude)) *
                np.cos(np.radians(hour_angle))
            )

            return np.degrees(elevation)

        except Exception as e:
            logger.error(f"Error calculating solar elevation: {e}")
            return np.zeros(len(df))

    def create_rolling_features(self, df, windows=[3, 6, 12, 24]):
        try:
            df = df.copy()
            df = df.sort_values('timestamp')

            for window in windows:
                for col in self.feature_columns:
                    if col in df.columns:
                        df[f'{col}_rolling_mean_{window}h'] = df[col].rolling(
                            window=window, min_periods=1
                        ).mean()

                        df[f'{col}_rolling_std_{window}h'] = df[col].rolling(
                            window=window, min_periods=1
                        ).std().fillna(0)

            logger.info(f"Rolling features created for windows: {windows}")
            return df

        except Exception as e:
            logger.error(f"Error creating rolling features: {e}")
            return df

    def create_lag_features(self, df, lags=[1, 2, 3, 6, 12, 24]):
        try:
            df = df.copy()
            df = df.sort_values('timestamp')

            for lag in lags:
                for col in self.feature_columns:
                    if col in df.columns:
                        df[f'{col}_lag_{lag}h'] = df[col].shift(lag)

            logger.info(f"Lag features created for lags: {lags}")
            return df

        except Exception as e:
            logger.error(f"Error creating lag features: {e}")
            return df

    def create_derived_features(self, df):
        try:
            df = df.copy()

            if 'panel_temp' in df.columns and 'temperature' in df.columns:
                df['temp_difference'] = df['panel_temp'] - df['temperature']

            if 'voltage' in df.columns and 'current' in df.columns:
                df['power'] = df['voltage'] * df['current']

            if 'power' in df.columns and 'power_factor' in df.columns:
                df['apparent_power'] = np.where(
                    df['power_factor'] > 0,
                    df['power'] / df['power_factor'],
                    0
                )

            if 'temperature' in df.columns and 'humidity' in df.columns:
                df['heat_index'] = self._calculate_heat_index(df['temperature'], df['humidity'])

            if 'wind_speed' in df.columns and 'panel_temp' in df.columns:
                df['wind_cooling'] = df['wind_speed'] / (df['panel_temp'] + 1)

            logger.info("Derived features created successfully")
            return df

        except Exception as e:
            logger.error(f"Error creating derived features: {e}")
            return df

    def _calculate_heat_index(self, temp, humidity):
        try:
            hi = 0.5 * (temp + 61.0 + ((temp - 68.0) * 1.2) + (humidity * 0.094))

            mask = hi >= 80
            if mask.any():
                hi_complex = (-42.379 + 2.04901523 * temp + 10.14333127 * humidity
                             - 0.22475541 * temp * humidity - 0.00683783 * temp**2
                             - 0.05481717 * humidity**2 + 0.00122874 * temp**2 * humidity
                             + 0.00085282 * temp * humidity**2 - 0.00000199 * temp**2 * humidity**2)
                hi = np.where(mask, hi_complex, hi)

            return hi

        except Exception as e:
            logger.error(f"Error calculating heat index: {e}")
            return temp

    def create_maintenance_features(self, df):
        try:
            df = df.copy()

            if 'dust_level' in df.columns:
                df['dust_trend'] = df['dust_level'].rolling(window=12, min_periods=1).apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
                )

            if 'panel_temp' in df.columns:
                df['temp_stress'] = np.where(df['panel_temp'] > 45, 1, 0)
                df['temp_stress_cumulative'] = df['temp_stress'].rolling(window=24, min_periods=1).sum()

            if 'voltage' in df.columns:
                df['voltage_stability'] = df['voltage'].rolling(window=6, min_periods=1).std().fillna(0)

            df['days_since_maintenance'] = np.random.randint(0, 90, len(df))

            logger.info("Maintenance features created successfully")
            return df

        except Exception as e:
            logger.error(f"Error creating maintenance features: {e}")
            return df

    def prepare_features(self, df, include_all=True):
        try:
            logger.info("Starting feature preparation")

            df = self.create_temporal_features(df)

            if include_all:
                df = self.create_rolling_features(df)
                df = self.create_lag_features(df)

            df = self.create_derived_features(df)
            df = self.create_maintenance_features(df)

            df = df.ffill().bfill().fillna(0)

            logger.info(f"Feature preparation completed. Shape: {df.shape}")
            return df

        except Exception as e:
            logger.error(f"Error in feature preparation: {e}")
            return df

    def get_feature_columns(self, df, additional_targets=None):
        if additional_targets is None:
            additional_targets = []

        exclude_columns = self.target_columns + additional_targets + ['timestamp']

        feature_columns = [col for col in df.columns if col not in exclude_columns]
        return feature_columns

def create_features_for_inference(df):
    engineer = FeatureEngineer()
    return engineer.prepare_features(df, include_all=False)

if __name__ == "__main__":
    from data_ingestion import DataIngestion

    ingestion = DataIngestion()
    df = ingestion.load_historical_data()

    engineer = FeatureEngineer()
    df_features = engineer.prepare_features(df)

    print(f"Original shape: {df.shape}")
    print(f"With features shape: {df_features.shape}")

    feature_cols = engineer.get_feature_columns(df_features)
    print(f"Feature columns: {len(feature_cols)}")
    print(f"Target columns excluded: {[col for col in df.columns if col not in feature_cols and col != 'timestamp']}")

