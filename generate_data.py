
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_comprehensive_data():
    print("Generating comprehensive historical data...")

    np.random.seed(42)

    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 2, 15)

    timestamps = []
    current_time = start_date
    while len(timestamps) < 10000 and current_time <= end_date:
        timestamps.append(current_time)
        current_time += timedelta(hours=1)

    timestamps = timestamps[:10000]

    data = []

    for i, ts in enumerate(timestamps):
        hour = ts.hour
        day_of_year = ts.timetuple().tm_yday
        month = ts.month

        seasonal_temp_factor = 8 * np.sin((day_of_year - 80) * 2 * np.pi / 365)
        daily_temp_factor = 6 * np.sin((hour - 6) * np.pi / 12) if 6 <= hour <= 18 else -2
        base_temp = 28 + seasonal_temp_factor + daily_temp_factor + np.random.normal(0, 2)

        if 6 <= hour <= 18:
            seasonal_solar = 0.7 + 0.5 * np.sin((day_of_year - 80) * 2 * np.pi / 365)
            daily_solar = np.sin((hour - 6) * np.pi / 12)
            weather_factor = np.random.uniform(0.6, 1.0)

            solar_irradiance = seasonal_solar * daily_solar * 1200 * weather_factor
            solar_irradiance = max(0, solar_irradiance + np.random.normal(0, 50))
        else:
            solar_irradiance = 0

        panel_temp = base_temp + (solar_irradiance * 0.02) + np.random.normal(0, 1.5)

        base_humidity = 70 - (base_temp - 25) * 1.5
        if hour < 6 or hour > 20:
            base_humidity += 10
        humidity = max(20, min(95, base_humidity + np.random.normal(0, 8)))

        base_wind = 2.5 + (0.5 if 10 <= hour <= 16 else 0)
        wind_speed = max(0, base_wind + np.random.normal(0, 1.2))

        base_efficiency = 0.18

        temp_efficiency_loss = max(0, (panel_temp - 25) * 0.004)
        dust_level = np.random.uniform(0.05, 0.9)
        dust_efficiency_loss = dust_level * 0.15

        actual_efficiency = base_efficiency - temp_efficiency_loss - dust_efficiency_loss
        actual_efficiency = max(0.05, actual_efficiency)

        energy_output = (solar_irradiance / 1000) * actual_efficiency * 100
        energy_output = max(0, energy_output + np.random.normal(0, 0.3))

        voltage = 24.0 + np.random.normal(0, 0.8) - (panel_temp - 25) * 0.02
        voltage = max(20, min(28, voltage))

        current = energy_output * 1000 / voltage if voltage > 0 else 0
        current = max(0, current + np.random.normal(0, 2))

        power_factor = np.random.uniform(0.92, 0.99)

        maintenance_prob = 0.02

        if dust_level > 0.7:
            maintenance_prob += 0.03
        if panel_temp > 45:
            maintenance_prob += 0.02
        if energy_output < 2 and solar_irradiance > 500:
            maintenance_prob += 0.05

        if month in [4, 10]:
            maintenance_prob += 0.02

        maintenance_needed = 1 if np.random.random() < maintenance_prob else 0

        if maintenance_needed:
            energy_output *= np.random.uniform(0.3, 0.8)
            voltage *= np.random.uniform(0.9, 0.98)
            current *= np.random.uniform(0.85, 0.95)
            power_factor *= np.random.uniform(0.85, 0.95)

        data.append({
            'timestamp': ts,
            'energy_output': round(energy_output, 2),
            'temperature': round(base_temp, 1),
            'humidity': round(humidity, 1),
            'wind_speed': round(wind_speed, 1),
            'solar_irradiance': round(solar_irradiance, 1),
            'panel_temp': round(panel_temp, 1),
            'voltage': round(voltage, 1),
            'current': round(current, 1),
            'power_factor': round(power_factor, 2),
            'dust_level': round(dust_level, 2),
            'maintenance_needed': maintenance_needed
        })

        if (i + 1) % 1000 == 0:
            print(f"Generated {i + 1} records...")

    df = pd.DataFrame(data)

    missing_indices = np.random.choice(df.index, size=int(0.01 * len(df)), replace=False)
    missing_columns = ['humidity', 'wind_speed', 'dust_level']
    for idx in missing_indices:
        col = np.random.choice(missing_columns)
        df.loc[idx, col] = np.nan

    outlier_indices = np.random.choice(df.index, size=int(0.005 * len(df)), replace=False)
    for idx in outlier_indices:
        if np.random.random() < 0.5:
            df.loc[idx, 'energy_output'] *= np.random.uniform(2, 3)
        else:
            df.loc[idx, 'panel_temp'] += np.random.uniform(15, 25)

    os.makedirs('data', exist_ok=True)
    df.to_csv('data/historical_data.csv', index=False)

    print(f"\nGenerated {len(df)} records successfully!")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Maintenance events: {df['maintenance_needed'].sum()} ({df['maintenance_needed'].mean()*100:.1f}%)")
    print(f"Average daily energy: {df.groupby(df['timestamp'].dt.date)['energy_output'].sum().mean():.1f} kWh")
    print(f"Data saved to: data/historical_data.csv")

    print("\nSample Statistics:")
    print(df.describe())

    return df

if __name__ == "__main__":
    generate_comprehensive_data()

