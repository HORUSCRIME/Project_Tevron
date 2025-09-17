import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import matplotlib.pyplot as plt

class SolarDataGenerator:
    def __init__(self, seed=42):
        np.random.seed(seed)
        random.seed(seed)
        
    def generate_weather_data(self, n_samples=1000, days=30):
        """Generate realistic weather patterns"""
        timestamps = [datetime.now() - timedelta(days=days) + timedelta(hours=i*24//(n_samples//days)) 
                     for i in range(n_samples)]
        
        hours = [(ts.hour + ts.minute/60) for ts in timestamps]
        base_irradiance = np.array([max(0, 1000 * np.sin(np.pi * (h - 6) / 12)) if 6 <= h <= 18 else 0 
                                   for h in hours])
        
        weather_factor = np.random.normal(1, 0.2, n_samples)
        weather_factor = np.clip(weather_factor, 0.1, 1.2)
        
        solar_irradiance = base_irradiance * weather_factor
        solar_irradiance = np.clip(solar_irradiance, 0, 1200)
        
        base_temp = 20 + 10 * np.sin(np.pi * np.array(hours) / 12)
        temperature = base_temp + np.random.normal(0, 3, n_samples)
        
        humidity = 80 - temperature + np.random.normal(0, 10, n_samples)
        humidity = np.clip(humidity, 20, 95)
        
        wind_speed = np.random.exponential(2, n_samples)
        wind_speed = np.clip(wind_speed, 0, 15)
        
        return {
            'timestamp': timestamps,
            'solar_irradiance': solar_irradiance,
            'temperature': temperature,
            'humidity': humidity,
            'wind_speed': wind_speed
        }
    
    def generate_panel_electrical_data(self, weather_data):
        """Generate panel electrical characteristics based on weather"""
        n_samples = len(weather_data['solar_irradiance'])
        
        rated_voltage = 24
        rated_current = 8.33
        rated_power = 200
        
        temp_coeff_v = -0.0023 
        panel_voltage = rated_voltage * (1 + temp_coeff_v * (weather_data['temperature'] - 25))
        
        panel_current = rated_current * (weather_data['solar_irradiance'] / 1000)
        
        panel_temp = weather_data['temperature'] + weather_data['solar_irradiance'] * 0.03
        
        power_output = panel_voltage * panel_current
        
        panel_voltage += np.random.normal(0, 0.5, n_samples)
        panel_current += np.random.normal(0, 0.2, n_samples)
        power_output = np.clip(power_output + np.random.normal(0, 5, n_samples), 0, 250)
        
        return {
            'panel_voltage': np.clip(panel_voltage, 0, 30),
            'panel_current': np.clip(panel_current, 0, 10),
            'power_output': power_output,
            'panel_temp': panel_temp
        }
    
    def generate_maintenance_data(self, n_samples):
        """Generate maintenance-related features"""
        dust_level = []
        hours_since_cleaning = []
        days_since_maintenance = []
        
        current_dust = 0
        current_hours = 0
        current_days = 0
        
        for i in range(n_samples):
            dust_increase = np.random.normal(0.02, 0.01)  
            current_dust = min(1.0, current_dust + dust_increase)
            
            if np.random.random() < 0.15:  
                current_dust = np.random.uniform(0, 0.1)  
                current_hours = 0
            else:
                current_hours += np.random.uniform(6, 18)  
            if np.random.random() < 0.02:  
                current_days = 0
            else:
                current_days += np.random.uniform(0.5, 1.5)
            
            dust_level.append(current_dust)
            hours_since_cleaning.append(current_hours)
            days_since_maintenance.append(current_days)
        
        return {
            'dust_level': np.array(dust_level),
            'hours_since_cleaning': np.array(hours_since_cleaning),
            'days_since_maintenance': np.array(days_since_maintenance)
        }
    
    def calculate_efficiency_and_maintenance(self, data):
        """Calculate efficiency and maintenance targets"""
        n_samples = len(data['solar_irradiance'])
        
        theoretical_power = 200 * (data['solar_irradiance'] / 1000)
        base_efficiency = data['power_output'] / (theoretical_power + 1e-6)
        
        dust_factor = 1 - (data['dust_level'] * 0.3)  
        maintenance_factor = 1 - np.clip(data['days_since_maintenance'] / 365, 0, 0.2)  
        
        efficiency = base_efficiency * dust_factor * maintenance_factor
        efficiency = np.clip(efficiency, 0.3, 1.2) 
        
        days_until_maintenance = []
        for i in range(n_samples):
            if efficiency[i] < 0.7:
                days = max(1, np.random.uniform(1, 5))  
            elif efficiency[i] < 0.8:
                days = np.random.uniform(5, 15)  
            else:
                days = np.random.uniform(15, 60)  
            
            days_until_maintenance.append(days)
        
        return {
            'efficiency': efficiency,
            'days_until_maintenance': np.array(days_until_maintenance)
        }
    
    def add_anomalies(self, data, anomaly_rate=0.05):
        """Add realistic anomalies to the dataset"""
        n_samples = len(data['solar_irradiance'])
        n_anomalies = int(n_samples * anomaly_rate)
        anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
        
        data_copy = data.copy()
        
        for idx in anomaly_indices:
            anomaly_type = np.random.choice(['voltage_drop', 'current_spike', 'power_loss', 'sensor_error'])
            
            if anomaly_type == 'voltage_drop':
                data_copy['panel_voltage'][idx] *= 0.5 
            elif anomaly_type == 'current_spike':
                data_copy['panel_current'][idx] *= 2.0  
            elif anomaly_type == 'power_loss':
                data_copy['power_output'][idx] *= 0.3  
            elif anomaly_type == 'sensor_error':
                data_copy['panel_temp'][idx] += 50  
        
        return data_copy
    
    def generate_dataset(self, n_samples=2000, days=60, save_path='solar_training_data.csv'):
        """Generate complete synthetic dataset"""
        print(f"Generating {n_samples} samples over {days} days...")
        
        weather = self.generate_weather_data(n_samples, days)
        electrical = self.generate_panel_electrical_data(weather)
        maintenance = self.generate_maintenance_data(n_samples)
        
        combined_data = {**weather, **electrical, **maintenance}
        
        targets = self.calculate_efficiency_and_maintenance(combined_data)
        combined_data.update(targets)
        
        combined_data = self.add_anomalies(combined_data)
        
        df = pd.DataFrame(combined_data)
        
        df.to_csv(save_path, index=False)
        print(f"Dataset saved to {save_path}")
        
        print("\nDataset Summary:")
        print(f"Total samples: {len(df)}")
        print(f"Efficiency range: {df['efficiency'].min():.2f} - {df['efficiency'].max():.2f}")
        print(f"Average power output: {df['power_output'].mean():.1f}W")
        print(f"Anomalies (efficiency < 0.5): {(df['efficiency'] < 0.5).sum()}")
        
        return df
    
    def plot_sample_data(self, df, n_points=200):
        """Plot sample data for visualization"""
        sample_df = df.head(n_points)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        axes[0,0].plot(sample_df.index, sample_df['solar_irradiance'], label='Irradiance')
        axes[0,0].set_ylabel('Solar Irradiance (W/m²)')
        axes[0,0].set_title('Solar Irradiance Pattern')
        
        ax2 = axes[0,0].twinx()
        ax2.plot(sample_df.index, sample_df['power_output'], 'r-', label='Power')
        ax2.set_ylabel('Power Output (W)')
        
        axes[0,1].plot(sample_df.index, sample_df['temperature'], 'g-', label='Ambient')
        axes[0,1].plot(sample_df.index, sample_df['panel_temp'], 'r-', label='Panel')
        axes[0,1].set_ylabel('Temperature (°C)')
        axes[0,1].set_title('Temperature')
        axes[0,1].legend()
        
        axes[1,0].plot(sample_df.index, sample_df['efficiency'])
        axes[1,0].set_ylabel('Efficiency')
        axes[1,0].set_title('Panel Efficiency')
        axes[1,0].set_xlabel('Sample Index')
        
        axes[1,1].plot(sample_df.index, sample_df['dust_level'], label='Dust Level')
        axes[1,1].plot(sample_df.index, sample_df['days_since_maintenance']/100, label='Days Since Maintenance/100')
        axes[1,1].set_ylabel('Level')
        axes[1,1].set_title('Maintenance Factors')
        axes[1,1].set_xlabel('Sample Index')
        axes[1,1].legend()
        
        plt.tight_layout()
        plt.savefig('sample_data_visualization.png', dpi=150, bbox_inches='tight')
        plt.show()

def main():
    """Generate training data and train the model"""
    print("Solar Panel Synthetic Data Generator")
    print("=" * 50)
    
    generator = SolarDataGenerator()
    df = generator.generate_dataset(n_samples=3000, days=90)
    
    test_df = generator.generate_dataset(n_samples=500, days=15, save_path='solar_test_data.csv')
    
    try:
        generator.plot_sample_data(df)
    except ImportError:
        print("matplotlib not available - skipping visualization")
    
    print("\nTraining ML models...")
    from solar_ml_model import SolarPanelML
    
    solar_ml = SolarPanelML()
    
    solar_ml.train_efficiency_model(df)
    solar_ml.train_maintenance_model(df) 
    solar_ml.train_anomaly_detector(df)
    
    solar_ml.save_models()
    
    print("\nTraining complete! Models saved.")
    print("You can now run the main model on real sensor data.")

if __name__ == "__main__":
    main()



