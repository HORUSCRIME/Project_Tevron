import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import configparser
import firebase_admin
from firebase_admin import credentials, db
import os
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataIngestion:
    def __init__(self, config_path='config.ini'):
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        self.firebase_initialized = False
        self._initialize_firebase()

    def _initialize_firebase(self):
        try:
            if not firebase_admin._apps and os.path.exists('serviceAccountKey.json'):
                with open('serviceAccountKey.json', 'r') as f:
                    key_content = f.read()
                    if 'YOUR_PRIVATE_KEY_HERE' not in key_content:
                        cred = credentials.Certificate('serviceAccountKey.json')
                        firebase_admin.initialize_app(cred, {
                            'databaseURL': self.config.get('FIREBASE', 'database_url', fallback='')
                        })
                        self.firebase_initialized = True
                        logger.info("Firebase initialized successfully")
                        return
            
            logger.warning("Firebase not configured - using mock data mode")
            self.firebase_initialized = False
        except Exception as e:
            logger.warning(f"Firebase initialization failed, using mock data: {e}")
            self.firebase_initialized = False

    def get_weather_data(self):
        try:
            api_key = self.config.get('API', 'openweather_api_key', fallback='')
            lat = self.config.get('LOCATION', 'latitude', fallback='11.0168')
            lon = self.config.get('LOCATION', 'longitude', fallback='76.9558')

            if not api_key or api_key == 'your_openweather_api_key_here':
                logger.warning("OpenWeatherMap API key not configured, using mock data")
                return self._get_mock_weather_data()

            url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            data = response.json()
            weather_data = {
                'temperature': data['main']['temp'],
                'humidity': data['main']['humidity'],
                'wind_speed': data['wind']['speed'],
                'timestamp': datetime.now()
            }

            logger.info("Weather data fetched successfully")
            return weather_data

        except Exception as e:
            logger.error(f"Error fetching weather data: {e}")
            return self._get_mock_weather_data()

    def _get_mock_weather_data(self):
        current_hour = datetime.now().hour
        base_temp = 25 + 8 * np.sin((current_hour - 6) * np.pi / 12)

        return {
            'temperature': round(base_temp + np.random.normal(0, 2), 1),
            'humidity': round(60 + 20 * np.sin((current_hour - 12) * np.pi / 12) + np.random.normal(0, 5), 1),
            'wind_speed': round(2.5 + np.random.normal(0, 0.5), 1),
            'timestamp': datetime.now()
        }

    def get_sensor_data(self):
        try:
            if not self.firebase_initialized:
                logger.warning("Firebase not initialized, using mock sensor data")
                return self._get_mock_sensor_data()

            ref = db.reference('sensors/solar_panel')
            data = ref.get()

            if data:
                logger.info("Sensor data fetched from Firebase")
                return data
            else:
                logger.warning("No sensor data in Firebase, using mock data")
                return self._get_mock_sensor_data()

        except Exception as e:
            logger.error(f"Error fetching sensor data: {e}")
            return self._get_mock_sensor_data()

    def _get_mock_sensor_data(self):
        current_hour = datetime.now().hour

        if 6 <= current_hour <= 18:
            irradiance = 1000 * np.sin((current_hour - 6) * np.pi / 12) + np.random.normal(0, 50)
            irradiance = max(0, irradiance)
        else:
            irradiance = 0

        energy_output = irradiance * 0.01 + np.random.normal(0, 0.5)
        energy_output = max(0, energy_output)

        return {
            'energy_output': round(energy_output, 2),
            'solar_irradiance': round(irradiance, 1),
            'panel_temp': round(25 + irradiance * 0.015 + np.random.normal(0, 2), 1),
            'voltage': round(24 + np.random.normal(0, 0.5), 1),
            'current': round(energy_output * 4 + np.random.normal(0, 1), 1),
            'power_factor': round(0.95 + np.random.normal(0, 0.02), 2),
            'dust_level': round(np.random.uniform(0.1, 0.8), 2),
            'timestamp': datetime.now()
        }

    def save_sensor_data(self, data):
        try:
            if not self.firebase_initialized:
                logger.warning("Firebase not initialized, cannot save data")
                return False

            ref = db.reference('sensors/solar_panel')
            ref.push(data)
            logger.info("Sensor data saved to Firebase")
            return True

        except Exception as e:
            logger.error(f"Error saving sensor data: {e}")
            return False

    def get_combined_data(self):
        try:
            weather_data = self.get_weather_data()
            sensor_data = self.get_sensor_data()

            combined_data = {**weather_data, **sensor_data}
            combined_data['timestamp'] = datetime.now()

            return combined_data

        except Exception as e:
            logger.error(f"Error getting combined data: {e}")
            return None

    def load_historical_data(self, file_path='data/historical_data.csv'):
        try:
            df = pd.read_csv(file_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            logger.info(f"Loaded {len(df)} historical records")
            return df

        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
            return self._generate_sample_data()

    def _generate_sample_data(self, days=365):
        logger.info("Generating sample historical data")

        start_date = datetime.now() - timedelta(days=days)
        timestamps = pd.date_range(start=start_date, periods=days*24, freq='H')

        data = []
        for ts in timestamps:
            hour = ts.hour
            day_of_year = ts.timetuple().tm_yday

            if 6 <= hour <= 18:
                seasonal_factor = 0.8 + 0.4 * np.sin((day_of_year - 80) * 2 * np.pi / 365)
                irradiance = seasonal_factor * 1000 * np.sin((hour - 6) * np.pi / 12)
                irradiance += np.random.normal(0, 100)
                irradiance = max(0, irradiance)
            else:
                irradiance = 0

            temp = 25 + 10 * np.sin((day_of_year - 80) * 2 * np.pi / 365)
            temp += 8 * np.sin((hour - 6) * np.pi / 12)
            temp += np.random.normal(0, 3)

            energy = irradiance * 0.01 + np.random.normal(0, 0.5)
            energy = max(0, energy)

            maintenance = 1 if np.random.random() < 0.05 else 0

            data.append({
                'timestamp': ts,
                'energy_output': round(energy, 2),
                'temperature': round(temp, 1),
                'humidity': round(60 + 20 * np.sin((hour - 12) * np.pi / 12) + np.random.normal(0, 10), 1),
                'wind_speed': round(2.5 + np.random.normal(0, 1), 1),
                'solar_irradiance': round(irradiance, 1),
                'panel_temp': round(temp + irradiance * 0.015 + np.random.normal(0, 2), 1),
                'voltage': round(24 + np.random.normal(0, 1), 1),
                'current': round(energy * 4 + np.random.normal(0, 2), 1),
                'power_factor': round(0.95 + np.random.normal(0, 0.03), 2),
                'dust_level': round(np.random.uniform(0.1, 1.0), 2),
                'maintenance_needed': maintenance
            })

        df = pd.DataFrame(data)

        os.makedirs('data', exist_ok=True)
        df.to_csv('data/historical_data.csv', index=False)
        logger.info(f"Generated and saved {len(df)} historical records")

        return df

if __name__ == "__main__":
    ingestion = DataIngestion()

    print("Testing weather data:")
    weather = ingestion.get_weather_data()
    print(weather)

    print("\nTesting sensor data:")
    sensor = ingestion.get_sensor_data()
    print(sensor)

    print("\nTesting combined data:")
    combined = ingestion.get_combined_data()
    print(combined)
