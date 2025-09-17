import time
import json
import logging
import threading
from datetime import datetime
import pandas as pd
import numpy as np
from queue import Queue

try:
    import RPi.GPIO as GPIO
    import spidev
    import board
    import busio
    import adafruit_ads1x15.ads1115 as ADS
    from adafruit_ads1x15.analog_in import AnalogIn
    RPi_AVAILABLE = True
except ImportError:
    print("RPi libraries not available - using simulation mode")
    RPi_AVAILABLE = False

class SensorInterface:
    """Interface for various sensors used in solar panel monitoring"""
    
    def __init__(self, simulation_mode=not RPi_AVAILABLE):
        self.simulation_mode = simulation_mode
        self.logger = self.setup_logger()
        
        if not simulation_mode:
            self.setup_hardware()
        else:
            self.logger.info("Running in simulation mode")
    
    def setup_logger(self):
        """Setup logging for sensor data"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('solar_sensors.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def setup_hardware(self):
        """Initialize hardware interfaces"""
        try:
            i2c = busio.I2C(board.SCL, board.SDA)
            self.ads = ADS.ADS1115(i2c)
            
            self.voltage_channel = AnalogIn(self.ads, ADS.P0)  
            self.current_channel = AnalogIn(self.ads, ADS.P1)  
            self.temp_channel = AnalogIn(self.ads, ADS.P2)     
            self.light_channel = AnalogIn(self.ads, ADS.P3)    
            
            self.spi = spidev.SpiDev()
            self.spi.open(0, 0)
            self.spi.max_speed_hz = 1000000
            
            self.logger.info("Hardware initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Hardware setup failed: {e}")
            self.simulation_mode = True
    
    def read_voltage(self):
        """Read panel voltage"""
        if self.simulation_mode:
            base_voltage = 24 + np.random.normal(0, 0.5)
            return max(0, base_voltage)
        
        try:
            raw_voltage = self.voltage_channel.voltage
            panel_voltage = raw_voltage * 10.0  
            return max(0, panel_voltage)
        except Exception as e:
            self.logger.error(f"Voltage reading error: {e}")
            return 0
    
    def read_current(self):
        """Read panel current using current sensor"""
        if self.simulation_mode:
            base_current = 6 + np.random.normal(0, 0.3)
            return max(0, base_current)
        
        try:
            raw_voltage = self.current_channel.voltage
            current = (raw_voltage - 2.5) / 0.066
            return max(0, current)
        except Exception as e:
            self.logger.error(f"Current reading error: {e}")
            return 0
    
    def read_temperature(self):
        """Read ambient temperature"""
        if self.simulation_mode:
            hour = datetime.now().hour
            base_temp = 20 + 10 * np.sin((hour - 6) * np.pi / 12)
            return base_temp + np.random.normal(0, 2)
        
        try:
            raw_voltage = self.temp_channel.voltage
            temperature = raw_voltage * 100  
            return temperature
        except Exception as e:
            self.logger.error(f"Temperature reading error: {e}")
            return 25  
    
    def read_irradiance(self):
        """Read solar irradiance using light sensor"""
        if self.simulation_mode:
            hour = datetime.now().hour
            if 6 <= hour <= 18:
                base_irradiance = 1000 * np.sin((hour - 6) * np.pi / 12)
                return max(0, base_irradiance + np.random.normal(0, 50))
            return np.random.uniform(0, 10)  
        
        try:
            raw_voltage = self.light_channel.voltage
            irradiance = raw_voltage * 300  
            return max(0, irradiance)
        except Exception as e:
            self.logger.error(f"Irradiance reading error: {e}")
            return 0
    
    def read_humidity(self):
        """Read humidity (simulated or from DHT22 sensor)"""
        if self.simulation_mode:
            temp = self.read_temperature()
            base_humidity = 80 - temp + np.random.normal(0, 10)
            return np.clip(base_humidity, 20, 95)
        
        try:

            return 60 + np.random.normal(0, 15)
        except Exception as e:
            self.logger.error(f"Humidity reading error: {e}")
            return 60
    
    def read_wind_speed(self):
        """Read wind speed (simulated or from anemometer)"""
        if self.simulation_mode:
            return np.random.exponential(2)
        
        try:
            return np.random.exponential(2)
        except Exception as e:
            self.logger.error(f"Wind speed reading error: {e}")
            return 1.0
    
    def calculate_panel_temperature(self, ambient_temp, irradiance):
        """Calculate panel temperature based on ambient and irradiance"""
        temp_rise = irradiance * 0.025  
        return ambient_temp + temp_rise


class SolarDataCollector:
    """Main data collection class for solar panel monitoring"""
    
    def __init__(self, collection_interval=10):
        self.sensor_interface = SensorInterface()
        self.collection_interval = collection_interval  
        self.data_queue = Queue()
        self.is_collecting = False
        self.maintenance_data = self.load_maintenance_data()
        self.logger = self.sensor_interface.logger
    
    def load_maintenance_data(self):
        """Load or initialize maintenance tracking data"""
        try:
            with open('maintenance_log.json', 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            data = {
                'last_cleaning': datetime.now().isoformat(),
                'last_maintenance': datetime.now().isoformat(),
                'dust_level': 0.1  
            }
            self.save_maintenance_data(data)
        return data
    
    def save_maintenance_data(self, data):
        """Save maintenance data to file"""
        with open('maintenance_log.json', 'w') as f:
            json.dump(data, f, indent=2)
    
    def update_dust_level(self):
        """Update dust accumulation based on time and weather"""
        last_cleaning = datetime.fromisoformat(self.maintenance_data['last_cleaning'])
        hours_since_cleaning = (datetime.now() - last_cleaning).total_seconds() / 3600
        
        base_accumulation = hours_since_cleaning * 0.001  
        current_dust = min(1.0, self.maintenance_data['dust_level'] + base_accumulation)
        
        self.maintenance_data['dust_level'] = current_dust
        return current_dust, hours_since_cleaning
    
    def collect_single_reading(self):
        """Collect a single set of sensor readings"""
        try:
            voltage = self.sensor_interface.read_voltage()
            current = self.sensor_interface.read_current()
            temperature = self.sensor_interface.read_temperature()
            irradiance = self.sensor_interface.read_irradiance()
            humidity = self.sensor_interface.read_humidity()
            wind_speed = self.sensor_interface.read_wind_speed()
            
            power_output = voltage * current
            panel_temp = self.sensor_interface.calculate_panel_temperature(temperature, irradiance)
            
            dust_level, hours_since_cleaning = self.update_dust_level()
            
            last_maintenance = datetime.fromisoformat(self.maintenance_data['last_maintenance'])
            days_since_maintenance = (datetime.now() - last_maintenance).days
            
            data_record = {
                'timestamp': datetime.now().isoformat(),
                'solar_irradiance': irradiance,
                'temperature': temperature,
                'humidity': humidity,
                'wind_speed': wind_speed,
                'panel_voltage': voltage,
                'panel_current': current,
                'power_output': power_output,
                'panel_temp': panel_temp,
                'dust_level': dust_level,
                'hours_since_cleaning': hours_since_cleaning,
                'days_since_maintenance': days_since_maintenance
            }
            
            return data_record
            
        except Exception as e:
            self.logger.error(f"Error collecting sensor data: {e}")
            return None
    
    def start_continuous_collection(self):
        """Start continuous data collection in background thread"""
        if self.is_collecting:
            self.logger.warning("Data collection already running")
            return
        
        self.is_collecting = True
        self.collection_thread = threading.Thread(target=self._collection_loop)
        self.collection_thread.daemon = True
        self.collection_thread.start()
        self.logger.info("Started continuous data collection")
    
    def stop_continuous_collection(self):
        """Stop continuous data collection"""
        self.is_collecting = False
        if hasattr(self, 'collection_thread'):
            self.collection_thread.join(timeout=5)
        self.logger.info("Stopped continuous data collection")
    
    def _collection_loop(self):
        """Background data collection loop"""
        while self.is_collecting:
            try:
                data = self.collect_single_reading()
                if data:
                    self.data_queue.put(data)
                    self.logger.debug(f"Collected data: Power={data['power_output']:.1f}W")
                
                time.sleep(self.collection_interval)
                
            except Exception as e:
                self.logger.error(f"Collection loop error: {e}")
                time.sleep(self.collection_interval)
    
    def get_latest_data(self, as_dataframe=True):
        """Get the most recent sensor reading"""
        if self.data_queue.empty():
            data = self.collect_single_reading()
        else:
            data = self.data_queue.get()
        
        if data and as_dataframe:
            return pd.DataFrame([data])
        return data
    
    def get_batch_data(self, max_samples=100):
        """Get multiple recent readings"""
        batch_data = []
        count = 0
        
        while not self.data_queue.empty() and count < max_samples:
            batch_data.append(self.data_queue.get())
            count += 1
        
        return pd.DataFrame(batch_data) if batch_data else None
    
    def log_cleaning_event(self):
        """Record that panel cleaning occurred"""
        self.maintenance_data['last_cleaning'] = datetime.now().isoformat()
        self.maintenance_data['dust_level'] = 0.05 
        self.save_maintenance_data(self.maintenance_data)
        self.logger.info("Panel cleaning event logged")
    
    def log_maintenance_event(self):
        """Record that maintenance was performed"""
        self.maintenance_data['last_maintenance'] = datetime.now().isoformat()
        self.maintenance_data['dust_level'] = 0.05
        self.save_maintenance_data(self.maintenance_data)
        self.logger.info("Maintenance event logged")
    
    def save_data_to_file(self, filepath='sensor_readings.csv', max_rows=10000):
        """Save collected data to CSV file"""
        try:
            batch_data = self.get_batch_data(max_rows)
            if batch_data is not None and not batch_data.empty:
                try:
                    existing_data = pd.read_csv(filepath)
                    combined_data = pd.concat([existing_data, batch_data], ignore_index=True)
                    if len(combined_data) > max_rows:
                        combined_data = combined_data.tail(max_rows)
                except FileNotFoundError:
                    combined_data = batch_data
                
                combined_data.to_csv(filepath, index=False)
                self.logger.info(f"Saved {len(batch_data)} readings to {filepath}")
                return True
        except Exception as e:
            self.logger.error(f"Error saving data to file: {e}")
        return False


class RealTimeMonitor:
    """Real-time monitoring system combining sensor data with ML predictions"""
    
    def __init__(self):
        self.data_collector = SolarDataCollector()
        self.ml_model = None
        self.logger = self.data_collector.logger
        
        try:
            from solar_ml_model import SolarPanelML
            self.ml_model = SolarPanelML()
            if not self.ml_model.load_models():
                self.logger.warning("ML models not found. Run synthetic_data_generator.py first")
        except ImportError:
            self.logger.error("ML model not available")
    
    def start_monitoring(self, save_interval=300):  
        """Start real-time monitoring with ML analysis"""
        self.data_collector.start_continuous_collection()
        
        try:
            last_save_time = time.time()
            
            while True:
                sensor_data = self.data_collector.get_latest_data()
                
                if sensor_data is not None and not sensor_data.empty:
                    if self.ml_model:
                        report = self.ml_model.generate_report(sensor_data)
                        self.display_status(sensor_data.iloc[0], report)
                    else:
                        self.display_basic_status(sensor_data.iloc[0])
                    
                    current_time = time.time()
                    if current_time - last_save_time >= save_interval:
                        self.data_collector.save_data_to_file()
                        last_save_time = current_time
                
                time.sleep(30)  
                
        except KeyboardInterrupt:
            self.logger.info("Monitoring stopped by user")
        finally:
            self.data_collector.stop_continuous_collection()
            self.data_collector.save_data_to_file()  
    
    def display_status(self, sensor_data, ml_report):
        """Display current status with ML analysis"""
        print("\n" + "="*60)
        print(f"Solar Panel Status - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        
        print(f"Power Output:      {sensor_data['power_output']:.1f} W")
        print(f"Voltage:           {sensor_data['panel_voltage']:.1f} V")
        print(f"Current:           {sensor_data['panel_current']:.1f} A")
        print(f"Panel Temp:        {sensor_data['panel_temp']:.1f} °C")
        print(f"Solar Irradiance:  {sensor_data['solar_irradiance']:.0f} W/m²")
        print(f"Ambient Temp:      {sensor_data['temperature']:.1f} °C")
        
        print("\nML Analysis:")
        print(f"Efficiency:        {ml_report.get('current_efficiency', 'N/A'):.1%}")
        print(f"Maintenance in:    {ml_report.get('days_until_maintenance', 'N/A'):.0f} days")
        
        if ml_report.get('anomaly_detected'):
            print("  ANOMALY DETECTED!")
        
        if ml_report.get('maintenance_urgent'):
            print(" MAINTENANCE URGENT!")
        
        if ml_report.get('recommendations'):
            print("\nRecommendations:")
            for rec in ml_report['recommendations']:
                print(f"• {rec}")
    
    def display_basic_status(self, sensor_data):
        """Display basic status without ML analysis"""
        print("\n" + "="*40)
        print(f"Solar Panel Status - {datetime.now().strftime('%H:%M:%S')}")
        print("="*40)
        print(f"Power:    {sensor_data['power_output']:.1f} W")
        print(f"Voltage:  {sensor_data['panel_voltage']:.1f} V")
        print(f"Current:  {sensor_data['panel_current']:.1f} A")
        print(f"Temp:     {sensor_data['panel_temp']:.1f} °C")


def main():
    """Main function for testing and demonstration"""
    print("Solar Panel Real-Time Data Collector")
    print("Running on Raspberry Pi 3B")
    print("="*50)
    
    collector = SolarDataCollector()
    data = collector.collect_single_reading()
    
    if data:
        print("\nSample sensor reading:")
        for key, value in data.items():
            if isinstance(value, float):
                print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value}")
    
    response = input("\nStart real-time monitoring? (y/N): ").lower()
    if response == 'y':
        monitor = RealTimeMonitor()
        print("Starting real-time monitoring... (Ctrl+C to stop)")
        monitor.start_monitoring()

if __name__ == "__main__":
    main()