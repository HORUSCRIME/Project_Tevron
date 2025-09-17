import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class SolarPanelML:
    def __init__(self):
        self.efficiency_model = None
        self.maintenance_model = None
        self.anomaly_detector = None
        self.scaler = StandardScaler()
        self.feature_names = [
            'solar_irradiance', 'temperature', 'humidity', 'wind_speed',
            'panel_voltage', 'panel_current', 'power_output', 'panel_temp',
            'dust_level', 'hours_since_cleaning', 'days_since_maintenance'
        ]
        
    def prepare_features(self, data):
        """Extract and engineer features from raw sensor data"""
        features = data[self.feature_names].copy()
        
        features['power_density'] = features['power_output'] / (features['solar_irradiance'] + 1e-6)
        features['temp_diff'] = features['panel_temp'] - features['temperature']
        features['efficiency_ratio'] = features['power_output'] / (features['panel_voltage'] * features['panel_current'] + 1e-6)
        features['maintenance_urgency'] = (
            features['dust_level'] * 0.3 + 
            features['hours_since_cleaning'] / 24 * 0.4 +
            features['days_since_maintenance'] * 0.3
        )
        
        return features
    
    def train_efficiency_model(self, data):
        """Train efficiency prediction model"""
        print("Training efficiency prediction model...")
        
        features = self.prepare_features(data)
        target = data['efficiency']
        
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.efficiency_model = RandomForestRegressor(
            n_estimators=50,  
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=2  
        )
        
        self.efficiency_model.fit(X_train_scaled, y_train)
        
        y_pred = self.efficiency_model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"Efficiency Model - MAE: {mae:.3f}, RMSE: {rmse:.3f}")
        return mae, rmse
    
    def train_maintenance_model(self, data):
        """Train maintenance prediction model"""
        print("Training maintenance prediction model...")
        
        features = self.prepare_features(data)
        target = data['days_until_maintenance']
        
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42
        )
        
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.maintenance_model = RandomForestRegressor(
            n_estimators=30,
            max_depth=8,
            min_samples_split=5,
            random_state=42,
            n_jobs=2
        )
        
        self.maintenance_model.fit(X_train_scaled, y_train)
        
        y_pred = self.maintenance_model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        
        print(f"Maintenance Model - MAE: {mae:.2f} days")
        return mae
    
    def train_anomaly_detector(self, data):
        """Train anomaly detection model"""
        print("Training anomaly detection model...")
        
        features = self.prepare_features(data)
        features_scaled = self.scaler.transform(features)
        
        self.anomaly_detector = IsolationForest(
            contamination=0.1,  
            random_state=42,
            n_jobs=2
        )
        
        self.anomaly_detector.fit(features_scaled)
        print("Anomaly detector trained successfully")
    
    def predict_efficiency(self, sensor_data):
        """Predict current efficiency"""
        if self.efficiency_model is None:
            raise ValueError("Efficiency model not trained")
        
        features = self.prepare_features(sensor_data)
        features_scaled = self.scaler.transform(features)
        
        efficiency = self.efficiency_model.predict(features_scaled)
        return efficiency[0] if len(efficiency) == 1 else efficiency
    
    def predict_maintenance(self, sensor_data):
        """Predict days until maintenance needed"""
        if self.maintenance_model is None:
            raise ValueError("Maintenance model not trained")
        
        features = self.prepare_features(sensor_data)
        features_scaled = self.scaler.transform(features)
        
        days_until = self.maintenance_model.predict(features_scaled)
        return max(0, days_until[0] if len(days_until) == 1 else days_until)
    
    def detect_anomaly(self, sensor_data):
        """Detect anomalies in sensor readings"""
        if self.anomaly_detector is None:
            raise ValueError("Anomaly detector not trained")
        
        features = self.prepare_features(sensor_data)
        features_scaled = self.scaler.transform(features)
        
        anomaly_score = self.anomaly_detector.decision_function(features_scaled)
        is_anomaly = self.anomaly_detector.predict(features_scaled)
        
        return {
            'is_anomaly': is_anomaly[0] == -1,
            'anomaly_score': anomaly_score[0],
            'confidence': abs(anomaly_score[0])
        }
    
    def get_feature_importance(self):
        """Get feature importance for efficiency model"""
        if self.efficiency_model is None:
            return None
        
        importance = self.efficiency_model.feature_importances_
        feature_names = list(self.prepare_features(pd.DataFrame(columns=self.feature_names)).columns)
        
        return dict(zip(feature_names, importance))
    
    def save_models(self, filepath='solar_models.joblib'):
        """Save trained models"""
        models = {
            'efficiency_model': self.efficiency_model,
            'maintenance_model': self.maintenance_model,
            'anomaly_detector': self.anomaly_detector,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }
        joblib.dump(models, filepath)
        print(f"Models saved to {filepath}")
    
    def load_models(self, filepath='solar_models.joblib'):
        """Load trained models"""
        try:
            models = joblib.load(filepath)
            self.efficiency_model = models['efficiency_model']
            self.maintenance_model = models['maintenance_model']
            self.anomaly_detector = models['anomaly_detector']
            self.scaler = models['scaler']
            self.feature_names = models['feature_names']
            print(f"Models loaded from {filepath}")
            return True
        except FileNotFoundError:
            print(f"Model file {filepath} not found")
            return False
    
    def generate_report(self, sensor_data):
        """Generate comprehensive analysis report"""
        try:
            efficiency = self.predict_efficiency(sensor_data)
            maintenance_days = self.predict_maintenance(sensor_data)
            anomaly_result = self.detect_anomaly(sensor_data)
            
            report = {
                'timestamp': pd.Timestamp.now(),
                'current_efficiency': efficiency,
                'days_until_maintenance': maintenance_days,
                'maintenance_urgent': maintenance_days < 3,
                'anomaly_detected': anomaly_result['is_anomaly'],
                'anomaly_confidence': anomaly_result['confidence'],
                'recommendations': []
            }
            
            if efficiency < 0.75:
                report['recommendations'].append("Low efficiency detected - check for dust or shading")
            
            if maintenance_days < 7:
                report['recommendations'].append(f"Maintenance recommended within {maintenance_days:.0f} days")
            
            if anomaly_result['is_anomaly']:
                report['recommendations'].append("Anomaly detected - inspect system immediately")
            
            return report
            
        except Exception as e:
            return {'error': str(e)}

def main():
    """Example usage and testing"""
    print("Solar Panel ML System - Raspberry Pi 3B")
    print("=" * 50)
    
    solar_ml = SolarPanelML()
    
    if solar_ml.load_models():
        print("Using existing trained models")
    else:
        print("No existing models found. Please run training first.")
        print("Use synthetic_data_generator.py to create training data")
        return
    
    sample_data = pd.DataFrame({
        'solar_irradiance': [800],
        'temperature': [25],
        'humidity': [60],
        'wind_speed': [2.5],
        'panel_voltage': [24.5],
        'panel_current': [8.2],
        'power_output': [180],
        'panel_temp': [35],
        'dust_level': [0.3],
        'hours_since_cleaning': [72],
        'days_since_maintenance': [45]
    })
    
    report = solar_ml.generate_report(sample_data)
    
    print("\nSample Analysis Report:")
    print("-" * 30)
    for key, value in report.items():
        if key == 'recommendations':
            print(f"{key}: {', '.join(value) if value else 'None'}")
        else:
            print(f"{key}: {value}")

if __name__ == "__main__":
    main()

