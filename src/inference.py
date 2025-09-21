
import pandas as pd
import numpy as np
import joblib
import logging
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')

from data_ingestion import DataIngestion
from feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)

class ModelInference:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = {}
        self.engineer = FeatureEngineer()
        self.ingestion = DataIngestion()

        self._load_models()

    def _load_models(self):
        try:
            model_files = {
                'maintenance': 'models/maintenance_classifier.pkl',
                'performance': 'models/performance_regressor.pkl',
                'anomaly': 'models/anomaly_detector.pkl'
            }

            feature_files = {
                'maintenance': 'models/maintenance_features.pkl',
                'performance': 'models/performance_features.pkl',
                'anomaly': 'models/anomaly_features.pkl'
            }

            scaler_files = {
                'anomaly': 'models/anomaly_scaler.pkl'
            }

            for name, file_path in model_files.items():
                if os.path.exists(file_path):
                    self.models[name] = joblib.load(file_path)
                    logger.info(f"Loaded {name} model")
                else:
                    logger.warning(f"Model file not found: {file_path}")

            for name, file_path in feature_files.items():
                if os.path.exists(file_path):
                    self.feature_columns[name] = joblib.load(file_path)
                    logger.info(f"Loaded {name} features")

            for name, file_path in scaler_files.items():
                if os.path.exists(file_path):
                    self.scalers[name] = joblib.load(file_path)
                    logger.info(f"Loaded {name} scaler")

            if not self.models:
                logger.warning("No models loaded. Run training first.")

        except Exception as e:
            logger.error(f"Error loading models: {e}")

    def prepare_input_data(self, data, model_type):
        try:
            if isinstance(data, dict):
                df = pd.DataFrame([data])
            elif isinstance(data, pd.DataFrame):
                df = data.copy()
            else:
                raise ValueError("Data must be dict or DataFrame")

            if 'timestamp' not in df.columns:
                df['timestamp'] = datetime.now()

            df = self.engineer.prepare_features(df, include_all=True)

            if model_type in self.feature_columns:
                required_features = self.feature_columns[model_type]

                for feature in required_features:
                    if feature not in df.columns:
                        df[feature] = 0

                df_features = df[required_features]

                return df_features
            else:
                logger.error(f"Feature columns not found for model: {model_type}")
                return None

        except Exception as e:
            logger.error(f"Error preparing input data: {e}")
            return None

    def predict_maintenance(self, data):
        try:
            if 'maintenance' not in self.models:
                logger.error("Maintenance model not loaded")
                return None

            X = self.prepare_input_data(data, 'maintenance')
            if X is None:
                return None

            model = self.models['maintenance']
            prediction = model.predict(X)[0]
            probability = model.predict_proba(X)[0]

            result = {
                'maintenance_needed': bool(prediction),
                'maintenance_probability': float(probability[1]),
                'confidence': float(max(probability)),
                'prediction_time': datetime.now().isoformat()
            }

            logger.info(f"Maintenance prediction: {result}")
            return result

        except Exception as e:
            logger.error(f"Error in maintenance prediction: {e}")
            return None

    def predict_performance(self, data):
        try:
            if 'performance' not in self.models:
                logger.error("Performance model not loaded")
                return None

            X = self.prepare_input_data(data, 'performance')
            if X is None:
                return None

            model = self.models['performance']
            prediction = model.predict(X)[0]

            result = {
                'predicted_energy_output': float(prediction),
                'prediction_time': datetime.now().isoformat()
            }

            logger.info(f"Performance prediction: {result}")
            return result

        except Exception as e:
            logger.error(f"Error in performance prediction: {e}")
            return None

    def detect_anomaly(self, data):
        try:
            if 'anomaly' not in self.models or 'anomaly' not in self.scalers:
                logger.error("Anomaly model or scaler not loaded")
                return None

            X = self.prepare_input_data(data, 'anomaly')
            if X is None:
                return None

            scaler = self.scalers['anomaly']
            X_scaled = scaler.transform(X)

            model = self.models['anomaly']
            anomaly_score = model.decision_function(X_scaled)[0]
            is_anomaly = model.predict(X_scaled)[0] == -1

            result = {
                'is_anomaly': bool(is_anomaly),
                'anomaly_score': float(anomaly_score),
                'severity': self._get_anomaly_severity(anomaly_score),
                'prediction_time': datetime.now().isoformat()
            }

            logger.info(f"Anomaly detection: {result}")
            return result

        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}")
            return None

    def _get_anomaly_severity(self, score):
        if score > 0:
            return "Normal"
        elif score > -0.2:
            return "Low"
        elif score > -0.4:
            return "Medium"
        else:
            return "High"

    def predict_all(self, data):
        try:
            results = {
                'input_data': data if isinstance(data, dict) else data.to_dict('records')[0],
                'predictions': {},
                'timestamp': datetime.now().isoformat()
            }

            maintenance_result = self.predict_maintenance(data)
            if maintenance_result:
                results['predictions']['maintenance'] = maintenance_result

            performance_result = self.predict_performance(data)
            if performance_result:
                results['predictions']['performance'] = performance_result

            anomaly_result = self.detect_anomaly(data)
            if anomaly_result:
                results['predictions']['anomaly'] = anomaly_result

            return results

        except Exception as e:
            logger.error(f"Error in predict_all: {e}")
            return None

    def predict_future_performance(self, hours_ahead=24):
        try:
            logger.info(f"Predicting performance for next {hours_ahead} hours")

            current_data = self.ingestion.get_combined_data()
            if not current_data:
                logger.error("Could not get current data")
                return None

            predictions = []
            base_time = datetime.now()

            for hour in range(1, hours_ahead + 1):
                future_time = base_time + timedelta(hours=hour)
                future_data = current_data.copy()
                future_data['timestamp'] = future_time

                hour_of_day = future_time.hour
                if 6 <= hour_of_day <= 18:
                    solar_factor = np.sin((hour_of_day - 6) * np.pi / 12)
                    future_data['solar_irradiance'] = 1000 * solar_factor * np.random.uniform(0.8, 1.2)
                else:
                    future_data['solar_irradiance'] = 0

                performance_result = self.predict_performance(future_data)
                if performance_result:
                    predictions.append({
                        'hour': hour,
                        'timestamp': future_time.isoformat(),
                        'predicted_energy': performance_result['predicted_energy_output'],
                        'solar_irradiance': future_data['solar_irradiance']
                    })

            return predictions

        except Exception as e:
            logger.error(f"Error predicting future performance: {e}")
            return None

    def get_model_status(self):
        status = {
            'models_loaded': list(self.models.keys()),
            'models_available': len(self.models),
            'last_check': datetime.now().isoformat()
        }

        for model_name in ['maintenance', 'performance', 'anomaly']:
            status[f'{model_name}_ready'] = model_name in self.models

        return status

    def get_feature_importance(self, model_type):
        try:
            if model_type not in self.models:
                logger.warning(f"Model {model_type} not loaded")
                return None

            model = self.models[model_type]

            if hasattr(model, 'feature_importances_'):
                feature_names = self.feature_columns.get(model_type, [])
                importances = model.feature_importances_

                if len(feature_names) == len(importances):
                    importance_df = pd.DataFrame({
                        'feature': feature_names,
                        'importance': importances
                    }).sort_values('importance', ascending=False)

                    return importance_df
                else:
                    logger.warning(f"Feature names and importances length mismatch for {model_type}")
                    feature_names = [f'feature_{i}' for i in range(len(importances))]
                    importance_df = pd.DataFrame({
                        'feature': feature_names,
                        'importance': importances
                    }).sort_values('importance', ascending=False)
                    return importance_df
            else:
                logger.warning(f"Model {model_type} does not have feature importance")
                return None

        except Exception as e:
            logger.error(f"Error getting feature importance for {model_type}: {e}")
            return None

    def run_real_time_inference(self):
        try:
            current_data = self.ingestion.get_combined_data()
            if not current_data:
                logger.error("Could not get current data for inference")
                return None

            results = self.predict_all(current_data)

            if results:
                results['system_status'] = self.get_model_status()

            return results

        except Exception as e:
            logger.error(f"Error in real-time inference: {e}")
            return None

def main():
    inference = ModelInference()

    sample_data = {
        'energy_output': 8.5,
        'temperature': 32.1,
        'humidity': 55.2,
        'wind_speed': 3.2,
        'solar_irradiance': 850.0,
        'panel_temp': 38.5,
        'voltage': 24.2,
        'current': 35.1,
        'power_factor': 0.98,
        'dust_level': 0.3
    }

    print("Testing inference with sample data:")
    print(f"Input: {sample_data}")

    print("\n1. Maintenance Prediction:")
    maintenance_result = inference.predict_maintenance(sample_data)
    print(maintenance_result)

    print("\n2. Performance Prediction:")
    performance_result = inference.predict_performance(sample_data)
    print(performance_result)

    print("\n3. Anomaly Detection:")
    anomaly_result = inference.detect_anomaly(sample_data)
    print(anomaly_result)

    print("\n4. All Predictions:")
    all_results = inference.predict_all(sample_data)
    print(all_results)

    print("\n5. Model Status:")
    status = inference.get_model_status()
    print(status)

if __name__ == "__main__":
    main()

