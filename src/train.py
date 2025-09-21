import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import logging
import os
from datetime import datetime

from data_ingestion import DataIngestion
from feature_engineering import FeatureEngineer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = {}

        os.makedirs('models', exist_ok=True)

    def prepare_data(self):
        try:
            logger.info("Loading and preparing data...")

            ingestion = DataIngestion()
            df = ingestion.load_historical_data()

            if len(df) < 100:
                logger.warning("Insufficient data, generating sample data")
                df = ingestion._generate_sample_data(days=365)

            engineer = FeatureEngineer()
            df = engineer.prepare_features(df, include_all=True)

            logger.info(f"Data prepared successfully. Shape: {df.shape}")
            return df, engineer

        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            raise

    def train_maintenance_classifier(self, df, engineer):
        try:
            logger.info("Training maintenance classifier...")

            feature_cols = engineer.get_feature_columns(df, ['maintenance_needed', 'timestamp'])
            X = df[feature_cols].select_dtypes(include=[np.number])
            y = df['maintenance_needed']

            from sklearn.utils.class_weight import compute_class_weight
            classes = np.unique(y)
            class_weights = compute_class_weight('balanced', classes=classes, y=y)
            class_weight_dict = dict(zip(classes, class_weights))

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            model = lgb.LGBMClassifier(
                objective='binary',
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                num_leaves=31,
                class_weight=class_weight_dict,
                random_state=42,
                verbose=-1
            )

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]

            logger.info("Maintenance Classifier Results:")
            logger.info(f"\n{classification_report(y_test, y_pred)}")

            cv_scores = cross_val_score(model, X, y, cv=5, scoring='f1')
            logger.info(f"Cross-validation F1 scores: {cv_scores}")
            logger.info(f"Mean CV F1 score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

            self.models['maintenance'] = model
            self.feature_columns['maintenance'] = list(X.columns)

            joblib.dump(model, 'models/maintenance_classifier.pkl')
            joblib.dump(list(X.columns), 'models/maintenance_features.pkl')

            logger.info("Maintenance classifier trained and saved successfully")
            return model

        except Exception as e:
            logger.error(f"Error training maintenance classifier: {e}")
            raise

    def train_performance_regressor(self, df, engineer):
        try:
            logger.info("Training performance regressor...")

            feature_cols = engineer.get_feature_columns(df, ['energy_output', 'timestamp'])
            X = df[feature_cols].select_dtypes(include=[np.number])
            y = df['energy_output']

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            model = lgb.LGBMRegressor(
                objective='regression',
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                num_leaves=31,
                random_state=42,
                verbose=-1
            )

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            logger.info("Performance Regressor Results:")
            logger.info(f"MSE: {mse:.3f}")
            logger.info(f"MAE: {mae:.3f}")
            logger.info(f"R2 Score: {r2:.3f}")

            cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
            logger.info(f"Cross-validation R2 scores: {cv_scores}")
            logger.info(f"Mean CV R2 score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

            self.models['performance'] = model
            self.feature_columns['performance'] = list(X.columns)

            joblib.dump(model, 'models/performance_regressor.pkl')
            joblib.dump(list(X.columns), 'models/performance_features.pkl')

            logger.info("Performance regressor trained and saved successfully")
            return model

        except Exception as e:
            logger.error(f"Error training performance regressor: {e}")
            raise

    def train_anomaly_detector(self, df, engineer):
        try:
            logger.info("Training anomaly detector...")

            feature_cols = engineer.get_feature_columns(df, ['maintenance_needed', 'timestamp'])
            X = df[feature_cols].select_dtypes(include=[np.number])

            normal_data = X[df['maintenance_needed'] == 0]

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(normal_data)

            model = IsolationForest(
                contamination=0.1,
                random_state=42,
                n_estimators=100
            )

            model.fit(X_scaled)

            X_full_scaled = scaler.transform(X)
            anomaly_scores = model.decision_function(X_full_scaled)
            anomaly_predictions = model.predict(X_full_scaled)

            anomaly_binary = (anomaly_predictions == -1).astype(int)

            logger.info(f"Anomalies detected: {anomaly_binary.sum()} out of {len(anomaly_binary)} samples")
            logger.info(f"Anomaly rate: {anomaly_binary.mean():.3f}")

            self.models['anomaly'] = model
            self.scalers['anomaly'] = scaler
            self.feature_columns['anomaly'] = list(X.columns)

            joblib.dump(model, 'models/anomaly_detector.pkl')
            joblib.dump(scaler, 'models/anomaly_scaler.pkl')
            joblib.dump(list(X.columns), 'models/anomaly_features.pkl')

            logger.info("Anomaly detector trained and saved successfully")
            return model, scaler

        except Exception as e:
            logger.error(f"Error training anomaly detector: {e}")
            raise

    def get_feature_importance(self, model_name):
        try:
            if model_name not in self.models:
                logger.warning(f"Model {model_name} not found")
                return None

            model = self.models[model_name]
            feature_cols = self.feature_columns[model_name]

            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': feature_cols,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)

                return importance_df
            else:
                logger.warning(f"Model {model_name} does not have feature importances")
                return None

        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return None

    def save_training_metadata(self):
        try:
            metadata = {
                'training_date': datetime.now().isoformat(),
                'models_trained': list(self.models.keys()),
                'feature_counts': {name: len(cols) for name, cols in self.feature_columns.items()}
            }

            joblib.dump(metadata, 'models/training_metadata.pkl')
            logger.info("Training metadata saved successfully")

        except Exception as e:
            logger.error(f"Error saving training metadata: {e}")

    def train_all_models(self):
        try:
            logger.info("Starting model training pipeline...")

            df, engineer = self.prepare_data()

            maintenance_model = self.train_maintenance_classifier(df, engineer)
            performance_model = self.train_performance_regressor(df, engineer)
            anomaly_model, anomaly_scaler = self.train_anomaly_detector(df, engineer)

            self.save_training_metadata()

            for model_name in ['maintenance', 'performance']:
                importance_df = self.get_feature_importance(model_name)
                if importance_df is not None:
                    logger.info(f"\nTop 10 features for {model_name} model:")
                    logger.info(f"\n{importance_df.head(10)}")

            logger.info("All models trained successfully!")
            return True

        except Exception as e:
            logger.error(f"Error in training pipeline: {e}")
            return False

def main():
    trainer = ModelTrainer()
    success = trainer.train_all_models()

    if success:
        print("Training completed successfully!")
        print("Models saved in 'models/' directory")
    else:
        print("Training failed. Check logs for details.")

if __name__ == "__main__":
    main()
