
import sys
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    try:
        print("Testing imports...")

        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns

        import lightgbm as lgb
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import IsolationForest

        import streamlit as st

        import plotly.express as px
        import plotly.graph_objects as go

        print("[SUCCESS] All imports successful!")
        return True

    except ImportError as e:
        print(f"[ERROR] Import error: {e}")
        return False

def test_data_loading():
    try:
        print("\nTesting data loading...")

        from src.data_ingestion import DataIngestion

        ingestion = DataIngestion()
        df = ingestion.load_historical_data()

        if df is not None and not df.empty:
            print(f"[SUCCESS] Data loaded successfully! Shape: {df.shape}")
            print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            return True
        else:
            print("[ERROR] Failed to load data")
            return False

    except Exception as e:
        print(f"[ERROR] Data loading error: {e}")
        return False

def test_feature_engineering():
    try:
        print("\nTesting feature engineering...")

        from src.data_ingestion import DataIngestion
        from src.feature_engineering import FeatureEngineer

        ingestion = DataIngestion()
        engineer = FeatureEngineer()

        df = ingestion.load_historical_data()
        df_features = engineer.prepare_features(df.head(100))

        if df_features is not None and len(df_features.columns) > len(df.columns):
            print(f"[SUCCESS] Feature engineering successful!")
            print(f"   Original features: {len(df.columns)}")
            print(f"   Engineered features: {len(df_features.columns)}")
            return True
        else:
            print("[ERROR] Feature engineering failed")
            return False

    except Exception as e:
        print(f"[ERROR] Feature engineering error: {e}")
        return False

def test_model_training():
    try:
        print("\nTesting model training...")

        from src.train import ModelTrainer

        trainer = ModelTrainer()
        success = trainer.train_all_models()

        if success:
            print("[SUCCESS] Model training successful!")

            model_files = [
                'models/maintenance_classifier.pkl',
                'models/performance_regressor.pkl',
                'models/anomaly_detector.pkl'
            ]

            for file_path in model_files:
                if os.path.exists(file_path):
                    print(f"   [SUCCESS] {file_path} created")
                else:
                    print(f"   [ERROR] {file_path} missing")

            return True
        else:
            print("[ERROR] Model training failed")
            return False

    except Exception as e:
        print(f"[ERROR] Model training error: {e}")
        return False

def test_inference():
    try:
        print("\nTesting inference...")

        from src.inference import ModelInference

        inference = ModelInference()

        sample_data = {
            'energy_output': 5.2,
            'temperature': 32.1,
            'humidity': 55.2,
            'wind_speed': 3.2,
            'solar_irradiance': 850.0,
            'panel_temp': 38.5,
            'voltage': 24.2,
            'current': 21.5,
            'power_factor': 0.98,
            'dust_level': 0.3
        }

        results = inference.predict_all(sample_data)

        if results and 'predictions' in results:
            print("[SUCCESS] Inference successful!")
            print(f"   Predictions generated: {list(results['predictions'].keys())}")
            return True
        else:
            print("[ERROR] Inference failed")
            return False

    except Exception as e:
        print(f"[ERROR] Inference error: {e}")
        return False

def main():
    print("Solar Panel ML System Installation Test")
    print("=" * 50)

    tests = [
        ("Import Test", test_imports),
        ("Data Loading Test", test_data_loading),
        ("Feature Engineering Test", test_feature_engineering),
        ("Model Training Test", test_model_training),
        ("Inference Test", test_inference)
    ]

    results = []

    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"[ERROR] {test_name} failed with exception: {e}")
            results.append((test_name, False))

    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)

    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:<25} {status}")
        if result:
            passed += 1

    print(f"\nTests passed: {passed}/{len(tests)}")

    if passed == len(tests):
        print("\n[SUCCESS] All tests passed! System is ready to use.")
        print("\nTo start the dashboard, run:")
        print("streamlit run src/dashboard.py")
    else:
        print(f"\n[WARNING] {len(tests) - passed} test(s) failed. Please check the errors above.")

    return passed == len(tests)

if __name__ == "__main__":
    main()
