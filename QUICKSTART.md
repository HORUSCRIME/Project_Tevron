# Solar Panel ML Application - Quick Start Guide

## 🚀 Quick Setup (5 minutes)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Automated Setup
```bash
python setup.py
```

This will:
- Create necessary directories
- Generate 10,000 rows of historical data
- Train all ML models
- Run system tests

### 3. Start the Dashboard
```bash
streamlit run src/dashboard.py
```

Navigate to `http://localhost:8501` to access the dashboard.

## 📊 Dashboard Features

### Real-Time Monitor
- Live energy production metrics
- AI-powered maintenance predictions
- Anomaly detection alerts
- System performance gauges

### Historical Analysis
- Energy production trends
- Performance correlations
- Seasonal patterns
- Data quality insights

### Performance Forecasting
- 24-72 hour energy predictions
- Weather-based forecasting
- Peak production analysis

### System Diagnostics
- Model status monitoring
- Data connectivity checks
- System health indicators

## 🔧 Configuration

### API Keys (Optional)
Edit `config.ini`:
```ini
[API]
openweather_api_key = your_api_key_here
```

### Firebase (Optional)
Replace `serviceAccountKey.json` with your Firebase credentials.

## 📈 Analysis Scripts

Run individual analysis scripts:

```bash
# Data profiling report
python analysis/01_data_profiling.py

# Comprehensive visualizations
python analysis/02_visualizations.py

# Correlation analysis
python analysis/03_correlation_analysis.py
```

## 🤖 ML Models

The application includes three trained models:

1. **Maintenance Classifier** - Predicts when maintenance is needed
2. **Performance Regressor** - Forecasts energy output
3. **Anomaly Detector** - Identifies unusual system behavior

## 📁 Project Structure

```
solar_panel_ml/
├── src/                    # Source code
│   ├── dashboard.py        # Streamlit dashboard
│   ├── train.py           # Model training
│   ├── inference.py       # Predictions
│   ├── data_ingestion.py  # Data collection
│   └── feature_engineering.py
├── analysis/              # Analysis scripts
├── data/                  # Historical data
├── models/               # Trained models
├── config.ini           # Configuration
└── requirements.txt     # Dependencies
```

## 🔍 Testing

Verify installation:
```bash
python test_installation.py
```

## 🆘 Troubleshooting

### Common Issues

1. **Import Errors**: Run `pip install -r requirements.txt`
2. **No Data**: Run `python generate_data.py`
3. **Model Errors**: Run `python src/train.py`
4. **Dashboard Issues**: Check Streamlit version: `streamlit --version`

### System Requirements

- Python 3.8+
- 4GB RAM (minimum)
- 1GB disk space
- Internet connection (for weather API)

### Hardware Compatibility

- ✅ PC/Laptop
- ✅ Raspberry Pi 4B (4GB+)
- ✅ NVIDIA Jetson Nano
- ✅ Cloud instances

## 📞 Support

For issues:
1. Check the logs in `logs/` directory
2. Run `python test_installation.py`
3. Verify all files are present
4. Check Python version compatibility

## 🎯 Next Steps

1. **Customize Models**: Modify training parameters in `src/train.py`
2. **Add Features**: Extend feature engineering in `src/feature_engineering.py`
3. **Connect Real Data**: Update data ingestion for your sensors
4. **Deploy**: Use Docker or cloud platforms for production

## 📊 Sample Data

The generated dataset includes:
- 10,000 hourly records
- 13 months of data
- Realistic solar patterns
- Weather correlations
- Maintenance events
- Performance variations

Enjoy monitoring your solar panels with AI! ☀️🤖