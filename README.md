# Solar Panel ML Monitoring System

A comprehensive machine learning application for solar panel monitoring with advanced analytics, predictive maintenance, performance forecasting, and beautiful interactive dashboards. Features 50+ visualizations, 6 EDA modules, and modern UI design.

## Features

### 🚀 Core Capabilities
- **Real-time Monitoring**: Live dashboard with current solar panel metrics
- **Predictive Maintenance**: ML-based maintenance need prediction using LightGBM
- **Performance Forecasting**: Energy output prediction with weather integration
- **Anomaly Detection**: Isolation Forest for detecting unusual patterns
- **Historical Analysis**: Comprehensive data visualization and trend analysis

### 📊 Enhanced Analytics
- **Advanced EDA**: 6 comprehensive analysis modules with 50+ visualizations
- **Interactive Dashboards**: Plotly-based interactive exploration tools
- **Statistical Analysis**: Time series decomposition, correlation networks, PCA
- **Performance Optimization**: Efficiency analysis and optimization recommendations
- **Comprehensive Reports**: Automated HTML reports with actionable insights

### 🎨 Modern UI/UX
- **Enhanced Dashboard**: Beautiful dark theme with animations and gradients
- **3D Visualizations**: Interactive 3D performance analysis
- **Real-time Alerts**: Smart notification system with severity levels
- **Mobile Responsive**: Optimized for desktop, tablet, and mobile devices
- **Professional Design**: Modern CSS with hover effects and smooth transitions

## System Requirements

- Python 3.8+
- Compatible with PC, Raspberry Pi 4B, and Jetson Nano
- Internet connection for weather API and Firebase integration

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd solar_panel_ml
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure settings:
   - Copy `config.ini.template` to `config.ini`
   - Add your OpenWeatherMap API key
   - Configure Firebase credentials in `serviceAccountKey.json`

4. Generate initial data and train models:
```bash
python src/train.py
```

5. Run the dashboard:
```bash
streamlit run src/dashboard.py
```

## Configuration

### config.ini
```ini
[API]
openweather_api_key = your_api_key_here

[LOCATION]
latitude = 11.0168
longitude = 76.9558
city = Coimbatore

[FIREBASE]
database_url = your_firebase_url
```

### Firebase Setup
1. Create a Firebase project
2. Generate service account key
3. Save as `serviceAccountKey.json` in project root
4. Update database URL in config.ini

## Usage

### Quick Start
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate data and train models
python setup.py

# 3. Launch enhanced dashboard
streamlit run src/enhanced_dashboard.py
```

### Dashboard Access
- **Enhanced Dashboard**: `streamlit run src/enhanced_dashboard.py`
- **Original Dashboard**: `streamlit run src/dashboard.py`
- Navigate to `http://localhost:8501` after running
- Use the sidebar to switch between different monitoring pages

### Model Training
```bash
python src/train.py
```

### Comprehensive EDA Analysis
```bash
# Run complete EDA pipeline (recommended)
python run_complete_eda.py

# Or run individual analysis modules
python analysis/01_data_profiling.py          # Data quality & profiling
python analysis/02_visualizations.py          # Statistical visualizations
python analysis/03_correlation_analysis.py    # Correlation analysis
python analysis/04_advanced_eda.py           # Advanced statistical analysis
python analysis/05_interactive_eda.py        # Interactive dashboards
python analysis/06_comprehensive_report.py   # Complete HTML report
```

## Project Structure

```
solar_panel_ml/
├── analysis/                          # Comprehensive EDA modules
│   ├── 01_data_profiling.py          # Data quality & profiling
│   ├── 02_visualizations.py          # Statistical visualizations
│   ├── 03_correlation_analysis.py    # Correlation & feature analysis
│   ├── 04_advanced_eda.py           # Advanced statistical analysis
│   ├── 05_interactive_eda.py        # Interactive dashboard generation
│   ├── 06_comprehensive_report.py   # Complete HTML report generation
│   ├── interactive/                  # Generated interactive dashboards
│   └── reports/                      # Generated analysis reports
├── data/                             # Historical data storage
│   └── historical_data.csv          # 10,000 rows of realistic data
├── models/                           # Trained ML models
│   ├── maintenance_classifier.pkl   # LightGBM maintenance predictor
│   ├── performance_regressor.pkl    # LightGBM performance forecaster
│   └── anomaly_detector.pkl         # Isolation Forest anomaly detector
├── src/                              # Core source code
│   ├── data_ingestion.py            # Firebase & API integration
│   ├── feature_engineering.py       # Advanced feature creation
│   ├── train.py                     # ML model training pipeline
│   ├── inference.py                 # Unified prediction interface
│   ├── dashboard.py                 # Original Streamlit dashboard
│   └── enhanced_dashboard.py        # Enhanced modern dashboard
├── config.ini                        # Configuration settings
├── requirements.txt                  # Python dependencies
├── setup.py                         # Automated setup script
├── run_complete_eda.py              # Master EDA execution script
├── generate_data.py                 # Realistic data generation
├── test_installation.py             # System verification tests
└── QUICKSTART.md                    # Quick start guide
```

## Models

1. **Maintenance Classifier**: Predicts maintenance needs (LightGBM)
2. **Performance Regressor**: Forecasts energy output (LightGBM)
3. **Anomaly Detector**: Identifies unusual patterns (Isolation Forest)

## API Integration

- **OpenWeatherMap**: Real-time weather data
- **Firebase**: Real-time database for sensor data

## Hardware Compatibility

Optimized for edge devices:
- Raspberry Pi 4B (4GB+ recommended)
- NVIDIA Jetson Nano
- Standard PC/Laptop

## Analysis & Monitoring Features

### 📈 Real-time Monitoring
- Live energy production metrics with advanced gauges
- Weather correlation analysis with 3D visualizations
- Smart maintenance alerts with ML predictions
- Performance trends with seasonal decomposition
- Anomaly notifications with severity classification

### 🔍 Advanced Analytics
- **Data Profiling**: Automated data quality assessment with ydata-profiling
- **Statistical Analysis**: Normality tests, outlier detection, correlation networks
- **Time Series Analysis**: Seasonal decomposition, stationarity tests, autocorrelation
- **Performance Analysis**: Efficiency optimization, temperature impact analysis
- **Clustering Analysis**: Operational pattern identification with K-means
- **Anomaly Detection**: Multi-algorithm anomaly detection (Isolation Forest, One-Class SVM)

### 📊 Interactive Visualizations
- **Overview Dashboard**: Comprehensive system overview with 9 interactive charts
- **Time Series Explorer**: Multi-variable time series analysis with maintenance events
- **Performance Analyzer**: Efficiency analysis with dust impact and temperature effects
- **Anomaly Explorer**: Advanced anomaly detection with pattern analysis
- **3D Performance Plots**: Interactive 3D scatter plots and surface visualizations
- **Correlation Networks**: Feature relationship mapping and importance analysis

### 📋 Comprehensive Reporting
- **Executive Summary**: KPI dashboard with key performance indicators
- **Data Quality Report**: Missing data analysis, outlier detection, completeness metrics
- **Performance Analysis**: Hourly/monthly patterns, efficiency trends, optimization opportunities
- **Maintenance Analysis**: Maintenance patterns, triggers, and cost optimization
- **ML Insights**: Model performance, feature importance, predictive capabilities
- **Strategic Recommendations**: Actionable insights with implementation roadmap

## 🎯 Quick Commands

```bash
# Complete setup and analysis
python setup.py                      # Full system setup
python run_complete_eda.py          # Complete EDA pipeline
streamlit run src/enhanced_dashboard.py  # Launch enhanced dashboard

# Individual components
python generate_data.py             # Generate sample data
python src/train.py                 # Train ML models
python test_installation.py        # Verify installation

# Analysis modules
python analysis/01_data_profiling.py     # Data profiling
python analysis/05_interactive_eda.py    # Interactive dashboards
python analysis/06_comprehensive_report.py  # Full report
```

## 📊 Output Files

The system generates numerous analysis outputs:

- **Interactive Dashboards**: `analysis/interactive/index.html`
- **Comprehensive Reports**: `analysis/reports/comprehensive_eda_report_*.html`
- **Data Profiling**: `data_profiling_report.html`
- **Visualizations**: `analysis/*.png` (50+ charts and plots)
- **Execution Logs**: `eda_execution.log`, `training.log`
- **Model Files**: `models/*.pkl` (trained ML models)

## 🔧 System Requirements

- **Python**: 3.8+ (tested on 3.9, 3.10, 3.11)
- **Memory**: 4GB RAM minimum, 8GB recommended
- **Storage**: 2GB free space for data and outputs
- **CPU**: Multi-core recommended for faster analysis
- **Browser**: Modern browser for interactive dashboards

## 🚀 Performance Optimization

- **Caching**: Streamlit caching for faster dashboard loading
- **Parallel Processing**: Multi-threaded analysis where possible
- **Memory Management**: Efficient data handling for large datasets
- **Progressive Loading**: Incremental dashboard updates
- **Optimized Visualizations**: Hardware-accelerated plotting with Plotly

## License

MIT License - see LICENSE file for details

## Support

For issues and questions:
1. Check the comprehensive documentation in generated reports
2. Review execution logs for detailed error information
3. Run `python test_installation.py` to verify system health
4. Create an issue in the repository with detailed information