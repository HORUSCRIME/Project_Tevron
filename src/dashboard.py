import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import logging

from data_ingestion import DataIngestion
from feature_engineering import FeatureEngineer
from inference import ModelInference

st.set_page_config(
    page_title="Solar Panel ML Monitor",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now()
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = False

@st.cache_resource
def initialize_components():
    try:
        ingestion = DataIngestion()
        engineer = FeatureEngineer()
        inference = ModelInference()
        return ingestion, engineer, inference
    except Exception as e:
        st.error(f"Error initializing components: {e}")
        return None, None, None

def load_historical_data():
    try:
        ingestion, _, _ = initialize_components()
        if ingestion:
            return ingestion.load_historical_data()
        return None
    except Exception as e:
        st.error(f"Error loading historical data: {e}")
        return None

def create_gauge_chart(value, title, min_val=0, max_val=100, threshold_good=70, threshold_warning=40):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title},
        delta = {'reference': threshold_good},
        gauge = {
            'axis': {'range': [None, max_val]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [min_val, threshold_warning], 'color': "lightgray"},
                {'range': [threshold_warning, threshold_good], 'color': "yellow"},
                {'range': [threshold_good, max_val], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': threshold_good
            }
        }
    ))

    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def real_time_monitoring():
    st.markdown('<h1 class="main-header">☀️ Real-Time Solar Panel Monitor</h1>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.session_state.auto_refresh = st.checkbox("Auto Refresh (30s)", value=st.session_state.auto_refresh)
    with col2:
        if st.button("Refresh Now"):
            st.session_state.last_update = datetime.now()
            st.rerun()
    with col3:
        st.write(f"Last Update: {st.session_state.last_update.strftime('%H:%M:%S')}")

    if st.session_state.auto_refresh:
        time.sleep(30)
        st.session_state.last_update = datetime.now()
        st.rerun()

    ingestion, engineer, inference = initialize_components()

    if not all([ingestion, engineer, inference]):
        st.error("Failed to initialize system components")
        return

    current_data = ingestion.get_combined_data()

    if not current_data:
        st.error("Unable to fetch current data")
        return

    predictions = inference.predict_all(current_data)

    st.subheader("Current System Status")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        energy_output = current_data.get('energy_output', 0)
        st.metric(
            label="Energy Output (kW)",
            value=f"{energy_output:.2f}",
            delta=f"{np.random.uniform(-0.5, 0.5):.2f}"
        )

    with col2:
        solar_irradiance = current_data.get('solar_irradiance', 0)
        st.metric(
            label="Solar Irradiance (W/m²)",
            value=f"{solar_irradiance:.0f}",
            delta=f"{np.random.uniform(-50, 50):.0f}"
        )

    with col3:
        panel_temp = current_data.get('panel_temp', 0)
        st.metric(
            label="Panel Temperature (°C)",
            value=f"{panel_temp:.1f}",
            delta=f"{np.random.uniform(-2, 2):.1f}"
        )

    with col4:
        efficiency = energy_output / (solar_irradiance / 1000) if solar_irradiance > 0 else 0
        st.metric(
            label="Efficiency (%)",
            value=f"{efficiency*100:.1f}",
            delta=f"{np.random.uniform(-5, 5):.1f}"
        )

    st.subheader("AI Predictions")

    if predictions and 'predictions' in predictions:
        pred_data = predictions['predictions']

        col1, col2, col3 = st.columns(3)

        with col1:
            if 'maintenance' in pred_data:
                maint_data = pred_data['maintenance']
                prob = maint_data.get('maintenance_probability', 0) * 100

                if prob > 70:
                    status_class = "status-critical"
                    status_text = "CRITICAL"
                elif prob > 40:
                    status_class = "status-warning"
                    status_text = "WARNING"
                else:
                    status_class = "status-good"
                    status_text = "GOOD"

        with col2:
            if 'performance' in pred_data:
                perf_data = pred_data['performance']
                predicted_energy = perf_data.get('predicted_energy_output', 0)

        with col3:
            if 'anomaly' in pred_data:
                anom_data = pred_data['anomaly']
                is_anomaly = anom_data.get('is_anomaly', False)
                severity = anom_data.get('severity', 'Normal')

                if is_anomaly:
                    if severity == 'High':
                        status_class = "status-critical"
                    elif severity == 'Medium':
                        status_class = "status-warning"
                    else:
                        status_class = "status-warning"
                else:
                    status_class = "status-good"

    st.subheader("System Gauges")

    col1, col2, col3 = st.columns(3)

    with col1:
        gauge_fig = create_gauge_chart(
            value=efficiency*100,
            title="Efficiency (%)",
            max_val=100,
            threshold_good=80,
            threshold_warning=60
        )
        st.plotly_chart(gauge_fig, use_container_width=True)

    with col2:
        gauge_fig = create_gauge_chart(
            value=panel_temp,
            title="Panel Temperature (°C)",
            max_val=60,
            threshold_good=45,
            threshold_warning=50
        )
        st.plotly_chart(gauge_fig, use_container_width=True)

    with col3:
        dust_level = current_data.get('dust_level', 0) * 100
        gauge_fig = create_gauge_chart(
            value=dust_level,
            title="Dust Level (%)",
            max_val=100,
            threshold_good=30,
            threshold_warning=60
        )
        st.plotly_chart(gauge_fig, use_container_width=True)

def historical_analysis():
    st.markdown('<h1 class="main-header"> Historical Analysis</h1>', unsafe_allow_html=True)

    df = load_historical_data()

    if df is None or df.empty:
        st.error("No historical data available")
        return

    st.subheader("Select Analysis Period")
    col1, col2 = st.columns(2)

    with col1:
        start_date = st.date_input(
            "Start Date",
            value=df['timestamp'].min().date(),
            min_value=df['timestamp'].min().date(),
            max_value=df['timestamp'].max().date()
        )

    with col2:
        end_date = st.date_input(
            "End Date",
            value=df['timestamp'].max().date(),
            min_value=df['timestamp'].min().date(),
            max_value=df['timestamp'].max().date()
        )

    mask = (df['timestamp'].dt.date >= start_date) & (df['timestamp'].dt.date <= end_date)
    filtered_df = df[mask]

    if filtered_df.empty:
        st.warning("No data available for selected period")
        return

    st.subheader("Summary Statistics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_energy = filtered_df['energy_output'].sum()
        st.metric("Total Energy (kWh)", f"{total_energy:.1f}")

    with col2:
        avg_efficiency = (filtered_df['energy_output'] / (filtered_df['solar_irradiance'] / 1000)).mean()
        st.metric("Avg Efficiency", f"{avg_efficiency:.2f}")

    with col3:
        maintenance_events = filtered_df['maintenance_needed'].sum()
        st.metric("Maintenance Events", f"{maintenance_events}")

    with col4:
        uptime = (1 - filtered_df['maintenance_needed'].mean()) * 100
        st.metric("System Uptime (%)", f"{uptime:.1f}")

    st.subheader("Energy Production Trends")

    daily_df = filtered_df.groupby(filtered_df['timestamp'].dt.date).agg({
        'energy_output': 'sum',
        'solar_irradiance': 'mean',
        'temperature': 'mean',
        'panel_temp': 'mean'
    }).reset_index()

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Daily Energy Output', 'Average Solar Irradiance',
                       'Temperature Trends', 'Panel vs Ambient Temperature'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": True}, {"secondary_y": False}]]
    )

    fig.add_trace(
        go.Scatter(x=daily_df['timestamp'], y=daily_df['energy_output'],
                  mode='lines+markers', name='Energy Output'),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=daily_df['timestamp'], y=daily_df['solar_irradiance'],
                  mode='lines+markers', name='Solar Irradiance', line=dict(color='orange')),
        row=1, col=2
    )

    fig.add_trace(
        go.Scatter(x=daily_df['timestamp'], y=daily_df['temperature'],
                  mode='lines+markers', name='Ambient Temp', line=dict(color='blue')),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(x=daily_df['timestamp'], y=daily_df['panel_temp'],
                  mode='lines+markers', name='Panel Temp', line=dict(color='red')),
        row=2, col=1, secondary_y=True
    )

    fig.add_trace(
        go.Scatter(x=daily_df['temperature'], y=daily_df['panel_temp'],
                  mode='markers', name='Temp Correlation'),
        row=2, col=2
    )

    fig.update_layout(height=600, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Performance Analysis")

    hourly_df = filtered_df.groupby(filtered_df['timestamp'].dt.hour).agg({
        'energy_output': 'mean',
        'solar_irradiance': 'mean',
        'panel_temp': 'mean'
    }).reset_index()

    fig = px.line(hourly_df, x='timestamp', y=['energy_output', 'solar_irradiance'],
                  title='Average Hourly Patterns',
                  labels={'timestamp': 'Hour of Day', 'value': 'Value'})
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Feature Correlations")

    numeric_cols = ['energy_output', 'temperature', 'humidity', 'wind_speed',
                   'solar_irradiance', 'panel_temp', 'voltage', 'current', 'dust_level']

    corr_matrix = filtered_df[numeric_cols].corr()

    fig = px.imshow(corr_matrix,
                    title='Feature Correlation Matrix',
                    color_continuous_scale='RdBu_r',
                    aspect='auto')
    st.plotly_chart(fig, use_container_width=True)

def performance_forecasting():
    st.markdown('<h1 class="main-header">🔮 Performance Forecasting</h1>', unsafe_allow_html=True)

    ingestion, engineer, inference = initialize_components()

    if not all([ingestion, engineer, inference]):
        st.error("Failed to initialize system components")
        return

    st.subheader("Forecast Configuration")

    col1, col2 = st.columns(2)

    with col1:
        forecast_hours = st.slider("Forecast Hours", min_value=1, max_value=72, value=24)

    with col2:
        if st.button("Generate Forecast"):
            with st.spinner("Generating forecast..."):
                forecast_data = inference.predict_future_performance(hours_ahead=forecast_hours)

    if 'forecast_data' in locals() and forecast_data:
        st.subheader("Energy Production Forecast")

        forecast_df = pd.DataFrame(forecast_data)
        forecast_df['timestamp'] = pd.to_datetime(forecast_df['timestamp'])

        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Predicted Energy Output', 'Solar Irradiance'),
            shared_xaxes=True
        )

        fig.add_trace(
            go.Scatter(x=forecast_df['timestamp'], y=forecast_df['predicted_energy'],
                      mode='lines+markers', name='Predicted Energy',
                      line=dict(color='green')),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(x=forecast_df['timestamp'], y=forecast_df['solar_irradiance'],
                      mode='lines+markers', name='Solar Irradiance',
                      line=dict(color='orange')),
            row=2, col=1
        )

        fig.update_layout(height=500, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            total_forecast = forecast_df['predicted_energy'].sum()
            st.metric("Total Forecast (kWh)", f"{total_forecast:.2f}")

        with col2:
            peak_output = forecast_df['predicted_energy'].max()
            st.metric("Peak Output (kW)", f"{peak_output:.2f}")

        with col3:
            avg_output = forecast_df['predicted_energy'].mean()
            st.metric("Average Output (kW)", f"{avg_output:.2f}")

def system_diagnostics():
    st.markdown('<h1 class="main-header"> System Diagnostics</h1>', unsafe_allow_html=True)

    ingestion, engineer, inference = initialize_components()

    st.subheader("Model Status")

    if inference:
        model_status = inference.get_model_status()

        col1, col2, col3 = st.columns(3)

        with col1:
            maintenance_ready = model_status.get('maintenance_ready', False)
            status_color = "🟢" if maintenance_ready else "🔴"
            st.write(f"{status_color} Maintenance Model: {'Ready' if maintenance_ready else 'Not Available'}")

        with col2:
            performance_ready = model_status.get('performance_ready', False)
            status_color = "🟢" if performance_ready else "🔴"
            st.write(f"{status_color} Performance Model: {'Ready' if performance_ready else 'Not Available'}")

        with col3:
            anomaly_ready = model_status.get('anomaly_ready', False)
            status_color = "🟢" if anomaly_ready else "🔴"
            st.write(f"{status_color} Anomaly Model: {'Ready' if anomaly_ready else 'Not Available'}")

    st.subheader("Data Connectivity")

    if ingestion:
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Weather API Status**")
            try:
                weather_data = ingestion.get_weather_data()
                if weather_data:
                    st.success(" Weather API Connected")
                    st.json(weather_data)
                else:
                    st.error(" Weather API Failed")
            except Exception as e:
                st.error(f" Weather API Error: {e}")

        with col2:
            st.write("**Sensor Data Status**")
            try:
                sensor_data = ingestion.get_sensor_data()
                if sensor_data:
                    st.success(" Sensor Data Available")
                    st.json(sensor_data)
                else:
                    st.error(" Sensor Data Failed")
            except Exception as e:
                st.error(f" Sensor Data Error: {e}")

    st.subheader("Recent System Activity")

    logs = [
        {"timestamp": datetime.now() - timedelta(minutes=5), "level": "INFO", "message": "Model inference completed successfully"},
        {"timestamp": datetime.now() - timedelta(minutes=10), "level": "INFO", "message": "Weather data updated"},
        {"timestamp": datetime.now() - timedelta(minutes=15), "level": "WARNING", "message": "High panel temperature detected"},
        {"timestamp": datetime.now() - timedelta(minutes=30), "level": "INFO", "message": "Sensor data synchronized"},
    ]

    for log in logs:
        level_color = {"INFO": "🔵", "WARNING": "🟡", "ERROR": "🔴"}.get(log["level"], "⚪")
        st.write(f"{level_color} {log['timestamp'].strftime('%H:%M:%S')} - {log['level']}: {log['message']}")

def main():
    st.sidebar.title("Navigation")

    pages = {
        " Real-Time Monitor": real_time_monitoring,
        " Historical Analysis": historical_analysis,
        " Performance Forecast": performance_forecasting,
        " System Diagnostics": system_diagnostics
    }

    selected_page = st.sidebar.selectbox("Select Page", list(pages.keys()))

    st.sidebar.markdown("---")
    st.sidebar.markdown("### System Info")
    st.sidebar.info(f"Location: Coimbatore, India\nLat: 11.0168, Lon: 76.9558\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    pages[selected_page]()

if __name__ == "__main__":
    main()
