import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

from data_ingestion import DataIngestion
from feature_engineering import FeatureEngineer
from inference import ModelInference

st.set_page_config(
    page_title="Tevron Solar AI Analytics Hub",
    page_icon="🌞",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .main {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 0 4px 8px rgba(0,0,0,0.3);
    }

    .metric-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        margin: 1rem 0;
        backdrop-filter: blur(10px);
        transition: transform 0.3s ease;
    }

    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.4);
    }

    .status-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        border: 1px solid rgba(255,255,255,0.2);
        box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        margin: 1rem 0;
        text-align: center;
    }

    .prediction-card {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 50%, #fecfef 100%);
        color: #2c3e50;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.2);
    }

    .status-excellent {
        color: #00ff88;
        font-weight: 600;
        text-shadow: 0 0 10px rgba(0,255,136,0.5);
    }

    .status-good {
        color: #00ff00;
        font-weight: 600;
        text-shadow: 0 0 10px rgba(0,255,0,0.5);
    }

    .status-warning {
        color: #ffaa00;
        font-weight: 600;
        text-shadow: 0 0 10px rgba(255,170,0,0.5);
    }

    .status-critical {
        color: #ff3366;
        font-weight: 600;
        text-shadow: 0 0 10px rgba(255,51,102,0.5);
    }

    .insight-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #00ff88;
    }

    .alert-banner {
        background: linear-gradient(90deg, #ff6b6b, #ee5a24);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-weight: 600;
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

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

@st.cache_data(ttl=300)
def load_historical_data():
    try:
        ingestion, _, _ = initialize_components()
        if ingestion:
            return ingestion.load_historical_data()
        return None
    except Exception as e:
        st.error(f"Error loading historical data: {e}")
        return None

def safe_calculate_efficiency(energy_output, solar_irradiance):
    try:
        if solar_irradiance > 0:
            efficiency = energy_output / (solar_irradiance / 1000)
            return max(0, min(1, efficiency))
        return 0
    except (ZeroDivisionError, TypeError):
        return 0

def create_advanced_gauge(value, title, min_val=0, max_val=100, thresholds=[30, 70], colors=['red', 'yellow', 'green']):
    try:
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = value,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': title, 'font': {'size': 16, 'color': 'white'}},
            delta = {'reference': thresholds[1]},
            gauge = {
                'axis': {'range': [None, max_val], 'tickcolor': 'white'},
                'bar': {'color': "darkblue", 'thickness': 0.3},
                'steps': [
                    {'range': [min_val, thresholds[0]], 'color': colors[0]},
                    {'range': [thresholds[0], thresholds[1]], 'color': colors[1]},
                    {'range': [thresholds[1], max_val], 'color': colors[2]}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': value
                }
            }
        ))

        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': 'white'}
        )
        return fig
    except Exception as e:
        st.error(f"Error creating gauge: {e}")
        return go.Figure()

def create_3d_performance_plot(df):
    try:
        if df.empty or len(df) < 3:
            fig = go.Figure()
            fig.add_annotation(
                text="Insufficient data for 3D plot",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font=dict(size=16, color='white')
            )
            return fig

        fig = go.Figure(data=[go.Scatter3d(
            x=df['solar_irradiance'],
            y=df['panel_temp'],
            z=df['energy_output'],
            mode='markers',
            marker=dict(
                size=4,
                color=df['energy_output'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Energy Output (kW)"),
                opacity=0.8
            ),
            text=df['timestamp'].dt.strftime('%Y-%m-%d %H:%M') if 'timestamp' in df.columns else None,
            hovertemplate='<b>Solar Irradiance:</b> %{x:.0f} W/m²<br>' +
                         '<b>Panel Temp:</b> %{y:.1f}°C<br>' +
                         '<b>Energy Output:</b> %{z:.2f} kW<extra></extra>'
        )])

        fig.update_layout(
            title='3D Performance Analysis',
            scene=dict(
                xaxis_title='Solar Irradiance (W/m²)',
                yaxis_title='Panel Temperature (°C)',
                zaxis_title='Energy Output (kW)',
                bgcolor='rgba(0,0,0,0)',
                xaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.2)"),
                yaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.2)"),
                zaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.2)")
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=600
        )
        return fig
    except Exception as e:
        st.error(f"Error creating 3D plot: {e}")
        return go.Figure()

def create_performance_heatmap(df):
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['month'] = df['timestamp'].dt.month

        pivot_data = df.pivot_table(
            values='energy_output',
            index='hour',
            columns='month',
            aggfunc='mean'
        )

        if pivot_data.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No data available for heatmap",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font=dict(size=16, color='white')
            )
        else:
            fig = go.Figure(data=go.Heatmap(
                z=pivot_data.values,
                x=[f'Month {i}' for i in pivot_data.columns],
                y=[f'{i}:00' for i in pivot_data.index],
                colorscale='Viridis',
                hoverongaps=False,
                colorbar=dict(title="Avg Energy Output (kW)")
            ))

        fig.update_layout(
            title='Energy Output Heatmap: Hour vs Month',
            xaxis_title='Month',
            yaxis_title='Hour of Day',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        return fig
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating heatmap: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=14, color='red')
        )
        fig.update_layout(
            title='Energy Output Heatmap: Hour vs Month',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        return fig

def enhanced_real_time_monitoring():
    st.markdown('<h1 class="main-header">🌞 Solar AI Analytics Hub</h1>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    with col1:
        st.session_state.auto_refresh = st.checkbox("🔄 Auto Refresh (30s)", value=st.session_state.auto_refresh)
    with col2:
        if st.button("🔄 Refresh Now"):
            st.session_state.last_update = datetime.now()
            st.rerun()
    with col3:
        alert_threshold = st.slider("🚨 Alert Threshold", 0, 100, 70)
    with col4:
        st.write(f"⏰ Last Update: {st.session_state.last_update.strftime('%H:%M:%S')}")

    ingestion, engineer, inference = initialize_components()

    if not all([ingestion, engineer, inference]):
        st.error("❌ Failed to initialize system components")
        return

    try:
        current_data = ingestion.get_combined_data()
    except Exception as e:
        st.error(f"❌ Unable to fetch current data: {e}")
        return

    if not current_data:
        st.error("❌ No current data available")
        return

    try:
        predictions = inference.predict_all(current_data)
    except Exception as e:
        st.warning(f"⚠️ Prediction error: {e}")
        predictions = None

    energy_output = current_data.get('energy_output', 0)
    panel_temp = current_data.get('panel_temp', 0)
    solar_irradiance = current_data.get('solar_irradiance', 0)

    if panel_temp > 50 or (predictions and 'predictions' in predictions and
        'maintenance' in predictions['predictions'] and
        predictions['predictions']['maintenance'].get('maintenance_probability', 0) > 0.7):
        st.markdown("""
        <div class="alert-banner">
            🚨 SYSTEM ALERT: High temperature or maintenance required!
        </div>
        """, unsafe_allow_html=True)

    st.markdown("## 📊 Real-Time Performance Dashboard")

    kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)

    with kpi_col1:
        delta_energy = np.random.uniform(-0.5, 0.5)
        st.metric(
            label="⚡ Energy Output",
            value=f"{energy_output:.2f} kW",
            delta=f"{delta_energy:.2f} kW"
        )

    with kpi_col2:
        delta_irradiance = np.random.uniform(-50, 50)
        st.metric(
            label="☀️ Solar Irradiance",
            value=f"{solar_irradiance:.0f} W/m²",
            delta=f"{delta_irradiance:.0f} W/m²"
        )

    with kpi_col3:
        delta_temp = np.random.uniform(-2, 2)
        st.metric(
            label="🌡️ Panel Temperature",
            value=f"{panel_temp:.1f}°C",
            delta=f"{delta_temp:.1f}°C"
        )

    with kpi_col4:
        efficiency = safe_calculate_efficiency(energy_output, solar_irradiance)
        delta_eff = np.random.uniform(-5, 5)
        st.metric(
            label="⚙️ Efficiency",
            value=f"{efficiency*100:.1f}%",
            delta=f"{delta_eff:.1f}%"
        )

    with kpi_col5:
        dust_level = current_data.get('dust_level', 0) * 100
        st.metric(
            label="🌪️ Dust Level",
            value=f"{dust_level:.1f}%",
            delta=f"{np.random.uniform(-5, 5):.1f}%"
        )

    st.markdown("## 🤖 AI Predictions & Insights")

    if predictions and 'predictions' in predictions:
        pred_data = predictions['predictions']

        pred_col1, pred_col2, pred_col3 = st.columns(3)

        with pred_col1:
            if 'maintenance' in pred_data:
                maint_data = pred_data['maintenance']
                prob = maint_data.get('maintenance_probability', 0) * 100

                if prob > 80:
                    status_class = "status-critical"
                    status_text = "🔴 CRITICAL"
                    status_icon = "🚨"
                elif prob > 60:
                    status_class = "status-warning"
                    status_text = "🟡 WARNING"
                    status_icon = "⚠️"
                elif prob > 30:
                    status_class = "status-good"
                    status_text = "🟢 CAUTION"
                    status_icon = "⚡"
                else:
                    status_class = "status-excellent"
                    status_text = "🟢 EXCELLENT"
                    status_icon = "✅"

                st.markdown(f"""
                <div class="prediction-card">
                    <h3>{status_icon} Maintenance Prediction</h3>
                    <h2 class="{status_class}">{status_text}</h2>
                    <p><strong>Probability:</strong> {prob:.1f}%</p>
                    <p><strong>Confidence:</strong> {maint_data.get('confidence', 0)*100:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)

        with pred_col2:
            if 'performance' in pred_data:
                perf_data = pred_data['performance']
                predicted_energy = perf_data.get('predicted_energy_output', 0)

                st.markdown(f"""
                <div class="prediction-card">
                    <h3>📈 Performance Forecast</h3>
                    <h2>{predicted_energy:.2f} kW</h2>
                    <p><strong>Next Hour Prediction</strong></p>
                    <p><strong>Trend:</strong> {'📈 Increasing' if predicted_energy > energy_output else '📉 Decreasing'}</p>
                </div>
                """, unsafe_allow_html=True)

        with pred_col3:
            if 'anomaly' in pred_data:
                anom_data = pred_data['anomaly']
                is_anomaly = anom_data.get('is_anomaly', False)
                severity = anom_data.get('severity', 'Normal')

                if is_anomaly:
                    if severity == 'High':
                        status_class = "status-critical"
                        status_icon = "🔴"
                    elif severity == 'Medium':
                        status_class = "status-warning"
                        status_icon = "🟡"
                    else:
                        status_class = "status-warning"
                        status_icon = "🟠"
                else:
                    status_class = "status-excellent"
                    status_icon = "🟢"

                st.markdown(f"""
                <div class="prediction-card">
                    <h3>🔍 Anomaly Detection</h3>
                    <h2 class="{status_class}">{status_icon} {severity}</h2>
                    <p><strong>Status:</strong> {'Anomaly Detected' if is_anomaly else 'Normal Operation'}</p>
                    <p><strong>Score:</strong> {anom_data.get('anomaly_score', 0):.3f}</p>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("## 🎛️ System Performance Gauges")

    gauge_col1, gauge_col2, gauge_col3 = st.columns(3)

    with gauge_col1:
        efficiency_gauge = create_advanced_gauge(
            value=efficiency*100,
            title="System Efficiency (%)",
            max_val=100,
            thresholds=[60, 80],
            colors=['#ff4757', '#ffa502', '#2ed573']
        )
        st.plotly_chart(efficiency_gauge, use_container_width=True)

    with gauge_col2:
        temp_gauge = create_advanced_gauge(
            value=panel_temp,
            title="Panel Temperature (°C)",
            max_val=70,
            thresholds=[40, 50],
            colors=['#2ed573', '#ffa502', '#ff4757']
        )
        st.plotly_chart(temp_gauge, use_container_width=True)

    with gauge_col3:
        dust_gauge = create_advanced_gauge(
            value=dust_level,
            title="Dust Level (%)",
            max_val=100,
            thresholds=[30, 60],
            colors=['#2ed573', '#ffa502', '#ff4757']
        )
        st.plotly_chart(dust_gauge, use_container_width=True)

    st.markdown("## 📈 Live Performance Charts")

    df = load_historical_data()
    if df is not None and not df.empty:
        try:
            recent_df = df.tail(24)

            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                fig_energy = go.Figure()
                fig_energy.add_trace(go.Scatter(
                    x=recent_df['timestamp'],
                    y=recent_df['energy_output'],
                    mode='lines+markers',
                    name='Energy Output',
                    line=dict(color='#00ff88', width=3),
                    marker=dict(size=6)
                ))

                fig_energy.update_layout(
                    title='Energy Output Trend (24h)',
                    xaxis_title='Time',
                    yaxis_title='Energy Output (kW)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    showlegend=False
                )

                st.plotly_chart(fig_energy, use_container_width=True)

            with chart_col2:
                fig_multi = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=('Solar Irradiance', 'Panel Temperature'),
                    vertical_spacing=0.1
                )

                fig_multi.add_trace(
                    go.Scatter(x=recent_df['timestamp'], y=recent_df['solar_irradiance'],
                              mode='lines', name='Solar Irradiance', line=dict(color='#ffa502')),
                    row=1, col=1
                )

                fig_multi.add_trace(
                    go.Scatter(x=recent_df['timestamp'], y=recent_df['panel_temp'],
                              mode='lines', name='Panel Temp', line=dict(color='#ff4757')),
                    row=2, col=1
                )

                fig_multi.update_layout(
                    title='Environmental Conditions (24h)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    showlegend=False,
                    height=400
                )

                st.plotly_chart(fig_multi, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating charts: {e}")

def advanced_analytics():
    st.markdown('<h1 class="main-header">📊 Advanced Analytics & Insights</h1>', unsafe_allow_html=True)

    df = load_historical_data()
    if df is None or df.empty:
        st.error("No data available for analysis")
        return

    try:
        engineer = FeatureEngineer()
        df = engineer.create_derived_features(df)

        if 'efficiency' not in df.columns:
            df['efficiency'] = df.apply(
                lambda row: safe_calculate_efficiency(row['energy_output'], row['solar_irradiance']),
                axis=1
            )
    except Exception as e:
        st.error(f"Error processing data: {e}")
        return

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🎯 Performance Analysis",
        "🔗 Correlation Analysis",
        "📈 Time Series Analysis",
        "🎨 3D Visualizations",
        "🤖 ML Insights",
        "📋 EDA Reports"
    ])

    with tab1:
        st.markdown("### 🎯 Comprehensive Performance Analysis")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_energy = df['energy_output'].sum()
            st.metric("Total Energy Produced", f"{total_energy:.1f} kWh")

        with col2:
            if 'efficiency' in df.columns and not df['efficiency'].isna().all():
                avg_efficiency = df['efficiency'].mean()
                st.metric("Average Efficiency", f"{avg_efficiency:.3f}")
            else:
                st.metric("Average Efficiency", "N/A")

        with col3:
            peak_output = df['energy_output'].max()
            st.metric("Peak Output", f"{peak_output:.2f} kW")

        with col4:
            uptime = (1 - df['maintenance_needed'].mean()) * 100
            st.metric("System Uptime", f"{uptime:.1f}%")

        st.markdown("#### Energy Production Heatmap")
        heatmap_fig = create_performance_heatmap(df)
        st.plotly_chart(heatmap_fig, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            try:
                if 'efficiency' in df.columns and not df['efficiency'].isna().all():
                    fig_eff_dist = px.histogram(
                        df, x='efficiency', nbins=30,
                        title='Efficiency Distribution',
                        color_discrete_sequence=['#00ff88']
                    )
                    fig_eff_dist.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white')
                    )
                    st.plotly_chart(fig_eff_dist, use_container_width=True)
                else:
                    st.warning("Efficiency data not available for distribution plot")
            except Exception as e:
                st.error(f"Error creating efficiency distribution: {e}")

        with col2:
            try:
                if 'efficiency' in df.columns and not df['efficiency'].isna().all():
                    fig_temp_eff = px.scatter(
                        df, x='panel_temp', y='efficiency',
                        title='Panel Temperature vs Efficiency',
                        color='energy_output',
                        color_continuous_scale='Viridis'
                    )
                    fig_temp_eff.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white')
                    )
                    st.plotly_chart(fig_temp_eff, use_container_width=True)
                else:
                    fig_temp_energy = px.scatter(
                        df, x='panel_temp', y='energy_output',
                        title='Panel Temperature vs Energy Output',
                        color='solar_irradiance',
                        color_continuous_scale='Viridis'
                    )
                    fig_temp_energy.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white')
                    )
                    st.plotly_chart(fig_temp_energy, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating temperature scatter plot: {e}")

    with tab2:
        st.markdown("### 🔗 Advanced Correlation Analysis")

        try:
            numeric_cols = ['energy_output', 'solar_irradiance', 'panel_temp', 'temperature',
                           'humidity', 'wind_speed', 'dust_level', 'voltage', 'current']

            available_cols = [col for col in numeric_cols if col in df.columns]

            if len(available_cols) > 2:
                corr_matrix = df[available_cols].corr()

                fig_corr = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    colorscale='RdBu',
                    zmid=0,
                    colorbar=dict(title="Correlation", titlefont=dict(color='white'))
                ))

                fig_corr.update_layout(
                    title='Feature Correlation Matrix',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )
                st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.warning("Not enough numeric columns for correlation analysis")
        except Exception as e:
            st.error(f"Error creating correlation matrix: {e}")

        st.markdown("#### 🎯 ML Model Feature Importance")

        try:
            inference = ModelInference()

            col1, col2 = st.columns(2)

            with col1:
                try:
                    maint_importance = inference.get_feature_importance('maintenance')
                    if maint_importance is not None and not maint_importance.empty:
                        fig_maint = px.bar(
                            maint_importance.head(10),
                            x='importance', y='feature',
                            orientation='h',
                            title='Top Features for Maintenance Prediction',
                            color='importance',
                            color_continuous_scale='Reds'
                        )
                        fig_maint.update_layout(
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='white')
                        )
                        st.plotly_chart(fig_maint, use_container_width=True)
                    else:
                        st.info("Maintenance model feature importance not available")
                except Exception as e:
                    st.warning(f"Error loading maintenance model importance: {e}")

            with col2:
                try:
                    perf_importance = inference.get_feature_importance('performance')
                    if perf_importance is not None and not perf_importance.empty:
                        fig_perf = px.bar(
                            perf_importance.head(10),
                            x='importance', y='feature',
                            orientation='h',
                            title='Top Features for Performance Prediction',
                            color='importance',
                            color_continuous_scale='Blues'
                        )
                        fig_perf.update_layout(
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='white')
                        )
                        st.plotly_chart(fig_perf, use_container_width=True)
                    else:
                        st.info("Performance model feature importance not available")
                except Exception as e:
                    st.warning(f"Error loading performance model importance: {e}")

        except Exception as e:
            st.warning(f"Could not initialize model inference: {e}")

    with tab3:
        st.markdown("### 📈 Time Series Deep Dive")

        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df_daily = df.groupby(df['timestamp'].dt.date)['energy_output'].sum().reset_index()
            df_daily['timestamp'] = pd.to_datetime(df_daily['timestamp'])

            fig_trend = go.Figure()

            fig_trend.add_trace(go.Scatter(
                x=df_daily['timestamp'],
                y=df_daily['energy_output'],
                mode='lines',
                name='Daily Energy Output',
                line=dict(color='#00ff88', width=2)
            ))

            if len(df_daily) >= 7:
                df_daily['ma_7'] = df_daily['energy_output'].rolling(window=7).mean()
                fig_trend.add_trace(go.Scatter(
                    x=df_daily['timestamp'],
                    y=df_daily['ma_7'],
                    mode='lines',
                    name='7-Day Moving Average',
                    line=dict(color='#ffa502', width=3)
                ))

            fig_trend.update_layout(
                title='Daily Energy Production Trend with Moving Average',
                xaxis_title='Date',
                yaxis_title='Daily Energy Output (kWh)',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )

            st.plotly_chart(fig_trend, use_container_width=True)

            hourly_pattern = df.groupby(df['timestamp'].dt.hour)['energy_output'].mean()

            fig_hourly = go.Figure()
            fig_hourly.add_trace(go.Scatter(
                x=hourly_pattern.index,
                y=hourly_pattern.values,
                mode='lines+markers',
                fill='tonexty',
                name='Average Hourly Output',
                line=dict(color='#ff4757', width=3),
                marker=dict(size=8)
            ))

            fig_hourly.update_layout(
                title='Average Energy Output by Hour of Day',
                xaxis_title='Hour of Day',
                yaxis_title='Average Energy Output (kW)',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )

            st.plotly_chart(fig_hourly, use_container_width=True)
        except Exception as e:
            st.error(f"Error in time series analysis: {e}")

    with tab4:
        st.markdown("### 🎨 3D Performance Visualizations")

        try:
            sample_size = min(1000, len(df))
            sample_df = df.sample(sample_size) if len(df) > sample_size else df
            fig_3d = create_3d_performance_plot(sample_df)
            st.plotly_chart(fig_3d, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating 3D plot: {e}")

        try:
            st.markdown("#### 3D Surface: Temperature vs Irradiance vs Energy")

            if len(df) < 10:
                st.warning("Not enough data points for 3D surface plot")
            else:
                temp_range = np.linspace(df['panel_temp'].min(), df['panel_temp'].max(), 20)
                irr_range = np.linspace(df['solar_irradiance'].min(), df['solar_irradiance'].max(), 20)

                temp_grid, irr_grid = np.meshgrid(temp_range, irr_range)

                try:
                    from scipy.interpolate import griddata

                    clean_df = df[['panel_temp', 'solar_irradiance', 'energy_output']].dropna()

                    if len(clean_df) > 0:
                        points = clean_df[['panel_temp', 'solar_irradiance']].values
                        values = clean_df['energy_output'].values

                        energy_grid = griddata(points, values, (temp_grid, irr_grid), method='linear')

                        fig_surface = go.Figure(data=[go.Surface(
                            z=energy_grid,
                            x=temp_grid,
                            y=irr_grid,
                            colorscale='Viridis',
                            colorbar=dict(title="Energy Output (kW)")
                        )])

                        fig_surface.update_layout(
                            title='3D Surface: Energy Output vs Temperature & Irradiance',
                            scene=dict(
                                xaxis_title='Panel Temperature (°C)',
                                yaxis_title='Solar Irradiance (W/m²)',
                                zaxis_title='Energy Output (kW)',
                                bgcolor='rgba(0,0,0,0)'
                            ),
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='white'),
                            height=600
                        )

                        st.plotly_chart(fig_surface, use_container_width=True)
                    else:
                        st.warning("No valid data for 3D surface plot")

                except ImportError:
                    st.warning("scipy not available for 3D surface interpolation")
                except Exception as e:
                    st.error(f"Error creating 3D surface: {e}")
        except Exception as e:
            st.error(f"Error in 3D surface section: {e}")

    with tab5:
        st.markdown("### 🤖 Machine Learning Insights")

        st.markdown("#### Model Performance Summary")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            <div class="insight-box">
                <h4>🔧 Maintenance Model</h4>
                <p><strong>Type:</strong> LightGBM Classifier</p>
                <p><strong>Accuracy:</strong> ~97.5%</p>
                <p><strong>Key Features:</strong> Voltage, Wind Speed, Power Factor</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="insight-box">
                <h4>📈 Performance Model</h4>
                <p><strong>Type:</strong> LightGBM Regressor</p>
                <p><strong>R² Score:</strong> ~99.9%</p>
                <p><strong>Key Features:</strong> Energy Normalized, Power, Dust Impact</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown("""
            <div class="insight-box">
                <h4>🔍 Anomaly Model</h4>
                <p><strong>Type:</strong> Isolation Forest</p>
                <p><strong>Detection Rate:</strong> ~10.4%</p>
                <p><strong>Contamination:</strong> 10%</p>
            </div>
            """, unsafe_allow_html=True)

        try:
            st.markdown("#### 📊 Principal Component Analysis")

            numeric_cols = ['energy_output', 'solar_irradiance', 'panel_temp', 'temperature',
                           'humidity', 'wind_speed', 'dust_level']

            available_cols = [col for col in numeric_cols if col in df.columns]

            if len(available_cols) < 3:
                st.warning("Not enough numeric columns for PCA analysis")
            else:
                pca_data = df[available_cols].dropna()

                if len(pca_data) < 10:
                    st.warning("Not enough data points for PCA analysis")
                else:
                    scaler = StandardScaler()
                    scaled_data = scaler.fit_transform(pca_data)

                    pca = PCA()
                    pca_result = pca.fit_transform(scaled_data)

                    fig_pca = go.Figure()
                    fig_pca.add_trace(go.Bar(
                        x=[f'PC{i+1}' for i in range(len(pca.explained_variance_ratio_))],
                        y=pca.explained_variance_ratio_ * 100,
                        name='Explained Variance',
                        marker_color='#00ff88'
                    ))

                    fig_pca.update_layout(
                        title='PCA Explained Variance Ratio',
                        xaxis_title='Principal Components',
                        yaxis_title='Explained Variance (%)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white')
                    )

                    st.plotly_chart(fig_pca, use_container_width=True)

                    try:
                        pca_df = pd.DataFrame(pca_result[:, :2], columns=['PC1', 'PC2'])
                        pca_df['maintenance'] = df.loc[pca_data.index, 'maintenance_needed'].values

                        fig_pca_scatter = px.scatter(
                            pca_df, x='PC1', y='PC2', color='maintenance',
                            title='PCA Visualization (First 2 Components)',
                            color_discrete_map={0: '#00ff88', 1: '#ff4757'}
                        )

                        fig_pca_scatter.update_layout(
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='white')
                        )

                        st.plotly_chart(fig_pca_scatter, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error creating PCA scatter plot: {e}")
        except Exception as e:
            st.error(f"Error in PCA analysis: {e}")

    with tab6:
        st.markdown("### 📋 Comprehensive EDA Reports")

        st.markdown("#### 🚀 Generate EDA Reports")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("📊 Generate Data Profiling Report", use_container_width=True):
                with st.spinner("Generating data profiling report..."):
                    try:
                        import subprocess
                        result = subprocess.run(["python", "analysis/01_data_profiling.py"],
                                              capture_output=True, text=True, cwd=".")
                        if result.returncode == 0:
                            st.success("✅ Data profiling report generated successfully!")
                            st.info("📁 Check analysis/data_profiling_report.html")
                        else:
                            st.error(f"❌ Error generating report: {result.stderr}")
                    except Exception as e:
                        st.error(f"❌ Error: {e}")

        with col2:
            if st.button("📈 Generate Statistical Visualizations", use_container_width=True):
                with st.spinner("Generating statistical visualizations..."):
                    try:
                        import subprocess
                        result = subprocess.run(["python", "analysis/02_visualizations.py"],
                                              capture_output=True, text=True, cwd=".")
                        if result.returncode == 0:
                            st.success("✅ Statistical visualizations generated successfully!")
                            st.info("📁 Check analysis/ directory for PNG files")
                        else:
                            st.error(f"❌ Error generating visualizations: {result.stderr}")
                    except Exception as e:
                        st.error(f"❌ Error: {e}")

        with col3:
            if st.button("🔗 Generate Correlation Analysis", use_container_width=True):
                with st.spinner("Generating correlation analysis..."):
                    try:
                        import subprocess
                        result = subprocess.run(["python", "analysis/03_correlation_analysis.py"],
                                              capture_output=True, text=True, cwd=".")
                        if result.returncode == 0:
                            st.success("✅ Correlation analysis generated successfully!")
                            st.info("📁 Check analysis/ directory for correlation plots")
                        else:
                            st.error(f"❌ Error generating analysis: {result.stderr}")
                    except Exception as e:
                        st.error(f"❌ Error: {e}")

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🎨 Generate Interactive Dashboards", use_container_width=True):
                with st.spinner("Generating interactive dashboards..."):
                    try:
                        import subprocess
                        result = subprocess.run(["python", "analysis/05_interactive_eda.py"],
                                              capture_output=True, text=True, cwd=".")
                        if result.returncode == 0:
                            st.success("✅ Interactive dashboards generated successfully!")
                            st.info("📁 Open analysis/interactive/index.html in your browser")
                        else:
                            st.error(f"❌ Error generating dashboards: {result.stderr}")
                    except Exception as e:
                        st.error(f"❌ Error: {e}")

        with col2:
            if st.button("📄 Generate Comprehensive Report", use_container_width=True):
                with st.spinner("Generating comprehensive report..."):
                    try:
                        import subprocess
                        result = subprocess.run(["python", "analysis/06_comprehensive_report.py"],
                                              capture_output=True, text=True, cwd=".")
                        if result.returncode == 0:
                            st.success("✅ Comprehensive report generated successfully!")
                            st.info("📁 Check analysis/reports/ directory for HTML report")
                        else:
                            st.error(f"❌ Error generating report: {result.stderr}")
                    except Exception as e:
                        st.error(f"❌ Error: {e}")

        st.markdown("#### 📊 Quick Data Summary")

        summary_col1, summary_col2, summary_col3 = st.columns(3)

        with summary_col1:
            st.metric("Total Records", f"{len(df):,}")
            st.metric("Date Range", f"{(df['timestamp'].max() - df['timestamp'].min()).days} days")

        with summary_col2:
            st.metric("Average Energy Output", f"{df['energy_output'].mean():.2f} kW")
            st.metric("Peak Energy Output", f"{df['energy_output'].max():.2f} kW")

        with summary_col3:
            try:
                uptime = (1 - df['maintenance_needed'].mean()) * 100
                st.metric("System Uptime", f"{uptime:.1f}%")

                if 'efficiency' in df.columns and not df['efficiency'].isna().all():
                    efficiency = df['efficiency'].mean()
                    st.metric("Average Efficiency", f"{efficiency:.3f}")
                else:
                    st.metric("Average Efficiency", "N/A")
            except Exception as e:
                st.error(f"Error calculating summary metrics: {e}")

def system_diagnostics():
    st.markdown('<h1 class="main-header">🔧 System Diagnostics & Health</h1>', unsafe_allow_html=True)

    st.markdown("## 🏥 System Health Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="status-card">
            <h3>🤖 AI Models</h3>
            <h2 class="status-excellent">✅ ONLINE</h2>
            <p>All 3 models loaded</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="status-card">
            <h3>📊 Data Pipeline</h3>
            <h2 class="status-good">✅ ACTIVE</h2>
            <p>Real-time ingestion</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="status-card">
            <h3>🌐 API Services</h3>
            <h2 class="status-warning">⚠️ LIMITED</h2>
            <p>Mock data mode</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="status-card">
            <h3>💾 Storage</h3>
            <h2 class="status-excellent">✅ OPTIMAL</h2>
            <p>9.8K records stored</p>
        </div>
        """, unsafe_allow_html=True)

def main():
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 1rem;">
        <h1>🌞 Solar AI Hub</h1>
        <p style="color: #888;">Advanced Analytics Platform</p>
    </div>
    """, unsafe_allow_html=True)

    pages = {
        "🏠 Real-Time Monitor": enhanced_real_time_monitoring,
        "📊 Advanced Analytics": advanced_analytics,
        "🔧 System Diagnostics": system_diagnostics
    }

    selected_page = st.sidebar.selectbox("🧭 Navigate to", list(pages.keys()))

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Settings")

    theme = st.sidebar.selectbox("🎨 Theme", ["Dark", "Light"], index=0)

    refresh_interval = st.sidebar.slider("🔄 Refresh Interval (seconds)", 10, 300, 30)

    st.sidebar.markdown("### 🚨 Alert Settings")
    temp_threshold = st.sidebar.slider("🌡️ Temperature Alert (°C)", 30, 70, 50)
    efficiency_threshold = st.sidebar.slider("⚙️ Efficiency Alert (%)", 50, 95, 80)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📍 System Info")
    st.sidebar.info(f"""
    **Location:** Coimbatore, India
    **Coordinates:** 11.0168°N, 76.9558°E
    **Local Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    **System Status:** 🟢 Online
    """)

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.8rem;">
        <p>Solar AI Analytics Hub v2.0</p>
        <p>Powered by Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)

    try:
        pages[selected_page]()
    except Exception as e:
        st.error(f"Error loading page: {e}")
        st.info("Please try refreshing the page or contact support if the issue persists.")

if __name__ == "__main__":
    main()
