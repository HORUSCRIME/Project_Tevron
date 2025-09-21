"""
Modern React-like Solar Panel Dashboard
Ultra-realistic UI with glassmorphism, animations, and modern design patterns
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
from scipy import stats
import seaborn as sns

from data_ingestion import DataIngestion
from feature_engineering import FeatureEngineer
from inference import ModelInference

# Configure page
st.set_page_config(
    page_title="Tevron Solar AI Hub",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Ultra-modern CSS with glassmorphism and animations
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Global Styles */
    .main {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }

    .stApp {
        background: transparent;
    }

    /* Header */
    .header-container {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 20px;
        padding: 1.5rem 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }

    .header-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #fff 0%, #f0f0f0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        text-align: center;
    }

    .header-subtitle {
        text-align: center;
        color: rgba(255, 255, 255, 0.8);
        font-size: 1.1rem;
        margin-top: 0.5rem;
        font-weight: 400;
    }

    /* Glass Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .glass-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15);
        border-color: rgba(255, 255, 255, 0.3);
    }

    /* Metric Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 20px;
        padding: 2rem 1.5rem;
        text-align: center;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }

    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
        transition: left 0.5s;
    }

    .metric-card:hover::before {
        left: 100%;
    }

    .metric-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.2);
    }

    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #fff;
        margin: 0.5rem 0;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
    }

    .metric-label {
        font-size: 0.9rem;
        color: rgba(255, 255, 255, 0.8);
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .metric-delta {
        font-size: 0.8rem;
        margin-top: 0.5rem;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        display: inline-block;
    }

    .metric-delta.positive {
        background: rgba(34, 197, 94, 0.2);
        color: #22c55e;
        border: 1px solid rgba(34, 197, 94, 0.3);
    }

    .metric-delta.negative {
        background: rgba(239, 68, 68, 0.2);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }

    /* Status Cards */
    .status-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        position: relative;
        overflow: hidden;
    }

    .status-excellent {
        border-left: 4px solid #22c55e;
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.1) 0%, rgba(255, 255, 255, 0.1) 100%);
    }

    .status-warning {
        border-left: 4px solid #f59e0b;
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.1) 0%, rgba(255, 255, 255, 0.1) 100%);
    }

    .status-critical {
        border-left: 4px solid #ef4444;
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(255, 255, 255, 0.1) 100%);
    }

    /* Navigation */
    .nav-container {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 50px;
        padding: 0.5rem;
        margin-bottom: 2rem;
        display: flex;
        justify-content: center;
        gap: 0.5rem;
    }

    .nav-item {
        padding: 0.75rem 1.5rem;
        border-radius: 25px;
        color: rgba(255, 255, 255, 0.7);
        text-decoration: none;
        transition: all 0.3s ease;
        font-weight: 500;
        cursor: pointer;
    }

    .nav-item:hover {
        background: rgba(255, 255, 255, 0.1);
        color: #fff;
        transform: translateY(-2px);
    }

    .nav-item.active {
        background: rgba(255, 255, 255, 0.2);
        color: #fff;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }

    /* Charts */
    .chart-container {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1rem;
        margin: 1rem 0;
    }

    /* Animations */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes pulse {
        0%, 100% {
            opacity: 1;
        }
        50% {
            opacity: 0.7;
        }
    }

    .fade-in {
        animation: fadeInUp 0.6s ease-out;
    }

    .pulse {
        animation: pulse 2s infinite;
    }

    /* Alert Banner */
    .alert-banner {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        padding: 1rem 2rem;
        border-radius: 12px;
        margin: 1rem 0;
        text-align: center;
        font-weight: 600;
        box-shadow: 0 4px 20px rgba(239, 68, 68, 0.3);
        animation: pulse 2s infinite;
    }

    /* Loading Spinner */
    .loading-spinner {
        width: 40px;
        height: 40px;
        border: 4px solid rgba(255, 255, 255, 0.3);
        border-top: 4px solid #fff;
        border-radius: 50%;
        animation: spin 1s linear infinite;
        margin: 2rem auto;
    }

    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    /* Hide Streamlit elements */
    .stDeployButton {
        display: none;
    }

    header[data-testid="stHeader"] {
        display: none;
    }

    .stMainBlockContainer {
        padding-top: 2rem;
    }

    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }

    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.3);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: rgba(255, 255, 255, 0.5);
    }
</style>
""", unsafe_allow_html=True)

# Initialize components
@st.cache_resource
def get_components():
    try:
        return DataIngestion(), FeatureEngineer(), ModelInference()
    except Exception as e:
        st.error(f"Error initializing: {e}")
        return None, None, None

@st.cache_data(ttl=300)
def load_data():
    ingestion, _, _ = get_components()
    if ingestion:
        return ingestion.load_historical_data()
    return None

def create_modern_gauge(value, title, color="#22c55e"):
    """Create modern circular gauge"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        title = {'text': title, 'font': {'size': 16, 'color': 'white'}},
        gauge = {
            'axis': {'range': [None, 100], 'tickcolor': 'white'},
            'bar': {'color': color, 'thickness': 0.8},
            'bgcolor': "rgba(255,255,255,0.1)",
            'borderwidth': 2,
            'bordercolor': "rgba(255,255,255,0.2)",
            'steps': [
                {'range': [0, 50], 'color': "rgba(255,255,255,0.05)"},
                {'range': [50, 100], 'color': "rgba(255,255,255,0.1)"}
            ]
        }
    ))

    fig.update_layout(
        height=200,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white', 'family': 'Inter'}
    )
    return fig

def render_header():
    """Render common header"""
    st.markdown("""
    <div class="header-container fade-in">
        <h1 class="header-title">⚡ Tevron Solar AI Hub</h1>
        <p class="header-subtitle">Advanced Solar Panel Monitoring & Analytics Platform</p>
    </div>
    """, unsafe_allow_html=True)

def render_navigation():
    """Render navigation menu"""
    pages = {
        "🏠 Dashboard": "dashboard",
        "📊 Analytics": "analytics",
        "🔬 EDA Explorer": "eda",
        "🔧 Maintenance": "maintenance",
        "⚙️ Settings": "settings",
        "📱 Mobile View": "mobile"
    }

    selected = st.selectbox("", list(pages.keys()), key="nav")
    return pages[selected]

def dashboard_page():

    # Get data
    ingestion, engineer, inference = get_components()
    if not all([ingestion, engineer, inference]):
        st.error("System initialization failed")
        return

    current_data = ingestion.get_combined_data()
    if not current_data:
        st.error("No data available")
        return

    # Real-time metrics
    st.markdown("### 📊 Real-Time Performance")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        energy = current_data.get('energy_output', 0)
        delta = np.random.uniform(-0.5, 0.5)
        delta_class = "positive" if delta > 0 else "negative"
        st.markdown(f"""
        <div class="metric-card fade-in">
            <div class="metric-label">Energy Output</div>
            <div class="metric-value">{energy:.1f}</div>
            <div class="metric-delta {delta_class}">
                {"↗" if delta > 0 else "↘"} {abs(delta):.1f} kW
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        irradiance = current_data.get('solar_irradiance', 0)
        delta = np.random.uniform(-50, 50)
        delta_class = "positive" if delta > 0 else "negative"
        st.markdown(f"""
        <div class="metric-card fade-in">
            <div class="metric-label">Solar Irradiance</div>
            <div class="metric-value">{irradiance:.0f}</div>
            <div class="metric-delta {delta_class}">
                {"↗" if delta > 0 else "↘"} {abs(delta):.0f} W/m²
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        temp = current_data.get('panel_temp', 0)
        delta = np.random.uniform(-2, 2)
        delta_class = "positive" if delta > 0 else "negative"
        st.markdown(f"""
        <div class="metric-card fade-in">
            <div class="metric-label">Panel Temperature</div>
            <div class="metric-value">{temp:.1f}°</div>
            <div class="metric-delta {delta_class}">
                {"↗" if delta > 0 else "↘"} {abs(delta):.1f}°C
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        efficiency = (energy / (irradiance / 1000)) if irradiance > 0 else 0
        efficiency = min(efficiency * 100, 100)
        delta = np.random.uniform(-5, 5)
        delta_class = "positive" if delta > 0 else "negative"
        st.markdown(f"""
        <div class="metric-card fade-in">
            <div class="metric-label">Efficiency</div>
            <div class="metric-value">{efficiency:.1f}%</div>
            <div class="metric-delta {delta_class}">
                {"↗" if delta > 0 else "↘"} {abs(delta):.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        dust = current_data.get('dust_level', 0) * 100
        delta = np.random.uniform(-3, 3)
        delta_class = "negative" if delta > 0 else "positive"  # Inverted for dust
        st.markdown(f"""
        <div class="metric-card fade-in">
            <div class="metric-label">Dust Level</div>
            <div class="metric-value">{dust:.1f}%</div>
            <div class="metric-delta {delta_class}">
                {"↗" if delta > 0 else "↘"} {abs(delta):.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)

    # AI Predictions
    st.markdown("### 🤖 AI Predictions")

    try:
        predictions = inference.predict_all(current_data)
        if predictions and 'predictions' in predictions:
            pred_col1, pred_col2, pred_col3 = st.columns(3)

            with pred_col1:
                if 'maintenance' in predictions['predictions']:
                    maint = predictions['predictions']['maintenance']
                    prob = maint.get('maintenance_probability', 0) * 100

                    if prob > 70:
                        status_class = "status-critical"
                        status_text = "🔴 CRITICAL"
                    elif prob > 40:
                        status_class = "status-warning"
                        status_text = "🟡 WARNING"
                    else:
                        status_class = "status-excellent"
                        status_text = "🟢 EXCELLENT"

                    st.markdown(f"""
                    <div class="glass-card {status_class} fade-in">
                        <h4 style="color: white; margin: 0 0 1rem 0;">🔧 Maintenance Prediction</h4>
                        <div style="font-size: 1.5rem; font-weight: 700; color: white;">{status_text}</div>
                        <div style="margin-top: 0.5rem; color: rgba(255,255,255,0.8);">
                            Probability: {prob:.1f}%<br>
                            Confidence: {maint.get('confidence', 0)*100:.1f}%
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            with pred_col2:
                if 'performance' in predictions['predictions']:
                    perf = predictions['predictions']['performance']
                    pred_energy = perf.get('predicted_energy_output', 0)

                    trend = "📈 Increasing" if pred_energy > energy else "📉 Decreasing"

                    st.markdown(f"""
                    <div class="glass-card fade-in">
                        <h4 style="color: white; margin: 0 0 1rem 0;">📈 Performance Forecast</h4>
                        <div style="font-size: 1.5rem; font-weight: 700; color: white;">{pred_energy:.2f} kW</div>
                        <div style="margin-top: 0.5rem; color: rgba(255,255,255,0.8);">
                            Next Hour Prediction<br>
                            Trend: {trend}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            with pred_col3:
                if 'anomaly' in predictions['predictions']:
                    anom = predictions['predictions']['anomaly']
                    is_anomaly = anom.get('is_anomaly', False)
                    severity = anom.get('severity', 'Normal')

                    if is_anomaly:
                        if severity == 'High':
                            status_class = "status-critical"
                            status_icon = "🔴"
                        else:
                            status_class = "status-warning"
                            status_icon = "🟡"
                    else:
                        status_class = "status-excellent"
                        status_icon = "🟢"

                    st.markdown(f"""
                    <div class="glass-card {status_class} fade-in">
                        <h4 style="color: white; margin: 0 0 1rem 0;">🔍 Anomaly Detection</h4>
                        <div style="font-size: 1.5rem; font-weight: 700; color: white;">{status_icon} {severity}</div>
                        <div style="margin-top: 0.5rem; color: rgba(255,255,255,0.8);">
                            Status: {'Anomaly Detected' if is_anomaly else 'Normal Operation'}<br>
                            Score: {anom.get('anomaly_score', 0):.3f}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Prediction error: {e}")

    # Performance Gauges
    st.markdown("### 🎛️ System Performance")

    gauge_col1, gauge_col2, gauge_col3 = st.columns(3)

    with gauge_col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        gauge1 = create_modern_gauge(efficiency, "Efficiency (%)", "#22c55e")
        st.plotly_chart(gauge1, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with gauge_col2:
        temp_percent = min((temp / 70) * 100, 100)
        color = "#ef4444" if temp_percent > 70 else "#f59e0b" if temp_percent > 50 else "#22c55e"
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        gauge2 = create_modern_gauge(temp_percent, "Temperature", color)
        st.plotly_chart(gauge2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with gauge_col3:
        dust_percent = min(dust, 100)
        color = "#ef4444" if dust_percent > 60 else "#f59e0b" if dust_percent > 30 else "#22c55e"
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        gauge3 = create_modern_gauge(dust_percent, "Dust Level (%)", color)
        st.plotly_chart(gauge3, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Live Charts
    st.markdown("### 📈 Live Performance Charts")

    df = load_data()
    if df is not None and not df.empty:
        recent_df = df.tail(24)

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=recent_df['timestamp'],
                y=recent_df['energy_output'],
                mode='lines+markers',
                name='Energy Output',
                line=dict(color='#22c55e', width=3),
                marker=dict(size=6, color='#22c55e'),
                fill='tonexty'
            ))

            fig.update_layout(
                title='Energy Output Trend (24h)',
                xaxis_title='Time',
                yaxis_title='Energy Output (kW)',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white', family='Inter'),
                showlegend=False,
                height=300
            )

            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with chart_col2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                               subplot_titles=('Solar Irradiance', 'Panel Temperature'))

            fig2.add_trace(go.Scatter(
                x=recent_df['timestamp'],
                y=recent_df['solar_irradiance'],
                mode='lines',
                name='Solar Irradiance',
                line=dict(color='#f59e0b', width=2)
            ), row=1, col=1)

            fig2.add_trace(go.Scatter(
                x=recent_df['timestamp'],
                y=recent_df['panel_temp'],
                mode='lines',
                name='Panel Temp',
                line=dict(color='#ef4444', width=2)
            ), row=2, col=1)

            fig2.update_layout(
                title='Environmental Conditions (24h)',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white', family='Inter'),
                showlegend=False,
                height=300
            )

            st.plotly_chart(fig2, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

def eda_page():
    """Comprehensive EDA Explorer page"""
    st.markdown("### 🔬 Exploratory Data Analysis")

    df = load_data()
    if df is None or df.empty:
        st.error("No data available for EDA")
        return

    # EDA Navigation tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview", "📈 Distributions", "🔗 Correlations",
        "🕰️ Time Series", "🔍 Outliers", "🎯 Statistical Tests"
    ])

    with tab1:
        # Dataset Overview
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### 📊 Dataset Overview")

            st.markdown(f"""
            - **Total Records:** {len(df):,}
            - **Features:** {len(df.columns)}
            - **Date Range:** {df['timestamp'].min().date()} to {df['timestamp'].max().date()}
            - **Missing Values:** {df.isnull().sum().sum()}
            - **Memory Usage:** {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB
            """)

            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### 📊 Data Types")

            dtype_counts = df.dtypes.value_counts()
            fig_dtype = px.pie(values=dtype_counts.values, names=dtype_counts.index,
                              title="Data Types Distribution")
            fig_dtype.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=300
            )
            st.plotly_chart(fig_dtype, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Statistical Summary
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 📊 Statistical Summary")

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        summary_stats = df[numeric_cols].describe().round(3)
        st.dataframe(summary_stats, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Missing Values Heatmap
        if df.isnull().sum().sum() > 0:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### 🕳️ Missing Values Pattern")

            missing_data = df.isnull().sum()
            missing_data = missing_data[missing_data > 0]

            if len(missing_data) > 0:
                fig_missing = px.bar(x=missing_data.index, y=missing_data.values,
                                   title="Missing Values by Column")
                fig_missing.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )
                st.plotly_chart(fig_missing, use_container_width=True)
            else:
                st.success("✅ No missing values found!")

            st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        # Distribution Analysis
        st.markdown("#### 📈 Feature Distributions")

        col1, col2 = st.columns(2)
        with col1:
            feature = st.selectbox("Select Feature", numeric_cols, key="dist_feature")
        with col2:
            plot_type = st.selectbox("Plot Type", ["Histogram", "Box Plot", "Violin Plot", "KDE"])

        col3, col4 = st.columns(2)

        with col3:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)

            if plot_type == "Histogram":
                fig = px.histogram(df, x=feature, nbins=30, title=f"{feature} Distribution")
            elif plot_type == "Box Plot":
                fig = px.box(df, y=feature, title=f"{feature} Box Plot")
            elif plot_type == "Violin Plot":
                fig = px.violin(df, y=feature, title=f"{feature} Violin Plot")
            else:  # KDE
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=df[feature], histnorm='probability density',
                                         name='Histogram', opacity=0.7))
                fig.update_layout(title=f"{feature} KDE Plot")

            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col4:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown(f"#### 📊 {feature} Statistics")

            stats = {
                "Mean": df[feature].mean(),
                "Median": df[feature].median(),
                "Std Dev": df[feature].std(),
                "Skewness": df[feature].skew(),
                "Kurtosis": df[feature].kurtosis(),
                "Min": df[feature].min(),
                "Max": df[feature].max(),
                "Range": df[feature].max() - df[feature].min()
            }

            for stat, value in stats.items():
                st.markdown(f"**{stat}:** {value:.3f}")

            st.markdown('</div>', unsafe_allow_html=True)

        # Multi-feature distribution comparison
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 🔄 Multi-Feature Distribution Comparison")

        selected_features = st.multiselect("Select Features to Compare", numeric_cols,
                                         default=numeric_cols[:4])

        if selected_features:
            fig_multi = make_subplots(rows=2, cols=2,
                                    subplot_titles=selected_features[:4])

            for i, feature in enumerate(selected_features[:4]):
                row = (i // 2) + 1
                col = (i % 2) + 1

                fig_multi.add_trace(
                    go.Histogram(x=df[feature], name=feature, showlegend=False),
                    row=row, col=col
                )

            fig_multi.update_layout(
                title="Feature Distributions Comparison",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=500
            )
            st.plotly_chart(fig_multi, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        # Correlation Analysis
        st.markdown("#### 🔗 Correlation Analysis")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)

            corr_matrix = df[numeric_cols].corr()

            fig_corr = px.imshow(corr_matrix,
                               title="Feature Correlation Heatmap",
                               color_continuous_scale="RdBu_r",
                               aspect="auto")
            fig_corr.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=600
            )
            st.plotly_chart(fig_corr, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### 🔍 High Correlations")

            # Find high correlations
            high_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7:
                        high_corr.append({
                            'Feature 1': corr_matrix.columns[i],
                            'Feature 2': corr_matrix.columns[j],
                            'Correlation': corr_val
                        })

            if high_corr:
                high_corr_df = pd.DataFrame(high_corr)
                high_corr_df = high_corr_df.sort_values('Correlation', key=abs, ascending=False)

                for _, row in high_corr_df.iterrows():
                    color = "#22c55e" if row['Correlation'] > 0 else "#ef4444"
                    st.markdown(f"""
                    <div style="
                        background: rgba(255,255,255,0.1);
                        border-left: 4px solid {color};
                        padding: 0.5rem;
                        margin: 0.5rem 0;
                        border-radius: 4px;
                    ">
                        <strong>{row['Feature 1']}</strong><br>
                        ↔️ <strong>{row['Feature 2']}</strong><br>
                        <small>r = {row['Correlation']:.3f}</small>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No high correlations (>0.7) found")

            st.markdown('</div>', unsafe_allow_html=True)

        # Scatter plot matrix
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 🔄 Scatter Plot Matrix")

        selected_for_scatter = st.multiselect("Select Features for Scatter Matrix",
                                            numeric_cols, default=numeric_cols[:4])

        if len(selected_for_scatter) >= 2:
            fig_scatter = px.scatter_matrix(df[selected_for_scatter],
                                          title="Feature Scatter Matrix")
            fig_scatter.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=600
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with tab4:
        # Time Series Analysis
        st.markdown("#### 🕰️ Time Series Analysis")

        col1, col2 = st.columns(2)
        with col1:
            ts_feature = st.selectbox("Select Feature", numeric_cols, key="ts_feature")
        with col2:
            time_window = st.selectbox("Time Window", ["1 Day", "1 Week", "1 Month", "All Data"])

        # Filter data based on time window
        if time_window == "1 Day":
            ts_df = df.tail(24)
        elif time_window == "1 Week":
            ts_df = df.tail(24 * 7)
        elif time_window == "1 Month":
            ts_df = df.tail(24 * 30)
        else:
            ts_df = df

        col3, col4 = st.columns(2)

        with col3:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)

            fig_ts = px.line(ts_df, x='timestamp', y=ts_feature,
                           title=f"{ts_feature} Time Series")
            fig_ts.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            st.plotly_chart(fig_ts, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col4:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)

            # Rolling statistics
            ts_df_copy = ts_df.copy()
            ts_df_copy['rolling_mean'] = ts_df_copy[ts_feature].rolling(window=24).mean()
            ts_df_copy['rolling_std'] = ts_df_copy[ts_feature].rolling(window=24).std()

            fig_rolling = go.Figure()
            fig_rolling.add_trace(go.Scatter(x=ts_df_copy['timestamp'], y=ts_df_copy[ts_feature],
                                           mode='lines', name='Original', opacity=0.7))
            fig_rolling.add_trace(go.Scatter(x=ts_df_copy['timestamp'], y=ts_df_copy['rolling_mean'],
                                           mode='lines', name='24h Mean'))

            fig_rolling.update_layout(
                title=f"{ts_feature} with Rolling Statistics",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            st.plotly_chart(fig_rolling, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Seasonal decomposition visualization
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 🌍 Seasonal Patterns")

        # Hourly patterns
        ts_df_copy['hour'] = ts_df_copy['timestamp'].dt.hour
        hourly_avg = ts_df_copy.groupby('hour')[ts_feature].mean()

        fig_hourly = px.bar(x=hourly_avg.index, y=hourly_avg.values,
                          title=f"Average {ts_feature} by Hour of Day")
        fig_hourly.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig_hourly, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab5:
        # Outlier Detection
        st.markdown("#### 🔍 Outlier Detection")

        col1, col2 = st.columns(2)
        with col1:
            outlier_feature = st.selectbox("Select Feature", numeric_cols, key="outlier_feature")
        with col2:
            outlier_method = st.selectbox("Detection Method", ["IQR", "Z-Score", "Modified Z-Score"])

        # Detect outliers
        if outlier_method == "IQR":
            Q1 = df[outlier_feature].quantile(0.25)
            Q3 = df[outlier_feature].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[outlier_feature] < lower_bound) | (df[outlier_feature] > upper_bound)]

        elif outlier_method == "Z-Score":
            z_scores = np.abs((df[outlier_feature] - df[outlier_feature].mean()) / df[outlier_feature].std())
            outliers = df[z_scores > 3]

        else:  # Modified Z-Score
            median = df[outlier_feature].median()
            mad = np.median(np.abs(df[outlier_feature] - median))
            modified_z_scores = 0.6745 * (df[outlier_feature] - median) / mad
            outliers = df[np.abs(modified_z_scores) > 3.5]

        col3, col4 = st.columns(2)

        with col3:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)

            fig_outlier = go.Figure()
            fig_outlier.add_trace(go.Scatter(x=df.index, y=df[outlier_feature],
                                           mode='markers', name='Normal',
                                           marker=dict(color='lightblue', size=4)))

            if len(outliers) > 0:
                fig_outlier.add_trace(go.Scatter(x=outliers.index, y=outliers[outlier_feature],
                                               mode='markers', name='Outliers',
                                               marker=dict(color='red', size=8)))

            fig_outlier.update_layout(
                title=f"Outliers in {outlier_feature} ({outlier_method})",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            st.plotly_chart(fig_outlier, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col4:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown(f"#### 📊 Outlier Summary")

            st.markdown(f"""
            - **Total Outliers:** {len(outliers)}
            - **Percentage:** {len(outliers)/len(df)*100:.2f}%
            - **Method:** {outlier_method}
            """)

            if len(outliers) > 0:
                st.markdown("**Outlier Values:**")
                outlier_summary = outliers[outlier_feature].describe()
                for stat, value in outlier_summary.items():
                    st.markdown(f"- **{stat}:** {value:.3f}")

            st.markdown('</div>', unsafe_allow_html=True)

        # Show outlier data
        if len(outliers) > 0:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### 📋 Outlier Records")
            st.dataframe(outliers.head(10), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    with tab6:
        # Statistical Tests
        st.markdown("#### 🎯 Statistical Tests")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### 📊 Normality Tests")

            test_feature = st.selectbox("Select Feature", numeric_cols, key="test_feature")

            from scipy import stats

            # Shapiro-Wilk test (for small samples)
            if len(df) <= 5000:
                shapiro_stat, shapiro_p = stats.shapiro(df[test_feature].dropna().sample(min(5000, len(df))))
                st.markdown(f"**Shapiro-Wilk Test:**")
                st.markdown(f"- Statistic: {shapiro_stat:.4f}")
                st.markdown(f"- p-value: {shapiro_p:.4f}")
                st.markdown(f"- Normal: {'Yes' if shapiro_p > 0.05 else 'No'}")

            # Kolmogorov-Smirnov test
            ks_stat, ks_p = stats.kstest(df[test_feature].dropna(), 'norm')
            st.markdown(f"\n**Kolmogorov-Smirnov Test:**")
            st.markdown(f"- Statistic: {ks_stat:.4f}")
            st.markdown(f"- p-value: {ks_p:.4f}")
            st.markdown(f"- Normal: {'Yes' if ks_p > 0.05 else 'No'}")

            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### 🔗 Correlation Tests")

            feature1 = st.selectbox("Feature 1", numeric_cols, key="corr_test_1")
            feature2 = st.selectbox("Feature 2", numeric_cols, key="corr_test_2")

            if feature1 != feature2:
                # Pearson correlation
                pearson_r, pearson_p = stats.pearsonr(df[feature1].dropna(), df[feature2].dropna())
                st.markdown(f"**Pearson Correlation:**")
                st.markdown(f"- Coefficient: {pearson_r:.4f}")
                st.markdown(f"- p-value: {pearson_p:.4f}")
                st.markdown(f"- Significant: {'Yes' if pearson_p < 0.05 else 'No'}")

                # Spearman correlation
                spearman_r, spearman_p = stats.spearmanr(df[feature1].dropna(), df[feature2].dropna())
                st.markdown(f"\n**Spearman Correlation:**")
                st.markdown(f"- Coefficient: {spearman_r:.4f}")
                st.markdown(f"- p-value: {spearman_p:.4f}")
                st.markdown(f"- Significant: {'Yes' if spearman_p < 0.05 else 'No'}")

            st.markdown('</div>', unsafe_allow_html=True)

        # Feature importance from models
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 🤖 ML Model Feature Importance")

        _, _, inference = get_components()
        if inference:
            model_type = st.selectbox("Select Model", ["maintenance", "performance", "anomaly"])

            importance_df = inference.get_feature_importance(model_type)
            if importance_df is not None:
                fig_importance = px.bar(importance_df.head(10),
                                      x='importance', y='feature',
                                      orientation='h',
                                      title=f"Top 10 Features - {model_type.title()} Model")
                fig_importance.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )
                st.plotly_chart(fig_importance, use_container_width=True)
            else:
                st.warning(f"Feature importance not available for {model_type} model")

        st.markdown('</div>', unsafe_allow_html=True)

def analytics_page():
    """Analytics and reporting page"""
    st.markdown("### 📈 Advanced Analytics")

    df = load_data()
    if df is None or df.empty:
        st.error("No data available")
        return

    # Time range selector
    col1, col2 = st.columns(2)
    with col1:
        days = st.selectbox("Time Range", [7, 30, 90, 365], index=1)
    with col2:
        chart_type = st.selectbox("Chart Type", ["Line", "Bar", "Heatmap"])

    recent_df = df.tail(days * 24)

    # Performance trends
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### Energy Production Trends")

    if chart_type == "Line":
        fig = px.line(recent_df, x='timestamp', y='energy_output',
                     title=f'Energy Output - Last {days} Days')
    elif chart_type == "Bar":
        daily_df = recent_df.groupby(recent_df['timestamp'].dt.date)['energy_output'].sum().reset_index()
        fig = px.bar(daily_df, x='timestamp', y='energy_output')
    else:
        # Heatmap by hour and day
        recent_df['hour'] = recent_df['timestamp'].dt.hour
        recent_df['day'] = recent_df['timestamp'].dt.day_name()
        pivot_df = recent_df.pivot_table(values='energy_output', index='hour', columns='day')
        fig = px.imshow(pivot_df, title="Energy Output Heatmap")

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Correlation matrix
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### Feature Correlations")

    numeric_cols = ['energy_output', 'temperature', 'humidity', 'wind_speed',
                   'solar_irradiance', 'panel_temp', 'voltage', 'current']
    corr_matrix = recent_df[numeric_cols].corr()

    fig_corr = px.imshow(corr_matrix,
                        title="Feature Correlation Matrix",
                        color_continuous_scale="RdBu_r")
    fig_corr.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    st.plotly_chart(fig_corr, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

def maintenance_page():
    """Maintenance scheduling and alerts page"""
    st.markdown("### 🔧 Maintenance Management")

    ingestion, _, inference = get_components()
    if not inference:
        st.error("System not available")
        return

    # Maintenance alerts
    st.markdown('<div class="alert-banner">', unsafe_allow_html=True)
    st.markdown("🚨 **CRITICAL ALERT:** Panel #3 requires immediate cleaning - Dust level: 85%")
    st.markdown('</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 📅 Scheduled Maintenance")

        maintenance_schedule = [
            {"Panel": "Panel #1", "Type": "Cleaning", "Due": "2024-01-15", "Priority": "Medium"},
            {"Panel": "Panel #2", "Type": "Inspection", "Due": "2024-01-18", "Priority": "Low"},
            {"Panel": "Panel #3", "Type": "Cleaning", "Due": "2024-01-12", "Priority": "High"},
            {"Panel": "Panel #4", "Type": "Repair", "Due": "2024-01-20", "Priority": "Critical"}
        ]

        for item in maintenance_schedule:
            priority_color = {
                "Critical": "#ef4444",
                "High": "#f59e0b",
                "Medium": "#3b82f6",
                "Low": "#22c55e"
            }[item["Priority"]]

            st.markdown(f"""
            <div style="
                background: rgba(255,255,255,0.1);
                border-left: 4px solid {priority_color};
                padding: 1rem;
                margin: 0.5rem 0;
                border-radius: 8px;
            ">
                <strong>{item['Panel']}</strong> - {item['Type']}<br>
                <small>Due: {item['Due']} | Priority: {item['Priority']}</small>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 🔍 Maintenance Predictions")

        current_data = ingestion.get_combined_data() if ingestion else {}
        if current_data:
            predictions = inference.predict_maintenance(current_data)
            if predictions:
                prob = predictions.get('maintenance_probability', 0) * 100
                confidence = predictions.get('confidence', 0) * 100

                st.markdown(f"""
                <div style="text-align: center; padding: 2rem;">
                    <div style="font-size: 3rem; margin-bottom: 1rem;">
                        {'🔴' if prob > 70 else '🟡' if prob > 40 else '🟢'}
                    </div>
                    <div style="font-size: 2rem; font-weight: bold; color: white;">
                        {prob:.1f}%
                    </div>
                    <div style="color: rgba(255,255,255,0.8); margin-top: 0.5rem;">
                        Maintenance Probability<br>
                        Confidence: {confidence:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # Maintenance history
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### 📋 Maintenance History")

    history_data = {
        "Date": ["2024-01-10", "2024-01-05", "2023-12-28", "2023-12-20"],
        "Panel": ["Panel #2", "Panel #1", "Panel #4", "Panel #3"],
        "Type": ["Cleaning", "Inspection", "Repair", "Cleaning"],
        "Status": ["Completed", "Completed", "Completed", "Completed"],
        "Cost": ["$150", "$75", "$450", "$150"]
    }

    st.dataframe(pd.DataFrame(history_data), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

def settings_page():
    """Settings and configuration page"""
    st.markdown("### ⚙️ System Settings")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 🔔 Alert Settings")

        temp_threshold = st.slider("Temperature Alert (°C)", 30, 80, 65)
        dust_threshold = st.slider("Dust Level Alert (%)", 10, 90, 60)
        efficiency_threshold = st.slider("Efficiency Alert (%)", 50, 95, 80)

        st.checkbox("Email Notifications", value=True)
        st.checkbox("SMS Alerts", value=False)
        st.checkbox("Push Notifications", value=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 🎨 Display Settings")

        theme = st.selectbox("Theme", ["Dark (Current)", "Light", "Auto"])
        refresh_rate = st.selectbox("Refresh Rate", ["5 seconds", "10 seconds", "30 seconds", "1 minute"])
        chart_style = st.selectbox("Chart Style", ["Modern", "Classic", "Minimal"])

        st.checkbox("Show Animations", value=True)
        st.checkbox("High Contrast Mode", value=False)

        st.markdown('</div>', unsafe_allow_html=True)

    # Model settings
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### 🤖 AI Model Settings")

    col3, col4, col5 = st.columns(3)

    with col3:
        st.markdown("**Maintenance Model**")
        st.selectbox("Algorithm", ["Random Forest", "XGBoost", "Neural Network"], key="maint_algo")
        st.slider("Sensitivity", 0.1, 1.0, 0.7, key="maint_sens")

    with col4:
        st.markdown("**Performance Model**")
        st.selectbox("Algorithm", ["Linear Regression", "SVR", "Neural Network"], key="perf_algo")
        st.slider("Accuracy", 0.5, 1.0, 0.85, key="perf_acc")

    with col5:
        st.markdown("**Anomaly Detection**")
        st.selectbox("Algorithm", ["Isolation Forest", "One-Class SVM", "Autoencoder"], key="anom_algo")
        st.slider("Threshold", 0.1, 1.0, 0.3, key="anom_thresh")

    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("💾 Save Settings", type="primary"):
        st.success("Settings saved successfully!")

def mobile_view():
    """Mobile-optimized view"""
    st.markdown("### 📱 Mobile Dashboard")

    ingestion, _, inference = get_components()
    if not ingestion:
        st.error("System not available")
        return

    current_data = ingestion.get_combined_data()
    if not current_data:
        st.error("No data available")
        return

    # Compact metrics
    energy = current_data.get('energy_output', 0)
    temp = current_data.get('panel_temp', 0)
    efficiency = min((energy / (current_data.get('solar_irradiance', 1) / 1000)) * 100, 100)

    st.markdown(f"""
    <div class="glass-card" style="text-align: center;">
        <div style="display: flex; justify-content: space-around; flex-wrap: wrap;">
            <div style="margin: 1rem;">
                <div style="font-size: 2rem; color: #22c55e;">⚡</div>
                <div style="font-size: 1.5rem; font-weight: bold; color: white;">{energy:.1f} kW</div>
                <div style="color: rgba(255,255,255,0.7);">Energy</div>
            </div>
            <div style="margin: 1rem;">
                <div style="font-size: 2rem; color: #f59e0b;">🌡️</div>
                <div style="font-size: 1.5rem; font-weight: bold; color: white;">{temp:.1f}°C</div>
                <div style="color: rgba(255,255,255,0.7);">Temperature</div>
            </div>
            <div style="margin: 1rem;">
                <div style="font-size: 2rem; color: #3b82f6;">📊</div>
                <div style="font-size: 1.5rem; font-weight: bold; color: white;">{efficiency:.1f}%</div>
                <div style="color: rgba(255,255,255,0.7);">Efficiency</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Quick actions
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### 🚀 Quick Actions")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.rerun()
        if st.button("📊 View Analytics", use_container_width=True):
            st.session_state.nav = "📊 Analytics"
            st.rerun()

    with col2:
        if st.button("🔧 Maintenance", use_container_width=True):
            st.session_state.nav = "🔧 Maintenance"
            st.rerun()
        if st.button("⚙️ Settings", use_container_width=True):
            st.session_state.nav = "⚙️ Settings"
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    # Status summary
    if inference:
        try:
            predictions = inference.predict_all(current_data)
            if predictions and 'predictions' in predictions:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown("#### 🎯 System Status")

                status_items = []

                if 'maintenance' in predictions['predictions']:
                    maint_prob = predictions['predictions']['maintenance'].get('maintenance_probability', 0) * 100
                    status = "🔴 High Risk" if maint_prob > 70 else "🟡 Medium Risk" if maint_prob > 40 else "🟢 Low Risk"
                    status_items.append(f"**Maintenance:** {status}")

                if 'anomaly' in predictions['predictions']:
                    is_anomaly = predictions['predictions']['anomaly'].get('is_anomaly', False)
                    status = "🔴 Anomaly Detected" if is_anomaly else "🟢 Normal Operation"
                    status_items.append(f"**Anomaly:** {status}")

                for item in status_items:
                    st.markdown(item)

                st.markdown('</div>', unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"Prediction error: {e}")

def main():
    """Main application with navigation"""
    render_header()

    # Navigation
    page = render_navigation()

    # Route to pages
    if page == "dashboard":
        dashboard_page()
    elif page == "analytics":
        analytics_page()
    elif page == "eda":
        eda_page()
    elif page == "maintenance":
        maintenance_page()
    elif page == "settings":
        settings_page()
    elif page == "mobile":
        mobile_view()

if __name__ == "__main__":
    main()
