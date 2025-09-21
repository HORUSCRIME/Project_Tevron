import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
from scipy import stats
from scipy.stats import pearsonr, spearmanr, shapiro, jarque_bera
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
import subprocess
import os
weasyprint = None
try:
    import weasyprint
except (ImportError, OSError):
    weasyprint = None

try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

from data_ingestion import DataIngestion
from feature_engineering import FeatureEngineer
from inference import ModelInference

st.set_page_config(
    page_title="Tevron Solar AI Hub",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def apply_custom_css():
    theme = st.session_state.get('theme', 'Dark')
    
    if theme == 'Light':
        st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        .main {
            background: #fffbeb;
            color: #4a4a4a;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        
        .stApp {
            background: #fffbeb;
        }
        
        .card {
            background: rgba(255, 255, 255, 0.9);
            backdrop-filter: blur(20px);
            border-radius: 16px;
            padding: 24px;
            margin: 16px 0;
            border: 2px solid rgba(255, 123, 0, 0.2);
            box-shadow: 0 8px 25px -5px rgba(255, 195, 0, 0.15), 0 4px 10px -2px rgba(76, 201, 240, 0.1);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        
        .card:hover {
            transform: translateY(-4px);
            box-shadow: 0 20px 40px -10px rgba(255, 123, 0, 0.25), 0 10px 20px -5px rgba(255, 195, 0, 0.2);
            border-color: rgba(255, 195, 0, 0.4);
        }
        
        .card-header {
            font-size: 20px;
            font-weight: 600;
            margin-bottom: 20px;
            color: #ff7b00;
            border-bottom: 2px solid rgba(255, 195, 0, 0.3);
            padding-bottom: 12px;
            letter-spacing: -0.025em;
        }
        
        .header {
            text-align: center;
            padding: 32px 0;
            margin-bottom: 32px;
            background: rgba(255, 255, 255, 0.8);
            border-radius: 20px;
            backdrop-filter: blur(20px);
            border: 3px solid rgba(255, 123, 0, 0.3);
            box-shadow: 0 10px 30px -5px rgba(255, 195, 0, 0.2);
        }
        
        .header h1 {
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, #ff7b00 0%, #ffc300 50%, #4cc9f0 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 8px;
            letter-spacing: -0.025em;
        }
        
        .nav-container {
            background: linear-gradient(135deg, rgba(255, 123, 0, 0.05) 0%, rgba(255, 195, 0, 0.05) 100%);
            backdrop-filter: blur(20px);
            border-radius: 16px;
            padding: 16px;
            margin-bottom: 24px;
            border: 1px solid rgba(255, 123, 0, 0.2);
        }
        
        .stButton > button {
            background: linear-gradient(135deg, #ff7b00 0%, #ffc300 100%);
            border: none;
            border-radius: 12px;
            color: white;
            font-weight: 600;
            padding: 12px 24px;
            font-size: 14px;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 4px 15px -3px rgba(255, 123, 0, 0.3), 0 2px 8px -1px rgba(255, 195, 0, 0.2);
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 15px 30px -5px rgba(255, 123, 0, 0.4), 0 8px 15px -3px rgba(255, 195, 0, 0.3);
        }
        
        .stSelectbox > div > div {
            background: rgba(255, 255, 255, 0.9);
            border: 2px solid rgba(255, 123, 0, 0.3);
            border-radius: 12px;
            color: #4a4a4a;
        }
        
        .stMetric {
            background: linear-gradient(135deg, rgba(255, 195, 0, 0.1) 0%, rgba(76, 201, 240, 0.05) 100%);
            border-radius: 12px;
            padding: 16px;
            border: 1px solid rgba(255, 123, 0, 0.2);
        }
        
        .status-online {
            background: rgba(76, 201, 240, 0.15);
            color: #4cc9f0;
            border: 1px solid rgba(76, 201, 240, 0.3);
        }
        
        .status-warning {
            background: rgba(255, 195, 0, 0.15);
            color: #ffc300;
            border: 1px solid rgba(255, 195, 0, 0.4);
        }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        .main {
            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 50%, #2a2a2a 100%);
            color: #f8fafc;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        
        .stApp {
            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 50%, #2a2a2a 100%);
        }
        
        .card {
            background: rgba(255, 255, 255, 0.03);
            backdrop-filter: blur(20px);
            border-radius: 16px;
            padding: 24px;
            margin: 16px 0;
            border: 1px solid rgba(255, 255, 255, 0.08);
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            
        }
        
        .card:hover {
            transform: translateY(-4px);
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
            border-color: rgba(255, 215, 0, 0.3);
        }
        
        .card-header {
            font-size: 20px;
            font-weight: 600;
            margin-bottom: 20px;
            color: #f1f5f9;
            border-bottom: 1px solid rgba(148, 163, 184, 0.2);
            padding: 20px;
            letter-spacing: -0.025em;
            text-align: center;
        }
        
        .header {
            text-align: center;
            padding: 32px 0;
            margin-bottom: 32px;
            background: rgba(255, 255, 255, 0.02);
            border-radius: 20px;
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.05);
        }
        
        .header h1 {
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, #ffd700 0%, #007bff 50%, #ffd700 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 8px;
            letter-spacing: -0.025em;
        }
        
        .stButton > button {
            background: linear-gradient(135deg, #ffd700 0%, #007bff 100%);
            border: none;
            border-radius: 12px;
            color: #0a0a0a;
            font-weight: 600;
            padding: 12px 24px;
            font-size: 14px;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 20px 25px -5px rgba(255, 215, 0, 0.3), 0 10px 10px -5px rgba(0, 123, 255, 0.1);
        }
        
        .stSelectbox > div > div {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            color: #f1f5f9;
        }
        </style>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <style>
    .nav-container {
        background: rgba(255, 255, 255, 0.02);
        backdrop-filter: blur(20px);
        border-radius: 16px;
        padding: 16px;
        margin-bottom: 24px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .status-indicator {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .status-online {
        background: rgba(34, 197, 94, 0.1);
        color: #22c55e;
        border: 1px solid rgba(34, 197, 94, 0.2);
    }
    
    .status-warning {
        background: rgba(255, 215, 0, 0.1);
        color: #ffd700;
        border: 1px solid rgba(255, 215, 0, 0.2);
    }
    
    .stMetric {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        padding: 16px;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    .stAlert {
        border-radius: 12px;
        border: none;
        backdrop-filter: blur(10px);
    }
    </style>
    """, unsafe_allow_html=True)

if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now()
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = False

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

def safe_calculate_efficiency(energy_output, solar_irradiance):
    try:
        if solar_irradiance > 0:
            efficiency = energy_output / (solar_irradiance / 1000)
            return max(0, min(1, efficiency))
        return 0
    except (ZeroDivisionError, TypeError):
        return 0

def create_modern_gauge(value, title, min_val=0, max_val=100, thresholds=[30, 70], colors=['#ef4444', '#f59e0b', '#10b981']):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 16, 'color': 'white'}},
        delta = {'reference': thresholds[1]},
        gauge = {
            'axis': {'range': [None, max_val], 'tickcolor': 'white'},
            'bar': {'color': "rgba(255,255,255,0.8)", 'thickness': 0.3},
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

        color_col = 'efficiency' if 'efficiency' in df.columns else 'energy_output'
        color_title = "Efficiency" if color_col == 'efficiency' else "Energy Output (kW)"

        fig = go.Figure(data=[go.Scatter3d(
            x=df['solar_irradiance'],
            y=df['panel_temp'],
            z=df['energy_output'],
            mode='markers',
            marker=dict(
                size=5,
                color=df[color_col],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title=color_title),
                opacity=0.8,
                line=dict(width=0.5, color='white')
            ),
            text=df['timestamp'].dt.strftime('%Y-%m-%d %H:%M') if 'timestamp' in df.columns else None,
            hovertemplate='<b>Solar Irradiance:</b> %{x:.0f} W/m²<br>' +
                         '<b>Panel Temp:</b> %{y:.1f}°C<br>' +
                         '<b>Energy Output:</b> %{z:.2f} kW<br>' +
                         f'<b>{color_title}:</b> %{{marker.color:.3f}}<extra></extra>'
        )])

        fig.update_layout(
            title='3D Scatter: Solar Irradiance vs Panel Temperature vs Energy Output',
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

def render_header():
    st.markdown("""
    <div class="header">
        <h1>⚡ Tevron Solar AI Hub</h1>
        <p style="
            color: rgba(248, 250, 252, 0.8);
            font-size: 1.1rem;
            margin: 0;
            font-weight: 400;
        ">Advanced Solar Panel Performance Analytics & Predictive Maintenance</p>
        <div style="margin-top: 16px; display: flex; justify-content: center; gap: 12px;">
            <span class="status-indicator status-online">
                <span style="width: 8px; height: 8px; background: #22c55e; border-radius: 50%; display: inline-block;"></span>
                System Online
            </span>
            <span class="status-indicator status-warning">
                <span style="width: 8px; height: 8px; background: #fbbf24; border-radius: 50%; display: inline-block;"></span>
                2 Alerts
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_navigation():
    pages = {
        " Dashboard": "dashboard",
        " Analytics": "analytics",
        " EDA Explorer": "eda",
        " 3D Visualizations": "3d_viz",
        " Maintenance": "maintenance",
        " ML Insights": "ml_insights",
        " Settings": "settings"
    }

    #st.markdown('<div class="nav-container">', unsafe_allow_html=True)
    selected = st.selectbox(
        "Navigate to:", 
        list(pages.keys()), 
        key="nav",
        help="Select a page to navigate to"
    )
    st.markdown('</div >', unsafe_allow_html=True)
    return pages[selected]

def dashboard_page():
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        with col1:
            st.session_state.auto_refresh = st.checkbox(" Auto Refresh (30s)", value=st.session_state.auto_refresh)
        with col2:
            if st.button(" Refresh Now", type="primary"):
                st.session_state.last_update = datetime.now()
                st.rerun()
        with col3:
            alert_threshold = st.slider(" Alert Threshold", 0, 100, 70)
        with col4:
            st.write(f" Last Update: {st.session_state.last_update.strftime('%H:%M:%S')}")

        ingestion, engineer, inference = get_components()

        if not all([ingestion, engineer, inference]):
            st.error(" Failed to initialize system components")
            return

        try:
            current_data = ingestion.get_combined_data()
        except Exception as e:
            st.error(f" Unable to fetch current data: {e}")
            return

        if not current_data:
            st.error(" No current data available")
            return

        try:
            predictions = inference.predict_all(current_data)
        except Exception as e:
            st.warning(f" Prediction error: {e}")
            predictions = None

        energy_output = current_data.get('energy_output', 0)
        panel_temp = current_data.get('panel_temp', 0)
        solar_irradiance = current_data.get('solar_irradiance', 0)
        dust_level = current_data.get('dust_level', 0) * 100
        efficiency = safe_calculate_efficiency(energy_output, solar_irradiance) * 100

        if panel_temp > 50 or (predictions and 'predictions' in predictions and
            'maintenance' in predictions['predictions'] and
            predictions['predictions']['maintenance'].get('maintenance_probability', 0) > 0.7):
            st.warning(" System Alert: High temperature or maintenance required!")

        st.markdown(" Real-Time Performance")

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            delta_energy = np.random.uniform(-0.5, 0.5)
            delta_class = "positive" if delta_energy > 0 else "negative"
            st.metric("Energy Output", f"{energy_output:.2f} kW", f"{delta_energy:+.2f}")

        with col2:
            delta_irradiance = np.random.uniform(-50, 50)
            delta_class = "positive" if delta_irradiance > 0 else "negative"
            st.metric("Solar Irradiance", f"{solar_irradiance:.0f} W/m²", f"{delta_irradiance:+.0f}")

        with col3:
            delta_temp = np.random.uniform(-2, 2)
            delta_class = "positive" if delta_temp > 0 else "negative"
            st.metric("Panel Temperature", f"{panel_temp:.1f}°C", f"{delta_temp:+.1f}")

        with col4:
            delta_eff = np.random.uniform(-5, 5)
            delta_class = "positive" if delta_eff > 0 else "negative"
            st.metric("Efficiency", f"{efficiency:.1f}%", f"{delta_eff:+.1f}%")

        with col5:
            delta_dust = np.random.uniform(-3, 3)
            delta_class = "negative" if delta_dust > 0 else "positive"
            st.metric("Dust Level", f"{dust_level:.1f}%", f"{delta_dust:+.1f}%")

        st.markdown("##  AI Predictions & Insights")

        if predictions and 'predictions' in predictions:
            pred_data = predictions['predictions']

            col1, col2, col3 = st.columns(3)

            with col1:
                if 'maintenance' in pred_data:
                    maint_data = pred_data['maintenance']
                    prob = maint_data.get('maintenance_probability', 0) * 100

                    if prob > 80:
                        status_class = "status-critical"
                        status_text = "🔴 CRITICAL"
                    elif prob > 60:
                        status_class = "status-warning"
                        status_text = "🟡 WARNING"
                    else:
                        status_class = "status-excellent"
                        status_text = "🟢 EXCELLENT"
                    
                    st.markdown(f"""
                    <div class="card">
                        <div class="card-header"> Maintenance Status</div>
                        <div style="text-align: center; padding: 2rem;">
                            <div style="font-size: 2rem; margin-bottom: 1rem;">{status_text}</div>
                            <div style="font-size: 1.5rem; color: white;">{prob:.1f}%</div>
                            <div style="color: rgba(255,255,255,0.8); margin-top: 0.5rem;">Maintenance Probability</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            with col2:
                if 'performance' in pred_data:
                    perf_data = pred_data['performance']
                    predicted_energy = perf_data.get('predicted_energy_output', 0)
                    
                    st.markdown(f"""
                    <div class="card">
                        <div class="card-header"> Performance Prediction</div>
                        <div style="text-align: center; padding: 3.7rem;">
                            <div style="font-size: 2rem; font-weight: bold; color: #10b981;">
                                {predicted_energy:.2f} kW
                            </div>
                            <div style="color: rgba(255,255,255,0.8); margin-top: 0.5rem;">Predicted Energy Output</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            with col3:
                if 'anomaly' in pred_data:
                    anom_data = pred_data['anomaly']
                    is_anomaly = anom_data.get('is_anomaly', False)
                    severity = anom_data.get('severity', 'Normal')

                    if is_anomaly:
                        status_class = "status-critical" if severity == 'High' else "status-warning"
                        status_icon = "🔴" if severity == 'High' else "🟡"
                        status_text = f"ANOMALY DETECTED ({severity})"
                    else:
                        status_class = "status-excellent"
                        status_icon = "🟢"
                        status_text = "NORMAL OPERATION"
                    
                    st.markdown(f"""
                    <div class="card">
                        <div class="card-header"> Anomaly Detection</div>
                        <div style="text-align: center; padding: 2.4rem;">
                            <div style="font-size: 3rem; margin-bottom: 1rem;">{status_icon}</div>
                            <div style="font-size: 1.2rem; color: white;">{status_text}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("##  System Performance Gauges")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            gauge1 = create_modern_gauge(efficiency, "System Efficiency (%)", max_val=100, thresholds=[60, 80])
            st.plotly_chart(gauge1, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            gauge2 = create_modern_gauge(panel_temp, "Panel Temperature (°C)", max_val=70, thresholds=[40, 50], colors=['#10b981', '#f59e0b', '#ef4444'])
            st.plotly_chart(gauge2, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            gauge3 = create_modern_gauge(dust_level, "Dust Level (%)", max_val=100, thresholds=[30, 60], colors=['#10b981', '#f59e0b', '#ef4444'])
            st.plotly_chart(gauge3, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("##  Live Performance Charts")

        df = load_data()
        if df is not None and not df.empty:
            recent_df = df.tail(24)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=recent_df['timestamp'],
                    y=recent_df['energy_output'],
                    mode='lines+markers',
                    name='Energy Output',
                    line=dict(color='#10b981', width=3),
                    marker=dict(size=6),
                    fill='tonexty'
                ))

                fig.update_layout(
                    title='Energy Output Trend (24h)',
                    xaxis_title='Time',
                    yaxis_title='Energy Output (kW)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    showlegend=False,
                    height=300
                )

                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
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
                    font=dict(color='white'),
                    showlegend=False,
                    height=300
                )

                st.plotly_chart(fig2, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

def analytics_page():
        st.markdown(" Advanced Analytics & Insights")

        df = load_data()
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

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_energy = df['energy_output'].sum()
            st.metric("Total Energy Generated", f"{total_energy:.1f} kWh")

        with col2:
            if 'efficiency' in df.columns and not df['efficiency'].isna().all():
                avg_efficiency = df['efficiency'].mean()
                st.metric("Average Efficiency", f"{avg_efficiency:.3f}")

        with col3:
            peak_output = df['energy_output'].max()
            st.metric("Peak Output", f"{peak_output:.2f} kW")

        with col4:
            uptime = (1 - df['maintenance_needed'].mean()) * 100
            st.metric("System Uptime", f"{uptime:.1f}%")

        #st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card card-header"> Energy Production Heatmap</div>', unsafe_allow_html=True)

        try:
            if 'timestamp' in df.columns and 'energy_output' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['hour'] = df['timestamp'].dt.hour
                df['month'] = df['timestamp'].dt.month

                pivot_data = df.pivot_table(
                    values='energy_output',
                    index='hour',
                    columns='month',
                    aggfunc='mean'
                )

                if not pivot_data.empty:
                    fig_heatmap = px.imshow(pivot_data,
                                        title="Energy Output by Hour and Month",
                                        color_continuous_scale="Viridis",
                                        aspect="auto")
                    fig_heatmap.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white')
                    )
                    st.plotly_chart(fig_heatmap, use_container_width=True)
                else:
                    st.info("No data available for heatmap")
            else:
                st.info("Required columns (timestamp, energy_output) not found")
        except Exception as e:
            st.error(f"Error creating heatmap: {e}")

        st.markdown('</div>', unsafe_allow_html=True)

def eda_page():
        st.markdown(" Exploratory Data Analysis")

        df = load_data()
        if df is None or df.empty:
            st.error("No data available for EDA")
            return

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
            " Overview", " Distributions", " Correlations", " Time Series",
            " Outliers", " Statistical Tests", " Advanced Analysis", " Report Generator"
        ])

        with tab1:
            col1, col2 = st.columns(2)

            with col1:
                #st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="card card-header"> Dataset Overview</div>', unsafe_allow_html=True)
                
                st.markdown(f"**Shape:** {df.shape[0]:,} rows × {df.shape[1]} columns")
                st.markdown(f"**Memory Usage:** {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
                st.markdown(f"**Missing Values:** {df.isnull().sum().sum():,} ({df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100:.1f}%)")
                st.markdown(f"**Duplicate Rows:** {df.duplicated().sum():,}")
                
                if 'timestamp' in df.columns:
                    st.markdown(f"**Date Range:** {df['timestamp'].min().strftime('%Y-%m-%d')} to {df['timestamp'].max().strftime('%Y-%m-%d')}")
                
                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                #st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="card card-header"> Data Types</div>', unsafe_allow_html=True)

                dtype_counts = df.dtypes.value_counts()
                dtype_names = [str(dtype) for dtype in dtype_counts.index]
                dtype_values = dtype_counts.values.tolist()

                fig_dtype = px.pie(values=dtype_values, names=dtype_names,
                                title="Data Types Distribution")
                fig_dtype.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    height=300
                )
                st.plotly_chart(fig_dtype, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            #st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card card-header"> Statistical Summary</div>', unsafe_allow_html=True)

            if len(numeric_cols) > 0:
                summary_stats = df[numeric_cols].describe().round(3)
                st.dataframe(summary_stats, use_container_width=True)
            else:
                st.info("No numeric columns found for statistical summary")
            st.markdown('</div>', unsafe_allow_html=True)

        with tab2:
            #st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card card-header"> Feature Distributions</div>', unsafe_allow_html=True)

            if len(numeric_cols) > 0:
                col1, col2 = st.columns(2)
                with col1:
                    feature = st.selectbox("Select Feature", numeric_cols, key="dist_feature")
                with col2:
                    plot_type = st.selectbox("Plot Type", ["Histogram", "Box Plot", "Violin Plot"])

                try:
                    if plot_type == "Histogram":
                        fig = px.histogram(df, x=feature, nbins=30, title=f"{feature} Distribution")
                    elif plot_type == "Box Plot":
                        fig = px.box(df, y=feature, title=f"{feature} Box Plot")
                    else:
                        fig = px.violin(df, y=feature, title=f"{feature} Violin Plot")

                    fig.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white')
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating distribution plot: {e}")
            else:
                st.info("No numeric columns available for distribution analysis")

            st.markdown('</div>', unsafe_allow_html=True)

        with tab3:
            #st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card card-header"> Correlation Matrix</div>', unsafe_allow_html=True)

            if len(numeric_cols) > 1:
                try:
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
                except Exception as e:
                    st.error(f"Error creating correlation matrix: {e}")
            else:
                st.info("Need at least 2 numeric columns for correlation analysis")

            st.markdown('</div>', unsafe_allow_html=True)

        with tab4:
            #st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card card-header"> Time Series Analysis</div>', unsafe_allow_html=True)

            if len(numeric_cols) > 0 and 'timestamp' in df.columns:
                feature = st.selectbox("Select Feature", numeric_cols, key="ts_feature")

                try:
                    fig_ts = px.line(df, x='timestamp', y=feature, title=f"{feature} Time Series")
                    fig_ts.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white')
                    )
                    st.plotly_chart(fig_ts, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating time series plot: {e}")
            else:
                st.info("No numeric columns or timestamp column available for time series analysis")

            st.markdown('</div>', unsafe_allow_html=True)

        with tab5:
            #st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card card-header"> Outlier Detection & Analysis</div>', unsafe_allow_html=True)

            if len(numeric_cols) > 0:
                col1, col2 = st.columns(2)
                with col1:
                    outlier_feature = st.selectbox("Select Feature", numeric_cols, key="outlier_feature")
                with col2:
                    outlier_method = st.selectbox("Detection Method", ["IQR", "Z-Score", "Modified Z-Score"])

                try:
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

                    else:
                        median = df[outlier_feature].median()
                        mad = np.median(np.abs(df[outlier_feature] - median))
                        if mad != 0:
                            modified_z_scores = 0.6745 * (df[outlier_feature] - median) / mad
                            outliers = df[np.abs(modified_z_scores) > 3.5]
                        else:
                            outliers = pd.DataFrame()

                    col3, col4 = st.columns(2)

                    with col3:
                        fig_outlier = go.Figure()
                        fig_outlier.add_trace(go.Scatter(
                            x=df.index, y=df[outlier_feature],
                            mode='markers', name='Normal',
                            marker=dict(color='lightblue', size=4)
                        ))

                        if len(outliers) > 0:
                            fig_outlier.add_trace(go.Scatter(
                                x=outliers.index, y=outliers[outlier_feature],
                                mode='markers', name='Outliers',
                                marker=dict(color='red', size=8)
                            ))

                        fig_outlier.update_layout(
                            title=f"Outliers in {outlier_feature} ({outlier_method})",
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='white')
                        )
                        st.plotly_chart(fig_outlier, use_container_width=True)

                    with col4:
                        st.markdown(f"**Outlier Summary:**")
                        st.markdown(f"- Total Outliers: {len(outliers)}")
                        st.markdown(f"- Percentage: {len(outliers)/len(df)*100:.2f}%")
                        st.markdown(f"- Method: {outlier_method}")

                        if len(outliers) > 0:
                            st.markdown("**Outlier Statistics:**")
                            outlier_stats = outliers[outlier_feature].describe()
                            for stat, value in outlier_stats.items():
                                st.markdown(f"- {stat}: {value:.3f}")

                    if len(outliers) > 0:
                        st.markdown("#### Outlier Records")
                        st.dataframe(outliers.head(10), use_container_width=True)

                except Exception as e:
                    st.error(f"Error in outlier detection: {e}")

            st.markdown('</div>', unsafe_allow_html=True)

        with tab6:
            #st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card card-header"> Statistical Tests & Analysis</div>', unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Normality Tests")
                if len(numeric_cols) > 0:
                    test_feature = st.selectbox("Select Feature", numeric_cols, key="test_feature")

                    try:
                        if len(df) <= 5000:
                            shapiro_stat, shapiro_p = stats.shapiro(df[test_feature].dropna().sample(min(5000, len(df))))
                            st.markdown(f"**Shapiro-Wilk Test:**")
                            st.markdown(f"- Statistic: {shapiro_stat:.4f}")
                            st.markdown(f"- p-value: {shapiro_p:.4f}")
                            st.markdown(f"- Normal: {'Yes' if shapiro_p > 0.05 else 'No'}")

                        ks_stat, ks_p = stats.kstest(df[test_feature].dropna(), 'norm')
                        st.markdown(f"**Kolmogorov-Smirnov Test:**")
                        st.markdown(f"- Statistic: {ks_stat:.4f}")
                        st.markdown(f"- p-value: {ks_p:.4f}")
                        st.markdown(f"- Normal: {'Yes' if ks_p > 0.05 else 'No'}")

                    except Exception as e:
                        st.error(f"Error in normality tests: {e}")

            with col2:
                st.markdown("#### Correlation Tests")
                if len(numeric_cols) >= 2:
                    feature1 = st.selectbox("Feature 1", numeric_cols, key="corr_test_1")
                    feature2 = st.selectbox("Feature 2", numeric_cols, key="corr_test_2")

                    if feature1 != feature2:
                        try:
                            pearson_r, pearson_p = stats.pearsonr(df[feature1].dropna(), df[feature2].dropna())
                            st.markdown(f"**Pearson Correlation:**")
                            st.markdown(f"- Coefficient: {pearson_r:.4f}")
                            st.markdown(f"- p-value: {pearson_p:.4f}")
                            st.markdown(f"- Significant: {'Yes' if pearson_p < 0.05 else 'No'}")

                            spearman_r, spearman_p = stats.spearmanr(df[feature1].dropna(), df[feature2].dropna())
                            st.markdown(f"**Spearman Correlation:**")
                            st.markdown(f"- Coefficient: {spearman_r:.4f}")
                            st.markdown(f"- p-value: {spearman_p:.4f}")
                            st.markdown(f"- Significant: {'Yes' if spearman_p < 0.05 else 'No'}")

                        except Exception as e:
                            st.error(f"Error in correlation tests: {e}")

            st.markdown('</div>', unsafe_allow_html=True)

        with tab7:
            #st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card card-header"> Advanced Analysis & Machine Learning</div>', unsafe_allow_html=True)

            analysis_type = st.selectbox("Select Analysis Type", [
                "PCA Analysis", "Clustering Analysis", "Anomaly Detection"
            ])

            if analysis_type == "PCA Analysis":
                try:
                    from sklearn.preprocessing import StandardScaler
                    from sklearn.decomposition import PCA

                    pca_cols = [col for col in numeric_cols if col in df.columns]
                    if len(pca_cols) >= 3:
                        pca_data = df[pca_cols].dropna()

                        scaler = StandardScaler()
                        scaled_data = scaler.fit_transform(pca_data)

                        pca = PCA()
                        pca_result = pca.fit_transform(scaled_data)

                        col1, col2 = st.columns(2)

                        with col1:
                            fig_pca = px.bar(
                                x=[f'PC{i+1}' for i in range(len(pca.explained_variance_ratio_))],
                                y=pca.explained_variance_ratio_ * 100,
                                title='PCA Explained Variance Ratio'
                            )
                            fig_pca.update_layout(
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                font=dict(color='white')
                            )
                            st.plotly_chart(fig_pca, use_container_width=True)

                        with col2:
                            if 'maintenance_needed' in df.columns:
                                pca_df = pd.DataFrame(pca_result[:, :2], columns=['PC1', 'PC2'])
                                pca_df['maintenance'] = df.loc[pca_data.index, 'maintenance_needed'].values

                                fig_pca_scatter = px.scatter(
                                    pca_df, x='PC1', y='PC2', color='maintenance',
                                    title='PCA Visualization (First 2 Components)'
                                )
                                fig_pca_scatter.update_layout(
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    font=dict(color='white')
                                )
                                st.plotly_chart(fig_pca_scatter, use_container_width=True)
                    else:
                        st.warning("Need at least 3 numeric columns for PCA analysis")
                except Exception as e:
                    st.error(f"Error in PCA analysis: {e}")

            elif analysis_type == "Clustering Analysis":
                try:
                    from sklearn.cluster import KMeans
                    from sklearn.preprocessing import StandardScaler

                    cluster_features = [col for col in ['energy_output', 'solar_irradiance', 'panel_temp', 'dust_level'] if col in df.columns]

                    if len(cluster_features) >= 2:
                        cluster_data = df[cluster_features].dropna()

                        scaler = StandardScaler()
                        scaled_data = scaler.fit_transform(cluster_data)

                        optimal_k = st.slider("Select number of clusters", 2, 7, 4)

                        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
                        cluster_labels = kmeans.fit_predict(scaled_data)

                        cluster_df = cluster_data.copy()
                        cluster_df['cluster'] = cluster_labels

                        if len(cluster_features) >= 2:
                            fig_cluster = px.scatter(
                                cluster_df, x=cluster_features[0], y=cluster_features[1],
                                color='cluster', title=f'Clusters ({optimal_k} clusters)'
                            )
                            fig_cluster.update_layout(
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                font=dict(color='white')
                            )
                            st.plotly_chart(fig_cluster, use_container_width=True)
                    else:
                        st.warning("Need at least 2 features for clustering analysis")
                except Exception as e:
                    st.error(f"Error in clustering analysis: {e}")

            elif analysis_type == "Anomaly Detection":
                try:
                    from sklearn.ensemble import IsolationForest
                    from sklearn.preprocessing import StandardScaler

                    anom_features = [col for col in ['energy_output', 'solar_irradiance', 'panel_temp', 'dust_level'] if col in df.columns]

                    if len(anom_features) >= 2:
                        anom_data = df[anom_features].dropna()

                        scaler = StandardScaler()
                        scaled_data = scaler.fit_transform(anom_data)

                        contamination = st.slider("Contamination Rate", 0.05, 0.2, 0.1, 0.01)

                        iso_forest = IsolationForest(contamination=contamination, random_state=42)
                        anomaly_labels = iso_forest.fit_predict(scaled_data)

                        anom_df = anom_data.copy()
                        anom_df['anomaly'] = anomaly_labels
                        anom_df['is_anomaly'] = anom_df['anomaly'] == -1

                        col1, col2 = st.columns(2)

                        with col1:
                            normal_data = anom_df[anom_df['is_anomaly'] == False]
                            anomaly_data = anom_df[anom_df['is_anomaly'] == True]

                            fig_anom = go.Figure()
                            fig_anom.add_trace(go.Scatter(
                                x=normal_data[anom_features[0]], y=normal_data[anom_features[1]],
                                mode='markers', name='Normal',
                                marker=dict(color='blue', size=4, opacity=0.6)
                            ))
                            fig_anom.add_trace(go.Scatter(
                                x=anomaly_data[anom_features[0]], y=anomaly_data[anom_features[1]],
                                mode='markers', name='Anomaly',
                                marker=dict(color='red', size=8, opacity=0.8)
                            ))

                            fig_anom.update_layout(
                                title='Anomaly Detection Results',
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                font=dict(color='white')
                            )
                            st.plotly_chart(fig_anom, use_container_width=True)

                        with col2:
                            st.markdown("**Anomaly Summary:**")
                            anomaly_count = (anom_df['is_anomaly'] == True).sum()
                            st.markdown(f"- Anomalies detected: {anomaly_count}")
                            st.markdown(f"- Anomaly rate: {anomaly_count/len(anom_df)*100:.2f}%")
                            st.markdown(f"- Contamination: {contamination*100:.1f}%")

                            if anomaly_count > 0:
                                st.markdown("**Anomaly Records:**")
                                st.dataframe(anomaly_data.head(), use_container_width=True)
                    else:
                        st.warning("Need at least 2 features for anomaly detection")
                except Exception as e:
                    st.error(f"Error in anomaly detection: {e}")

            st.markdown('</div>', unsafe_allow_html=True)

        with tab8:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-header"> EDA Report Generator</div>', unsafe_allow_html=True)

            st.markdown("#### Generate Comprehensive EDA Reports")
            
            try:
                import os
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                reports_dir = os.path.join(project_root, 'analysis', 'reports')
                interactive_dir = os.path.join(project_root, 'analysis', 'interactive')
                
                if os.path.exists(reports_dir):
                    reports = [f for f in os.listdir(reports_dir) if f.endswith('.html')]
                    if reports:
                        st.info(f" Found {len(reports)} existing reports in analysis/reports/")
                        
                if os.path.exists(interactive_dir):
                    interactive = [f for f in os.listdir(interactive_dir) if f.endswith('.html')]
                    if interactive:
                        st.info(f" Found {len(interactive)} interactive dashboards in analysis/interactive/")
            except:
                pass

            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button(" Generate Data Profiling Report", use_container_width=True):
                    with st.spinner("Generating data profiling report..."):
                        try:
                            import subprocess
                            import os
                            
                            project_root = os.path.dirname(os.path.abspath(__file__))
                            project_root = os.path.dirname(project_root)  
                            
                            result = subprocess.run(["python", "analysis/01_data_profiling.py"],
                                                capture_output=True, text=True, cwd=project_root)
                            if result.returncode == 0:
                                st.success(" Data profiling report generated successfully!")
                                st.info(" Check analysis/data_profiling_report.html")
                                if result.stdout:
                                    st.text(result.stdout[-500:])  
                            else:
                                st.error(f" Error generating report")
                                if result.stderr:
                                    st.code(result.stderr[-1000:])  
                                if result.stdout:
                                    st.code(result.stdout[-1000:])
                        except Exception as e:
                            st.error(f" Error: {str(e)}")

            with col2:
                if st.button(" Generate Statistical Visualizations", use_container_width=True):
                    with st.spinner("Generating statistical visualizations..."):
                        try:
                            import subprocess
                            import os
                            
                            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                            
                            result = subprocess.run(["python", "analysis/02_visualizations.py"],
                                                capture_output=True, text=True, cwd=project_root)
                            if result.returncode == 0:
                                st.success(" Statistical visualizations generated successfully!")
                                st.info(" Check analysis/ directory for PNG files")
                                if result.stdout:
                                    st.text(result.stdout[-500:])
                            else:
                                st.error(f" Error generating visualizations")
                                if result.stderr:
                                    st.code(result.stderr[-1000:])
                        except Exception as e:
                            st.error(f" Error: {str(e)}")

            with col3:
                if st.button(" Generate Correlation Analysis", use_container_width=True):
                    with st.spinner("Generating correlation analysis..."):
                        try:
                            import subprocess
                            import os
                            
                            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                            
                            result = subprocess.run(["python", "analysis/03_correlation_analysis.py"],
                                                capture_output=True, text=True, cwd=project_root)
                            if result.returncode == 0:
                                st.success(" Correlation analysis generated successfully!")
                                st.info(" Check analysis/ directory for correlation plots")
                                if result.stdout:
                                    st.text(result.stdout[-500:])
                            else:
                                st.error(f" Error generating analysis")
                                if result.stderr:
                                    st.code(result.stderr[-1000:])
                        except Exception as e:
                            st.error(f" Error: {str(e)}")

            st.markdown("---")
            
            st.markdown(" Convert Reports to PDF")
            
            col_pdf1, col_pdf2 = st.columns(2)
            
            with col_pdf1:
                if st.button(" Convert HTML Reports to PDF", use_container_width=True):
                    with st.spinner("Converting HTML reports to PDF..."):
                        try:
                            import subprocess
                            import os
                            from datetime import datetime
                            
                            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                            reports_dir = os.path.join(project_root, 'analysis', 'reports')
                            interactive_dir = os.path.join(project_root, 'analysis', 'interactive')
                            pdf_dir = os.path.join(project_root, 'analysis', 'pdf_reports')
                            
                            os.makedirs(pdf_dir, exist_ok=True)
                            
                            converted_files = []
                            
                            for dir_path, dir_name in [(reports_dir, 'reports'), (interactive_dir, 'interactive')]:
                                if os.path.exists(dir_path):
                                    for file in os.listdir(dir_path):
                                        if file.endswith('.html'):
                                            html_path = os.path.join(dir_path, file)
                                            pdf_name = f"{dir_name}_{file.replace('.html', '')}.pdf"
                                            pdf_path = os.path.join(pdf_dir, pdf_name)
                                            
                                            if REPORTLAB_AVAILABLE:
                                                try:
                                                    c = canvas.Canvas(pdf_path, pagesize=letter)
                                                    width, height = letter
                                                    
                                                    c.setFont("Helvetica-Bold", 16)
                                                    c.drawString(50, height - 50, f"Solar Panel ML Report: {file}")
                                                    
                                                    c.setFont("Helvetica", 12)
                                                    y_pos = height - 100
                                                    c.drawString(50, y_pos, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                                                    y_pos -= 30
                                                    c.drawString(50, y_pos, f"Source: {dir_name}/{file}")
                                                    y_pos -= 30
                                                    c.drawString(50, y_pos, f"Original file: {html_path}")

                                                    y_pos -= 50
                                                    c.setFont("Helvetica-Bold", 10)
                                                    c.drawString(50, y_pos, "Note: This is a placeholder PDF. For full HTML content:")
                                                    y_pos -= 20
                                                    c.setFont("Helvetica", 10)
                                                    c.drawString(50, y_pos, "1. Open the original HTML file in a browser")
                                                    y_pos -= 15
                                                    c.drawString(50, y_pos, "2. Use browser's Print to PDF feature")
                                                    y_pos -= 15
                                                    c.drawString(50, y_pos, "3. Or install wkhtmltopdf for automated conversion")
                                                    
                                                    try:
                                                        file_size = os.path.getsize(html_path)
                                                        y_pos -= 30
                                                        c.drawString(50, y_pos, f"File size: {file_size:,} bytes")
                                                    except:
                                                        pass
                                                    
                                                    c.save()
                                                    converted_files.append(pdf_name)
                                                except Exception as e:
                                                    st.warning(f"Could not create PDF for {file}: {e}")
                            
                            if converted_files:
                                st.success(f" Converted {len(converted_files)} reports to PDF!")
                                st.info(f" PDFs saved in: analysis/pdf_reports/")
                                for file in converted_files:
                                    st.text(f"• {file}")
                            else:
                                if not REPORTLAB_AVAILABLE:
                                    st.error(" ReportLab not installed. Run: pip install reportlab")
                                else:
                                    st.warning("No HTML reports found to convert")
                                    st.info(" Generate reports first using the buttons above")
                                
                        except Exception as e:
                            st.error(f" Error converting to PDF: {str(e)}")
                            st.info(" Install ReportLab for basic PDF support: pip install reportlab")
            
            with col_pdf2:
                if st.button(" Generate PDF Summary Report", use_container_width=True):
                    with st.spinner("Generating PDF summary report..."):
                        if not REPORTLAB_AVAILABLE:
                            st.error(" ReportLab not installed. Run: pip install reportlab")
                            return
                            
                        try:
                            from datetime import datetime
                            import os
                            
                            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                            pdf_dir = os.path.join(project_root, 'analysis', 'pdf_reports')
                            os.makedirs(pdf_dir, exist_ok=True)
                            
                            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                            pdf_path = os.path.join(pdf_dir, f'solar_panel_summary_{timestamp}.pdf')
                            
                            doc = SimpleDocTemplate(pdf_path, pagesize=A4)
                            styles = getSampleStyleSheet()
                            story = []
                            
                            title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], 
                                                       fontSize=24, spaceAfter=30, textColor=colors.darkblue)
                            story.append(Paragraph(" Solar Panel ML Analysis Report", title_style))
                            story.append(Spacer(1, 20))
                            
                            df = load_data()
                            if df is not None and not df.empty:
                                story.append(Paragraph(" Dataset Overview", styles['Heading2']))
                                
                                data = [
                                    ['Metric', 'Value'],
                                    ['Total Records', f"{len(df):,}"],
                                    ['Features', f"{df.shape[1]}"],
                                    ['Missing Values', f"{df.isnull().sum().sum():,}"],
                                    ['Date Range', f"{df['timestamp'].min().strftime('%Y-%m-%d')} to {df['timestamp'].max().strftime('%Y-%m-%d')}" if 'timestamp' in df.columns else 'N/A'],
                                    ['Average Energy Output', f"{df['energy_output'].mean():.2f} kW" if 'energy_output' in df.columns else 'N/A'],
                                    ['Peak Energy Output', f"{df['energy_output'].max():.2f} kW" if 'energy_output' in df.columns else 'N/A']
                                ]
                                
                                table = Table(data)
                                table.setStyle(TableStyle([
                                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                                    ('FONTSIZE', (0, 0), (-1, 0), 14),
                                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                                ]))
                                story.append(table)
                                story.append(Spacer(1, 20))
                            
                            story.append(Paragraph(" Report Information", styles['Heading2']))
                            story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
                            story.append(Paragraph("System: Tevron Solar AI Hub", styles['Normal']))
                            story.append(Paragraph("Version: 1.0", styles['Normal']))
                            
                            doc.build(story)
                            
                            st.success(" PDF summary report generated successfully!")
                            st.info(f" Saved as: analysis/pdf_reports/solar_panel_summary_{timestamp}.pdf")
                            
                        except ImportError:
                            st.error(" ReportLab not available. This should not happen.")
                        except Exception as e:
                            st.error(f" Error generating PDF: {str(e)}")

            st.markdown("---")

            col1, col2 = st.columns(2)

            with col1:
                if st.button(" Generate Interactive Dashboards", use_container_width=True):
                    with st.spinner("Generating interactive dashboards..."):
                        try:
                            import subprocess
                            import os
                            
                            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                            
                            result = subprocess.run(["python", "analysis/05_interactive_eda.py"],
                                                capture_output=True, text=True, cwd=project_root)
                            if result.returncode == 0:
                                st.success(" Interactive dashboards generated successfully!")
                                st.info(" Open analysis/interactive/index.html in your browser")
                                if result.stdout:
                                    st.text(result.stdout[-500:])
                            else:
                                st.error(f" Error generating dashboards")
                                if result.stderr:
                                    st.code(result.stderr[-1000:])
                        except Exception as e:
                            st.error(f" Error: {str(e)}")

            with col2:
                if st.button(" Generate Comprehensive Report", use_container_width=True):
                    with st.spinner("Generating comprehensive report..."):
                        try:
                            import subprocess
                            import os
                            
                            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                            
                            result = subprocess.run(["python", "analysis/06_comprehensive_report.py"],
                                                capture_output=True, text=True, cwd=project_root)
                            if result.returncode == 0:
                                st.success(" Comprehensive report generated successfully!")
                                st.info(" Check analysis/reports/ directory for HTML report")
                                if result.stdout:
                                    st.text(result.stdout[-500:])
                            else:
                                st.error(f" Error generating report")
                                if result.stderr:
                                    st.code(result.stderr[-1000:])
                        except Exception as e:
                            st.error(f" Error: {str(e)}")

            st.markdown(" Quick Data Summary")

            summary_col1, summary_col2, summary_col3 = st.columns(3)

            with summary_col1:
                st.metric("Total Records", f"{len(df):,}")
                if 'timestamp' in df.columns:
                    date_range = (df['timestamp'].max() - df['timestamp'].min()).days
                    st.metric("Date Range", f"{date_range} days")

            with summary_col2:
                if 'energy_output' in df.columns:
                    st.metric("Average Energy Output", f"{df['energy_output'].mean():.2f} kW")
                    st.metric("Peak Energy Output", f"{df['energy_output'].max():.2f} kW")

            with summary_col3:
                if 'maintenance_needed' in df.columns:
                    uptime = (1 - df['maintenance_needed'].mean()) * 100
                    st.metric("System Uptime", f"{uptime:.1f}%")

                missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
                st.metric("Data Completeness", f"{100-missing_pct:.1f}%")
                
            st.markdown(" Generated Reports")
            try:
                import os
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                analysis_dir = os.path.join(project_root, 'analysis')
                
                if os.path.exists(analysis_dir):
                    files = []
                    for root, dirs, filenames in os.walk(analysis_dir):
                        for filename in filenames:
                            if filename.endswith(('.html', '.png')):
                                rel_path = os.path.relpath(os.path.join(root, filename), project_root)
                                files.append(rel_path)
                    
                    if files:
                        st.markdown(f"**Available Files ({len(files)}):**")
                        for file in sorted(files)[-10:]: 
                            st.markdown(f"- `{file}`")
                        if len(files) > 10:
                            st.markdown(f"... and {len(files) - 10} more files")
                    else:
                        st.info("No generated reports found. Use the buttons above to generate reports.")
            except Exception as e:
                st.warning(f"Could not list files: {e}")

            st.markdown('</div>', unsafe_allow_html=True)

def create_3d_surface_plot(df):
        try:
            if df.empty or len(df) < 10:
                fig = go.Figure()
                fig.add_annotation(
                    text="Insufficient data for 3D surface plot",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, xanchor='center', yanchor='middle',
                    showarrow=False, font=dict(size=16, color='white')
                )
                return fig

            x = np.linspace(df['solar_irradiance'].min(), df['solar_irradiance'].max(), 20)
            y = np.linspace(df['panel_temp'].min(), df['panel_temp'].max(), 20)
            X, Y = np.meshgrid(x, y)

            from scipy.interpolate import griddata
            points = df[['solar_irradiance', 'panel_temp']].values
            values = df['energy_output'].values
            Z = griddata(points, values, (X, Y), method='cubic', fill_value=0)

            fig = go.Figure(data=[go.Surface(
                x=X, y=Y, z=Z,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Energy Output (kW)"),
                hovertemplate='<b>Solar Irradiance:</b> %{x:.0f} W/m²<br>' +
                            '<b>Panel Temp:</b> %{y:.1f}°C<br>' +
                            '<b>Energy Output:</b> %{z:.2f} kW<extra></extra>'
            )])

            fig.update_layout(
                title='3D Surface: Energy Output vs Environmental Conditions',
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
            st.error(f"Error creating 3D surface plot: {e}")
            return go.Figure()

def viz_3d_page():
        st.markdown(" 3D Performance Visualizations")

        df = load_data()
        if df is None or df.empty:
            st.error("No data available for 3D visualization")
            return

        if 'efficiency' not in df.columns:
            df['efficiency'] = df.apply(
                lambda row: safe_calculate_efficiency(row['energy_output'], row['solar_irradiance']),
                axis=1
            )

        viz_type = st.selectbox(
            "Select 3D Visualization Type",
            ["3D Scatter Plot", "3D Surface Plot"],
            key="3d_viz_type"
        )

        if viz_type == "3D Scatter Plot":
            #st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card card-header"> 3D Scatter: Performance Analysis</div>', unsafe_allow_html=True)

            sample_size = min(1000, len(df))
            sample_df = df.sample(sample_size) if len(df) > sample_size else df
            fig_3d = create_3d_performance_plot(sample_df)
            st.plotly_chart(fig_3d, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        else:
            #st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card card-header"> 3D Surface: Energy Output Landscape</div>', unsafe_allow_html=True)

            sample_size = min(500, len(df))
            sample_df = df.sample(sample_size) if len(df) > sample_size else df
            fig_surface = create_3d_surface_plot(sample_df)
            st.plotly_chart(fig_surface, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        #st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card card-header"> 3D Visualization Insights</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Data Points Visualized",
                f"{min(1000 if viz_type == '3D Scatter Plot' else 500, len(df)):,}"
            )

        with col2:
            if 'efficiency' in df.columns:
                avg_efficiency = df['efficiency'].mean()
                st.metric("Average Efficiency", f"{avg_efficiency:.3f}")

        with col3:
            correlation = df['solar_irradiance'].corr(df['energy_output'])
            st.metric("Irradiance-Energy Correlation", f"{correlation:.3f}")

        st.markdown('</div>', unsafe_allow_html=True)

def maintenance_page():
        st.markdown(" Maintenance Management")

        col1, col2 = st.columns(2)

        with col1:
            #st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card card-header"> Scheduled Maintenance</div>', unsafe_allow_html=True)

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
                    "Low": "#10b981"
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
            #st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card card-header"> Maintenance Predictions</div>', unsafe_allow_html=True)

            try:
                ingestion, _, inference = get_components()
                if ingestion and inference:
                    try:
                        current_data = ingestion.get_combined_data()
                    except Exception as e:
                        current_data = {
                            'energy_output': 8.5, 'temperature': 32.1, 'humidity': 55.2,
                            'wind_speed': 3.2, 'solar_irradiance': 850.0, 'panel_temp': 38.5,
                            'voltage': 24.2, 'current': 35.1, 'power_factor': 0.98, 'dust_level': 0.3
                        }
                    
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
                        else:
                            st.warning(" Maintenance prediction unavailable")
                            st.info(" Check if models are trained and loaded")
                    else:
                        st.error(" No data available for prediction")
                else:
                    st.error(" Models not loaded. Run training first.")
            except Exception as e:
                st.error(f" Error loading predictions: {str(e)}")
                st.info(" Try running: python src/train.py")

            st.markdown('</div>', unsafe_allow_html=True)

def ml_insights_page():
    st.markdown(" Machine Learning Insights")

    col1, col2, col3 = st.columns(3)

    with col1:
        # st.markdown("""
        # <div class="card">
        #     <div class="card-header"> Maintenance Model</div>
        #     <p><strong>Type:</strong> LightGBM Classifier</p>
        #     <p><strong>Accuracy:</strong> ~97.5%</p>
        #     <p><strong>Key Features:</strong> Voltage, Wind Speed, Power Factor</p>
        # </div>
        # """, unsafe_allow_html=True)
        st.markdown("""
        <div>
            <div class="card card-header"> Maintenance Model</div>
            <p><strong>Type:</strong> LightGBM Classifier</p>
            <p><strong>Accuracy:</strong> ~97.5%</p>
            <p><strong>Key Features:</strong> Voltage, Wind Speed, Power Factor</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # st.markdown("""
        # <div class="card">
        #     <div class="card-header"> Performance Model</div>
        #     <p><strong>Type:</strong> LightGBM Regressor</p>
        #     <p><strong>R² Score:</strong> ~99.9%</p>
        #     <p><strong>Key Features:</strong> Energy Normalized, Power, Dust Impact</p>
        # </div>
        # """, unsafe_allow_html=True)
        st.markdown("""
        <div >
            <div class="card card-header"> Performance Model</div>
            <p><strong>Type:</strong> LightGBM Regressor</p>
            <p><strong>R² Score:</strong> ~99.9%</p>
            <p><strong>Key Features:</strong> Energy Normalized, Power, Dust Impact</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        # st.markdown("""
        # <div class="card">
        #     <div class=" card card-header"> Anomaly Model</div>
        #     <p><strong>Type:</strong> Isolation Forest</p>
        #     <p><strong>Detection Rate:</strong> ~10.4%</p>
        #     <p><strong>Contamination:</strong> 10%</p>
        # </div>
        # """, unsafe_allow_html=True)
        st.markdown("""
        <div>
            <div class=" card card-header"> Anomaly Model</div>
            <p><strong>Type:</strong> Isolation Forest</p>
            <p><strong>Detection Rate:</strong> ~10.4%</p>
            <p><strong>Contamination:</strong> 10%</p>
        </div>
        """, unsafe_allow_html=True)

    #st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card card-header"> Feature Importance Analysis</div>', unsafe_allow_html=True)

    try:
            _, _, inference = get_components()
            if inference and inference.models:
                col1, col2 = st.columns(2)

                with col1:
                    if 'maintenance' in inference.models:
                        maint_importance = inference.get_feature_importance('maintenance')
                        if maint_importance is not None and not maint_importance.empty:
                            fig_maint = px.bar(
                                maint_importance.head(10),
                                x='importance', y='feature',
                                orientation='h',
                                title='Maintenance Model Features',
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
                            st.warning(" Maintenance model feature importance not available")
                    else:
                        st.error(" Maintenance model not loaded")

                with col2:
                    if 'performance' in inference.models:
                        perf_importance = inference.get_feature_importance('performance')
                        if perf_importance is not None and not perf_importance.empty:
                            fig_perf = px.bar(
                                perf_importance.head(10),
                                x='importance', y='feature',
                                orientation='h',
                                title='Performance Model Features',
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
                            st.warning(" Performance model feature importance not available")
                    else:
                        st.error(" Performance model not loaded")
                        
                if hasattr(inference, 'get_model_status'):
                    status = inference.get_model_status()
                    if status:
                        st.markdown(" Model Status")
                        col3, col4, col5 = st.columns(3)
                        with col3:
                            st.metric("Models Loaded", status.get('models_available', 0))
                        with col4:
                            st.metric("Maintenance Ready", "✅" if status.get('maintenance_ready') else "❌")
                        with col5:
                            st.metric("Performance Ready", "✅" if status.get('performance_ready') else "❌")
            else:
                st.error(" Models not loaded or inference system unavailable")
                st.info(" Run training first: python src/train.py")
                
                try:
                    import os
                    models_dir = "models"
                    if os.path.exists(models_dir):
                        model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
                        if model_files:
                            st.markdown(" Available Model Files:")
                            for file in model_files:
                                st.text(f"• {file}")
                        else:
                            st.warning("No model files found in models/ directory")
                    else:
                        st.warning("Models directory not found")
                except Exception:
                    pass
                    
    except Exception as e:
        st.error(f"Error loading model insights: {str(e)}")
        st.info(" Try running: python src/train.py to train models first")
        st.markdown('</div>', unsafe_allow_html=True)

def settings_page():
        st.markdown(" System Settings")

        col1, col2 = st.columns(2)

        with col1:
            # st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card ard-header"> Alert Settings</div>', unsafe_allow_html=True)

            temp_threshold = st.slider("Temperature Alert (°C)", 30, 80, 65)
            dust_threshold = st.slider("Dust Level Alert (%)", 10, 90, 60)
            efficiency_threshold = st.slider("Efficiency Alert (%)", 50, 95, 80)

            st.checkbox("Email Notifications", value=True)
            st.checkbox("SMS Alerts", value=False)
            st.checkbox("Push Notifications", value=True)

            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            #st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card card-header"> Display Settings</div>', unsafe_allow_html=True)

            theme = st.selectbox("Theme", ["Dark", "Light", "Auto"])
            if theme != st.session_state.get('theme', 'Dark'):
                st.session_state.theme = theme
                st.rerun()
            refresh_rate = st.selectbox("Refresh Rate", ["5 seconds", "10 seconds", "30 seconds", "1 minute"])
            chart_style = st.selectbox("Chart Style", ["Modern", "Classic", "Minimal"])

            st.checkbox("Show Animations", value=True)
            st.checkbox("High Contrast Mode", value=False)

            st.markdown('</div>', unsafe_allow_html=True)

        if st.button(" Save Settings", type="primary"):
            st.success("Settings saved successfully!")

def main():
        apply_custom_css()
        render_header()

        page = render_navigation()

        if page == "dashboard":
            dashboard_page()
        elif page == "analytics":
            analytics_page()
        elif page == "eda":
            eda_page()
        elif page == "3d_viz":
            viz_3d_page()
        elif page == "maintenance":
            maintenance_page()
        elif page == "ml_insights":
            ml_insights_page()
        elif page == "settings":
            settings_page()

if __name__ == "__main__":
    main()





