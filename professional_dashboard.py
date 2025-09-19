import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="SolarVision Pro",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Professional React-like CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    .main {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        min-height: 100vh;
        color: #ffffff;
    }
    
    /* Header */
    .header {
        background: rgba(15, 23, 42, 0.95);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(59, 130, 246, 0.2);
        border-radius: 12px;
        padding: 1.5rem 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    
    .header-title {
        font-size: 2rem;
        font-weight: 700;
        color: #ffffff;
        text-align: center;
        margin: 0;
    }
    
    .header-subtitle {
        font-size: 1rem;
        color: rgba(148, 163, 184, 0.8);
        text-align: center;
        margin-top: 0.5rem;
    }
    
    /* Navigation */
    .nav-container {
        background: rgba(15, 23, 42, 0.9);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(59, 130, 246, 0.2);
        border-radius: 12px;
        padding: 0.75rem;
        margin-bottom: 2rem;
        display: flex;
        gap: 0.5rem;
    }
    
    /* Cards */
    .card {
        background: rgba(15, 23, 42, 0.8);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(59, 130, 246, 0.15);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.2);
    }
    
    .card:hover {
        transform: translateY(-2px);
        border-color: rgba(59, 130, 246, 0.3);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: rgba(148, 163, 184, 0.8);
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 600;
        color: #ffffff;
        margin: 0;
        line-height: 1;
    }
    
    .metric-icon {
        font-size: 1.2rem;
        margin-right: 0.5rem;
        opacity: 0.8;
    }
    
    /* Chart Container */
    .chart-container {
        background: rgba(15, 23, 42, 0.8);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(59, 130, 246, 0.15);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }
    
    .chart-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 1rem;
    }
    
    /* Status Badge */
    .status-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 500;
        font-size: 0.875rem;
        margin: 0.5rem;
    }
    
    .status-excellent {
        background: rgba(34, 197, 94, 0.2);
        color: #22c55e;
        border: 1px solid rgba(34, 197, 94, 0.4);
    }
    
    .status-good {
        background: rgba(59, 130, 246, 0.2);
        color: #3b82f6;
        border: 1px solid rgba(59, 130, 246, 0.4);
    }
    
    .status-warning {
        background: rgba(245, 158, 11, 0.2);
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.4);
    }
    
    .status-critical {
        background: rgba(239, 68, 68, 0.2);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.4);
    }
    
    /* Streamlit Overrides */
    .stButton > button {
        background: rgba(59, 130, 246, 0.1) !important;
        border: 1px solid rgba(59, 130, 246, 0.3) !important;
        color: rgba(255, 255, 255, 0.9) !important;
        font-weight: 500 !important;
        border-radius: 8px !important;
        padding: 0.5rem 1rem !important;
        transition: all 0.2s ease !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    .stButton > button:hover {
        background: rgba(59, 130, 246, 0.2) !important;
        border-color: rgba(59, 130, 246, 0.5) !important;
        color: #ffffff !important;
        transform: translateY(-1px) !important;
    }
    
    .stSelectbox > div > div {
        background: rgba(15, 23, 42, 0.8) !important;
        border: 1px solid rgba(59, 130, 246, 0.3) !important;
        border-radius: 8px !important;
        color: #ffffff !important;
    }
    
    .stDataFrame {
        background: rgba(15, 23, 42, 0.5) !important;
        border-radius: 8px !important;
        border: 1px solid rgba(59, 130, 246, 0.2) !important;
    }
    
    .stAlert {
        background: rgba(15, 23, 42, 0.8) !important;
        border: 1px solid rgba(59, 130, 246, 0.3) !important;
        border-radius: 8px !important;
        color: #ffffff !important;
    }
    
    /* Text Colors */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
        color: #ffffff !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    .stMarkdown p, .stMarkdown li {
        color: rgba(255, 255, 255, 0.9) !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Hide Streamlit Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(15, 23, 42, 0.5);
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(59, 130, 246, 0.5);
        border-radius: 3px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache solar panel data"""
    try:
        df = pd.read_csv('solar_training_data.csv')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except FileNotFoundError:
        try:
            df = pd.read_csv('solar_test_data.csv')
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        except FileNotFoundError:
            return generate_sample_data()

def generate_sample_data():
    """Generate sample data"""
    np.random.seed(42)
    dates = pd.date_range(start='2025-01-01', end='2025-01-31', freq='H')
    n_samples = len(dates)
    
    hour = dates.hour
    solar_base = np.where((hour >= 6) & (hour <= 18), 
                         800 * np.sin(np.pi * (hour - 6) / 12), 0)
    
    data = {
        'timestamp': dates,
        'solar_irradiance': np.maximum(0, solar_base + np.random.normal(0, 100, n_samples)),
        'temperature': 20 + 10 * np.sin(2 * np.pi * dates.dayofyear / 365) + np.random.normal(0, 3, n_samples),
        'humidity': 50 + 20 * np.sin(2 * np.pi * dates.dayofyear / 365 + np.pi/4) + np.random.normal(0, 10, n_samples),
        'wind_speed': np.maximum(0, 5 + np.random.normal(0, 2, n_samples)),
        'panel_voltage': 24 + np.random.normal(0, 1, n_samples),
        'panel_current': np.maximum(0, solar_base / 100 + np.random.normal(0, 0.5, n_samples)),
        'power_output': np.maximum(0, solar_base / 5 + np.random.normal(0, 20, n_samples)),
        'panel_temp': 25 + (solar_base / 50) + np.random.normal(0, 3, n_samples),
        'dust_level': np.maximum(0, np.random.exponential(0.1, n_samples)),
        'hours_since_cleaning': np.random.uniform(0, 168, n_samples),
        'days_since_maintenance': np.random.uniform(0, 90, n_samples),
        'efficiency': np.random.uniform(0.3, 1.2, n_samples),
        'days_until_maintenance': np.maximum(1, 60 - np.random.exponential(20, n_samples))
    }
    
    return pd.DataFrame(data)

def create_metric_card(title, value, unit="", icon="📊"):
    """Create metric card"""
    return f"""
    <div class="card">
        <div class="metric-label">
            <span class="metric-icon">{icon}</span>{title}
        </div>
        <div class="metric-value">{value}{unit}</div>
    </div>
    """

def get_system_status(df):
    """Calculate system status"""
    if len(df) == 0:
        return "Offline", "critical"
    
    latest = df.iloc[-1]
    efficiency = latest['efficiency']
    dust_level = latest['dust_level']
    days_until_maintenance = latest['days_until_maintenance']
    
    health_score = (
        (efficiency * 0.4) +
        ((1 - min(dust_level, 1)) * 0.3) +
        (min(days_until_maintenance / 30, 1) * 0.3)
    )
    
    if health_score > 0.8:
        return "Excellent", "excellent"
    elif health_score > 0.6:
        return "Good", "good"
    elif health_score > 0.4:
        return "Warning", "warning"
    else:
        return "Critical", "critical"

def create_main_chart(df):
    """Create main dashboard chart"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Power Generation', 'Solar Irradiance', 'System Efficiency', 'Temperature'],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    recent_data = df.tail(100) if len(df) > 100 else df
    
    colors = {
        'power': '#22c55e',
        'solar': '#f59e0b', 
        'efficiency': '#3b82f6',
        'temp': '#ef4444'
    }
    
    # Power Output
    fig.add_trace(
        go.Scatter(
            x=recent_data['timestamp'], 
            y=recent_data['power_output'],
            mode='lines',
            name='Power',
            line=dict(color=colors['power'], width=3),
            fill='tonexty'
        ),
        row=1, col=1
    )
    
    # Solar Irradiance
    fig.add_trace(
        go.Scatter(
            x=recent_data['timestamp'], 
            y=recent_data['solar_irradiance'],
            mode='lines',
            name='Solar',
            line=dict(color=colors['solar'], width=3)
        ),
        row=1, col=2
    )
    
    # Efficiency
    fig.add_trace(
        go.Scatter(
            x=recent_data['timestamp'], 
            y=recent_data['efficiency'],
            mode='lines+markers',
            name='Efficiency',
            line=dict(color=colors['efficiency'], width=3),
            marker=dict(size=4)
        ),
        row=2, col=1
    )
    
    # Temperature
    fig.add_trace(
        go.Scatter(
            x=recent_data['timestamp'], 
            y=recent_data['panel_temp'],
            mode='lines',
            name='Temperature',
            line=dict(color=colors['temp'], width=3)
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        height=600,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter", size=12, color='#ffffff'),
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)', color='#ffffff')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)', color='#ffffff')
    
    return fig

def create_gauge(value, title, max_value=100):
    """Create gauge chart"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0.1, 0.9]},
        title={'text': title, 'font': {'size': 16, 'family': 'Inter', 'color': '#ffffff'}},
        number={'font': {'size': 24, 'color': '#ffffff'}},
        gauge={
            'axis': {
                'range': [None, max_value], 
                'tickfont': {'size': 12, 'color': '#ffffff'}
            },
            'bar': {'color': "#22c55e", 'thickness': 0.8},
            'steps': [
                {'range': [0, max_value * 0.5], 'color': "rgba(239, 68, 68, 0.2)"},
                {'range': [max_value * 0.5, max_value * 0.8], 'color': "rgba(245, 158, 11, 0.2)"},
                {'range': [max_value * 0.8, max_value], 'color': "rgba(34, 197, 94, 0.2)"}
            ],
            'bordercolor': 'rgba(255,255,255,0.3)',
            'threshold': {
                'line': {'color': "#3b82f6", 'width': 3},
                'thickness': 0.75,
                'value': max_value * 0.9
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter", color='#ffffff'),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def create_analytics_chart(df):
    """Create analytics chart"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Daily Power', 'Efficiency Distribution', 'Weather Impact', 'Performance Trends']
    )
    
    # Daily Power
    df['date'] = df['timestamp'].dt.date
    daily_power = df.groupby('date')['power_output'].sum()
    fig.add_trace(
        go.Bar(x=daily_power.index, y=daily_power.values, name='Daily Power', marker_color='#22c55e'),
        row=1, col=1
    )
    
    # Efficiency Distribution
    fig.add_trace(
        go.Histogram(x=df['efficiency'], nbinsx=20, name='Efficiency', marker_color='#3b82f6'),
        row=1, col=2
    )
    
    # Weather Impact
    fig.add_trace(
        go.Scatter(x=df['temperature'], y=df['power_output'], mode='markers', 
                  name='Temp vs Power', marker=dict(color='#f59e0b', size=6)),
        row=2, col=1
    )
    
    # Performance Trends
    df['rolling_power'] = df['power_output'].rolling(window=24).mean()
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['rolling_power'], mode='lines',
                  name='Power Trend', line=dict(color='#ef4444', width=3)),
        row=2, col=2
    )
    
    fig.update_layout(
        height=700,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter", size=12, color='#ffffff'),
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)', color='#ffffff')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)', color='#ffffff')
    
    return fig

def main():
    # Header
    st.markdown("""
    <div class="header">
        <h1 class="header-title">⚡ SolarVision Pro</h1>
        <p class="header-subtitle">Professional Solar Analytics Dashboard</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    
    # Navigation
    #st.markdown('<div class="nav-container chart-container">', unsafe_allow_html=True)
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        dashboard_btn = st.button("🏠 Dashboard", use_container_width=True)
    with col2:
        analytics_btn = st.button("📊 Analytics", use_container_width=True)
    with col3:
        live_btn = st.button("⚡ Live Data", use_container_width=True)
    with col4:
        maintenance_btn = st.button("🔧 Maintenance", use_container_width=True)
    with col5:
        reports_btn = st.button("📈 Reports", use_container_width=True)
    
    st.markdown('</div class="chart-container">', unsafe_allow_html=True)
    
    # Determine selected section
    if analytics_btn:
        selected = "Analytics"
    elif live_btn:
        selected = "Live Data"
    elif maintenance_btn:
        selected = "Maintenance"
    elif reports_btn:
        selected = "Reports"
    else:
        selected = "Dashboard"
    
    if selected == "Dashboard":
        # System Status
        if len(df) > 0:
            status_text, status_type = get_system_status(df)
            st.markdown(f"""
            <div style="text-align: center; margin: 2rem 0;">
                <span class="status-badge status-{status_type}">
                    System Status: {status_text}
                </span>
            </div>
            """, unsafe_allow_html=True)
            
            # Key Metrics
            latest = df.iloc[-1]
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(create_metric_card(
                    "Power Output", 
                    f"{latest['power_output']:.1f}", 
                    " W", "⚡"
                ), unsafe_allow_html=True)
            
            with col2:
                st.markdown(create_metric_card(
                    "Efficiency", 
                    f"{latest['efficiency']:.2f}", 
                    "", "📊"
                ), unsafe_allow_html=True)
            
            with col3:
                st.markdown(create_metric_card(
                    "Temperature", 
                    f"{latest['panel_temp']:.1f}", 
                    "°C", "🌡️"
                ), unsafe_allow_html=True)
            
            with col4:
                st.markdown(create_metric_card(
                    "Maintenance", 
                    f"{latest['days_until_maintenance']:.0f}", 
                    " days", "🔧"
                ), unsafe_allow_html=True)
            
            # Gauges
            col1, col2, col3 = st.columns(3)
            
            with col1:
                #st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.markdown('<div class="chart-container chart-title">System Efficiency</div>', unsafe_allow_html=True)
                efficiency_gauge = create_gauge(
                    latest['efficiency'] * 100, 
                    "", 
                    120
                )
                st.plotly_chart(efficiency_gauge, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                #st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.markdown('<div class="chart-container chart-title">Power Output</div>', unsafe_allow_html=True)
                power_gauge = create_gauge(
                    latest['power_output'], 
                    "", 
                    max(df['power_output'].max(), 100)
                )
                st.plotly_chart(power_gauge, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                #st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.markdown('<div class="chart-container chart-title">System Health</div>', unsafe_allow_html=True)
                health_score = (latest['efficiency'] * 0.6 + (1 - min(latest['dust_level'], 1)) * 0.4) * 100
                health_gauge = create_gauge(
                    health_score, 
                    "", 
                    100
                )
                st.plotly_chart(health_gauge, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Main Chart
            #st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown('<div class="chart-container chart-title">Real-time System Monitoring</div>', unsafe_allow_html=True)
            main_chart = create_main_chart(df)
            st.plotly_chart(main_chart, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.error("No data available")
    
    elif selected == "Analytics":
        # st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown('<div class="chart-container chart-container chart-title">📊 Advanced Analytics</div>', unsafe_allow_html=True)
        if len(df) > 0:
            analytics_chart = create_analytics_chart(df)
            st.plotly_chart(analytics_chart, use_container_width=True)
        else:
            st.error("No data available for analytics")
        st.markdown('</div>', unsafe_allow_html=True)
        
        if len(df) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                #st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.markdown('<div class="chart-container chart-title">Performance Insights</div>', unsafe_allow_html=True)
                avg_efficiency = df['efficiency'].mean()
                peak_power = df['power_output'].max()
                avg_temp = df['panel_temp'].mean()
                
                st.markdown(f"""
                - **Average Efficiency:** {avg_efficiency:.1%}
                - **Peak Power:** {peak_power:.1f} W
                - **Average Temperature:** {avg_temp:.1f}°C
                - **Total Records:** {len(df):,}
                """)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                #st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.markdown('<div class=" chart-container chart-title">Recent Trends</div>', unsafe_allow_html=True)
                df['date'] = df['timestamp'].dt.date
                daily_stats = df.groupby('date').agg({
                    'power_output': 'mean',
                    'efficiency': 'mean'
                }).round(2)
                st.dataframe(daily_stats.tail(5), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
    
    elif selected == "Live Data":
        #st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown('<div class="chart-container chart-title">⚡ Live System Monitor</div>', unsafe_allow_html=True)
        
        if st.button("🔄 Refresh Data"):
            st.rerun()
        
        if len(df) > 0:
            latest = df.iloc[-1]
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            metrics = [
                ("Solar Irradiance", f"{latest['solar_irradiance']:.0f}", "W/m²", "☀️"),
                ("Power Output", f"{latest['power_output']:.1f}", "W", "⚡"),
                ("Voltage", f"{latest['panel_voltage']:.1f}", "V", "🔋"),
                ("Current", f"{latest['panel_current']:.2f}", "A", "⚡"),
                ("Efficiency", f"{latest['efficiency']:.1%}", "", "📊")
            ]
            
            for i, (label, value, unit, icon) in enumerate(metrics):
                with [col1, col2, col3, col4, col5][i]:
                    st.markdown(create_metric_card(label, value, unit, icon), unsafe_allow_html=True)
            
            live_chart = create_main_chart(df)
            st.plotly_chart(live_chart, use_container_width=True)
        else:
            st.error("No live data available")
        st.markdown('</div>', unsafe_allow_html=True)
    
    elif selected == "Maintenance":
        #st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown('<div class="chart-container chart-title ">🔧 Maintenance Dashboard</div>', unsafe_allow_html=True)
        
        if len(df) > 0:
            latest = df.iloc[-1]
            
            alerts = []
            if latest['days_until_maintenance'] < 7:
                alerts.append("🔧 Scheduled maintenance due within 7 days")
            if latest['dust_level'] > 0.5:
                alerts.append("🧹 Panel cleaning recommended")
            if latest['efficiency'] < 0.7:
                alerts.append("⚠️ Low efficiency detected")
            
            if alerts:
                st.markdown("Maintenance Alerts")
                for alert in alerts:
                    st.warning(alert)
            else:
                st.success("✅ All systems operating normally")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(create_metric_card(
                    "Hours Since Cleaning", 
                    f"{latest['hours_since_cleaning']:.0f}", 
                    " hrs", "🧹"
                ), unsafe_allow_html=True)
            
            with col2:
                st.markdown(create_metric_card(
                    "Days Since Maintenance", 
                    f"{latest['days_since_maintenance']:.0f}", 
                    " days", "🔧"
                ), unsafe_allow_html=True)
            
            with col3:
                st.markdown(create_metric_card(
                    "Dust Level", 
                    f"{latest['dust_level']:.2f}", 
                    "", "💨"
                ), unsafe_allow_html=True)
        else:
            st.error("No maintenance data available")
        st.markdown('</div>', unsafe_allow_html=True)
    
    elif selected == "Reports":
        #st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown('<div class="chart-container chart-title">📈 System Reports</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            #st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown('<div class="chart-container chart-title">Generate Report</div>', unsafe_allow_html=True)
            
            report_type = st.selectbox(
                "Select Report Type",
                ["📊 Daily Summary", "📈 Weekly Performance", "⚡ Efficiency Report"]
            )
            
            if st.button("🚀 Generate Report"):
                if len(df) > 0:
                    st.success(f"✅ {report_type} generated successfully!")
                    
                    if "Daily" in report_type:
                        df['date'] = df['timestamp'].dt.date
                        daily_report = df.groupby('date').agg({
                            'power_output': ['mean', 'max'],
                            'efficiency': 'mean'
                        }).round(2)
                        st.dataframe(daily_report.tail(7), use_container_width=True)
                    
                    elif "Efficiency" in report_type:
                        st.markdown("#### Efficiency Analysis")
                        st.markdown(f"- **Average Efficiency:** {df['efficiency'].mean():.2%}")
                        st.markdown(f"- **Peak Efficiency:** {df['efficiency'].max():.2%}")
                        st.markdown(f"- **Lowest Efficiency:** {df['efficiency'].min():.2%}")
                    
                    csv_data = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Dataset",
                        data=csv_data,
                        file_name="solar_report.csv",
                        mime="text/csv"
                    )
                else:
                    st.error("❌ No data available")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            #st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown('<div class=" chart-container chart-title">Quick Stats</div>', unsafe_allow_html=True)
            if len(df) > 0:
                st.markdown(f"- **Total Records:** {len(df):,}")
                st.markdown(f"- **Max Power:** {df['power_output'].max():.1f} W")
                st.markdown(f"- **Avg Efficiency:** {df['efficiency'].mean():.1%}")
                st.markdown(f"- **Peak Irradiance:** {df['solar_irradiance'].max():.0f} W/m²")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: rgba(255,255,255,0.7); font-size: 0.875rem; margin-top: 2rem;">
        <strong>⚡ SolarVision Pro</strong> | Professional Solar Analytics Dashboard
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
