import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


st.set_page_config(
    page_title="Solar Panel Monitoring Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B35;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    .stSelectbox > div > div {
        background-color: #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the solar panel data"""
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
            st.warning("No data files found. Generating sample data...")
            return generate_sample_data()

def generate_sample_data():
    """Generate sample data for demonstration"""
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

def create_correlation_heatmap(df):
    """Create an interactive correlation heatmap"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale='RdBu_r',
        title="Feature Correlation Matrix"
    )
    fig.update_layout(height=600)
    return fig

def create_time_series_analysis(df):
    """Create comprehensive time series analysis"""
    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=[
            'Solar Irradiance Over Time', 'Power Output Over Time',
            'Temperature vs Panel Temperature', 'Efficiency Trends',
            'Weather Conditions', 'Maintenance Indicators',
            'Daily Patterns', 'Performance Metrics'
        ],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": True}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['solar_irradiance'], 
                  name='Solar Irradiance', line=dict(color='orange')),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['power_output'], 
                  name='Power Output', line=dict(color='green')),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['temperature'], 
                  name='Ambient Temp', line=dict(color='blue')),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['panel_temp'], 
                  name='Panel Temp', line=dict(color='red')),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['efficiency'], 
                  name='Efficiency', line=dict(color='purple')),
        row=2, col=2
    )
    
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['humidity'], 
                  name='Humidity', line=dict(color='cyan')),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['wind_speed'], 
                  name='Wind Speed', line=dict(color='gray')),
        row=3, col=1, secondary_y=True
    )
    
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['dust_level'], 
                  name='Dust Level', line=dict(color='brown')),
        row=3, col=2
    )
    
    df['hour'] = df['timestamp'].dt.hour
    hourly_avg = df.groupby('hour')['power_output'].mean()
    fig.add_trace(
        go.Scatter(x=hourly_avg.index, y=hourly_avg.values, 
                  name='Hourly Avg Power', line=dict(color='gold')),
        row=4, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['panel_voltage'], 
                  name='Panel Voltage', line=dict(color='navy')),
        row=4, col=2
    )
    
    fig.update_layout(height=1200, showlegend=True, title_text="Comprehensive Time Series Analysis")
    return fig

def create_distribution_analysis(df):
    """Create distribution analysis plots"""
    numeric_cols = ['solar_irradiance', 'temperature', 'humidity', 'power_output', 
                   'efficiency', 'dust_level', 'panel_temp', 'wind_speed']
    
    fig = make_subplots(
        rows=2, cols=4,
        subplot_titles=numeric_cols,
        specs=[[{"secondary_y": False} for _ in range(4)] for _ in range(2)]
    )
    
    for i, col in enumerate(numeric_cols):
        row = i // 4 + 1
        col_pos = i % 4 + 1
        
        fig.add_trace(
            go.Histogram(x=df[col], name=col, nbinsx=30, opacity=0.7),
            row=row, col=col_pos
        )
    
    fig.update_layout(height=800, showlegend=False, title_text="Distribution Analysis")
    return fig

def create_performance_dashboard(df):
    """Create performance monitoring dashboard"""
    avg_power = df['power_output'].mean()
    max_power = df['power_output'].max()
    avg_efficiency = df['efficiency'].mean()
    current_efficiency = df['efficiency'].iloc[-1] if len(df) > 0 else 0
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Power Output Distribution', 'Efficiency vs Solar Irradiance',
                       'Temperature Impact on Performance', 'Maintenance Schedule'],
        specs=[[{"type": "histogram"}, {"type": "scatter"}],
               [{"type": "scatter"}, {"type": "bar"}]]
    )

    fig.add_trace(
        go.Histogram(x=df['power_output'], nbinsx=50, name='Power Distribution'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df['solar_irradiance'], y=df['efficiency'], 
                  mode='markers', name='Efficiency vs Irradiance',
                  marker=dict(color=df['temperature'], colorscale='Viridis', showscale=True)),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(x=df['panel_temp'], y=df['power_output'], 
                  mode='markers', name='Temp vs Power',
                  marker=dict(color=df['dust_level'], colorscale='Reds', showscale=True)),
        row=2, col=1
    )
    
    maintenance_bins = pd.cut(df['days_until_maintenance'], bins=5)
    maintenance_counts = maintenance_bins.value_counts().sort_index()
    fig.add_trace(
        go.Bar(x=[str(x) for x in maintenance_counts.index], y=maintenance_counts.values,
               name='Maintenance Schedule'),
        row=2, col=2
    )
    
    fig.update_layout(height=800, title_text="Performance Dashboard")
    return fig, avg_power, max_power, avg_efficiency, current_efficiency

def create_weather_impact_analysis(df):
    """Analyze weather impact on solar performance"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Humidity vs Power Output', 'Wind Speed Impact',
                       'Temperature Correlation', 'Weather Patterns'],
        specs=[[{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "scatter"}, {"type": "heatmap"}]]
    )
    
    fig.add_trace(
        go.Scatter(x=df['humidity'], y=df['power_output'], 
                  mode='markers', name='Humidity Impact',
                  marker=dict(color=df['solar_irradiance'], colorscale='Plasma')),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df['wind_speed'], y=df['efficiency'], 
                  mode='markers', name='Wind Speed Impact',
                  marker=dict(color=df['panel_temp'], colorscale='Turbo')),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(x=df['temperature'], y=df['panel_temp'], 
                  mode='markers', name='Temperature Correlation',
                  marker=dict(color=df['power_output'], colorscale='Viridis')),
        row=2, col=1
    )
    
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.day
    weather_pivot = df.pivot_table(values='power_output', index='hour', columns='day', aggfunc='mean')
    
    fig.add_trace(
        go.Heatmap(z=weather_pivot.values, x=weather_pivot.columns, y=weather_pivot.index,
                  colorscale='Viridis', name='Daily Power Patterns'),
        row=2, col=2
    )
    
    fig.update_layout(height=800, title_text="Weather Impact Analysis")
    return fig

def create_maintenance_analysis(df):
    """Create maintenance and health analysis"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Dust Accumulation Over Time', 'Cleaning Impact',
                       'Maintenance Prediction', 'System Health Score'],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    fig.add_trace(
        go.Scatter(x=df['hours_since_cleaning'], y=df['dust_level'], 
                  mode='markers', name='Dust vs Time Since Cleaning',
                  marker=dict(color=df['efficiency'], colorscale='RdYlGn_r')),
        row=1, col=1
    )
    
    cleaning_bins = pd.cut(df['hours_since_cleaning'], bins=10)
    cleaning_impact = df.groupby(cleaning_bins)['efficiency'].mean()
    fig.add_trace(
        go.Scatter(x=list(range(len(cleaning_impact))), y=cleaning_impact.values, 
                  mode='lines+markers', name='Cleaning Impact on Efficiency'),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(x=df['days_since_maintenance'], y=df['days_until_maintenance'], 
                  mode='markers', name='Maintenance Prediction',
                  marker=dict(color=df['efficiency'], colorscale='Spectral')),
        row=2, col=1
    )
    
    df['health_score'] = (df['efficiency'] * 0.4 + 
                         (1 - df['dust_level'] / df['dust_level'].max()) * 0.3 +
                         (df['days_until_maintenance'] / df['days_until_maintenance'].max()) * 0.3)
    
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['health_score'], 
                  mode='lines', name='System Health Score',
                  line=dict(color='green', width=3)),
        row=2, col=2
    )
    
    fig.update_layout(height=800, title_text="Maintenance & Health Analysis")
    return fig

def create_advanced_analytics(df):
    """Create advanced analytics and insights"""
    df['power_7d_avg'] = df['power_output'].rolling(window=168, min_periods=1).mean()  
    df['efficiency_trend'] = df['efficiency'].rolling(window=24, min_periods=1).mean()  
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Power Output Trends', 'Efficiency Patterns',
                       'Performance Anomalies', 'Predictive Insights'],
        specs=[[{"secondary_y": True}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['power_output'], 
                  name='Actual Power', opacity=0.3, line=dict(color='blue')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['power_7d_avg'], 
                  name='7-Day Average', line=dict(color='red', width=3)),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['efficiency'], 
                  name='Efficiency', opacity=0.5, line=dict(color='green')),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['efficiency_trend'], 
                  name='Efficiency Trend', line=dict(color='darkgreen', width=3)),
        row=1, col=2
    )
    
    df['power_zscore'] = np.abs((df['power_output'] - df['power_output'].mean()) / df['power_output'].std())
    anomalies = df[df['power_zscore'] > 2]
    
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['power_output'], 
                  mode='markers', name='Normal Operation', 
                  marker=dict(color='blue', size=4)),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=anomalies['timestamp'], y=anomalies['power_output'], 
                  mode='markers', name='Anomalies', 
                  marker=dict(color='red', size=8, symbol='x')),
        row=2, col=1
    )
    
    from sklearn.linear_model import LinearRegression
    df['timestamp_numeric'] = df['timestamp'].astype(np.int64) // 10**9
    
    X = df['timestamp_numeric'].values.reshape(-1, 1)
    y = df['efficiency'].values
    model = LinearRegression().fit(X, y)
    
    future_timestamps = pd.date_range(start=df['timestamp'].max(), periods=168, freq='H')[1:]
    future_numeric = future_timestamps.astype(np.int64) // 10**9
    future_predictions = model.predict(future_numeric.values.reshape(-1, 1))
    
    fig.add_trace(
        go.Scatter(x=future_timestamps, y=future_predictions, 
                  name='Predicted Efficiency', line=dict(color='orange', dash='dash')),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['efficiency'], 
                  name='Historical Efficiency', line=dict(color='purple')),
        row=2, col=2
    )
    
    fig.update_layout(height=800, title_text="Advanced Analytics & Predictions")
    return fig

def main():
    st.markdown('<h1 class="main-header"> Solar Panel Monitoring Dashboard</h1>', unsafe_allow_html=True)
    
    df = load_data()
    
    st.sidebar.title("Dashboard Controls")
    
    if len(df) > 0:
        min_date = df['timestamp'].min().date()
        max_date = df['timestamp'].max().date()
        
        date_range = st.sidebar.date_input(
            "Select Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        if len(date_range) == 2:
            start_date, end_date = date_range
            df_filtered = df[(df['timestamp'].dt.date >= start_date) & 
                           (df['timestamp'].dt.date <= end_date)]
        else:
            df_filtered = df
    else:
        df_filtered = df
    
    analysis_type = st.sidebar.selectbox(
        "Select Analysis Type",
        ["Overview", "Time Series Analysis", "Performance Dashboard", 
         "Weather Impact", "Maintenance Analysis", "Advanced Analytics", 
         "Statistical Analysis", "Correlation Analysis"]
    )
    
    if len(df_filtered) > 0:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Average Power Output",
                value=f"{df_filtered['power_output'].mean():.1f} W",
                delta=f"{df_filtered['power_output'].iloc[-1] - df_filtered['power_output'].mean():.1f} W"
            )
        
        with col2:
            st.metric(
                label="Current Efficiency",
                value=f"{df_filtered['efficiency'].iloc[-1]:.2f}",
                delta=f"{df_filtered['efficiency'].iloc[-1] - df_filtered['efficiency'].mean():.2f}"
            )
        
        with col3:
            st.metric(
                label="System Health",
                value=f"{((df_filtered['efficiency'].mean() * 0.6 + (1 - df_filtered['dust_level'].mean()) * 0.4) * 100):.1f}%"
            )
        
        with col4:
            st.metric(
                label="Days Until Maintenance",
                value=f"{df_filtered['days_until_maintenance'].iloc[-1]:.0f} days"
            )
    
    if analysis_type == "Overview":
        st.subheader("System Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Current Status")
            if len(df_filtered) > 0:
                latest = df_filtered.iloc[-1]
                st.write(f"**Solar Irradiance:** {latest['solar_irradiance']:.1f} W/m²")
                st.write(f"**Power Output:** {latest['power_output']:.1f} W")
                st.write(f"**Panel Temperature:** {latest['panel_temp']:.1f}°C")
                st.write(f"**Efficiency:** {latest['efficiency']:.2f}")
                st.write(f"**Dust Level:** {latest['dust_level']:.3f}")
        
        with col2:
            st.subheader("Quick Statistics")
            if len(df_filtered) > 0:
                st.write(f"**Total Records:** {len(df_filtered):,}")
                st.write(f"**Date Range:** {df_filtered['timestamp'].min().strftime('%Y-%m-%d')} to {df_filtered['timestamp'].max().strftime('%Y-%m-%d')}")
                st.write(f"**Max Power:** {df_filtered['power_output'].max():.1f} W")
                st.write(f"**Avg Efficiency:** {df_filtered['efficiency'].mean():.2f}")
                st.write(f"**Peak Irradiance:** {df_filtered['solar_irradiance'].max():.1f} W/m²")
        
        if len(df_filtered) > 0:
            fig_overview = make_subplots(
                rows=2, cols=2,
                subplot_titles=['Power Output Over Time', 'Solar Irradiance', 
                               'Efficiency Trends', 'Temperature Monitoring']
            )
            
            fig_overview.add_trace(
                go.Scatter(x=df_filtered['timestamp'], y=df_filtered['power_output'], 
                          name='Power Output', line=dict(color='green')),
                row=1, col=1
            )
            
            fig_overview.add_trace(
                go.Scatter(x=df_filtered['timestamp'], y=df_filtered['solar_irradiance'], 
                          name='Solar Irradiance', line=dict(color='orange')),
                row=1, col=2
            )
            
            fig_overview.add_trace(
                go.Scatter(x=df_filtered['timestamp'], y=df_filtered['efficiency'], 
                          name='Efficiency', line=dict(color='blue')),
                row=2, col=1
            )
            
            fig_overview.add_trace(
                go.Scatter(x=df_filtered['timestamp'], y=df_filtered['panel_temp'], 
                          name='Panel Temp', line=dict(color='red')),
                row=2, col=2
            )
            
            fig_overview.update_layout(height=600, title_text="System Overview")
            st.plotly_chart(fig_overview, use_container_width=True)
    
    elif analysis_type == "Time Series Analysis":
        st.subheader("Comprehensive Time Series Analysis")
        fig_ts = create_time_series_analysis(df_filtered)
        st.plotly_chart(fig_ts, use_container_width=True)
        
        st.subheader("Time Series Insights")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Daily Patterns:**")
            if len(df_filtered) > 0:
                df_filtered['hour'] = df_filtered['timestamp'].dt.hour
                hourly_stats = df_filtered.groupby('hour')['power_output'].agg(['mean', 'max', 'std'])
                st.dataframe(hourly_stats.round(2))
        
        with col2:
            st.write("**Weekly Patterns:**")
            if len(df_filtered) > 0:
                df_filtered['weekday'] = df_filtered['timestamp'].dt.day_name()
                weekly_stats = df_filtered.groupby('weekday')['efficiency'].agg(['mean', 'max', 'std'])
                st.dataframe(weekly_stats.round(3))
    
    elif analysis_type == "Performance Dashboard":
        st.subheader("Performance Monitoring Dashboard")
        fig_perf, avg_power, max_power, avg_eff, curr_eff = create_performance_dashboard(df_filtered)
        st.plotly_chart(fig_perf, use_container_width=True)
        
        st.subheader("Performance Insights")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Power Statistics:**")
            st.write(f"Average: {avg_power:.1f} W")
            st.write(f"Maximum: {max_power:.1f} W")
            st.write(f"Capacity Factor: {(avg_power/max_power)*100:.1f}%")
        
        with col2:
            st.write("**Efficiency Analysis:**")
            st.write(f"Average: {avg_eff:.3f}")
            st.write(f"Current: {curr_eff:.3f}")
            st.write(f"Performance: {'Good' if curr_eff > avg_eff else 'Needs Attention'}")
        
        with col3:
            st.write("**Recommendations:**")
            if curr_eff < 0.8:
                st.write("⚠️ Low efficiency detected")
            if df_filtered['dust_level'].iloc[-1] > 0.5:
                st.write("🧹 Cleaning recommended")
            if df_filtered['days_until_maintenance'].iloc[-1] < 7:
                st.write("🔧 Maintenance due soon")
    
    elif analysis_type == "Weather Impact":
        st.subheader("Weather Impact Analysis")
        fig_weather = create_weather_impact_analysis(df_filtered)
        st.plotly_chart(fig_weather, use_container_width=True)
        
        st.subheader("Weather Correlations")
        weather_cols = ['temperature', 'humidity', 'wind_speed', 'solar_irradiance']
        performance_cols = ['power_output', 'efficiency']
        
        correlations = df_filtered[weather_cols + performance_cols].corr()
        weather_corr = correlations.loc[weather_cols, performance_cols]
        
        fig_corr = px.imshow(weather_corr, text_auto=True, aspect="auto",
                            title="Weather vs Performance Correlations")
        st.plotly_chart(fig_corr, use_container_width=True)
    
    elif analysis_type == "Maintenance Analysis":
        st.subheader("Maintenance & Health Analysis")
        fig_maint = create_maintenance_analysis(df_filtered)
        st.plotly_chart(fig_maint, use_container_width=True)
        
        st.subheader("Maintenance Schedule")
        if len(df_filtered) > 0:
            urgent_maintenance = df_filtered[df_filtered['days_until_maintenance'] < 7]
            if len(urgent_maintenance) > 0:
                st.warning(f" {len(urgent_maintenance)} systems require maintenance within 7 days!")
                st.dataframe(urgent_maintenance[['timestamp', 'days_until_maintenance', 'efficiency', 'dust_level']].head())
            else:
                st.success(" No urgent maintenance required")
    
    elif analysis_type == "Advanced Analytics":
        st.subheader("Advanced Analytics & Predictions")
        fig_advanced = create_advanced_analytics(df_filtered)
        st.plotly_chart(fig_advanced, use_container_width=True)
        
        st.subheader("Machine Learning Insights")
        if len(df_filtered) > 100:  
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import mean_absolute_error, r2_score
            
            features = ['solar_irradiance', 'temperature', 'humidity', 'wind_speed', 
                       'panel_temp', 'dust_level', 'hours_since_cleaning']
            X = df_filtered[features].fillna(0)
            y = df_filtered['power_output']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Model Performance:**")
                st.write(f"MAE: {mae:.2f} W")
                st.write(f"R² Score: {r2:.3f}")
            
            with col2:
                st.write("**Feature Importance:**")
                importance = pd.DataFrame({
                    'feature': features,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                st.dataframe(importance)
    
    elif analysis_type == "Statistical Analysis":
        st.subheader("Statistical Analysis")
        
        fig_dist = create_distribution_analysis(df_filtered)
        st.plotly_chart(fig_dist, use_container_width=True)
        
        st.subheader("Statistical Summary")
        numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns
        stats_summary = df_filtered[numeric_cols].describe()
        st.dataframe(stats_summary.round(3))
        
        st.subheader("Outlier Analysis")
        Q1 = df_filtered['power_output'].quantile(0.25)
        Q3 = df_filtered['power_output'].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df_filtered[(df_filtered['power_output'] < (Q1 - 1.5 * IQR)) | 
                              (df_filtered['power_output'] > (Q3 + 1.5 * IQR))]
        
        st.write(f"**Outliers detected:** {len(outliers)} ({len(outliers)/len(df_filtered)*100:.1f}%)")
        if len(outliers) > 0:
            st.dataframe(outliers[['timestamp', 'power_output', 'solar_irradiance', 'efficiency']].head())
    
    elif analysis_type == "Correlation Analysis":
        st.subheader("Correlation Analysis")
        
        fig_corr = create_correlation_heatmap(df_filtered)
        st.plotly_chart(fig_corr, use_container_width=True)
        
        st.subheader("Key Correlations")
        numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns
        corr_matrix = df_filtered[numeric_cols].corr()
        
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_pairs.append({
                    'Variable 1': corr_matrix.columns[i],
                    'Variable 2': corr_matrix.columns[j],
                    'Correlation': corr_matrix.iloc[i, j]
                })
        
        corr_df = pd.DataFrame(corr_pairs)
        corr_df = corr_df.reindex(corr_df['Correlation'].abs().sort_values(ascending=False).index)
        
        st.dataframe(corr_df.head(10))
    
    st.markdown("---")
    st.markdown("**Solar Panel Monitoring Dashboard** | Built with Streamlit | Optimized for Raspberry Pi 3B")

if __name__ == "__main__":
    main()