
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from datetime import datetime, timedelta
import sys
import os
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_ingestion import DataIngestion
from src.feature_engineering import FeatureEngineer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InteractiveEDA:
    def __init__(self):
        self.df = None
        self.output_dir = 'analysis/interactive'

    def load_data(self):
        try:
            logger.info("Loading data for interactive EDA...")

            ingestion = DataIngestion()
            self.df = ingestion.load_historical_data()

            if self.df is None or self.df.empty:
                logger.warning("No data available, generating sample data")
                self.df = ingestion._generate_sample_data(days=365)

            engineer = FeatureEngineer()
            self.df = engineer.create_derived_features(self.df)

            os.makedirs(self.output_dir, exist_ok=True)

            logger.info(f"Data loaded successfully. Shape: {self.df.shape}")
            return True

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False

    def create_overview_dashboard(self):
        try:
            logger.info("Creating overview dashboard...")

            fig = make_subplots(
                rows=3, cols=3,
                subplot_titles=[
                    'Energy Output Over Time', 'Solar Irradiance Distribution', 'Temperature vs Energy',
                    'Efficiency Trends', 'Maintenance Events', 'Correlation Heatmap',
                    'Hourly Patterns', 'Monthly Patterns', 'System Performance'
                ],
                specs=[
                    [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                    [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                    [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]
                ],
                vertical_spacing=0.08,
                horizontal_spacing=0.08
            )

            fig.add_trace(
                go.Scatter(
                    x=self.df['timestamp'],
                    y=self.df['energy_output'],
                    mode='lines',
                    name='Energy Output',
                    line=dict(color='#00ff88', width=1),
                    opacity=0.7
                ),
                row=1, col=1
            )

            fig.add_trace(
                go.Histogram(
                    x=self.df['solar_irradiance'],
                    nbinsx=30,
                    name='Solar Irradiance',
                    marker_color='#ffa502',
                    opacity=0.7
                ),
                row=1, col=2
            )

            fig.add_trace(
                go.Scatter(
                    x=self.df['panel_temp'],
                    y=self.df['energy_output'],
                    mode='markers',
                    name='Temp vs Energy',
                    marker=dict(
                        color=self.df['solar_irradiance'],
                        colorscale='Viridis',
                        size=4,
                        opacity=0.6
                    )
                ),
                row=1, col=3
            )

            if 'efficiency' in self.df.columns:
                daily_eff = self.df.groupby(self.df['timestamp'].dt.date)['efficiency'].mean()
                fig.add_trace(
                    go.Scatter(
                        x=daily_eff.index,
                        y=daily_eff.values,
                        mode='lines+markers',
                        name='Daily Efficiency',
                        line=dict(color='#ff4757', width=2)
                    ),
                    row=2, col=1
                )

            maintenance_events = self.df[self.df['maintenance_needed'] == 1]
            fig.add_trace(
                go.Scatter(
                    x=maintenance_events['timestamp'],
                    y=maintenance_events['energy_output'],
                    mode='markers',
                    name='Maintenance Events',
                    marker=dict(color='red', size=8, symbol='x')
                ),
                row=2, col=2
            )

            numeric_cols = ['energy_output', 'solar_irradiance', 'panel_temp', 'temperature', 'humidity']
            corr_matrix = self.df[numeric_cols].corr()

            fig.add_trace(
                go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    colorscale='RdBu',
                    zmid=0,
                    name='Correlation'
                ),
                row=2, col=3
            )

            hourly_avg = self.df.groupby(self.df['timestamp'].dt.hour)['energy_output'].mean()
            fig.add_trace(
                go.Bar(
                    x=hourly_avg.index,
                    y=hourly_avg.values,
                    name='Hourly Average',
                    marker_color='#3742fa',
                    opacity=0.7
                ),
                row=3, col=1
            )

            monthly_avg = self.df.groupby(self.df['timestamp'].dt.month)['energy_output'].mean()
            fig.add_trace(
                go.Bar(
                    x=monthly_avg.index,
                    y=monthly_avg.values,
                    name='Monthly Average',
                    marker_color='#2ed573',
                    opacity=0.7
                ),
                row=3, col=2
            )

            if 'efficiency' in self.df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=self.df['dust_level'],
                        y=self.df['efficiency'],
                        mode='markers',
                        name='Efficiency vs Dust',
                        marker=dict(
                            color=self.df['panel_temp'],
                            colorscale='Plasma',
                            size=4,
                            opacity=0.6
                        )
                    ),
                    row=3, col=3
                )

            fig.update_layout(
                title='Solar Panel System - Comprehensive Overview Dashboard',
                height=1200,
                showlegend=False,
                template='plotly_dark',
                font=dict(size=10)
            )

            fig.write_html(f'{self.output_dir}/overview_dashboard.html')
            logger.info("Overview dashboard saved as HTML")

            return fig

        except Exception as e:
            logger.error(f"Error creating overview dashboard: {e}")
            return None

    def create_time_series_explorer(self):
        try:
            logger.info("Creating time series explorer...")

            ts_df = self.df.copy()
            ts_df['timestamp'] = pd.to_datetime(ts_df['timestamp'])
            ts_df = ts_df.sort_values('timestamp')

            fig = make_subplots(
                rows=4, cols=1,
                shared_xaxes=True,
                subplot_titles=[
                    'Energy Output & Solar Irradiance',
                    'Temperature Comparison',
                    'Environmental Conditions',
                    'System Health Indicators'
                ],
                specs=[
                    [{"secondary_y": True}],
                    [{"secondary_y": True}],
                    [{"secondary_y": True}],
                    [{"secondary_y": False}]
                ],
                vertical_spacing=0.05
            )

            fig.add_trace(
                go.Scatter(
                    x=ts_df['timestamp'],
                    y=ts_df['energy_output'],
                    name='Energy Output (kW)',
                    line=dict(color='#00ff88', width=2),
                    hovertemplate='<b>Energy:</b> %{y:.2f} kW<br><b>Time:</b> %{x}<extra></extra>'
                ),
                row=1, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=ts_df['timestamp'],
                    y=ts_df['solar_irradiance'],
                    name='Solar Irradiance (W/m²)',
                    line=dict(color='#ffa502', width=1),
                    yaxis='y2',
                    hovertemplate='<b>Irradiance:</b> %{y:.0f} W/m²<br><b>Time:</b> %{x}<extra></extra>'
                ),
                row=1, col=1, secondary_y=True
            )

            fig.add_trace(
                go.Scatter(
                    x=ts_df['timestamp'],
                    y=ts_df['temperature'],
                    name='Ambient Temp (°C)',
                    line=dict(color='#3742fa', width=2),
                    hovertemplate='<b>Ambient:</b> %{y:.1f}°C<br><b>Time:</b> %{x}<extra></extra>'
                ),
                row=2, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=ts_df['timestamp'],
                    y=ts_df['panel_temp'],
                    name='Panel Temp (°C)',
                    line=dict(color='#ff4757', width=2),
                    yaxis='y4',
                    hovertemplate='<b>Panel:</b> %{y:.1f}°C<br><b>Time:</b> %{x}<extra></extra>'
                ),
                row=2, col=1, secondary_y=True
            )

            fig.add_trace(
                go.Scatter(
                    x=ts_df['timestamp'],
                    y=ts_df['humidity'],
                    name='Humidity (%)',
                    line=dict(color='#2ed573', width=2),
                    hovertemplate='<b>Humidity:</b> %{y:.1f}%<br><b>Time:</b> %{x}<extra></extra>'
                ),
                row=3, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=ts_df['timestamp'],
                    y=ts_df['wind_speed'],
                    name='Wind Speed (m/s)',
                    line=dict(color='#5352ed', width=2),
                    yaxis='y6',
                    hovertemplate='<b>Wind:</b> %{y:.1f} m/s<br><b>Time:</b> %{x}<extra></extra>'
                ),
                row=3, col=1, secondary_y=True
            )

            fig.add_trace(
                go.Scatter(
                    x=ts_df['timestamp'],
                    y=ts_df['dust_level'],
                    name='Dust Level',
                    line=dict(color='#ff6348', width=2),
                    fill='tonexty',
                    hovertemplate='<b>Dust:</b> %{y:.2f}<br><b>Time:</b> %{x}<extra></extra>'
                ),
                row=4, col=1
            )

            maintenance_events = ts_df[ts_df['maintenance_needed'] == 1]['timestamp']
            for event_time in maintenance_events:
                fig.add_vline(
                    x=event_time,
                    line_dash="dash",
                    line_color="red",
                    opacity=0.7,
                    annotation_text="Maintenance"
                )

            fig.update_layout(
                title='Interactive Time Series Explorer - Solar Panel System',
                height=1000,
                template='plotly_dark',
                hovermode='x unified',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )

            fig.update_yaxes(title_text="Energy Output (kW)", row=1, col=1)
            fig.update_yaxes(title_text="Solar Irradiance (W/m²)", row=1, col=1, secondary_y=True)
            fig.update_yaxes(title_text="Ambient Temp (°C)", row=2, col=1)
            fig.update_yaxes(title_text="Panel Temp (°C)", row=2, col=1, secondary_y=True)
            fig.update_yaxes(title_text="Humidity (%)", row=3, col=1)
            fig.update_yaxes(title_text="Wind Speed (m/s)", row=3, col=1, secondary_y=True)
            fig.update_yaxes(title_text="Dust Level", row=4, col=1)

            fig.write_html(f'{self.output_dir}/time_series_explorer.html')
            logger.info("Time series explorer saved as HTML")

            return fig

        except Exception as e:
            logger.error(f"Error creating time series explorer: {e}")
            return None

    def create_performance_analyzer(self):
        try:
            logger.info("Creating performance analyzer...")

            if 'efficiency' not in self.df.columns:
                self.df['efficiency'] = np.where(
                    self.df['solar_irradiance'] > 0,
                    self.df['energy_output'] / (self.df['solar_irradiance'] / 1000),
                    0
                )

            fig = make_subplots(
                rows=2, cols=3,
                subplot_titles=[
                    'Efficiency Distribution', 'Performance vs Temperature', 'Energy vs Irradiance',
                    'Dust Impact Analysis', 'Performance Heatmap', 'Efficiency Trends'
                ],
                specs=[
                    [{"type": "histogram"}, {"type": "scatter"}, {"type": "scatter"}],
                    [{"type": "scatter"}, {"type": "heatmap"}, {"type": "scatter"}]
                ]
            )

            fig.add_trace(
                go.Histogram(
                    x=self.df['efficiency'],
                    nbinsx=50,
                    name='Efficiency Distribution',
                    marker_color='#00ff88',
                    opacity=0.7,
                    hovertemplate='<b>Efficiency:</b> %{x:.3f}<br><b>Count:</b> %{y}<extra></extra>'
                ),
                row=1, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=self.df['panel_temp'],
                    y=self.df['efficiency'],
                    mode='markers',
                    name='Efficiency vs Temp',
                    marker=dict(
                        color=self.df['solar_irradiance'],
                        colorscale='Viridis',
                        size=4,
                        opacity=0.6,
                        colorbar=dict(title="Solar Irradiance", x=0.45)
                    ),
                    hovertemplate='<b>Panel Temp:</b> %{x:.1f}°C<br><b>Efficiency:</b> %{y:.3f}<br><b>Irradiance:</b> %{marker.color:.0f} W/m²<extra></extra>'
                ),
                row=1, col=2
            )

            fig.add_trace(
                go.Scatter(
                    x=self.df['solar_irradiance'],
                    y=self.df['energy_output'],
                    mode='markers',
                    name='Energy vs Irradiance',
                    marker=dict(
                        color=self.df['panel_temp'],
                        colorscale='Plasma',
                        size=4,
                        opacity=0.6,
                        colorbar=dict(title="Panel Temp (°C)", x=0.78)
                    ),
                    hovertemplate='<b>Irradiance:</b> %{x:.0f} W/m²<br><b>Energy:</b> %{y:.2f} kW<br><b>Panel Temp:</b> %{marker.color:.1f}°C<extra></extra>'
                ),
                row=1, col=3
            )

            fig.add_trace(
                go.Scatter(
                    x=self.df['dust_level'],
                    y=self.df['efficiency'],
                    mode='markers',
                    name='Dust Impact',
                    marker=dict(
                        color=self.df['energy_output'],
                        colorscale='RdYlBu_r',
                        size=5,
                        opacity=0.7,
                        colorbar=dict(title="Energy Output (kW)", x=0.45, y=0.3)
                    ),
                    hovertemplate='<b>Dust Level:</b> %{x:.2f}<br><b>Efficiency:</b> %{y:.3f}<br><b>Energy:</b> %{marker.color:.2f} kW<extra></extra>'
                ),
                row=2, col=1
            )

            self.df['hour'] = self.df['timestamp'].dt.hour
            self.df['month'] = self.df['timestamp'].dt.month

            pivot_data = self.df.pivot_table(
                values='efficiency',
                index='hour',
                columns='month',
                aggfunc='mean'
            )

            fig.add_trace(
                go.Heatmap(
                    z=pivot_data.values,
                    x=[f'Month {i}' for i in pivot_data.columns],
                    y=[f'{i}:00' for i in pivot_data.index],
                    colorscale='Viridis',
                    name='Efficiency Heatmap',
                    hovertemplate='<b>Month:</b> %{x}<br><b>Hour:</b> %{y}<br><b>Avg Efficiency:</b> %{z:.3f}<extra></extra>',
                    colorbar=dict(title="Avg Efficiency", x=0.78, y=0.3)
                ),
                row=2, col=2
            )

            daily_efficiency = self.df.groupby(self.df['timestamp'].dt.date)['efficiency'].agg(['mean', 'std']).reset_index()
            daily_efficiency['timestamp'] = pd.to_datetime(daily_efficiency['timestamp'])

            fig.add_trace(
                go.Scatter(
                    x=daily_efficiency['timestamp'],
                    y=daily_efficiency['mean'],
                    mode='lines+markers',
                    name='Daily Avg Efficiency',
                    line=dict(color='#ff4757', width=2),
                    hovertemplate='<b>Date:</b> %{x}<br><b>Avg Efficiency:</b> %{y:.3f}<extra></extra>'
                ),
                row=2, col=3
            )

            fig.add_trace(
                go.Scatter(
                    x=daily_efficiency['timestamp'],
                    y=daily_efficiency['mean'] + daily_efficiency['std'],
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=2, col=3
            )

            fig.add_trace(
                go.Scatter(
                    x=daily_efficiency['timestamp'],
                    y=daily_efficiency['mean'] - daily_efficiency['std'],
                    mode='lines',
                    line=dict(width=0),
                    fill='tonexty',
                    fillcolor='rgba(255, 71, 87, 0.2)',
                    showlegend=False,
                    name='Std Dev',
                    hovertemplate='<b>Date:</b> %{x}<br><b>Std Dev:</b> %{y:.3f}<extra></extra>'
                ),
                row=2, col=3
            )

            fig.update_layout(
                title='Performance Analysis Dashboard - Solar Panel System',
                height=800,
                template='plotly_dark',
                showlegend=False
            )

            fig.write_html(f'{self.output_dir}/performance_analyzer.html')
            logger.info("Performance analyzer saved as HTML")

            return fig

        except Exception as e:
            logger.error(f"Error creating performance analyzer: {e}")
            return None

    def create_anomaly_explorer(self):
        try:
            logger.info("Creating anomaly explorer...")

            from sklearn.ensemble import IsolationForest
            from sklearn.preprocessing import StandardScaler

            features = ['energy_output', 'solar_irradiance', 'panel_temp', 'dust_level']
            X = self.df[features].dropna()

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            anomaly_labels = iso_forest.fit_predict(X_scaled)
            anomaly_scores = iso_forest.decision_function(X_scaled)

            anomaly_df = X.copy()
            anomaly_df['anomaly'] = anomaly_labels
            anomaly_df['anomaly_score'] = anomaly_scores
            anomaly_df['is_anomaly'] = anomaly_df['anomaly'] == -1

            fig = make_subplots(
                rows=2, cols=3,
                subplot_titles=[
                    'Anomaly Score Distribution', 'Anomalies in Feature Space', 'Anomaly Timeline',
                    'Anomaly vs Normal Comparison', 'Feature Importance for Anomalies', 'Anomaly Patterns'
                ]
            )

            fig.add_trace(
                go.Histogram(
                    x=anomaly_df['anomaly_score'],
                    nbinsx=50,
                    name='Anomaly Scores',
                    marker_color='#ff4757',
                    opacity=0.7,
                    hovertemplate='<b>Score:</b> %{x:.3f}<br><b>Count:</b> %{y}<extra></extra>'
                ),
                row=1, col=1
            )

            fig.add_vline(
                x=0,
                line_dash="dash",
                line_color="white",
                opacity=0.7,
                annotation_text="Threshold",
                row=1, col=1
            )

            normal_data = anomaly_df[anomaly_df['is_anomaly'] == False]
            anomaly_data = anomaly_df[anomaly_df['is_anomaly'] == True]

            fig.add_trace(
                go.Scatter(
                    x=normal_data['solar_irradiance'],
                    y=normal_data['energy_output'],
                    mode='markers',
                    name='Normal',
                    marker=dict(color='#2ed573', size=4, opacity=0.6),
                    hovertemplate='<b>Irradiance:</b> %{x:.0f} W/m²<br><b>Energy:</b> %{y:.2f} kW<br><b>Status:</b> Normal<extra></extra>'
                ),
                row=1, col=2
            )

            fig.add_trace(
                go.Scatter(
                    x=anomaly_data['solar_irradiance'],
                    y=anomaly_data['energy_output'],
                    mode='markers',
                    name='Anomaly',
                    marker=dict(color='#ff4757', size=8, opacity=0.8, symbol='x'),
                    hovertemplate='<b>Irradiance:</b> %{x:.0f} W/m²<br><b>Energy:</b> %{y:.2f} kW<br><b>Status:</b> Anomaly<extra></extra>'
                ),
                row=1, col=2
            )

            anomaly_df['timestamp'] = self.df.loc[X.index, 'timestamp']
            anomaly_timeline = anomaly_df[anomaly_df['is_anomaly'] == True]

            fig.add_trace(
                go.Scatter(
                    x=anomaly_timeline['timestamp'],
                    y=anomaly_timeline['anomaly_score'],
                    mode='markers',
                    name='Anomaly Timeline',
                    marker=dict(
                        color=anomaly_timeline['energy_output'],
                        colorscale='Reds',
                        size=8,
                        opacity=0.8,
                        colorbar=dict(title="Energy Output", x=0.78)
                    ),
                    hovertemplate='<b>Time:</b> %{x}<br><b>Score:</b> %{y:.3f}<br><b>Energy:</b> %{marker.color:.2f} kW<extra></extra>'
                ),
                row=1, col=3
            )

            comparison_data = []
            for feature in features:
                for status in ['Normal', 'Anomaly']:
                    if status == 'Normal':
                        values = normal_data[feature]
                    else:
                        values = anomaly_data[feature]

                    comparison_data.extend([{
                        'feature': feature,
                        'status': status,
                        'value': val
                    } for val in values])

            comparison_df = pd.DataFrame(comparison_data)

            for i, feature in enumerate(features):
                feature_data = comparison_df[comparison_df['feature'] == feature]

                for status in ['Normal', 'Anomaly']:
                    status_data = feature_data[feature_data['status'] == status]
                    color = '#2ed573' if status == 'Normal' else '#ff4757'

                    fig.add_trace(
                        go.Box(
                            y=status_data['value'],
                            name=f'{feature} - {status}',
                            marker_color=color,
                            opacity=0.7,
                            showlegend=False
                        ),
                        row=2, col=1
                    )

            feature_importance = []
            for feature in features:
                normal_mean = normal_data[feature].mean()
                anomaly_mean = anomaly_data[feature].mean()
                importance = abs(anomaly_mean - normal_mean) / normal_mean if normal_mean != 0 else 0
                feature_importance.append({'feature': feature, 'importance': importance})

            importance_df = pd.DataFrame(feature_importance).sort_values('importance', ascending=True)

            fig.add_trace(
                go.Bar(
                    x=importance_df['importance'],
                    y=importance_df['feature'],
                    orientation='h',
                    name='Feature Importance',
                    marker_color='#3742fa',
                    opacity=0.7,
                    hovertemplate='<b>Feature:</b> %{y}<br><b>Importance:</b> %{x:.3f}<extra></extra>'
                ),
                row=2, col=2
            )

            fig.add_trace(
                go.Scatter(
                    x=normal_data['panel_temp'],
                    y=normal_data['dust_level'],
                    mode='markers',
                    name='Normal Pattern',
                    marker=dict(color='#2ed573', size=4, opacity=0.6),
                    hovertemplate='<b>Panel Temp:</b> %{x:.1f}°C<br><b>Dust Level:</b> %{y:.2f}<br><b>Status:</b> Normal<extra></extra>'
                ),
                row=2, col=3
            )

            fig.add_trace(
                go.Scatter(
                    x=anomaly_data['panel_temp'],
                    y=anomaly_data['dust_level'],
                    mode='markers',
                    name='Anomaly Pattern',
                    marker=dict(color='#ff4757', size=8, opacity=0.8, symbol='x'),
                    hovertemplate='<b>Panel Temp:</b> %{x:.1f}°C<br><b>Dust Level:</b> %{y:.2f}<br><b>Status:</b> Anomaly<extra></extra>'
                ),
                row=2, col=3
            )

            fig.update_layout(
                title='Anomaly Detection Explorer - Solar Panel System',
                height=800,
                template='plotly_dark',
                showlegend=False
            )

            fig.write_html(f'{self.output_dir}/anomaly_explorer.html')
            logger.info("Anomaly explorer saved as HTML")

            return fig

        except Exception as e:
            logger.error(f"Error creating anomaly explorer: {e}")
            return None

    def generate_all_dashboards(self):
        try:
            logger.info("Generating all interactive dashboards...")

            dashboards = [
                ("Overview Dashboard", self.create_overview_dashboard),
                ("Time Series Explorer", self.create_time_series_explorer),
                ("Performance Analyzer", self.create_performance_analyzer),
                ("Anomaly Explorer", self.create_anomaly_explorer)
            ]

            results = {}

            for name, func in dashboards:
                try:
                    logger.info(f"Creating {name}...")
                    fig = func()
                    results[name] = fig is not None

                    if fig:
                        logger.info(f"[SUCCESS] {name} created successfully")
                    else:
                        logger.error(f"[ERROR] {name} failed")

                except Exception as e:
                    logger.error(f"[ERROR] {name} failed with error: {e}")
                    results[name] = False

            self._create_index_html(results)

            return results

        except Exception as e:
            logger.error(f"Error generating dashboards: {e}")
            return {}

    def _create_index_html(self, results):
        try:

            dashboards_info = {
                "Overview Dashboard": {
                    "file": "overview_dashboard.html",
                    "description": "Comprehensive system overview with key metrics, trends, and correlations.",
                    "icon": ""
                },
                "Time Series Explorer": {
                    "file": "time_series_explorer.html",
                    "description": "Interactive time series analysis with multiple variables and maintenance events.",
                    "icon": ""
                },
                "Performance Analyzer": {
                    "file": "performance_analyzer.html",
                    "description": "Deep dive into system performance, efficiency analysis, and optimization insights.",
                    "icon": ""
                },
                "Anomaly Explorer": {
                    "file": "anomaly_explorer.html",
                    "description": "Advanced anomaly detection and pattern analysis for system health monitoring.",
                    "icon": ""
                }
            }

            for name, info in dashboards_info.items():
                status = "success" if results.get(name, False) else "error"
                status_text = " Available" if status == "success" else " Failed"

                if status == "success":
                    html_content += f'<a href="{info["file"]}" target="_blank">Open Dashboard</a>'

                html_content += "</div>"

            with open(f'{self.output_dir}/index.html', 'w') as f:
                f.write(html_content)

            logger.info("Index HTML file created successfully")

        except Exception as e:
            logger.error(f"Error creating index HTML: {e}")

def main():
    print("Interactive EDA Dashboard Generator")
    print("=" * 50)

    eda = InteractiveEDA()

    if not eda.load_data():
        print("Failed to load data. Exiting.")
        return

    results = eda.generate_all_dashboards()

    print("\n" + "=" * 50)
    print("DASHBOARD GENERATION SUMMARY")
    print("=" * 50)

    for name, success in results.items():
        status = "[SUCCESS]" if success else "[ERROR] FAILED"
        print(f"{name:<25} {status}")

    successful = sum(results.values())
    total = len(results)

    print(f"\nDashboards created: {successful}/{total}")

    if successful > 0:
        print(f"\n[SUCCESS] Interactive dashboards generated successfully!")
        print(f"Location: {eda.output_dir}/")
        print(f"Open: {eda.output_dir}/index.html")
    else:
        print("\n[ERROR] No dashboards were created successfully.")

if __name__ == "__main__":
    main()
