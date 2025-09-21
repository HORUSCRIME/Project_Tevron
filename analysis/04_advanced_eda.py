
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.stats import normaltest, jarque_bera, shapiro
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import sys
import os
import logging
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_ingestion import DataIngestion
from src.feature_engineering import FeatureEngineer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

plt.style.use('dark_background')
sns.set_palette("husl")

class AdvancedEDA:
    def __init__(self):
        self.df = None
        self.numeric_cols = None

    def load_data(self):
        try:
            logger.info("Loading data for advanced EDA...")

            ingestion = DataIngestion()
            self.df = ingestion.load_historical_data()

            if self.df is None or self.df.empty:
                logger.warning("No data available, generating sample data")
                self.df = ingestion._generate_sample_data(days=365)

            engineer = FeatureEngineer()
            self.df = engineer.create_derived_features(self.df)

            self.numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()

            logger.info(f"Data loaded successfully. Shape: {self.df.shape}")
            return True

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False

    def statistical_summary(self):
        try:
            logger.info("Generating statistical summary...")

            print("ADVANCED STATISTICAL SUMMARY")
            print("=" * 80)

            print("\n1. DESCRIPTIVE STATISTICS")
            print("-" * 40)
            desc_stats = self.df[self.numeric_cols].describe()
            print(desc_stats)

            print("\n2. DISTRIBUTION CHARACTERISTICS")
            print("-" * 40)
            skew_kurt = pd.DataFrame({
                'Skewness': self.df[self.numeric_cols].skew(),
                'Kurtosis': self.df[self.numeric_cols].kurtosis(),
                'Variance': self.df[self.numeric_cols].var(),
                'Std_Dev': self.df[self.numeric_cols].std()
            })
            print(skew_kurt.round(3))

            print("\n3. NORMALITY TESTS")
            print("-" * 40)
            normality_results = []

            for col in self.numeric_cols[:10]:
                data = self.df[col].dropna()
                if len(data) > 20:
                    if len(data) <= 5000:
                        shapiro_stat, shapiro_p = shapiro(data)
                    else:
                        shapiro_stat, shapiro_p = np.nan, np.nan

                    jb_stat, jb_p = jarque_bera(data)

                    normality_results.append({
                        'Variable': col,
                        'Shapiro_Stat': shapiro_stat,
                        'Shapiro_p': shapiro_p,
                        'JB_Stat': jb_stat,
                        'JB_p': jb_p,
                        'Normal_Shapiro': shapiro_p > 0.05 if not np.isnan(shapiro_p) else 'N/A',
                        'Normal_JB': jb_p > 0.05
                    })

            normality_df = pd.DataFrame(normality_results)
            print(normality_df.round(4))

            print("\n4. CORRELATION STRENGTH ANALYSIS")
            print("-" * 40)
            corr_matrix = self.df[self.numeric_cols].corr()

            corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.3:
                        corr_pairs.append({
                            'Var1': corr_matrix.columns[i],
                            'Var2': corr_matrix.columns[j],
                            'Correlation': corr_val,
                            'Strength': self._correlation_strength(abs(corr_val))
                        })

            corr_df = pd.DataFrame(corr_pairs)
            corr_df = corr_df.sort_values('Correlation', key=abs, ascending=False)
            print(corr_df.head(15))

            return True

        except Exception as e:
            logger.error(f"Error in statistical summary: {e}")
            return False

    def _correlation_strength(self, corr_val):
        if corr_val >= 0.9:
            return "Very Strong"
        elif corr_val >= 0.7:
            return "Strong"
        elif corr_val >= 0.5:
            return "Moderate"
        elif corr_val >= 0.3:
            return "Weak"
        else:
            return "Very Weak"

    def time_series_analysis(self):
        try:
            logger.info("Performing time series analysis...")

            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
            self.df = self.df.sort_values('timestamp')

            ts_df = self.df.set_index('timestamp')

            print("\nTIME SERIES ANALYSIS")
            print("=" * 50)

            print("\n1. STATIONARITY TESTS (Augmented Dickey-Fuller)")
            print("-" * 50)

            for col in ['energy_output', 'solar_irradiance', 'panel_temp']:
                if col in ts_df.columns:
                    series = ts_df[col].dropna()
                    adf_result = adfuller(series)

                    print(f"\n{col}:")
                    print(f"  ADF Statistic: {adf_result[0]:.6f}")
                    print(f"  p-value: {adf_result[1]:.6f}")
                    print(f"  Critical Values: {adf_result[4]}")
                    print(f"  Stationary: {'Yes' if adf_result[1] < 0.05 else 'No'}")

            self._seasonal_decomposition(ts_df)

            self._autocorrelation_analysis(ts_df)

            return True

        except Exception as e:
            logger.error(f"Error in time series analysis: {e}")
            return False

    def _seasonal_decomposition(self, ts_df):
        try:
            print("\n2. SEASONAL DECOMPOSITION")
            print("-" * 30)

            daily_energy = ts_df['energy_output'].resample('D').sum()

            if len(daily_energy) > 30:
                decomposition = seasonal_decompose(daily_energy, model='additive', period=7)

                fig, axes = plt.subplots(4, 1, figsize=(15, 12))
                fig.suptitle('Seasonal Decomposition of Daily Energy Output', fontsize=16)

                decomposition.observed.plot(ax=axes[0], title='Original', color='white')
                decomposition.trend.plot(ax=axes[1], title='Trend', color='cyan')
                decomposition.seasonal.plot(ax=axes[2], title='Seasonal', color='yellow')
                decomposition.resid.plot(ax=axes[3], title='Residual', color='red')

                for ax in axes:
                    ax.grid(True, alpha=0.3)

                plt.tight_layout()
                plt.savefig('analysis/seasonal_decomposition.png', dpi=300, bbox_inches='tight')
                plt.show()

                print(f"Trend strength: {1 - np.var(decomposition.resid.dropna()) / np.var(decomposition.observed.dropna()):.3f}")
                print(f"Seasonal strength: {1 - np.var(decomposition.resid.dropna()) / np.var(decomposition.seasonal.dropna()):.3f}")

        except Exception as e:
            logger.error(f"Error in seasonal decomposition: {e}")

    def _autocorrelation_analysis(self, ts_df):
        try:
            from statsmodels.tsa.stattools import acf, pacf

            print("\n3. AUTOCORRELATION ANALYSIS")
            print("-" * 30)

            energy_series = ts_df['energy_output'].dropna()

            if len(energy_series) > 50:
                acf_values = acf(energy_series, nlags=48, fft=True)
                pacf_values = pacf(energy_series, nlags=48)

                significant_lags = []
                confidence_interval = 1.96 / np.sqrt(len(energy_series))

                for i, acf_val in enumerate(acf_values[1:], 1):
                    if abs(acf_val) > confidence_interval:
                        significant_lags.append((i, acf_val))

                print(f"Significant autocorrelation lags (first 10):")
                for lag, value in significant_lags[:10]:
                    print(f"  Lag {lag}: {value:.3f}")

                fig, axes = plt.subplots(2, 1, figsize=(12, 8))

                axes[0].plot(range(len(acf_values)), acf_values, 'o-', color='cyan')
                axes[0].axhline(y=confidence_interval, color='red', linestyle='--', alpha=0.7)
                axes[0].axhline(y=-confidence_interval, color='red', linestyle='--', alpha=0.7)
                axes[0].set_title('Autocorrelation Function (ACF)')
                axes[0].grid(True, alpha=0.3)

                axes[1].plot(range(len(pacf_values)), pacf_values, 'o-', color='yellow')
                axes[1].axhline(y=confidence_interval, color='red', linestyle='--', alpha=0.7)
                axes[1].axhline(y=-confidence_interval, color='red', linestyle='--', alpha=0.7)
                axes[1].set_title('Partial Autocorrelation Function (PACF)')
                axes[1].grid(True, alpha=0.3)

                plt.tight_layout()
                plt.savefig('analysis/autocorrelation_analysis.png', dpi=300, bbox_inches='tight')
                plt.show()

        except Exception as e:
            logger.error(f"Error in autocorrelation analysis: {e}")

    def advanced_visualizations(self):
        try:
            logger.info("Creating advanced visualizations...")

            self._create_3d_scatter()

            self._create_parallel_coordinates()

            self._create_radar_chart()

            self._create_sunburst_chart()

            self._create_animated_timeseries()

            return True

        except Exception as e:
            logger.error(f"Error creating advanced visualizations: {e}")
            return False

    def _create_3d_scatter(self):
        try:
            fig = go.Figure(data=[go.Scatter3d(
                x=self.df['solar_irradiance'],
                y=self.df['panel_temp'],
                z=self.df['energy_output'],
                mode='markers',
                marker=dict(
                    size=3,
                    color=self.df['efficiency'] if 'efficiency' in self.df.columns else self.df['energy_output'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Efficiency")
                ),
                text=self.df['timestamp'].dt.strftime('%Y-%m-%d %H:%M'),
                hovertemplate='<b>Solar Irradiance:</b> %{x}<br>' +
                             '<b>Panel Temp:</b> %{y}<br>' +
                             '<b>Energy Output:</b> %{z}<br>' +
                             '<b>Time:</b> %{text}<extra></extra>'
            )])

            fig.update_layout(
                title='3D Relationship: Solar Irradiance vs Panel Temperature vs Energy Output',
                scene=dict(
                    xaxis_title='Solar Irradiance (W/m²)',
                    yaxis_title='Panel Temperature (°C)',
                    zaxis_title='Energy Output (kW)',
                    bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="white", gridwidth=2),
                    yaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="white", gridwidth=2),
                    zaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="white", gridwidth=2)
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )

            fig.write_html('analysis/3d_scatter_plot.html')
            fig.show()

        except Exception as e:
            logger.error(f"Error creating 3D scatter plot: {e}")

    def _create_parallel_coordinates(self):
        try:
            key_vars = ['energy_output', 'solar_irradiance', 'panel_temp', 'temperature',
                       'humidity', 'wind_speed', 'dust_level']

            scaler = StandardScaler()
            normalized_data = scaler.fit_transform(self.df[key_vars])
            normalized_df = pd.DataFrame(normalized_data, columns=key_vars)
            normalized_df['maintenance_needed'] = self.df['maintenance_needed'].values

            fig = go.Figure(data=go.Parcoords(
                line=dict(color=normalized_df['maintenance_needed'],
                         colorscale='RdYlBu',
                         showscale=True,
                         colorbar=dict(title="Maintenance Needed")),
                dimensions=list([
                    dict(range=[-3, 3], label=var, values=normalized_df[var])
                    for var in key_vars
                ])
            ))

            fig.update_layout(
                title='Parallel Coordinates Plot - System Variables',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )

            fig.write_html('analysis/parallel_coordinates.html')
            fig.show()

        except Exception as e:
            logger.error(f"Error creating parallel coordinates plot: {e}")

    def _create_radar_chart(self):
        try:
            hourly_metrics = self.df.groupby(self.df['timestamp'].dt.hour).agg({
                'energy_output': 'mean',
                'efficiency': 'mean' if 'efficiency' in self.df.columns else lambda x: x.iloc[0],
                'panel_temp': 'mean',
                'dust_level': 'mean',
                'voltage': 'mean',
                'current': 'mean'
            }).reset_index()

            metrics = ['energy_output', 'efficiency', 'panel_temp', 'dust_level', 'voltage', 'current']
            for metric in metrics:
                if metric in hourly_metrics.columns:
                    hourly_metrics[f'{metric}_norm'] = (
                        hourly_metrics[metric] - hourly_metrics[metric].min()
                    ) / (hourly_metrics[metric].max() - hourly_metrics[metric].min())

            peak_hours = [10, 12, 14]

            fig = go.Figure()

            for hour in peak_hours:
                hour_data = hourly_metrics[hourly_metrics['timestamp'] == hour]
                if not hour_data.empty:
                    values = [hour_data[f'{metric}_norm'].iloc[0] for metric in metrics
                             if f'{metric}_norm' in hour_data.columns]

                    fig.add_trace(go.Scatterpolar(
                        r=values + [values[0]],
                        theta=metrics + [metrics[0]],
                        fill='toself',
                        name=f'{hour}:00',
                        line=dict(width=2)
                    ))

            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                title="Performance Radar Chart - Peak Hours Comparison",
                showlegend=True,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )

            fig.write_html('analysis/radar_chart.html')
            fig.show()

        except Exception as e:
            logger.error(f"Error creating radar chart: {e}")

    def _create_sunburst_chart(self):
        try:
            self.df['hour_category'] = pd.cut(self.df['timestamp'].dt.hour,
                                            bins=[0, 6, 12, 18, 24],
                                            labels=['Night', 'Morning', 'Afternoon', 'Evening'])

            self.df['temp_category'] = pd.cut(self.df['panel_temp'],
                                            bins=3,
                                            labels=['Cool', 'Moderate', 'Hot'])

            self.df['performance_category'] = pd.cut(self.df['energy_output'],
                                                   bins=3,
                                                   labels=['Low', 'Medium', 'High'])

            sunburst_data = self.df.groupby(['hour_category', 'temp_category', 'performance_category']).size().reset_index(name='count')

            ids = []
            labels = []
            parents = []
            values = []

            for hour_cat in sunburst_data['hour_category'].unique():
                if pd.notna(hour_cat):
                    ids.append(hour_cat)
                    labels.append(hour_cat)
                    parents.append("")
                    values.append(sunburst_data[sunburst_data['hour_category'] == hour_cat]['count'].sum())

            for _, row in sunburst_data.groupby(['hour_category', 'temp_category'])['count'].sum().reset_index().iterrows():
                if pd.notna(row['hour_category']) and pd.notna(row['temp_category']):
                    id_str = f"{row['hour_category']}-{row['temp_category']}"
                    ids.append(id_str)
                    labels.append(row['temp_category'])
                    parents.append(row['hour_category'])
                    values.append(row['count'])

            for _, row in sunburst_data.iterrows():
                if pd.notna(row['hour_category']) and pd.notna(row['temp_category']) and pd.notna(row['performance_category']):
                    id_str = f"{row['hour_category']}-{row['temp_category']}-{row['performance_category']}"
                    parent_str = f"{row['hour_category']}-{row['temp_category']}"
                    ids.append(id_str)
                    labels.append(row['performance_category'])
                    parents.append(parent_str)
                    values.append(row['count'])

            fig = go.Figure(go.Sunburst(
                ids=ids,
                labels=labels,
                parents=parents,
                values=values,
                branchvalues="total",
                hovertemplate='<b>%{label}</b><br>Count: %{value}<extra></extra>',
                maxdepth=3
            ))

            fig.update_layout(
                title="Hierarchical Analysis: Time → Temperature → Performance",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )

            fig.write_html('analysis/sunburst_chart.html')
            fig.show()

        except Exception as e:
            logger.error(f"Error creating sunburst chart: {e}")

    def _create_animated_timeseries(self):
        try:
            daily_data = self.df.groupby(self.df['timestamp'].dt.date).agg({
                'energy_output': 'sum',
                'solar_irradiance': 'mean',
                'panel_temp': 'mean',
                'efficiency': 'mean' if 'efficiency' in self.df.columns else lambda x: x.iloc[0]
            }).reset_index()

            daily_data['timestamp'] = pd.to_datetime(daily_data['timestamp'])
            daily_data['cumulative_energy'] = daily_data['energy_output'].cumsum()

            fig = px.scatter(daily_data,
                           x='solar_irradiance',
                           y='energy_output',
                           animation_frame=daily_data['timestamp'].dt.strftime('%Y-%m-%d'),
                           size='panel_temp',
                           color='efficiency' if 'efficiency' in daily_data.columns else 'energy_output',
                           hover_name='timestamp',
                           title='Animated Solar Panel Performance Over Time',
                           labels={
                               'solar_irradiance': 'Average Solar Irradiance (W/m²)',
                               'energy_output': 'Daily Energy Output (kWh)',
                               'efficiency': 'Efficiency'
                           })

            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )

            fig.write_html('analysis/animated_timeseries.html')
            fig.show()

        except Exception as e:
            logger.error(f"Error creating animated time series: {e}")

    def clustering_analysis(self):
        try:
            logger.info("Performing clustering analysis...")

            cluster_features = ['energy_output', 'solar_irradiance', 'panel_temp',
                              'temperature', 'humidity', 'wind_speed', 'dust_level']

            cluster_data = self.df[cluster_features].dropna()

            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(cluster_data)

            inertias = []
            k_range = range(2, 11)

            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(scaled_data)
                inertias.append(kmeans.inertia_)

            plt.figure(figsize=(10, 6))
            plt.plot(k_range, inertias, 'bo-', color='cyan')
            plt.xlabel('Number of Clusters (k)')
            plt.ylabel('Inertia')
            plt.title('Elbow Method for Optimal k')
            plt.grid(True, alpha=0.3)
            plt.savefig('analysis/elbow_curve.png', dpi=300, bbox_inches='tight')
            plt.show()

            optimal_k = 4
            kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(scaled_data)

            cluster_df = cluster_data.copy()
            cluster_df['cluster'] = cluster_labels

            print("\nCLUSTER ANALYSIS")
            print("=" * 50)

            cluster_summary = cluster_df.groupby('cluster').agg({
                'energy_output': ['mean', 'std'],
                'solar_irradiance': ['mean', 'std'],
                'panel_temp': ['mean', 'std'],
                'dust_level': ['mean', 'std']
            }).round(2)

            print("\nCluster Characteristics:")
            print(cluster_summary)

            self._visualize_clusters(cluster_df, scaled_data, kmeans)

            return cluster_df

        except Exception as e:
            logger.error(f"Error in clustering analysis: {e}")
            return None

    def _visualize_clusters(self, cluster_df, scaled_data, kmeans):
        try:
            pca = PCA(n_components=2)
            pca_data = pca.fit_transform(scaled_data)

            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Clustering Analysis Results', fontsize=16)

            scatter = axes[0, 0].scatter(pca_data[:, 0], pca_data[:, 1],
                                       c=cluster_df['cluster'], cmap='viridis', alpha=0.6)
            axes[0, 0].set_title('Clusters in PCA Space')
            axes[0, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
            axes[0, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
            plt.colorbar(scatter, ax=axes[0, 0])

            for cluster in cluster_df['cluster'].unique():
                cluster_data = cluster_df[cluster_df['cluster'] == cluster]
                axes[0, 1].scatter(cluster_data['solar_irradiance'], cluster_data['energy_output'],
                                 label=f'Cluster {cluster}', alpha=0.6)
            axes[0, 1].set_xlabel('Solar Irradiance (W/m²)')
            axes[0, 1].set_ylabel('Energy Output (kW)')
            axes[0, 1].set_title('Clusters: Energy vs Irradiance')
            axes[0, 1].legend()

            for cluster in cluster_df['cluster'].unique():
                cluster_data = cluster_df[cluster_df['cluster'] == cluster]
                axes[1, 0].scatter(cluster_data['panel_temp'], cluster_data['dust_level'],
                                 label=f'Cluster {cluster}', alpha=0.6)
            axes[1, 0].set_xlabel('Panel Temperature (°C)')
            axes[1, 0].set_ylabel('Dust Level')
            axes[1, 0].set_title('Clusters: Temperature vs Dust')
            axes[1, 0].legend()

            cluster_counts = cluster_df['cluster'].value_counts().sort_index()
            axes[1, 1].bar(cluster_counts.index, cluster_counts.values, color='skyblue', alpha=0.7)
            axes[1, 1].set_xlabel('Cluster')
            axes[1, 1].set_ylabel('Number of Points')
            axes[1, 1].set_title('Cluster Sizes')

            for ax in axes.flat:
                ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig('analysis/clustering_results.png', dpi=300, bbox_inches='tight')
            plt.show()

        except Exception as e:
            logger.error(f"Error visualizing clusters: {e}")

    def anomaly_detection_analysis(self):
        try:
            logger.info("Performing anomaly detection analysis...")

            from sklearn.ensemble import IsolationForest
            from sklearn.svm import OneClassSVM
            from sklearn.covariance import EllipticEnvelope

            features = ['energy_output', 'solar_irradiance', 'panel_temp', 'dust_level']
            X = self.df[features].dropna()

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            methods = {
                'Isolation Forest': IsolationForest(contamination=0.1, random_state=42),
                'One-Class SVM': OneClassSVM(nu=0.1),
                'Elliptic Envelope': EllipticEnvelope(contamination=0.1, random_state=42)
            }

            anomaly_results = {}

            for name, method in methods.items():
                predictions = method.fit_predict(X_scaled)
                anomalies = predictions == -1

                anomaly_results[name] = {
                    'predictions': predictions,
                    'anomaly_count': anomalies.sum(),
                    'anomaly_rate': anomalies.mean()
                }

                print(f"\n{name}:")
                print(f"  Anomalies detected: {anomalies.sum()}")
                print(f"  Anomaly rate: {anomalies.mean():.3f}")

            self._visualize_anomalies(X, anomaly_results)

            return anomaly_results

        except Exception as e:
            logger.error(f"Error in anomaly detection analysis: {e}")
            return None

    def _visualize_anomalies(self, X, anomaly_results):
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Anomaly Detection Results', fontsize=16)

            methods = list(anomaly_results.keys())

            for i, method in enumerate(methods[:3]):
                row = i // 2
                col = i % 2

                predictions = anomaly_results[method]['predictions']
                normal_mask = predictions == 1
                anomaly_mask = predictions == -1

                axes[row, col].scatter(X.loc[normal_mask, 'solar_irradiance'],
                                     X.loc[normal_mask, 'energy_output'],
                                     c='blue', alpha=0.6, label='Normal', s=20)
                axes[row, col].scatter(X.loc[anomaly_mask, 'solar_irradiance'],
                                     X.loc[anomaly_mask, 'energy_output'],
                                     c='red', alpha=0.8, label='Anomaly', s=30)

                axes[row, col].set_xlabel('Solar Irradiance (W/m²)')
                axes[row, col].set_ylabel('Energy Output (kW)')
                axes[row, col].set_title(f'{method}')
                axes[row, col].legend()
                axes[row, col].grid(True, alpha=0.3)

            method_names = list(anomaly_results.keys())
            anomaly_counts = [anomaly_results[method]['anomaly_count'] for method in method_names]

            axes[1, 1].bar(method_names, anomaly_counts, color=['blue', 'green', 'orange'], alpha=0.7)
            axes[1, 1].set_ylabel('Number of Anomalies')
            axes[1, 1].set_title('Anomaly Count Comparison')
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig('analysis/anomaly_detection.png', dpi=300, bbox_inches='tight')
            plt.show()

        except Exception as e:
            logger.error(f"Error visualizing anomalies: {e}")

    def generate_comprehensive_report(self):
        try:
            logger.info("Generating comprehensive EDA report...")

            report = []
            report.append("COMPREHENSIVE EXPLORATORY DATA ANALYSIS REPORT")
            report.append("=" * 80)
            report.append(f"Generated on: {pd.Timestamp.now()}")
            report.append(f"Dataset shape: {self.df.shape}")
            report.append(f"Date range: {self.df['timestamp'].min()} to {self.df['timestamp'].max()}")
            report.append("")

            report.append("DATA QUALITY SUMMARY")
            report.append("-" * 30)
            missing_data = self.df.isnull().sum()
            report.append(f"Missing values: {missing_data.sum()} total")
            report.append(f"Complete records: {len(self.df) - missing_data.max()}")
            report.append(f"Data completeness: {((len(self.df) - missing_data.max()) / len(self.df) * 100):.1f}%")
            report.append("")

            report.append("KEY INSIGHTS")
            report.append("-" * 15)

            total_energy = self.df['energy_output'].sum()
            avg_daily_energy = self.df.groupby(self.df['timestamp'].dt.date)['energy_output'].sum().mean()
            peak_energy = self.df['energy_output'].max()

            report.append(f"• Total energy produced: {total_energy:.1f} kWh")
            report.append(f"• Average daily production: {avg_daily_energy:.1f} kWh")
            report.append(f"• Peak instantaneous output: {peak_energy:.1f} kW")

            if 'efficiency' in self.df.columns:
                avg_efficiency = self.df['efficiency'].mean()
                report.append(f"• Average system efficiency: {avg_efficiency:.2f}")

            maintenance_rate = self.df['maintenance_needed'].mean()
            report.append(f"• Maintenance frequency: {maintenance_rate:.1%}")

            temp_corr = self.df['energy_output'].corr(self.df['panel_temp'])
            irradiance_corr = self.df['energy_output'].corr(self.df['solar_irradiance'])

            report.append(f"• Temperature correlation with output: {temp_corr:.3f}")
            report.append(f"• Irradiance correlation with output: {irradiance_corr:.3f}")

            with open('analysis/comprehensive_eda_report.txt', 'w') as f:
                f.write('\n'.join(report))

            print('\n'.join(report))

            logger.info("Comprehensive EDA report generated successfully")
            return True

        except Exception as e:
            logger.error(f"Error generating comprehensive report: {e}")
            return False

def main():
    print("Advanced Exploratory Data Analysis for Solar Panel System")
    print("=" * 70)

    os.makedirs('analysis', exist_ok=True)

    eda = AdvancedEDA()

    if not eda.load_data():
        print("Failed to load data. Exiting.")
        return

    analyses = [
        ("Statistical Summary", eda.statistical_summary),
        ("Time Series Analysis", eda.time_series_analysis),
        ("Advanced Visualizations", eda.advanced_visualizations),
        ("Clustering Analysis", eda.clustering_analysis),
        ("Anomaly Detection", eda.anomaly_detection_analysis),
        ("Comprehensive Report", eda.generate_comprehensive_report)
    ]

    for analysis_name, analysis_func in analyses:
        print(f"\n{'='*20} {analysis_name} {'='*20}")
        try:
            success = analysis_func()
            if success:
                print(f"[SUCCESS] {analysis_name} completed successfully")
            else:
                print(f"[ERROR] {analysis_name} failed")
        except Exception as e:
            print(f"[ERROR] {analysis_name} failed with error: {e}")

    print(f"\n{'='*70}")
    print("[SUCCESS] Advanced EDA completed!")
    print("Check the 'analysis/' directory for generated files and visualizations.")

if __name__ == "__main__":
    main()
