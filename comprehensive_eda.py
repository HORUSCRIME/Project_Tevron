import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

class SolarEDA:
    """Comprehensive EDA class for solar panel data analysis"""
    
    def __init__(self, data_file='solar_training_data.csv'):
        """Initialize with data file"""
        try:
            self.df = pd.read_csv(data_file)
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
            print(f"Loaded {len(self.df)} records from {data_file}")
        except FileNotFoundError:
            print(f"File {data_file} not found. Generating sample data...")
            self.df = self._generate_sample_data()
        
        self.numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        
    def _generate_sample_data(self):
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
    
    def basic_info(self):
        """Display basic information about the dataset"""
        print("DATASET OVERVIEW")
        print("=" * 50)
        print(f"Shape: {self.df.shape}")
        print(f"Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        print(f"Date range: {self.df['timestamp'].min()} to {self.df['timestamp'].max()}")
        print(f"Duration: {(self.df['timestamp'].max() - self.df['timestamp'].min()).days} days")
        
        print("\nCOLUMN INFORMATION")
        print("-" * 30)
        print(self.df.info())
        
        print("\n MISSING VALUES")
        print("-" * 20)
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        missing_df = pd.DataFrame({
            'Missing Count': missing,
            'Missing %': missing_pct
        })
        print(missing_df[missing_df['Missing Count'] > 0])
        
        return self.df.describe()
    
    def temporal_analysis(self):
        """Comprehensive temporal analysis"""
        print("\n TEMPORAL ANALYSIS")
        print("=" * 50)
        
        self.df['hour'] = self.df['timestamp'].dt.hour
        self.df['day_of_week'] = self.df['timestamp'].dt.dayofweek
        self.df['month'] = self.df['timestamp'].dt.month
        self.df['day_of_year'] = self.df['timestamp'].dt.dayofyear
        
        print("\n HOURLY PATTERNS")
        hourly_stats = self.df.groupby('hour').agg({
            'power_output': ['mean', 'max', 'std'],
            'solar_irradiance': ['mean', 'max'],
            'efficiency': ['mean', 'std']
        }).round(2)
        
        print("Power Output by Hour:")
        print(hourly_stats['power_output'])
        
        print("\nDAILY PATTERNS")
        daily_stats = self.df.groupby('day_of_week').agg({
            'power_output': 'mean',
            'efficiency': 'mean',
            'dust_level': 'mean'
        }).round(3)
        
        daily_stats.index = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        print(daily_stats)
        
        if self.df['timestamp'].dt.date.nunique() > 30:
            print("\n SEASONAL PATTERNS")
            monthly_stats = self.df.groupby('month').agg({
                'power_output': 'mean',
                'temperature': 'mean',
                'solar_irradiance': 'mean'
            }).round(2)
            print(monthly_stats)
        
        return hourly_stats, daily_stats
    
    def correlation_analysis(self):
        """Advanced correlation analysis"""
        print("\n CORRELATION ANALYSIS")
        print("=" * 50)
        
        corr_matrix = self.df[self.numeric_cols].corr()
        
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_pairs.append({
                    'Variable 1': corr_matrix.columns[i],
                    'Variable 2': corr_matrix.columns[j],
                    'Correlation': corr_matrix.iloc[i, j],
                    'Abs_Correlation': abs(corr_matrix.iloc[i, j])
                })
        
        corr_df = pd.DataFrame(corr_pairs)
        corr_df = corr_df.sort_values('Abs_Correlation', ascending=False)
        
        print(" STRONGEST CORRELATIONS:")
        print(corr_df.head(10)[['Variable 1', 'Variable 2', 'Correlation']])
        
        performance_vars = ['power_output', 'efficiency']
        environmental_vars = ['solar_irradiance', 'temperature', 'humidity', 'wind_speed']
        
        print("\n ENVIRONMENTAL vs PERFORMANCE CORRELATIONS:")
        for perf_var in performance_vars:
            print(f"\n{perf_var.upper()}:")
            for env_var in environmental_vars:
                corr = self.df[perf_var].corr(self.df[env_var])
                print(f"  {env_var}: {corr:.3f}")
        
        return corr_matrix, corr_df
    
    def distribution_analysis(self):
        """Analyze distributions of key variables"""
        print("\n DISTRIBUTION ANALYSIS")
        print("=" * 50)
        
        key_vars = ['power_output', 'efficiency', 'solar_irradiance', 'temperature']
        
        distribution_stats = {}
        
        for var in key_vars:
            data = self.df[var].dropna()
            
            stats_dict = {
                'mean': data.mean(),
                'median': data.median(),
                'std': data.std(),
                'skewness': stats.skew(data),
                'kurtosis': stats.kurtosis(data),
                'min': data.min(),
                'max': data.max(),
                'range': data.max() - data.min()
            }
            
            _, p_value = stats.normaltest(data)
            stats_dict['normal_p_value'] = p_value
            stats_dict['is_normal'] = p_value > 0.05
            
            distribution_stats[var] = stats_dict
            
            print(f"\n {var.upper()}:")
            print(f"  Mean: {stats_dict['mean']:.3f}")
            print(f"  Median: {stats_dict['median']:.3f}")
            print(f"  Std: {stats_dict['std']:.3f}")
            print(f"  Skewness: {stats_dict['skewness']:.3f}")
            print(f"  Kurtosis: {stats_dict['kurtosis']:.3f}")
            print(f"  Normal distribution: {'Yes' if stats_dict['is_normal'] else 'No'} (p={stats_dict['normal_p_value']:.3f})")
        
        return distribution_stats
    
    def outlier_analysis(self):
        """Comprehensive outlier detection"""
        print("\n OUTLIER ANALYSIS")
        print("=" * 50)
        
        outlier_results = {}
        
        for col in ['power_output', 'efficiency', 'solar_irradiance']:
            data = self.df[col].dropna()
            
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            iqr_outliers = data[(data < lower_bound) | (data > upper_bound)]
            
            z_scores = np.abs(stats.zscore(data))
            z_outliers = data[z_scores > 3]
            
            median = data.median()
            mad = np.median(np.abs(data - median))
            modified_z_scores = 0.6745 * (data - median) / mad
            modified_z_outliers = data[np.abs(modified_z_scores) > 3.5]
            
            outlier_results[col] = {
                'iqr_outliers': len(iqr_outliers),
                'z_outliers': len(z_outliers),
                'modified_z_outliers': len(modified_z_outliers),
                'iqr_percentage': (len(iqr_outliers) / len(data)) * 100,
                'z_percentage': (len(z_outliers) / len(data)) * 100,
                'modified_z_percentage': (len(modified_z_outliers) / len(data)) * 100
            }
            
            print(f"\n {col.upper()}:")
            print(f"  IQR outliers: {len(iqr_outliers)} ({(len(iqr_outliers)/len(data)*100):.1f}%)")
            print(f"  Z-score outliers: {len(z_outliers)} ({(len(z_outliers)/len(data)*100):.1f}%)")
            print(f"  Modified Z-score outliers: {len(modified_z_outliers)} ({(len(modified_z_outliers)/len(data)*100):.1f}%)")
        
        return outlier_results
    
    def performance_analysis(self):
        """Analyze system performance patterns"""
        print("\n⚡ PERFORMANCE ANALYSIS")
        print("=" * 50)
        
        self.df['power_category'] = pd.cut(self.df['power_output'], 
                                          bins=[0, 50, 100, 200, float('inf')],
                                          labels=['Low', 'Medium', 'High', 'Peak'])
        
        self.df['efficiency_category'] = pd.cut(self.df['efficiency'],
                                               bins=[0, 0.6, 0.8, 1.0, float('inf')],
                                               labels=['Poor', 'Fair', 'Good', 'Excellent'])
        
        print(" POWER OUTPUT DISTRIBUTION:")
        power_dist = self.df['power_category'].value_counts()
        for category, count in power_dist.items():
            percentage = (count / len(self.df)) * 100
            print(f"  {category}: {count} ({percentage:.1f}%)")
        
        print("\n EFFICIENCY DISTRIBUTION:")
        eff_dist = self.df['efficiency_category'].value_counts()
        for category, count in eff_dist.items():
            percentage = (count / len(self.df)) * 100
            print(f"  {category}: {count} ({percentage:.1f}%)")
        
        print("\n PERFORMANCE BY CONDITIONS:")
        
        high_irradiance = self.df[self.df['solar_irradiance'] > self.df['solar_irradiance'].median()]
        low_irradiance = self.df[self.df['solar_irradiance'] <= self.df['solar_irradiance'].median()]
        
        print(f"High irradiance avg power: {high_irradiance['power_output'].mean():.1f}W")
        print(f"Low irradiance avg power: {low_irradiance['power_output'].mean():.1f}W")
        
        clean_panels = self.df[self.df['dust_level'] < self.df['dust_level'].median()]
        dusty_panels = self.df[self.df['dust_level'] >= self.df['dust_level'].median()]
        
        print(f"Clean panels avg efficiency: {clean_panels['efficiency'].mean():.3f}")
        print(f"Dusty panels avg efficiency: {dusty_panels['efficiency'].mean():.3f}")
        
        return power_dist, eff_dist
    
    def maintenance_analysis(self):
        """Analyze maintenance patterns and needs"""
        print("\n MAINTENANCE ANALYSIS")
        print("=" * 50)
        
        urgent = self.df[self.df['days_until_maintenance'] < 7]
        soon = self.df[(self.df['days_until_maintenance'] >= 7) & 
                      (self.df['days_until_maintenance'] < 30)]
        later = self.df[self.df['days_until_maintenance'] >= 30]
        
        print(" MAINTENANCE URGENCY:")
        print(f"  Urgent (< 7 days): {len(urgent)} ({len(urgent)/len(self.df)*100:.1f}%)")
        print(f"  Soon (7-30 days): {len(soon)} ({len(soon)/len(self.df)*100:.1f}%)")
        print(f"  Later (> 30 days): {len(later)} ({len(later)/len(self.df)*100:.1f}%)")
        
        needs_cleaning = self.df[self.df['hours_since_cleaning'] > 72]  
        print(f"\n CLEANING NEEDED: {len(needs_cleaning)} systems ({len(needs_cleaning)/len(self.df)*100:.1f}%)")
        
        high_dust = self.df[self.df['dust_level'] > self.df['dust_level'].quantile(0.75)]
        low_dust = self.df[self.df['dust_level'] < self.df['dust_level'].quantile(0.25)]
        
        dust_impact = high_dust['efficiency'].mean() - low_dust['efficiency'].mean()
        print(f" DUST IMPACT: {dust_impact:.3f} efficiency reduction")
        
        maint_corr = self.df['days_since_maintenance'].corr(self.df['efficiency'])
        print(f" MAINTENANCE-PERFORMANCE CORRELATION: {maint_corr:.3f}")
        
        return urgent, soon, later
    
    def anomaly_detection(self):
        """Detect anomalies in the data"""
        print("\n ANOMALY DETECTION")
        print("=" * 50)
        
        features = ['solar_irradiance', 'temperature', 'humidity', 'power_output', 
                   'efficiency', 'dust_level']
        
        X = self.df[features].fillna(self.df[features].mean())
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        anomalies = iso_forest.fit_predict(X_scaled)
        
        anomaly_indices = np.where(anomalies == -1)[0]
        anomaly_data = self.df.iloc[anomaly_indices]
        
        print(f" ANOMALIES DETECTED: {len(anomaly_indices)} ({len(anomaly_indices)/len(self.df)*100:.1f}%)")
        
        if len(anomaly_indices) > 0:
            print("\n ANOMALY CHARACTERISTICS:")
            print("Average values for anomalies vs normal:")
            
            normal_data = self.df.iloc[np.where(anomalies == 1)[0]]
            
            for feature in features:
                anomaly_avg = anomaly_data[feature].mean()
                normal_avg = normal_data[feature].mean()
                print(f"  {feature}: {anomaly_avg:.2f} vs {normal_avg:.2f}")
        
        return anomaly_indices, anomaly_data
    
    def clustering_analysis(self):
        """Perform clustering analysis to identify operational patterns"""
        print("\n CLUSTERING ANALYSIS")
        print("=" * 50)
        
        features = ['solar_irradiance', 'temperature', 'power_output', 'efficiency']
        X = self.df[features].fillna(self.df[features].mean())
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        inertias = []
        k_range = range(2, 8)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            inertias.append(kmeans.inertia_)
        
        optimal_k = 4
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        self.df['cluster'] = clusters
        
        print(f" IDENTIFIED {optimal_k} OPERATIONAL CLUSTERS:")
        
        for i in range(optimal_k):
            cluster_data = self.df[self.df['cluster'] == i]
            cluster_size = len(cluster_data)
            cluster_pct = (cluster_size / len(self.df)) * 100
            
            print(f"\n CLUSTER {i} ({cluster_size} samples, {cluster_pct:.1f}%):")
            print(f"  Avg Power: {cluster_data['power_output'].mean():.1f}W")
            print(f"  Avg Efficiency: {cluster_data['efficiency'].mean():.3f}")
            print(f"  Avg Irradiance: {cluster_data['solar_irradiance'].mean():.1f}W/m²")
            print(f"  Avg Temperature: {cluster_data['temperature'].mean():.1f}°C")
        
        return clusters, kmeans
    
    def generate_insights(self):
        """Generate actionable insights from the analysis"""
        print("\n KEY INSIGHTS & RECOMMENDATIONS")
        print("=" * 50)
        
        insights = []
        
        avg_efficiency = self.df['efficiency'].mean()
        if avg_efficiency < 0.8:
            insights.append(" System efficiency is below optimal (< 0.8). Consider maintenance.")
        
        high_dust_eff = self.df[self.df['dust_level'] > self.df['dust_level'].quantile(0.75)]['efficiency'].mean()
        low_dust_eff = self.df[self.df['dust_level'] < self.df['dust_level'].quantile(0.25)]['efficiency'].mean()
        
        if (low_dust_eff - high_dust_eff) > 0.1:
            insights.append("Dust significantly impacts efficiency. Increase cleaning frequency.")
        
        temp_power_corr = self.df['temperature'].corr(self.df['power_output'])
        if temp_power_corr < -0.3:
            insights.append(" High temperatures negatively impact power output. Consider cooling solutions.")
        
        urgent_maintenance = len(self.df[self.df['days_until_maintenance'] < 7])
        if urgent_maintenance > len(self.df) * 0.1:
            insights.append(" High number of systems need urgent maintenance. Review maintenance schedule.")
        
        self.df['hour'] = self.df['timestamp'].dt.hour
        peak_hours = self.df.groupby('hour')['power_output'].mean().idxmax()
        insights.append(f" Peak performance typically occurs around {peak_hours}:00.")
        
        irradiance_power_corr = self.df['solar_irradiance'].corr(self.df['power_output'])
        if irradiance_power_corr > 0.8:
            insights.append(" System performance is highly dependent on solar irradiance.")
        
        for i, insight in enumerate(insights, 1):
            print(f"{i}. {insight}")
        
        return insights
    
    def run_complete_analysis(self):
        """Run the complete EDA analysis"""
        print(" COMPREHENSIVE SOLAR PANEL EDA")
        print("=" * 60)
        
        basic_stats = self.basic_info()
        
        hourly_stats, daily_stats = self.temporal_analysis()
        
        corr_matrix, corr_df = self.correlation_analysis()
        
        dist_stats = self.distribution_analysis()
        
        outlier_results = self.outlier_analysis()
        
        power_dist, eff_dist = self.performance_analysis()
        
        urgent, soon, later = self.maintenance_analysis()
        
        anomaly_indices, anomaly_data = self.anomaly_detection()
        
        clusters, kmeans_model = self.clustering_analysis()
        
        insights = self.generate_insights()
        
        print("\n ANALYSIS COMPLETE!")
        print(" Check the results above for detailed insights.")
        
        return {
            'basic_stats': basic_stats,
            'correlations': corr_matrix,
            'distributions': dist_stats,
            'outliers': outlier_results,
            'anomalies': anomaly_indices,
            'clusters': clusters,
            'insights': insights
        }

def main():
    """Main function to run the EDA"""
    print(" Starting Comprehensive Solar Panel EDA...")
    
    eda = SolarEDA()
    
    results = eda.run_complete_analysis()
    
    summary = {
        'total_records': len(eda.df),
        'date_range': f"{eda.df['timestamp'].min()} to {eda.df['timestamp'].max()}",
        'avg_power_output': eda.df['power_output'].mean(),
        'avg_efficiency': eda.df['efficiency'].mean(),
        'anomalies_detected': len(results['anomalies']),
        'insights_count': len(results['insights'])
    }
    
    print(f"\n ANALYSIS SUMMARY:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    main()