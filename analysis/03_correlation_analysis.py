import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr
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

def calculate_correlation_matrix(df, method='pearson'):
    """Calculate correlation matrix using specified method."""
    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if method == 'pearson':
            corr_matrix = df[numeric_cols].corr(method='pearson')
        elif method == 'spearman':
            corr_matrix = df[numeric_cols].corr(method='spearman')
        else:
            raise ValueError("Method must be 'pearson' or 'spearman'")

        return corr_matrix

    except Exception as e:
        logger.error(f"Error calculating correlation matrix: {e}")
        return None

def find_high_correlations(corr_matrix, threshold=0.7):
    """Find pairs of features with high correlation."""
    try:
        high_corr_pairs = []

        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) >= threshold:
                    high_corr_pairs.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': corr_val,
                        'abs_correlation': abs(corr_val)
                    })

        high_corr_pairs.sort(key=lambda x: x['abs_correlation'], reverse=True)

        return high_corr_pairs

    except Exception as e:
        logger.error(f"Error finding high correlations: {e}")
        return []

def create_correlation_heatmaps(df):
    """Create comprehensive correlation heatmaps."""
    try:
        logger.info("Creating correlation heatmaps...")

        pearson_corr = calculate_correlation_matrix(df, 'pearson')
        spearman_corr = calculate_correlation_matrix(df, 'spearman')

        if pearson_corr is None or spearman_corr is None:
            return

        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        fig.suptitle('Correlation Analysis: Pearson vs Spearman', fontsize=16, y=0.98)

        mask = np.triu(np.ones_like(pearson_corr, dtype=bool))
        sns.heatmap(pearson_corr,
                   mask=mask,
                   annot=True,
                   cmap='RdBu_r',
                   center=0,
                   square=True,
                   fmt='.2f',
                   cbar_kws={"shrink": .8},
                   ax=axes[0])
        axes[0].set_title('Pearson Correlation Matrix')

        mask = np.triu(np.ones_like(spearman_corr, dtype=bool))
        sns.heatmap(spearman_corr,
                   mask=mask,
                   annot=True,
                   cmap='RdBu_r',
                   center=0,
                   square=True,
                   fmt='.2f',
                   cbar_kws={"shrink": .8},
                   ax=axes[1])
        axes[1].set_title('Spearman Correlation Matrix')

        plt.tight_layout()
        plt.savefig('analysis/correlation_heatmaps.png', dpi=300, bbox_inches='tight')
        plt.show()

        logger.info("Correlation heatmaps created successfully")

        return pearson_corr, spearman_corr

    except Exception as e:
        logger.error(f"Error creating correlation heatmaps: {e}")
        return None, None

def analyze_target_correlations(df):
    """Analyze correlations with target variables."""
    try:
        logger.info("Analyzing target correlations...")

        target_vars = ['energy_output', 'maintenance_needed']
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col not in target_vars]

        results = {}

        for target in target_vars:
            if target in df.columns:
                correlations = []

                for feature in feature_cols:
                    pearson_corr, pearson_p = pearsonr(df[feature], df[target])

                    spearman_corr, spearman_p = spearmanr(df[feature], df[target])

                    correlations.append({
                        'feature': feature,
                        'pearson_corr': pearson_corr,
                        'pearson_p_value': pearson_p,
                        'spearman_corr': spearman_corr,
                        'spearman_p_value': spearman_p,
                        'abs_pearson': abs(pearson_corr),
                        'abs_spearman': abs(spearman_corr)
                    })

                correlations.sort(key=lambda x: x['abs_pearson'], reverse=True)
                results[target] = correlations

        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        fig.suptitle('Feature Correlations with Target Variables', fontsize=16, y=0.98)

        for i, (target, corr_data) in enumerate(results.items()):
            if i < 2: 
                features = [item['feature'] for item in corr_data[:15]]  
                pearson_values = [item['pearson_corr'] for item in corr_data[:15]]
                spearman_values = [item['spearman_corr'] for item in corr_data[:15]]

                x = np.arange(len(features))
                width = 0.35

                axes[i].bar(x - width/2, pearson_values, width,
                           label='Pearson', alpha=0.8, color='blue')
                axes[i].bar(x + width/2, spearman_values, width,
                           label='Spearman', alpha=0.8, color='red')

                axes[i].set_title(f'Correlations with {target}')
                axes[i].set_xlabel('Features')
                axes[i].set_ylabel('Correlation Coefficient')
                axes[i].set_xticks(x)
                axes[i].set_xticklabels(features, rotation=45, ha='right')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
                axes[i].axhline(y=0, color='white', linestyle='-', alpha=0.5)

        plt.tight_layout()
        plt.savefig('analysis/target_correlations.png', dpi=300, bbox_inches='tight')
        plt.show()

        logger.info("Target correlation analysis completed")

        return results

    except Exception as e:
        logger.error(f"Error analyzing target correlations: {e}")
        return {}

def create_scatter_matrix(df, features=None):
    """Create scatter plot matrix for key features."""
    try:
        logger.info("Creating scatter plot matrix...")

        if features is None:
            
            features = ['energy_output', 'solar_irradiance', 'panel_temp',
                       'temperature', 'dust_level', 'voltage', 'current']

        available_features = [f for f in features if f in df.columns]

        if len(available_features) < 2:
            logger.warning("Not enough features available for scatter matrix")
            return

        fig, axes = plt.subplots(len(available_features), len(available_features),
                                figsize=(15, 15))
        fig.suptitle('Feature Scatter Plot Matrix', fontsize=16, y=0.98)

        for i, feature1 in enumerate(available_features):
            for j, feature2 in enumerate(available_features):
                if i == j:
                    axes[i, j].hist(df[feature1], bins=30, alpha=0.7, color='skyblue')
                    axes[i, j].set_title(f'{feature1}')
                else:
                    axes[i, j].scatter(df[feature2], df[feature1],
                                     alpha=0.5, s=10, color='gold')

                    corr_coef = df[feature1].corr(df[feature2])
                    axes[i, j].text(0.05, 0.95, f'r={corr_coef:.2f}',
                                   transform=axes[i, j].transAxes,
                                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

                axes[i, j].grid(True, alpha=0.3)

                if i == len(available_features) - 1:
                    axes[i, j].set_xlabel(feature2)
                if j == 0:
                    axes[i, j].set_ylabel(feature1)

        plt.tight_layout()
        plt.savefig('analysis/scatter_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()

        logger.info("Scatter plot matrix created successfully")

    except Exception as e:
        logger.error(f"Error creating scatter matrix: {e}")

def analyze_feature_relationships(df):
    """Analyze specific feature relationships and interactions."""
    try:
        logger.info("Analyzing feature relationships...")

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Key Feature Relationships', fontsize=16, y=0.98)

        scatter = axes[0, 0].scatter(df['solar_irradiance'], df['energy_output'],
                                   c=df['temperature'], cmap='coolwarm',
                                   alpha=0.6, s=20)
        axes[0, 0].set_xlabel('Solar Irradiance (W/m²)')
        axes[0, 0].set_ylabel('Energy Output (kW)')
        axes[0, 0].set_title('Energy vs Irradiance (colored by Temperature)')
        plt.colorbar(scatter, ax=axes[0, 0], label='Temperature (°C)')

        axes[0, 1].scatter(df['temperature'], df['panel_temp'],
                          alpha=0.6, s=20, color='red')
        axes[0, 1].set_xlabel('Ambient Temperature (°C)')
        axes[0, 1].set_ylabel('Panel Temperature (°C)')
        axes[0, 1].set_title('Panel vs Ambient Temperature')

        z = np.polyfit(df['temperature'], df['panel_temp'], 1)
        p = np.poly1d(z)
        axes[0, 1].plot(df['temperature'], p(df['temperature']), "r--", alpha=0.8)

        if 'solar_irradiance' in df.columns:
            df['efficiency'] = np.where(df['solar_irradiance'] > 0,
                                       df['energy_output'] / (df['solar_irradiance'] / 1000),
                                       0)

            axes[0, 2].scatter(df['dust_level'], df['efficiency'],
                              alpha=0.6, s=20, color='brown')
            axes[0, 2].set_xlabel('Dust Level')
            axes[0, 2].set_ylabel('Efficiency')
            axes[0, 2].set_title('Efficiency vs Dust Level')

        axes[1, 0].scatter(df['voltage'], df['current'],
                          alpha=0.6, s=20, color='green')
        axes[1, 0].set_xlabel('Voltage (V)')
        axes[1, 0].set_ylabel('Current (A)')
        axes[1, 0].set_title('Voltage vs Current')

        axes[1, 1].scatter(df['wind_speed'], df['panel_temp'],
                          alpha=0.6, s=20, color='blue')
        axes[1, 1].set_xlabel('Wind Speed (m/s)')
        axes[1, 1].set_ylabel('Panel Temperature (°C)')
        axes[1, 1].set_title('Wind Speed vs Panel Temperature')

        axes[1, 2].scatter(df['humidity'], df['energy_output'],
                          alpha=0.6, s=20, color='purple')
        axes[1, 2].set_xlabel('Humidity (%)')
        axes[1, 2].set_ylabel('Energy Output (kW)')
        axes[1, 2].set_title('Humidity vs Energy Output')

        for ax in axes.flat:
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('analysis/feature_relationships.png', dpi=300, bbox_inches='tight')
        plt.show()

        logger.info("Feature relationship analysis completed")

    except Exception as e:
        logger.error(f"Error analyzing feature relationships: {e}")

def statistical_significance_test(df):
    """Perform statistical significance tests for correlations."""
    try:
        logger.info("Performing statistical significance tests...")

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        significant_pairs = []

        for i, col1 in enumerate(numeric_cols):
            for j, col2 in enumerate(numeric_cols):
                if i < j:  
                    corr_coef, p_value = pearsonr(df[col1], df[col2])

                    if p_value < 0.05:  
                        significant_pairs.append({
                            'feature1': col1,
                            'feature2': col2,
                            'correlation': corr_coef,
                            'p_value': p_value,
                            'significance_level': '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*'
                        })

        significant_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)

        print("\nStatistically Significant Correlations (p < 0.05):")
        print("=" * 80)
        print(f"{'Feature 1':<20} {'Feature 2':<20} {'Correlation':<12} {'P-value':<12} {'Significance'}")
        print("-" * 80)

        for pair in significant_pairs[:20]:  
            print(f"{pair['feature1']:<20} {pair['feature2']:<20} "
                  f"{pair['correlation']:<12.3f} {pair['p_value']:<12.6f} {pair['significance_level']}")

        return significant_pairs

    except Exception as e:
        logger.error(f"Error in statistical significance test: {e}")
        return []

def main():
    """Main function to run all correlation analyses."""
    print("Solar Panel Correlation Analysis")
    print("=" * 50)

    try:
        os.makedirs('analysis', exist_ok=True)

        ingestion = DataIngestion()
        df = ingestion.load_historical_data()

        if df is None or df.empty:
            logger.warning("No data available, generating sample data")
            df = ingestion._generate_sample_data(days=365)

        engineer = FeatureEngineer()
        df = engineer.create_derived_features(df)

        logger.info(f"Loaded data with shape: {df.shape}")

        pearson_corr, spearman_corr = create_correlation_heatmaps(df)

        target_correlations = analyze_target_correlations(df)

        if pearson_corr is not None:
            high_corr_pairs = find_high_correlations(pearson_corr, threshold=0.7)

            print("\nHigh Correlations (|r| >= 0.7):")
            print("=" * 50)
            for pair in high_corr_pairs:
                print(f"{pair['feature1']} - {pair['feature2']}: {pair['correlation']:.3f}")

        create_scatter_matrix(df)

        analyze_feature_relationships(df)

        significant_pairs = statistical_significance_test(df)

        print(f"\nCorrelation Analysis Summary:")
        print("=" * 50)
        print(f"Total features analyzed: {len(df.select_dtypes(include=[np.number]).columns)}")
        print(f"High correlations found: {len(high_corr_pairs) if 'high_corr_pairs' in locals() else 0}")
        print(f"Statistically significant pairs: {len(significant_pairs)}")

        print("\n" + "=" * 50)
        print("Correlation analysis completed successfully!")
        print("Check 'analysis/' directory for generated plots")

    except Exception as e:
        logger.error(f"Error in correlation analysis: {e}")
        print("Analysis failed. Check logs for details.")

if __name__ == "__main__":
    main()
