
import os
import sys
import subprocess
import logging
from datetime import datetime
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('eda_execution.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EDARunner:
    def __init__(self):
        self.start_time = datetime.now()
        self.results = {}

        self.output_dirs = [
            'analysis',
            'analysis/interactive',
            'analysis/reports',
            'logs'
        ]

        for directory in self.output_dirs:
            os.makedirs(directory, exist_ok=True)

    def run_script(self, script_path, description):
        try:
            logger.info(f"Starting: {description}")
            start_time = time.time()

            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=1800
            )

            execution_time = time.time() - start_time

            if result.returncode == 0:
                logger.info(f"[SUCCESS] {description} completed successfully in {execution_time:.1f}s")
                self.results[description] = {
                    'status': 'success',
                    'execution_time': execution_time,
                    'output': result.stdout
                }
                return True
            else:
                logger.error(f"[ERROR] {description} failed with return code {result.returncode}")
                logger.error(f"Error output: {result.stderr}")
                self.results[description] = {
                    'status': 'failed',
                    'execution_time': execution_time,
                    'error': result.stderr
                }
                return False

        except subprocess.TimeoutExpired:
            logger.error(f"[ERROR] {description} timed out after 30 minutes")
            self.results[description] = {
                'status': 'timeout',
                'execution_time': 1800,
                'error': 'Script execution timed out'
            }
            return False

        except Exception as e:
            logger.error(f"[ERROR] {description} failed with exception: {e}")
            self.results[description] = {
                'status': 'error',
                'execution_time': 0,
                'error': str(e)
            }
            return False

    def run_complete_eda(self):
        try:
            logger.info("Starting Complete EDA Pipeline")
            logger.info("=" * 60)

            eda_modules = [
                {
                    'script': 'analysis/01_data_profiling.py',
                    'description': 'Data Profiling & Quality Assessment',
                    'priority': 'high'
                },
                {
                    'script': 'analysis/02_visualizations.py',
                    'description': 'Statistical Visualizations',
                    'priority': 'high'
                },
                {
                    'script': 'analysis/03_correlation_analysis.py',
                    'description': 'Correlation & Feature Analysis',
                    'priority': 'high'
                },
                {
                    'script': 'analysis/04_advanced_eda.py',
                    'description': 'Advanced Statistical Analysis',
                    'priority': 'medium'
                },
                {
                    'script': 'analysis/05_interactive_eda.py',
                    'description': 'Interactive Dashboard Generation',
                    'priority': 'medium'
                },
                {
                    'script': 'analysis/06_comprehensive_report.py',
                    'description': 'Comprehensive Report Generation',
                    'priority': 'high'
                }
            ]

            high_priority_modules = [m for m in eda_modules if m['priority'] == 'high']
            medium_priority_modules = [m for m in eda_modules if m['priority'] == 'medium']

            logger.info("Phase 1: Running High Priority Modules")
            logger.info("-" * 40)

            for module in high_priority_modules:
                if os.path.exists(module['script']):
                    self.run_script(module['script'], module['description'])
                else:
                    logger.warning(f"Script not found: {module['script']}")
                    self.results[module['description']] = {
                        'status': 'not_found',
                        'execution_time': 0,
                        'error': 'Script file not found'
                    }

            logger.info("\nPhase 2: Running Medium Priority Modules")
            logger.info("-" * 40)

            for module in medium_priority_modules:
                if os.path.exists(module['script']):
                    self.run_script(module['script'], module['description'])
                else:
                    logger.warning(f"Script not found: {module['script']}")
                    self.results[module['description']] = {
                        'status': 'not_found',
                        'execution_time': 0,
                        'error': 'Script file not found'
                    }

            self.generate_execution_summary()

            return True

        except Exception as e:
            logger.error(f"Error in complete EDA pipeline: {e}")
            return False

    def generate_execution_summary(self):
        try:
            logger.info("Generating execution summary...")

            total_time = (datetime.now() - self.start_time).total_seconds()
            successful_modules = sum(1 for r in self.results.values() if r['status'] == 'success')
            total_modules = len(self.results)

            for module_name, result in self.results.items():
                status = result['status']
                exec_time = result['execution_time']

                status_class = f"status-{status}"
                status_text = {
                    'success': '[SUCCESS] Success',
                    'failed': '[ERROR] Failed',
                    'timeout': '[TIMEOUT] Timeout',
                    'not_found': '[NOT FOUND] Not Found',
                    'error': '[ERROR] Error'
                }.get(status, status)

                notes = ""
                if status == 'success':
                    notes = "Completed successfully"
                elif status == 'failed':
                    notes = "Check logs for details"
                elif status == 'timeout':
                    notes = "Execution exceeded 30 minutes"
                elif status == 'not_found':
                    notes = "Script file not found"

            generated_files = []

            file_patterns = [
                ('analysis/data_profiling_report.html', 'Data Profiling Report'),
                ('analysis/distribution_plots.png', 'Distribution Visualizations'),
                ('analysis/correlation_heatmaps.png', 'Correlation Analysis'),
                ('analysis/interactive/index.html', 'Interactive Dashboards'),
                ('analysis/reports/', 'Comprehensive Reports'),
                ('analysis/seasonal_decomposition.png', 'Time Series Analysis'),
                ('analysis/clustering_results.png', 'Clustering Analysis'),
                ('analysis/anomaly_detection.png', 'Anomaly Detection')
            ]

            for file_path, description in file_patterns:
                if os.path.exists(file_path):
                    generated_files.append((file_path, description))
                elif os.path.isdir(file_path):
                    try:
                        files_in_dir = os.listdir(file_path)
                        if files_in_dir:
                            generated_files.append((file_path, f"{description} ({len(files_in_dir)} files)"))
                    except:
                        pass

            for file_path, description in generated_files:

            summary_file = f"analysis/eda_execution_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(summary_html)

            logger.info(f"Execution summary saved: {summary_file}")

            print("\n" + "=" * 60)
            print("EDA PIPELINE EXECUTION SUMMARY")
            print("=" * 60)
            print(f"Total execution time: {total_time/60:.1f} minutes")
            print(f"Modules completed: {successful_modules}/{total_modules}")
            print(f"Success rate: {(successful_modules/total_modules*100):.0f}%")
            print(f"Summary report: {summary_file}")

            if successful_modules == total_modules:
                print("\n[SUCCESS] All modules completed successfully!")
            elif successful_modules > 0:
                print(f"\n[WARNING] {total_modules - successful_modules} module(s) failed. Check logs for details.")
            else:
                print("\n[ERROR] All modules failed. Check system requirements and data availability.")

            return summary_file

        except Exception as e:
            logger.error(f"Error generating execution summary: {e}")
            return None

def main():
    print("Solar Panel System - Complete EDA Pipeline")
    print("=" * 60)
    print("This will run all EDA modules and generate comprehensive analysis outputs.")
    print("Estimated execution time: 10-30 minutes depending on system performance.")
    print()

    response = input("Do you want to proceed? (y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print("EDA pipeline cancelled.")
        return

    runner = EDARunner()
    success = runner.run_complete_eda()

    if success:
        print("\n[SUCCESS] EDA pipeline completed!")
        print("Check the generated summary report for detailed results.")
    else:
        print("\n[ERROR] EDA pipeline encountered errors.")
        print("Check the execution log for details: eda_execution.log")

if __name__ == "__main__":
    main()
