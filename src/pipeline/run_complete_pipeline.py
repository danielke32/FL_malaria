"""
======================================================================
🇬🇭 GHANA MALARIA FEDERATED LEARNING - COMPLETE PIPELINE
======================================================================
MSc Thesis: Performance Evaluation of Federated Learning Algorithms
            for Privacy-Preserving Malaria Symptom Assessment

This pipeline runs the complete workflow:
  Stage 1: Data Extraction     (src/data/data_extraction.py)
  Stage 2: Data Preprocessing  (src/data/data_preprocessing.py)
  Stage 3: FL Scenario Creation (src/data/create_fl_scenarios.py)
  Stage 4: Centralized Training (src/training/train_centralized.py)
  Stage 5: Federated Training   (src/training/train_federated.py)
  Stage 6: Final Analysis       (src/evaluation/final_analysis.py)

Features:
  - Comprehensive logging to logs/ directory
  - Timing for each stage
  - Progress tracking
  - Error handling with detailed messages
  - Summary report generation
  - Configurable experiment types

Location: src/pipeline/run_complete_pipeline.py

Usage (from project root):
  python src/pipeline/run_complete_pipeline.py                    # Run full pipeline (baseline)
  python src/pipeline/run_complete_pipeline.py --experiment baseline
  python src/pipeline/run_complete_pipeline.py --experiment grid_search
  python src/pipeline/run_complete_pipeline.py --experiment ablation
  python src/pipeline/run_complete_pipeline.py --skip-extraction  # Skip data extraction if already done
  python src/pipeline/run_complete_pipeline.py --stages 1,2,3     # Run specific stages only

Author: Daniel Kwasi Kovor
Date: December 2025
======================================================================
"""

import os
import sys
import subprocess
import time
import logging
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import traceback

# Ensure proper UTF-8 handling on Windows
if sys.platform == 'win32':
    # Set console to UTF-8 mode
    try:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, errors='replace')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, errors='replace')
    except Exception:
        pass
    
    # Also set environment variable for child processes
    os.environ['PYTHONIOENCODING'] = 'utf-8'


# --- Configuration ---
class PipelineConfig:
    """Pipeline configuration."""
    
    # Get directory
    PIPELINE_DIR = Path(__file__).parent.resolve()
    SRC_DIR = PIPELINE_DIR.parent
    PROJECT_ROOT = SRC_DIR.parent
    
    # Directories for logs, results, data
    LOGS_DIR = PROJECT_ROOT / "logs"
    RESULTS_DIR = PROJECT_ROOT / "results"
    DATA_DIR = PROJECT_ROOT / "data"
    
    # Work flow script files 
    SCRIPTS = {
        1: SRC_DIR / "data" / "data_extraction.py",
        2: SRC_DIR / "data" / "data_preprocessing.py",
        3: SRC_DIR / "data" / "create_fl_scenarios.py",
        4: SRC_DIR / "training" / "train_centralized.py",
        5: SRC_DIR / "training" / "train_federated.py",
        6: SRC_DIR / "evaluation" / "final_analysis.py"
    }
    
    # Stage names
    STAGE_NAMES = {
        1: "Data Extraction",
        2: "Data Preprocessing",
        3: "FL Scenario Creation",
        4: "Centralized Training",
        5: "Federated Training",
        6: "Final Analysis"
    }
    
    # Stage descriptions
    STAGE_DESCRIPTIONS = {
        1: "Extract and merge Ghana DHS/MIS malaria survey data",
        2: "Clean, impute, and prepare data with MICE and symptom simulation",
        3: "Create IID, Non-IID, and Data Quality scenarios for FL",
        4: "Train centralized baseline models (LR, RF) with GridSearchCV",
        5: "Train federated models (FedAvg, FedProx) across scenarios",
        6: "Generate publication-ready figures, tables, and statistical analysis"
    }
    
    # Outputs for the various stage (for validation, relative to project root)
    EXPECTED_OUTPUTS = {
        1: ["data/merged/ghana_malaria_merged.csv"],
        2: ["data/cleaned/train_raw.csv", "data/cleaned/train_centralized.csv", 
            "data/cleaned/val_set.csv", "data/cleaned/test_set.csv"],
        3: ["data/fl_scenarios/s1_iid/client_0.csv", 
            "data/fl_scenarios/s2_noniid/client_0.csv",
            "data/fl_scenarios/s3_quality/client_0.csv",
            "data/fl_scenarios/heterogeneity_metrics.json"],
        4: ["results/centralized_results.json"],
        5: ["results/results_baseline.json"],
        6: ["results/comprehensive_statistical_analysis.json",
            "results/figures", "results/tables"]
    }
    
    # Timeout per stage (seconds)
    STAGE_TIMEOUTS = {
        1: 600,    # 10 minutes
        2: 600,    # 10 minutes
        3: 300,    # 5 minutes
        4: 1800,   # 30 minutes
        5: 7200,   # 2 hours (FL training can be slow)
        6: 1200    # 20 minutes
    }
    
    @classmethod
    def create_directories(cls):
        """Create required directories."""
        dirs = [
            cls.LOGS_DIR, 
            cls.RESULTS_DIR, 
            cls.DATA_DIR, 
            cls.DATA_DIR / "raw", 
            cls.DATA_DIR / "merged", 
            cls.DATA_DIR / "cleaned", 
            cls.DATA_DIR / "fl_scenarios",
            cls.RESULTS_DIR / "figures", 
            cls.RESULTS_DIR / "tables"
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_script_display_path(cls, stage: int) -> str:
        """Get display-friendly relative path for a script."""
        script = cls.SCRIPTS.get(stage)
        if script:
            try:
                return str(script.relative_to(cls.PROJECT_ROOT))
            except ValueError:
                return str(script)
        return "unknown"


# Logging Setup
class PipelineLogger:
    """Handles logging for the pipeline."""
    
    def __init__(self, log_dir: Path, experiment_name: str):
        """Initialize logger with file and console handlers."""
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create log directory
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Log files
        self.main_log = log_dir / f"pipeline_{experiment_name}_{self.timestamp}.log"
        self.summary_log = log_dir / f"summary_{experiment_name}_{self.timestamp}.json"
        self.timing_log = log_dir / f"timing_{experiment_name}_{self.timestamp}.csv"
        
        # Setup main logger
        self.logger = logging.getLogger("MalariaPipeline")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers = []  # Clear existing handlers
        
        # File handler (detailed) - with UTF-8 encoding for Windows
        file_handler = logging.FileHandler(self.main_log, encoding='utf-8', errors='replace')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler (info and above)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # Timing data
        self.timing_data = []
        self.stage_outputs = {}
        
    def info(self, msg: str):
        """Log info message."""
        self.logger.info(msg)
    
    def debug(self, msg: str):
        """Log debug message."""
        self.logger.debug(msg)
    
    def warning(self, msg: str):
        """Log warning message."""
        self.logger.warning(msg)
    
    def error(self, msg: str):
        """Log error message."""
        self.logger.error(msg)
    
    def success(self, msg: str):
        """Log success message with checkmark."""
        self.logger.info(f"✅ {msg}")
    
    def stage_start(self, stage: int, name: str, description: str):
        """Log stage start."""
        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info(f"STAGE {stage}: {name.upper()}")
        self.logger.info("=" * 70)
        self.logger.info(f"Description: {description}")
        self.logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("-" * 70)
    
    def stage_end(self, stage: int, name: str, duration: float, success: bool, 
                  outputs: List[str] = None):
        """Log stage completion."""
        self.logger.info("-" * 70)
        status = "COMPLETED" if success else "FAILED"
        symbol = "✅" if success else "❌"
        self.logger.info(f"{symbol} Stage {stage} ({name}) {status}")
        self.logger.info(f"   Duration: {self._format_duration(duration)}")
        
        if outputs:
            self.logger.info(f"   Outputs created: {len(outputs)}")
            for out in outputs[:5]:  # Show first 5
                self.logger.info(f"      - {out}")
            if len(outputs) > 5:
                self.logger.info(f"      ... and {len(outputs) - 5} more")
        
        # Record timing
        self.timing_data.append({
            'stage': stage,
            'name': name,
            'duration_sec': duration,
            'duration_formatted': self._format_duration(duration),
            'success': success,
            'timestamp': datetime.now().isoformat()
        })
        
        self.stage_outputs[stage] = outputs or []
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration as human-readable string."""
        if seconds < 60:
            return f"{seconds:.2f} seconds"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.2f} minutes"
        else:
            hours = seconds / 3600
            return f"{hours:.2f} hours"
    
    def save_summary(self, total_duration: float, stages_run: List[int], 
                     success: bool, error_msg: str = None):
        """Save summary JSON file."""
        summary = {
            "experiment": self.experiment_name,
            "timestamp": self.timestamp,
            "total_duration_sec": total_duration,
            "total_duration_formatted": self._format_duration(total_duration),
            "stages_run": stages_run,
            "overall_success": success,
            "error_message": error_msg,
            "stage_timing": self.timing_data,
            "stage_outputs": self.stage_outputs,
            "project_root": str(PipelineConfig.PROJECT_ROOT)
        }
        
        with open(self.summary_log, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.info(f"\n📝 Summary saved: {self.summary_log}")
    
    def save_timing_csv(self):
        """Save timing data as CSV."""
        import csv
        with open(self.timing_log, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'stage', 'name', 'duration_sec', 'duration_formatted', 
                'success', 'timestamp'
            ])
            writer.writeheader()
            writer.writerows(self.timing_data)
        
        self.info(f"📊 Timing log saved: {self.timing_log}")



# Pipeline Runner
class PipelineRunner:
    """Main pipeline execution class."""
    
    def __init__(self, experiment: str = "baseline", 
                 skip_extraction: bool = False,
                 stages: List[int] = None):
        """Initialize pipeline runner."""
        self.experiment = experiment
        self.skip_extraction = skip_extraction
        self.stages = stages or [1, 2, 3, 4, 5, 6]
        
        # Remove stage 1 if skipping extraction
        if skip_extraction and 1 in self.stages:
            self.stages = [s for s in self.stages if s != 1]
        
        # Setup directories
        PipelineConfig.create_directories()
        
        # Setup logging
        self.logger = PipelineLogger(PipelineConfig.LOGS_DIR, experiment)
        
        # Track results
        self.results = {}
        self.start_time = None
    
    def run(self) -> bool:
        """Run the complete pipeline."""
        self.start_time = time.time()
        overall_success = True
        error_message = None
        
        # Print header
        self._print_header()
        
        try:
            for stage in self.stages:
                script = PipelineConfig.SCRIPTS.get(stage)
                name = PipelineConfig.STAGE_NAMES.get(stage)
                description = PipelineConfig.STAGE_DESCRIPTIONS.get(stage)
                
                if not script:
                    self.logger.warning(f"Unknown stage: {stage}")
                    continue
                
                # Check if script exists
                if not script.exists():
                    self.logger.error(f"Script not found: {script}")
                    overall_success = False
                    error_message = f"Script not found: {script}"
                    break
                
                # Run stage
                success, duration, outputs = self._run_stage(stage, script, name, description)
                
                if not success:
                    overall_success = False
                    error_message = f"Stage {stage} ({name}) failed"
                    break
                
                # Validate outputs
                if not self._validate_outputs(stage):
                    self.logger.warning(f"  Some expected outputs missing for stage {stage}")
        
        except KeyboardInterrupt:
            self.logger.error("\n  Pipeline interrupted by user")
            overall_success = False
            error_message = "Interrupted by user"
        
        except Exception as e:
            self.logger.error(f"\n Pipeline error: {str(e)}")
            self.logger.debug(traceback.format_exc())
            overall_success = False
            error_message = str(e)
        
        # Calculate total duration
        total_duration = time.time() - self.start_time
        
        # Print summary
        self._print_summary(total_duration, overall_success)
        
        # Save logs
        self.logger.save_summary(total_duration, self.stages, overall_success, error_message)
        self.logger.save_timing_csv()
        
        return overall_success
    
    def _run_stage(self, stage: int, script: Path, name: str, 
                   description: str) -> Tuple[bool, float, List[str]]:
        """Run a single stage."""
        self.logger.stage_start(stage, name, description)
        
        start_time = time.time()
        outputs = []
        success = False
        
        try:
            # Build command
            cmd = [sys.executable, str(script)]
            
            # Add experiment flag for federated training
            if stage == 5:
                cmd.extend(["--experiment", self.experiment])
            
            # Run subprocess from project root directory
            self.logger.debug(f"Running: {' '.join(cmd)}")
            self.logger.debug(f"Working directory: {PipelineConfig.PROJECT_ROOT}")
            
            # Create stage-specific log file
            stage_log = PipelineConfig.LOGS_DIR / f"stage_{stage}_{name.lower().replace(' ', '_')}.log"
            
            with open(stage_log, 'w', encoding='utf-8', errors='replace') as log_file:
                # Using encoding with error handling for Windows compatibility
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    cwd=str(PipelineConfig.PROJECT_ROOT),  # Run from project root
                    env={**os.environ, 'PYTHONIOENCODING': 'utf-8'}  # Force UTF-8
                )
                
                # Stream output with proper encoding handling
                for line_bytes in process.stdout:
                    # Decode with error handling for Windows
                    try:
                        line = line_bytes.decode('utf-8')
                    except UnicodeDecodeError:
                        try:
                            line = line_bytes.decode('cp1252', errors='replace')
                        except:
                            line = line_bytes.decode('latin-1', errors='replace')
                    
                    # Write to stage log
                    log_file.write(line)
                    log_file.flush()
                    
                    # Also print to console (summarized)
                    line_stripped = line.strip()
                    if line_stripped:
                        # Show important lines
                        if any(kw in line_stripped.lower() for kw in 
                               ['error', 'warning', 'saved', 
                                'complete', 'loading', 'training', 'generating',
                                'stage', 'total', 'auc', 'accuracy', 'f1']):
                            self.logger.info(f"   {line_stripped}")
                        else:
                            self.logger.debug(f"   {line_stripped}")
                
                # Wait for completion
                process.wait()
                
                success = process.returncode == 0
            
            # Get outputs
            if success:
                outputs = self._get_created_outputs(stage)
            
        except subprocess.TimeoutExpired:
            self.logger.error(f"Stage {stage} timed out after {PipelineConfig.STAGE_TIMEOUTS[stage]}s")
            success = False
        
        except Exception as e:
            self.logger.error(f"Error in stage {stage}: {str(e)}")
            self.logger.debug(traceback.format_exc())
            success = False
        
        duration = time.time() - start_time
        self.logger.stage_end(stage, name, duration, success, outputs)
        
        return success, duration, outputs
    
    def _get_created_outputs(self, stage: int) -> List[str]:
        """Get list of outputs created by a stage."""
        outputs = []
        expected = PipelineConfig.EXPECTED_OUTPUTS.get(stage, [])
        
        for path in expected:
            p = PipelineConfig.PROJECT_ROOT / path
            if p.exists():
                if p.is_dir():
                    # Count files in directory
                    files = list(p.glob("*"))
                    outputs.append(f"{path}/ ({len(files)} files)")
                else:
                    outputs.append(str(path))
        
        return outputs
    
    def _validate_outputs(self, stage: int) -> bool:
        """Validate that expected outputs exist."""
        expected = PipelineConfig.EXPECTED_OUTPUTS.get(stage, [])
        all_exist = True
        
        for path in expected:
            p = PipelineConfig.PROJECT_ROOT / path
            if not p.exists():
                self.logger.warning(f"   Missing output: {path}")
                all_exist = False
        
        return all_exist
    
    def _print_header(self):
        """Print pipeline header."""
        self.logger.info("")
        self.logger.info("╔" + "═" * 68 + "╗")
        self.logger.info("║" + " " * 68 + "║")
        self.logger.info("║" + "   🇬🇭 GHANA MALARIA FEDERATED LEARNING PIPELINE".center(68) + "║")
        self.logger.info("║" + " " * 68 + "║")
        self.logger.info("║" + f"   Experiment: {self.experiment.upper()}".center(68) + "║")
        self.logger.info("║" + f"   Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(68) + "║")
        self.logger.info("║" + " " * 68 + "║")
        self.logger.info("╚" + "═" * 68 + "╝")
        self.logger.info("")
        
        self.logger.info(f" Project Root: {PipelineConfig.PROJECT_ROOT}")
        self.logger.info("")
        
        # Print stages to run
        self.logger.info(" Stages to execute:")
        for stage in self.stages:
            name = PipelineConfig.STAGE_NAMES.get(stage, f"Stage {stage}")
            script = PipelineConfig.get_script_display_path(stage)
            self.logger.info(f"   {stage}. {name} ({script})")
        self.logger.info("")
    
    def _print_summary(self, total_duration: float, success: bool):
        """Print pipeline summary."""
        self.logger.info("")
        self.logger.info("╔" + "═" * 68 + "╗")
        self.logger.info("║" + " PIPELINE SUMMARY ".center(68, "═") + "║")
        self.logger.info("╠" + "═" * 68 + "╣")
        
        status = " COMPLETED SUCCESSFULLY" if success else " FAILED"
        self.logger.info("║" + f"   Status: {status}".ljust(67) + "║")
        self.logger.info("║" + f"   Total Duration: {self.logger._format_duration(total_duration)}".ljust(67) + "║")
        self.logger.info("║" + f"   Experiment: {self.experiment}".ljust(67) + "║")
        self.logger.info("║" + f"   Stages Run: {len(self.stages)}".ljust(67) + "║")
        
        self.logger.info("╠" + "═" * 68 + "╣")
        self.logger.info("║" + " Stage Timing:".ljust(67) + "║")
        
        for timing in self.logger.timing_data:
            stage_status = "" if timing['success'] else "❌"
            line = f"   {stage_status} Stage {timing['stage']}: {timing['duration_formatted']}"
            self.logger.info("║" + line.ljust(67) + "║")
        
        self.logger.info("╠" + "═" * 68 + "╣")
        self.logger.info("║" + " Output Locations:".ljust(67) + "║")
        self.logger.info("║" + f"    Logs:    {PipelineConfig.LOGS_DIR}/".ljust(67) + "║")
        self.logger.info("║" + f"    Results: {PipelineConfig.RESULTS_DIR}/".ljust(67) + "║")
        self.logger.info("║" + f"    Data:    {PipelineConfig.DATA_DIR}/".ljust(67) + "║")
        
        self.logger.info("╚" + "═" * 68 + "╝")


# COMMAND LINE INTERFACE
def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the Ghana Malaria Federated Learning Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples (run from project root):
  python src/pipeline/run_complete_pipeline.py                        # Full pipeline (baseline)
  python src/pipeline/run_complete_pipeline.py --experiment grid_search
  python src/pipeline/run_complete_pipeline.py --experiment ablation
  python src/pipeline/run_complete_pipeline.py --skip-extraction      # Skip data extraction
  python src/pipeline/run_complete_pipeline.py --stages 4,5,6         # Training & analysis only
  python src/pipeline/run_complete_pipeline.py --stages 5 --experiment ablation

Project Structure:
  project_root/
  ├── src/
  │   ├── data/           # Data processing scripts (stages 1-3)
  │   ├── training/       # Model training scripts (stages 4-5)
  │   ├── evaluation/     # Analysis scripts (stage 6)
  │   └── pipeline/       # This pipeline script
  ├── data/               # Data files (raw, cleaned, fl_scenarios)
  ├── results/            # Output results, figures, tables
  └── logs/               # Pipeline logs and timing
        """
    )
    
    parser.add_argument(
        '--experiment', '-e',
        type=str,
        choices=['baseline', 'grid_search', 'ablation'],
        default='baseline',
        help='Experiment type for federated training (default: baseline)'
    )
    
    parser.add_argument(
        '--skip-extraction', '-s',
        action='store_true',
        help='Skip data extraction stage (use if raw data already processed)'
    )
    
    parser.add_argument(
        '--stages',
        type=str,
        default=None,
        help='Comma-separated list of stages to run (e.g., "1,2,3" or "4,5,6")'
    )
    
    parser.add_argument(
        '--list-stages',
        action='store_true',
        help='List all available stages and exit'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show all output (not just important lines)'
    )
    
    return parser.parse_args()


def list_stages():
    """Print available stages."""
    print("\n📋 Available Pipeline Stages:")
    print("-" * 70)
    for stage_num in sorted(PipelineConfig.STAGE_NAMES.keys()):
        name = PipelineConfig.STAGE_NAMES[stage_num]
        script = PipelineConfig.get_script_display_path(stage_num)
        desc = PipelineConfig.STAGE_DESCRIPTIONS[stage_num]
        print(f"\n  Stage {stage_num}: {name}")
        print(f"    Script: {script}")
        print(f"    Description: {desc}")
    print("\n" + "-" * 70)
    print(f"\n Project Root: {PipelineConfig.PROJECT_ROOT}")
    print(f" Logs Directory: {PipelineConfig.LOGS_DIR}")


# ======================================================================
# MAIN ENTRY POINT
# ======================================================================

def main():
    """Main entry point."""
    args = parse_arguments()
    
    # List stages and exit if requested
    if args.list_stages:
        list_stages()
        return 0
    
    # Parse stages
    stages = None
    if args.stages:
        try:
            stages = [int(s.strip()) for s in args.stages.split(',')]
            # Validate stage numbers
            for s in stages:
                if s not in PipelineConfig.STAGE_NAMES:
                    print(f" Invalid stage number: {s}")
                    print(f"   Valid stages: {list(PipelineConfig.STAGE_NAMES.keys())}")
                    return 1
        except ValueError:
            print(f" Invalid stages format: {args.stages}")
            print("   Use comma-separated numbers, e.g., '1,2,3' or '4,5,6'")
            return 1
    
    # Create and run pipeline
    pipeline = PipelineRunner(
        experiment=args.experiment,
        skip_extraction=args.skip_extraction,
        stages=stages
    )
    
    success = pipeline.run()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())