#!/usr/bin/env python3
"""
FTIR Spectral Analysis Pipeline

This script orchestrates the complete FTIR spectral analysis workflow:
1. Data loading and preprocessing
2. Model training and evaluation
3. Visualization generation
4. Results reporting

The script expects FTIR data in Excel format (.xlsx) organized in three groups:
- OSF
- Habit
- Normal

Author: Dr. Priya's Research Team
Version: 1.0.0
Date: January 2025
"""

import os
import logging
from datetime import datetime
from pathlib import Path
import sys
from data_processor import DataProcessor
from model_trainer import ModelTrainer
from visualizer import Visualizer


def setup_logging(output_dir):
    """
    Configure logging to both file and console.

    Args:
        output_dir (str): Directory where log file will be created

    Returns:
        None
    """
    log_file = os.path.join(output_dir, "analysis.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )


def process_ftir_data():
    """
    Main function to process FTIR data through the complete analysis pipeline.

    This function:
    1. Sets up output directory and logging
    2. Processes raw FTIR data
    3. Trains and evaluates multiple ML models
    4. Generates visualizations
    5. Creates a summary report

    Returns:
        tuple: (results dict, output directory path)
    """
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(current_dir)
    output_dir = os.path.join(project_dir, "output", f"analysis_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # Setup logging
    setup_logging(output_dir)

    try:
        # Log analysis start
        logging.info("Starting FTIR analysis")
        logging.info(
            f'Input directory: {os.path.abspath(os.path.join(project_dir, "data"))}'
        )
        logging.info(f"Output directory: {os.path.abspath(output_dir)}")

        # Initialize components
        processor = DataProcessor(
            data_dir=os.path.join(project_dir, "data"),
            osf_file="osmf.xlsx",
            habit_file="habit.xlsx",
            normal_file="normal.xlsx",
        )
        trainer = ModelTrainer(output_dir)
        visualizer = Visualizer(output_dir)

        # Process data
        logging.info("Processing data...")
        X, y = processor.process_data()
        logging.info(f"Processed data shape: X={X.shape}, y={y.shape}")

        # Train and evaluate models
        logging.info("Training and evaluating models...")
        results = trainer.train_and_evaluate(X, y)

        # Generate visualizations
        logging.info("Generating visualizations...")
        visualizer.create_visualizations(results, X, y, processor.std_wavenumbers)

        # Create summary report
        logging.info("Creating summary report...")
        create_summary_report(results, output_dir)

        logging.info("Analysis complete! Check the output directory for results.")
        return results, output_dir

    except Exception as e:
        logging.error(f"Error during analysis: {str(e)}")
        raise


def create_summary_report(results, output_dir):
    """
    Create a detailed summary report of the analysis results.

    Args:
        results (dict): Dictionary containing model results
        output_dir (str): Directory to save the report

    Returns:
        None
    """
    report_path = os.path.join(output_dir, "analysis_report.txt")

    with open(report_path, "w") as f:
        f.write("FTIR Analysis Summary Report\n")
        f.write("==========================\n\n")

        # Write timestamp
        f.write(f'Analysis Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')

        # Write model results
        f.write("Model Performance Summary:\n")
        f.write("-----------------------\n")

        # Find best model
        best_model = max(results.items(), key=lambda x: x[1]["metrics"]["f1"])

        for model_name, result in results.items():
            f.write(f"\n{model_name}:\n")
            f.write(f'  Accuracy:  {result["metrics"]["accuracy"]:.4f}\n')
            f.write(f'  Precision: {result["metrics"]["precision"]:.4f}\n')
            f.write(f'  Recall:    {result["metrics"]["recall"]:.4f}\n')
            f.write(f'  F1 Score:  {result["metrics"]["f1"]:.4f}\n')
            f.write(
                f'  CV Scores: Mean={result["cv_scores"].mean():.4f}, Std={result["cv_scores"].std():.4f}\n'
            )

        f.write(
            f'\nBest Model: {best_model[0]} (F1: {best_model[1]["metrics"]["f1"]:.4f})\n'
        )


def main():
    """Main entry point of the script."""
    try:
        process_ftir_data()
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
