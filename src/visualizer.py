"""
FTIR Visualization Module

This module handles the creation of visualizations for FTIR spectral analysis, including:
- ROC curves with confidence intervals
- Model performance comparisons
- Spectral plots and feature importance
- Classification results visualization

Author: Dr. Priya's Research Team
Version: 1.0.0
Date: January 2025
"""

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from scipy.interpolate import interp1d
from itertools import cycle

class Visualizer:
    """
    A class for creating and saving visualizations of FTIR analysis results.
    
    This class handles the generation of various plots and visualizations to help
    interpret the results of the FTIR spectral analysis and model performance.
    
    Attributes:
        output_dir (str): Directory where visualizations will be saved
        dpi (int): Resolution for saved figures
        figsize (tuple): Default figure size (width, height)
    """
    
    def __init__(self, output_dir, dpi=300, figsize=(10, 8)):
        """
        Initialize the Visualizer with output settings.
        
        Args:
            output_dir (str): Directory to save visualizations
            dpi (int): Resolution for saved figures
            figsize (tuple): Default figure size (width, height)
        """
        self.output_dir = output_dir
        self.dpi = dpi
        self.figsize = figsize
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up plotting style
        sns.set_theme(style='whitegrid')
        plt.rcParams['figure.figsize'] = figsize
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['axes.titlesize'] = 16

    def plot_roc_curves(self, results, X, y):
        """
        Plot ROC curves with confidence intervals for all models.
        
        Args:
            results (dict): Dictionary containing model results
            X (numpy.ndarray): Feature matrix
            y (numpy.ndarray): Target labels
        """
        try:
            plt.figure(figsize=self.figsize)
            
            # Colors for different models
            colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red'])
            
            # Plot ROC curve for each model
            for model_name, color in zip(results.keys(), colors):
                model = results[model_name]['model']
                
                # Get unique classes
                classes = np.unique(y)
                n_classes = len(classes)
                
                # Binarize the labels for ROC curve calculation
                y_bin = label_binarize(y, classes=classes)
                
                # Initialize arrays to store ROC curve data
                fpr = dict()
                tpr = dict()
                roc_auc = dict()
                
                # Compute ROC curve for each class
                y_pred = model.predict_proba(X)
                for i, class_label in enumerate(classes):
                    fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_pred[:, i])
                    roc_auc[i] = auc(fpr[i], tpr[i])
                
                # Compute micro-average ROC curve
                fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), y_pred.ravel())
                roc_auc_micro = auc(fpr_micro, tpr_micro)
                
                # Plot ROC curves
                plt.plot(
                    fpr_micro, tpr_micro,
                    label=f'{model_name} (AUC = {roc_auc_micro:.2f})',
                    color=color, linestyle='-', linewidth=2
                )
            
            # Plot random chance line
            plt.plot([0, 1], [0, 1], 'k--', label='Random')
            
            # Customize plot
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curves for Different Models')
            plt.legend(loc="lower right")
            plt.grid(True)
            
            # Save plot
            plt.savefig(os.path.join(self.output_dir, 'roc_curves.png'), 
                       dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            logging.info("Successfully created ROC curves plot")
            
        except Exception as e:
            logging.error(f"Error creating ROC curves plot: {str(e)}")
            raise

    def plot_model_comparison(self, results):
        """
        Create a bar plot comparing model performance metrics.
        
        Args:
            results (dict): Dictionary containing model results
        """
        try:
            # Prepare data for plotting
            models = list(results.keys())
            metrics = ['accuracy', 'precision', 'recall', 'f1']
            
            # Extract metrics from results
            metric_values = {metric: [] for metric in metrics}
            for model in models:
                for metric in metrics:
                    metric_values[metric].append(results[model]['metrics'][metric])
            
            # Create plot
            plt.figure(figsize=self.figsize)
            x = np.arange(len(models))
            width = 0.2
            multiplier = 0
            
            for metric in metrics:
                offset = width * multiplier
                plt.bar(x + offset, metric_values[metric], width, label=metric.capitalize())
                multiplier += 1
            
            # Customize plot
            plt.xlabel('Models')
            plt.ylabel('Score')
            plt.title('Model Performance Comparison')
            plt.xticks(x + width * 1.5, models, rotation=45, ha='right')
            plt.legend(loc='lower right')
            plt.grid(True, alpha=0.3)
            
            # Add value labels on top of each bar
            for i, metric in enumerate(metrics):
                for j, value in enumerate(metric_values[metric]):
                    plt.text(
                        x[j] + width * i,
                        value,
                        f'{value:.2f}',
                        ha='center',
                        va='bottom',
                        fontsize=8,
                        rotation=45
                    )
            
            # Adjust layout and save
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'model_comparison.png'),
                       dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            logging.info("Successfully created model comparison plot")
            
        except Exception as e:
            logging.error(f"Error creating model comparison plot: {str(e)}")
            raise

    def plot_feature_importance(self, results, feature_names=None):
        """
        Create feature importance plots for applicable models.
        
        Args:
            results (dict): Dictionary containing model results
            feature_names (list): Optional list of feature names
        """
        try:
            for model_name, result in results.items():
                model = result['model']
                
                # Check if model has feature importance attribute
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    indices = np.argsort(importances)[-20:]  # Top 20 features
                    
                    plt.figure(figsize=self.figsize)
                    plt.title(f'Top 20 Feature Importances ({model_name})')
                    plt.barh(range(20), importances[indices])
                    
                    if feature_names is not None:
                        names = [feature_names[i] for i in indices]
                    else:
                        names = [f'Feature {i}' for i in indices]
                    
                    plt.yticks(range(20), names)
                    plt.xlabel('Relative Importance')
                    
                    # Save plot
                    plt.tight_layout()
                    plt.savefig(
                        os.path.join(self.output_dir, f'feature_importance_{model_name}.png'),
                        dpi=self.dpi, bbox_inches='tight'
                    )
                    plt.close()
            
            logging.info("Successfully created feature importance plots")
            
        except Exception as e:
            logging.error(f"Error creating feature importance plots: {str(e)}")
            raise

    def create_visualizations(self, results, X, y, wavenumbers=None):
        """
        Create all visualizations for the analysis results.
        
        Args:
            results (dict): Dictionary containing model results
            X (numpy.ndarray): Feature matrix
            y (numpy.ndarray): Target labels
            wavenumbers (numpy.ndarray, optional): Wavenumber values
        """
        try:
            # Create ROC curves
            self.plot_roc_curves(results, X, y)
            
            # Create model comparison plot
            self.plot_model_comparison(results)
            
            # Create FTIR-specific visualizations if wavenumbers are provided
            if wavenumbers is not None:
                self.plot_feature_importance(results, wavenumbers)
                self._create_average_spectra(X, y, wavenumbers)
                self._create_peak_analysis(X, y, wavenumbers)
                self._create_spectral_variance_plot(X, y, wavenumbers)
            
            logging.info("Successfully created all visualizations")
            
        except Exception as e:
            logging.error(f"Error creating visualizations: {str(e)}")
            raise

    def _create_average_spectra(self, X, y, wavenumbers):
        """Create plot of average spectra for each class with confidence intervals"""
        try:
            plt.figure(figsize=self.figsize)

            # Use only spectral features for this plot
            spectral_features = X[:, :len(wavenumbers)]

            colors = {"OSF": "red", "Habit": "blue", "Normal": "green"}
            linestyles = {"OSF": "-", "Habit": "--", "Normal": ":"}

            # For each class
            for class_name in np.unique(y):
                # Get spectra for this class
                class_spectra = spectral_features[y == class_name]

                # Calculate average spectrum and std dev
                avg_spectrum = np.mean(class_spectra, axis=0)
                std_spectrum = np.std(class_spectra, axis=0)

                # Plot average spectrum
                plt.plot(
                    wavenumbers,
                    avg_spectrum,
                    label=f"{class_name} (n={len(class_spectra)})",
                    color=colors.get(class_name, "gray"),
                    linestyle=linestyles.get(class_name, "-"),
                    linewidth=2,
                )

                # Plot confidence interval
                plt.fill_between(
                    wavenumbers,
                    avg_spectrum - std_spectrum,
                    avg_spectrum + std_spectrum,
                    color=colors.get(class_name, "gray"),
                    alpha=0.2,
                )

            plt.xlabel("Wavenumber (cm⁻¹)")
            plt.ylabel("Absorbance")
            plt.title("Average FTIR Spectra by Class\n(Shaded areas show ±1 std dev)")
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Add text box with sample sizes
            sample_sizes = {name: sum(y == name) for name in np.unique(y)}
            plt.text(
                0.02,
                0.98,
                "Sample Sizes:\n" + "\n".join([f"{k}: {v}" for k, v in sample_sizes.items()]),
                transform=plt.gca().transAxes,
                bbox=dict(facecolor="white", alpha=0.8),
                verticalalignment="top",
            )

            plt.savefig(
                os.path.join(self.output_dir, "average_spectra.png"),
                dpi=self.dpi,
                bbox_inches="tight",
            )
            plt.close()
            logging.info("Successfully created average spectra plot")

        except Exception as e:
            logging.error(f"Error creating average spectra plot: {str(e)}")
            raise

    def _create_peak_analysis(self, X, y, wavenumbers):
        """Create plot showing peak analysis for each class"""
        try:
            from scipy.signal import find_peaks
            plt.figure(figsize=self.figsize)

            # Use only spectral features
            spectral_features = X[:, :len(wavenumbers)]
            colors = {"OSF": "red", "Habit": "blue", "Normal": "green"}

            # For each class
            for class_name in np.unique(y):
                # Get spectra for this class
                class_spectra = spectral_features[y == class_name]

                # Calculate mean spectrum
                mean_spectrum = np.mean(class_spectra, axis=0)

                # Plot mean spectrum
                plt.plot(
                    wavenumbers,
                    mean_spectrum,
                    label=f"{class_name} Mean",
                    color=colors.get(class_name, "gray"),
                )

                # Find peaks
                peaks, _ = find_peaks(mean_spectrum, distance=20, prominence=0.01)

                # Plot peak points
                plt.plot(
                    wavenumbers[peaks],
                    mean_spectrum[peaks],
                    "x",
                    color=colors.get(class_name, "gray"),
                    label=f"{class_name} Peaks",
                )

                # Annotate major peaks (top 5)
                peak_heights = mean_spectrum[peaks]
                top_peak_indices = np.argsort(peak_heights)[-5:]
                top_peaks = peaks[top_peak_indices]

                for peak in top_peaks:
                    plt.annotate(
                        f"{wavenumbers[peak]:.0f}",
                        xy=(wavenumbers[peak], mean_spectrum[peak]),
                        xytext=(5, 5),
                        textcoords="offset points",
                        fontsize=8,
                        color=colors.get(class_name, "gray"),
                    )

            plt.xlabel("Wavenumber (cm⁻¹)")
            plt.ylabel("Absorbance")
            plt.title("Peak Analysis by Class")
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.savefig(
                os.path.join(self.output_dir, "peak_analysis.png"),
                dpi=self.dpi,
                bbox_inches="tight",
            )
            plt.close()
            logging.info("Successfully created peak analysis plot")

        except Exception as e:
            logging.error(f"Error creating peak analysis plot: {str(e)}")
            raise

    def _create_spectral_variance_plot(self, X, y, wavenumbers):
        """Create plot showing spectral variance across classes"""
        try:
            plt.figure(figsize=self.figsize)

            # Use only spectral features
            spectral_features = X[:, :len(wavenumbers)]
            colors = {"OSF": "red", "Habit": "blue", "Normal": "green"}

            # Calculate overall variance
            total_variance = np.var(spectral_features, axis=0)
            plt.plot(wavenumbers, total_variance, "k--", label="Total Variance", alpha=0.5)

            # Calculate and plot variance for each class
            for class_name in np.unique(y):
                class_spectra = spectral_features[y == class_name]
                variance = np.var(class_spectra, axis=0)

                plt.plot(
                    wavenumbers,
                    variance,
                    label=f"{class_name} Variance",
                    color=colors.get(class_name, "gray"),
                )

            plt.xlabel("Wavenumber (cm⁻¹)")
            plt.ylabel("Variance")
            plt.title("Spectral Variance by Class")
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.savefig(
                os.path.join(self.output_dir, "spectral_variance.png"),
                dpi=self.dpi,
                bbox_inches="tight",
            )
            plt.close()
            logging.info("Successfully created spectral variance plot")

        except Exception as e:
            logging.error(f"Error creating spectral variance plot: {str(e)}")
            raise
