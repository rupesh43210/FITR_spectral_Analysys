"""
FTIR Data Processor Module

This module handles the preprocessing of FTIR spectral data, including:
- Loading data from Excel files
- Spectral preprocessing (baseline correction, normalization)
- Feature extraction (peak detection, derivatives)
- Data validation and cleaning

The processor expects data in Excel format with wavenumbers in the first column
and spectral measurements in subsequent columns.

Author: Dr. Priya's Research Team
Version: 1.0.0
Date: January 2025
"""

import os
import logging
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, find_peaks
from scipy.interpolate import interp1d
from sklearn.impute import SimpleImputer

class DataProcessor:
    """
    A class for processing FTIR spectral data.
    
    This class handles all aspects of data preprocessing including loading,
    cleaning, feature extraction, and preparation for model training.
    
    Attributes:
        data_dir (str): Directory containing input data files
        osf_file (str): Filename for OSF group data
        habit_file (str): Filename for Habit group data
        normal_file (str): Filename for Normal group data
        std_wavenumbers (numpy.ndarray): Standardized wavenumber range
    """
    
    def __init__(self, data_dir='data', osf_file='osmf.xlsx', 
                 habit_file='habit.xlsx', normal_file='normal.xlsx'):
        """
        Initialize the DataProcessor with file paths.
        
        Args:
            data_dir (str): Directory containing input files
            osf_file (str): Filename for OSF group data
            habit_file (str): Filename for Habit group data
            normal_file (str): Filename for Normal group data
        """
        self.data_dir = data_dir
        self.osf_file = os.path.join(data_dir, osf_file)
        self.habit_file = os.path.join(data_dir, habit_file)
        self.normal_file = os.path.join(data_dir, normal_file)
        self.std_wavenumbers = None
        self.imputer = SimpleImputer(strategy='mean')
    
    def _load_excel_data(self, filepath):
        """
        Load and validate data from an Excel file.
        
        Args:
            filepath (str): Path to Excel file
            
        Returns:
            tuple: (wavenumbers, spectra)
        """
        try:
            # Read Excel file
            df = pd.read_excel(filepath)
            
            # Handle empty dataframe
            if df.empty:
                raise ValueError(f"Empty data file: {filepath}")
            
            # Extract wavenumbers and spectra
            wavenumbers = df.iloc[:, 0].values
            spectra = df.iloc[:, 1:].values.T
            
            # Handle missing values in wavenumbers
            if np.any(pd.isna(wavenumbers)):
                logging.warning(f"Found missing values in wavenumbers in {filepath}. Interpolating...")
                wavenumbers = pd.Series(wavenumbers).interpolate(method='linear').values
            
            # Handle missing values in spectra
            if np.any(pd.isna(spectra)):
                logging.warning(f"Found missing values in spectra in {filepath}. Imputing with mean...")
                spectra = self.imputer.fit_transform(spectra)
            
            # Ensure wavenumbers are in descending order
            if not np.all(np.diff(wavenumbers) < 0):
                logging.warning(f"Wavenumbers in {filepath} are not in descending order. Sorting...")
                sort_idx = np.argsort(wavenumbers)[::-1]
                wavenumbers = wavenumbers[sort_idx]
                spectra = spectra[:, sort_idx]
            
            # Remove outliers (spectra with unrealistic values)
            valid_mask = self._remove_outliers(spectra)
            if not np.all(valid_mask):
                logging.warning(f"Removed {np.sum(~valid_mask)} outlier spectra from {filepath}")
                spectra = spectra[valid_mask]
            
            # Validate final data
            if len(spectra) == 0:
                raise ValueError(f"No valid spectra remaining in {filepath}")
            
            logging.info(f"Successfully loaded {len(spectra)} spectra from {filepath}")
            return wavenumbers, spectra
            
        except Exception as e:
            logging.error(f"Error loading {filepath}: {str(e)}")
            raise
    
    def _remove_outliers(self, spectra, zscore_threshold=3.0):
        """
        Remove outlier spectra using z-score method.
        
        Args:
            spectra (numpy.ndarray): Spectral data
            zscore_threshold (float): Z-score threshold for outlier detection
            
        Returns:
            numpy.ndarray: Boolean mask of valid spectra
        """
        # Calculate z-scores for each spectrum
        mean_intensities = np.mean(spectra, axis=1)
        std_intensities = np.std(spectra, axis=1)
        zscores = np.abs((mean_intensities - np.mean(mean_intensities)) / np.std(mean_intensities))
        
        # Create mask for valid spectra
        valid_mask = zscores < zscore_threshold
        
        # Ensure we don't remove too many spectra
        if np.sum(valid_mask) < 0.5 * len(spectra):
            logging.warning("Too many outliers detected. Adjusting threshold...")
            return np.ones(len(spectra), dtype=bool)
        
        return valid_mask
    
    def _preprocess_spectra(self, spectra):
        """
        Apply preprocessing steps to spectral data.
        
        Steps:
        1. Baseline correction
        2. Normalization
        3. Smoothing
        4. Feature extraction
        
        Args:
            spectra (numpy.ndarray): Raw spectral data
            
        Returns:
            numpy.ndarray: Processed spectral data
        """
        try:
            # Baseline correction
            baseline = np.min(spectra, axis=1, keepdims=True)
            spectra_baselined = spectra - baseline
            
            # Handle negative values
            spectra_baselined = np.maximum(spectra_baselined, 0)
            
            # Normalization with error checking
            max_vals = np.max(spectra_baselined, axis=1, keepdims=True)
            max_vals[max_vals == 0] = 1  # Avoid division by zero
            spectra_normalized = spectra_baselined / max_vals
            
            # Smoothing using Savitzky-Golay filter with error handling
            spectra_smoothed = []
            for spectrum in spectra_normalized:
                try:
                    smoothed = savgol_filter(spectrum, window_length=7, polyorder=2)
                except Exception as e:
                    logging.warning(f"Smoothing failed, using original spectrum: {str(e)}")
                    smoothed = spectrum
                spectra_smoothed.append(smoothed)
            
            return np.array(spectra_smoothed)
            
        except Exception as e:
            logging.error(f"Error in preprocessing: {str(e)}")
            return spectra
    
    def _extract_features(self, spectra):
        """
        Extract additional features from spectral data.
        
        Features:
        1. First derivative
        2. Second derivative
        3. Peak characteristics
        
        Args:
            spectra (numpy.ndarray): Preprocessed spectral data
            
        Returns:
            numpy.ndarray: Extended feature matrix
        """
        features = []
        
        for spectrum in spectra:
            try:
                # Calculate derivatives
                first_deriv = np.gradient(spectrum)
                second_deriv = np.gradient(first_deriv)
                
                # Find peaks with error handling
                try:
                    peaks, properties = find_peaks(spectrum, height=0.1, width=2)
                    n_peaks = len(peaks)
                    mean_height = np.mean(properties['peak_heights']) if len(peaks) > 0 else 0
                    mean_width = np.mean(properties['widths']) if len(peaks) > 0 else 0
                except Exception as e:
                    logging.warning(f"Peak detection failed: {str(e)}")
                    n_peaks, mean_height, mean_width = 0, 0, 0
                
                # Compile features
                spectrum_features = np.concatenate([
                    spectrum,  # Original spectrum
                    first_deriv,  # First derivative
                    second_deriv,  # Second derivative
                    [n_peaks],  # Number of peaks
                    [mean_height],  # Mean peak height
                    [mean_width]  # Mean peak width
                ])
                
                features.append(spectrum_features)
                
            except Exception as e:
                logging.warning(f"Feature extraction failed for a spectrum: {str(e)}")
                # Use zeros for failed feature extraction
                features.append(np.zeros_like(features[0]) if features else np.zeros(len(spectrum) * 3 + 3))
        
        return np.array(features)
    
    def process_data(self):
        """
        Process all data files and prepare for model training.
        
        Returns:
            tuple: (X, y) where X is the feature matrix and y is the label vector
        """
        try:
            # Load data with error handling for each file
            data_groups = []
            for filepath, label in [(self.osf_file, 'OSF'),
                                  (self.habit_file, 'Habit'),
                                  (self.normal_file, 'Normal')]:
                try:
                    wave, spectra = self._load_excel_data(filepath)
                    if self.std_wavenumbers is None:
                        self.std_wavenumbers = wave
                    
                    # Interpolate to standard wavenumbers if needed
                    if not np.array_equal(wave, self.std_wavenumbers):
                        logging.warning(f"Interpolating {label} data to match standard wavenumbers")
                        new_spectra = []
                        for spectrum in spectra:
                            f = interp1d(wave, spectrum, bounds_error=False, fill_value='extrapolate')
                            new_spectra.append(f(self.std_wavenumbers))
                        spectra = np.array(new_spectra)
                    
                    # Process spectra
                    processed = self._preprocess_spectra(spectra)
                    features = self._extract_features(processed)
                    
                    data_groups.append((features, label))
                    logging.info(f"Successfully processed {label} group: {len(features)} samples")
                    
                except Exception as e:
                    logging.error(f"Error processing {label} group: {str(e)}")
                    if not data_groups:
                        raise  # Raise error if no data has been processed yet
                    continue
            
            # Combine all processed data
            X = np.vstack([group[0] for group in data_groups])
            y = np.concatenate([np.repeat(group[1], len(group[0])) for group in data_groups])
            
            logging.info(f"Final processed data shape: X={X.shape}, y={y.shape}")
            return X, y
            
        except Exception as e:
            logging.error(f"Error processing data: {str(e)}")
            raise
