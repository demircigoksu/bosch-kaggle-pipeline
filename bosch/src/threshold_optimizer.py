"""
Threshold optimization for Matthews Correlation Coefficient (MCC).
"""

import numpy as np
from sklearn.metrics import matthews_corrcoef
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


def find_best_threshold_mcc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_thresholds: int = 100
) -> Tuple[float, float]:
    """
    Find the threshold that maximizes Matthews Correlation Coefficient.
    
    Args:
        y_true: True binary labels (0/1)
        y_prob: Predicted probabilities
        n_thresholds: Number of thresholds to evaluate (if using grid search)
    
    Returns:
        Tuple of (best_threshold, best_mcc)
    """
    if len(y_true) != len(y_prob):
        raise ValueError(f"y_true and y_prob must have same length: {len(y_true)} vs {len(y_prob)}")
    
    # Remove any NaN probabilities
    valid_mask = ~np.isnan(y_prob)
    if not np.all(valid_mask):
        logger.warning(f"Removing {np.sum(~valid_mask)} NaN predictions")
        y_true = y_true[valid_mask]
        y_prob = y_prob[valid_mask]
    
    if len(y_true) == 0:
        raise ValueError("No valid predictions after removing NaNs")
    
    # Strategy 1: Use unique probabilities (faster for large datasets)
    unique_probs = np.unique(y_prob)
    
    # Strategy 2: If too many unique values, use a grid
    if len(unique_probs) > n_thresholds:
        # Use a grid from min to max probability
        thresholds = np.linspace(y_prob.min(), y_prob.max(), n_thresholds)
        logger.info(f"Using grid search with {n_thresholds} thresholds")
    else:
        # Use unique probabilities
        thresholds = np.sort(unique_probs)
        logger.info(f"Using {len(thresholds)} unique probability thresholds")
    
    # Evaluate MCC for each threshold
    best_mcc = -np.inf
    best_threshold = 0.5  # Default fallback
    
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        mcc = matthews_corrcoef(y_true, y_pred)
        
        if mcc > best_mcc:
            best_mcc = mcc
            best_threshold = threshold
    
    logger.info(f"Best threshold: {best_threshold:.6f}, Best MCC: {best_mcc:.6f}")
    
    return best_threshold, best_mcc


def find_best_threshold_mcc_grid(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold_min: float = 0.01,
    threshold_max: float = 0.99,
    step: float = 0.01
) -> Tuple[float, float]:
    """
    Find best threshold using a fixed grid (alternative implementation).
    
    Args:
        y_true: True binary labels (0/1)
        y_prob: Predicted probabilities
        threshold_min: Minimum threshold to test
        threshold_max: Maximum threshold to test
        step: Step size for grid
    
    Returns:
        Tuple of (best_threshold, best_mcc)
    """
    thresholds = np.arange(threshold_min, threshold_max + step, step)
    
    best_mcc = -np.inf
    best_threshold = 0.5
    
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        mcc = matthews_corrcoef(y_true, y_pred)
        
        if mcc > best_mcc:
            best_mcc = mcc
            best_threshold = threshold
    
    return best_threshold, best_mcc

