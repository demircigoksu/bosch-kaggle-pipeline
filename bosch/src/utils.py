"""
Utility functions for logging, metrics reporting, and artifact saving.
"""

import json
import pickle
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import (
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

logger = logging.getLogger(__name__)


def setup_logging(log_dir: Optional[Path] = None, level: int = logging.INFO):
    """
    Setup logging configuration.
    
    Args:
        log_dir: Directory to save log files (optional)
        level: Logging level
    """
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)
    
    # File handler (if log_dir provided)
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"bosch_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_file}")


def set_random_seeds(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    
    # Try to set seeds for other libraries
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
    
    logger.info(f"Random seeds set to {seed}")


def save_oof_predictions(
    oof_preds: np.ndarray,
    oof_labels: np.ndarray,
    output_dir: Path,
    filename: str = "oof_predictions.csv"
):
    """
    Save out-of-fold predictions.
    
    Args:
        oof_preds: OOF probability predictions
        oof_labels: True labels
        output_dir: Output directory
        filename: Output filename
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    
    df = pd.DataFrame({
        'y_true': oof_labels,
        'y_pred_proba': oof_preds
    })
    df.to_csv(output_path, index=False)
    logger.info(f"Saved OOF predictions to {output_path}")


def save_feature_list(
    feature_names: list,
    output_dir: Path,
    filename: str = "feature_list.json"
):
    """
    Save feature list.
    
    Args:
        feature_names: List of feature names
        output_dir: Output directory
        filename: Output filename
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    
    with open(output_path, 'w') as f:
        json.dump(feature_names, f, indent=2)
    
    logger.info(f"Saved feature list ({len(feature_names)} features) to {output_path}")


def save_config_snapshot(
    config: Dict[str, Any],
    output_dir: Path,
    filename: str = "config_snapshot.yaml"
):
    """
    Save configuration snapshot.
    
    Args:
        config: Configuration dictionary
        output_dir: Output directory
        filename: Output filename
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"Saved config snapshot to {output_path}")


def save_model(
    model: Any,
    output_dir: Path,
    filename: str = "final_model.pkl"
):
    """
    Save trained model.
    
    Args:
        model: Trained model object
        output_dir: Output directory
        filename: Output filename
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    
    with open(output_path, 'wb') as f:
        pickle.dump(model, f)
    
    logger.info(f"Saved model to {output_path}")


def save_threshold(
    threshold: float,
    output_dir: Path,
    filename: str = "optimal_threshold.pkl"
):
    """
    Save optimal threshold.
    
    Args:
        threshold: Optimal threshold value
        output_dir: Output directory
        filename: Output filename
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    
    with open(output_path, 'wb') as f:
        pickle.dump(threshold, f)
    
    logger.info(f"Saved optimal threshold ({threshold:.6f}) to {output_path}")


def compute_metrics_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray] = None,
    threshold: Optional[float] = None
) -> Dict[str, Any]:
    """
    Compute comprehensive metrics report.
    
    Args:
        y_true: True labels
        y_pred: Binary predictions
        y_pred_proba: Probability predictions (optional)
        threshold: Threshold used (optional)
    
    Returns:
        Dictionary of metrics
    """
    mcc = matthews_corrcoef(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    report = {
        'mcc': float(mcc),
        'confusion_matrix': {
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp)
        },
        'accuracy': float((tp + tn) / (tp + tn + fp + fn)) if (tp + tn + fp + fn) > 0 else 0.0,
        'precision': float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
        'recall': float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        'f1_score': float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0,
    }
    
    if threshold is not None:
        report['threshold'] = float(threshold)
    
    if y_pred_proba is not None:
        from sklearn.metrics import roc_auc_score
        try:
            report['roc_auc'] = float(roc_auc_score(y_true, y_pred_proba))
        except ValueError:
            report['roc_auc'] = None
    
    return report


def save_metrics_report(
    metrics: Dict[str, Any],
    output_dir: Path,
    filename_json: str = "metrics_report.json",
    filename_md: str = "metrics_report.md"
):
    """
    Save metrics report in JSON and Markdown formats.
    
    Args:
        metrics: Metrics dictionary
        output_dir: Output directory
        filename_json: JSON filename
        filename_md: Markdown filename
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save JSON
    json_path = output_dir / filename_json
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics report (JSON) to {json_path}")
    
    # Save Markdown
    md_path = output_dir / filename_md
    with open(md_path, 'w') as f:
        f.write("# Metrics Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Summary\n\n")
        if 'mcc' in metrics:
            f.write(f"- **Matthews Correlation Coefficient (MCC):** {metrics['mcc']:.6f}\n")
        if 'roc_auc' in metrics and metrics['roc_auc'] is not None:
            f.write(f"- **ROC-AUC:** {metrics['roc_auc']:.6f}\n")
        if 'f1_score' in metrics:
            f.write(f"- **F1-Score:** {metrics['f1_score']:.6f}\n")
        if 'precision' in metrics:
            f.write(f"- **Precision:** {metrics['precision']:.6f}\n")
        if 'recall' in metrics:
            f.write(f"- **Recall:** {metrics['recall']:.6f}\n")
        if 'threshold' in metrics:
            f.write(f"- **Optimal Threshold:** {metrics['threshold']:.6f}\n")
        
        f.write("\n## Confusion Matrix\n\n")
        if 'confusion_matrix' in metrics:
            cm = metrics['confusion_matrix']
            f.write("| | Predicted: 0 | Predicted: 1 |\n")
            f.write("|---|---|---|\n")
            f.write(f"| **Actual: 0** | {cm['tn']:,} (TN) | {cm['fp']:,} (FP) |\n")
            f.write(f"| **Actual: 1** | {cm['fn']:,} (FN) | {cm['tp']:,} (TP) |\n")
        
        f.write("\n## Detailed Metrics\n\n")
        for key, value in metrics.items():
            if key != 'confusion_matrix':
                if isinstance(value, float):
                    f.write(f"- **{key}:** {value:.6f}\n")
                else:
                    f.write(f"- **{key}:** {value}\n")
    
    logger.info(f"Saved metrics report (Markdown) to {md_path}")


def save_cv_results(
    cv_results: Dict[str, Any],
    output_dir: Path,
    filename: str = "cv_results.json"
):
    """
    Save cross-validation results.
    
    Args:
        cv_results: CV results dictionary
        output_dir: Output directory
        filename: Output filename
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj
    
    cv_results_serializable = convert_numpy(cv_results)
    
    with open(output_path, 'w') as f:
        json.dump(cv_results_serializable, f, indent=2)
    
    logger.info(f"Saved CV results to {output_path}")

