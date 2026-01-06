"""
Inference and submission generation for Bosch Kaggle competition.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import logging
import pickle
import joblib

from .data_loader import load_kaggle_data
from .feature_engineering import FeatureEngineeringPipeline
from .utils import setup_logging, set_random_seeds

logger = logging.getLogger(__name__)


def load_trained_model(model_dir: Path):
    """
    Load trained model and related artifacts.
    
    Args:
        model_dir: Directory containing saved model files
    
    Returns:
        Dictionary with model, feature_pipeline, threshold, feature_names
    """
    logger.info(f"Loading model from {model_dir}")
    
    model_path = model_dir / "final_model.pkl"
    threshold_path = model_dir / "optimal_threshold.pkl"
    feature_list_path = model_dir / "feature_list.json"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = joblib.load(model_path)
    logger.info(f"Loaded model: {type(model).__name__}")
    
    threshold = None
    if threshold_path.exists():
        threshold = pickle.load(open(threshold_path, 'rb'))
        logger.info(f"Loaded threshold: {threshold:.6f}")
    else:
        logger.warning(f"Threshold file not found: {threshold_path}, using default 0.5")
        threshold = 0.5
    
    feature_names = None
    if feature_list_path.exists():
        import json
        with open(feature_list_path, 'r') as f:
            feature_names = json.load(f)
        logger.info(f"Loaded feature list: {len(feature_names)} features")
    
    return {
        'model': model,
        'threshold': threshold,
        'feature_names': feature_names
    }


def generate_submission(
    model: any,
    X_test: pd.DataFrame,
    test_ids: pd.Series,
    threshold: float,
    output_path: Path
) -> pd.DataFrame:
    """
    Generate submission CSV in exact format: Id,Response.
    
    Args:
        model: Trained model
        X_test: Test features
        test_ids: Test IDs
        threshold: Optimal threshold
        output_path: Path to save submission CSV
    
    Returns:
        Submission dataframe
    """
    logger.info("Generating submission...")
    logger.info(f"  Test shape: {X_test.shape}")
    logger.info(f"  Threshold: {threshold:.6f}")
    
    # Predict probabilities
    test_proba = model.predict_proba(X_test)[:, 1]
    
    # Convert to binary using threshold
    test_predictions = (test_proba >= threshold).astype(int)
    
    # Create submission dataframe
    submission = pd.DataFrame({
        'Id': test_ids.values,
        'Response': test_predictions
    })
    
    # Validate format
    assert list(submission.columns) == ['Id', 'Response'], \
        f"Columns must be ['Id', 'Response'], got {list(submission.columns)}"
    assert submission['Response'].isin([0, 1]).all(), \
        "Response must be binary 0/1"
    assert len(submission) == len(test_ids), \
        f"Row count mismatch: {len(submission)} vs {len(test_ids)}"
    
    # Save submission
    submission.to_csv(output_path, index=False)
    logger.info(f"Saved submission to {output_path}")
    logger.info(f"  Rows: {len(submission):,}")
    logger.info(f"  Positive predictions: {submission['Response'].sum():,}")
    logger.info(f"  Negative predictions: {(submission['Response'] == 0).sum():,}")
    
    # Print first few lines
    logger.info("\nFirst 5 lines of submission:")
    logger.info("\n" + submission.head().to_string(index=False))
    
    return submission


def generate_submission_from_trained(
    data_dir: Path,
    model_dir: Path,
    output_dir: Path,
    config: dict,
    submission_filename: str = "submission.csv"
) -> pd.DataFrame:
    """
    Complete pipeline: load model, load test data, generate submission.
    
    Args:
        data_dir: Directory containing Kaggle input files
        model_dir: Directory containing trained model
        output_dir: Directory to save submission
        config: Configuration dictionary
        submission_filename: Output filename
    
    Returns:
        Submission dataframe
    """
    logger.info("=" * 60)
    logger.info("Generating Submission")
    logger.info("=" * 60)
    
    set_random_seeds(config.get('random_seed', 42))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    artifacts = load_trained_model(model_dir)
    model = artifacts['model']
    threshold = artifacts['threshold']
    feature_names = artifacts['feature_names']
    
    # Load test data
    logger.info("\nLoading test data...")
    _, _, X_test, test_ids = load_kaggle_data(
        data_dir=data_dir,
        chunk_size=config.get('chunk_size', 50000),
        load_categorical=config.get('load_categorical', True),
        load_date=config.get('load_date', True)
    )
    
    # Load feature pipeline (saved during training)
    logger.info("\nLoading feature pipeline...")
    feature_pipeline_path = model_dir / "feature_pipeline.pkl"
    if feature_pipeline_path.exists():
        feature_pipeline = joblib.load(feature_pipeline_path)
        logger.info("Loaded fitted feature pipeline from training")
    else:
        logger.error(f"Feature pipeline not found at {feature_pipeline_path}")
        logger.error("Please ensure training completed successfully and feature pipeline was saved.")
        raise FileNotFoundError(f"Feature pipeline not found: {feature_pipeline_path}")
    
    X_test_fe = feature_pipeline.transform(X_test)
    
    # Handle missing values
    X_test_fe = X_test_fe.fillna(0.0).astype('float32')
    
    # Ensure feature order matches training
    if feature_names:
        missing_cols = set(feature_names) - set(X_test_fe.columns)
        if missing_cols:
            logger.warning(f"Missing features: {missing_cols}, filling with 0")
            for col in missing_cols:
                X_test_fe[col] = 0.0
        X_test_fe = X_test_fe[feature_names]
    
    # Generate submission
    submission_path = output_dir / submission_filename
    submission = generate_submission(
        model=model,
        X_test=X_test_fe,
        test_ids=test_ids,
        threshold=threshold,
        output_path=submission_path
    )
    
    logger.info("\n" + "=" * 60)
    logger.info("Submission Generated Successfully!")
    logger.info("=" * 60)
    
    return submission

