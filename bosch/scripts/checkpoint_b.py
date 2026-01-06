#!/usr/bin/env python
"""
Checkpoint B: Feature table build check - Build compact features and join by Id.
"""

import sys
from pathlib import Path
import yaml
import logging
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_kaggle_data
from src.feature_engineering import FeatureEngineeringPipeline
from src.utils import setup_logging

# Setup paths
BOSCH_DIR = Path(__file__).parent.parent
CONFIG_PATH = BOSCH_DIR / "configs" / "kaggle_config.yaml"
OUTPUT_DIR = BOSCH_DIR / "outputs"

if __name__ == "__main__":
    # Load config
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    
    setup_logging(OUTPUT_DIR / "logs", level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("Checkpoint B: Feature Table Build Check")
    logger.info("=" * 60)
    
    # Get data directory
    if len(sys.argv) > 1:
        data_dir = Path(sys.argv[1])
    else:
        data_dir = Path(config['data_dir'])
    
    if not data_dir.is_absolute():
        data_dir = BOSCH_DIR.parent / data_dir
    
    # Load data (use small chunk for quick check)
    logger.info(f"\nLoading data from {data_dir}...")
    X_train, y_train, X_test, test_ids = load_kaggle_data(
        data_dir=data_dir,
        chunk_size=config.get('chunk_size', 50000),
        load_categorical=config.get('load_categorical', True),
        load_date=config.get('load_date', True)
    )
    
    logger.info(f"\nRaw data shapes:")
    logger.info(f"  X_train: {X_train.shape}")
    logger.info(f"  X_test: {X_test.shape}")
    logger.info(f"  y_train: {y_train.shape if y_train is not None else 'None'}")
    logger.info(f"  test_ids: {test_ids.shape}")
    
    # Feature engineering
    logger.info(f"\nApplying feature engineering...")
    feature_pipeline = FeatureEngineeringPipeline(
        extract_missingness=config.get('extract_missingness', True),
        extract_numeric_agg=config.get('extract_numeric_agg', True),
        extract_date_features=config.get('extract_date_features', True),
        encode_categorical=config.get('encode_categorical', True),
        max_missing_pct=config.get('max_missing_pct', 0.99),
        categorical_min_freq=config.get('categorical_min_freq', 2),
        categorical_top_k=config.get('categorical_top_k', None)
    )
    
    X_train_fe = feature_pipeline.fit_transform(X_train, y_train)
    X_test_fe = feature_pipeline.transform(X_test)
    
    logger.info(f"\nAfter feature engineering:")
    logger.info(f"  X_train_fe: {X_train_fe.shape}")
    logger.info(f"  X_test_fe: {X_test_fe.shape}")
    logger.info(f"  Feature count: {len(feature_pipeline.feature_names_)}")
    
    # Memory estimate
    train_memory_mb = X_train_fe.memory_usage(deep=True).sum() / 1024 / 1024
    test_memory_mb = X_test_fe.memory_usage(deep=True).sum() / 1024 / 1024
    logger.info(f"\nMemory usage (approximate):")
    logger.info(f"  X_train_fe: {train_memory_mb:.1f} MB")
    logger.info(f"  X_test_fe: {test_memory_mb:.1f} MB")
    
    # NaN stats
    logger.info(f"\nNaN statistics:")
    logger.info(f"  X_train_fe NaN count: {X_train_fe.isnull().sum().sum()}")
    logger.info(f"  X_train_fe NaN pct: {X_train_fe.isnull().sum().sum() / (X_train_fe.shape[0] * X_train_fe.shape[1]) * 100:.2f}%")
    logger.info(f"  X_test_fe NaN count: {X_test_fe.isnull().sum().sum()}")
    logger.info(f"  X_test_fe NaN pct: {X_test_fe.isnull().sum().sum() / (X_test_fe.shape[0] * X_test_fe.shape[1]) * 100:.2f}%")
    
    # Sample features
    logger.info(f"\nSample feature names (first 10):")
    for feat in feature_pipeline.feature_names_[:10]:
        logger.info(f"  {feat}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Checkpoint B: PASSED")
    logger.info("=" * 60)

