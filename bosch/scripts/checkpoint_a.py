#!/usr/bin/env python
"""
Checkpoint A: Dry load check - Load only Id and Response from train_numeric.
"""

import sys
from pathlib import Path
import pandas as pd
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import setup_logging

# Setup paths
BOSCH_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BOSCH_DIR / "outputs"

if __name__ == "__main__":
    setup_logging(OUTPUT_DIR / "logs", level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("Checkpoint A: Dry Load Check")
    logger.info("=" * 60)
    
    # Get data directory
    if len(sys.argv) > 1:
        data_dir = Path(sys.argv[1])
    else:
        data_dir = Path("../input")
    
    if not data_dir.is_absolute():
        data_dir = BOSCH_DIR.parent / data_dir
    
    train_numeric_path = data_dir / "train_numeric.csv"
    test_numeric_path = data_dir / "test_numeric.csv"
    
    if not train_numeric_path.exists():
        logger.error(f"File not found: {train_numeric_path}")
        sys.exit(1)
    
    # Load only Id and Response from train
    logger.info(f"\nLoading Id and Response from {train_numeric_path}...")
    train_sample = pd.read_csv(train_numeric_path, usecols=['Id', 'Response'], nrows=100000)
    
    logger.info(f"\nTrain sample shape: {train_sample.shape}")
    logger.info(f"Train columns: {list(train_sample.columns)}")
    logger.info(f"\nResponse distribution:")
    logger.info(train_sample['Response'].value_counts().sort_index())
    
    if 'Response' in train_sample.columns:
        pos_count = (train_sample['Response'] == 1).sum()
        neg_count = (train_sample['Response'] == 0).sum()
        if pos_count > 0:
            imbalance_ratio = neg_count / pos_count
            logger.info(f"\nClass imbalance ratio: {imbalance_ratio:.1f}:1 (negative:positive)")
        else:
            logger.warning("No positive samples found in sample!")
    
    # Load only Id from test
    if test_numeric_path.exists():
        logger.info(f"\nLoading Id from {test_numeric_path}...")
        test_sample = pd.read_csv(test_numeric_path, usecols=['Id'], nrows=100000)
        logger.info(f"Test sample shape: {test_sample.shape}")
        logger.info(f"Test ID range: {test_sample['Id'].min()} to {test_sample['Id'].max()}")
    else:
        logger.warning(f"Test file not found: {test_numeric_path}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Checkpoint A: PASSED")
    logger.info("=" * 60)

