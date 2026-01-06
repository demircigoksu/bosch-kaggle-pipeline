#!/usr/bin/env python
"""
Checkpoint D: Submission format check - Validate submission CSV format.
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
SUBMISSION_PATH = OUTPUT_DIR / "submissions" / "submission.csv"

if __name__ == "__main__":
    setup_logging(OUTPUT_DIR / "logs", level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("Checkpoint D: Submission Format Check")
    logger.info("=" * 60)
    
    # Check if submission exists
    if not SUBMISSION_PATH.exists():
        logger.error(f"Submission file not found: {SUBMISSION_PATH}")
        logger.error("Please run submit_bosch.py first to generate submission.")
        sys.exit(1)
    
    # Load submission
    logger.info(f"\nLoading submission from {SUBMISSION_PATH}...")
    submission = pd.read_csv(SUBMISSION_PATH)
    
    # Validate format
    logger.info(f"\nSubmission shape: {submission.shape}")
    logger.info(f"Submission columns: {list(submission.columns)}")
    
    # Check header
    expected_columns = ['Id', 'Response']
    if list(submission.columns) != expected_columns:
        logger.error(f"Invalid columns! Expected {expected_columns}, got {list(submission.columns)}")
        sys.exit(1)
    else:
        logger.info("✓ Header is correct: ['Id', 'Response']")
    
    # Check Response values
    if not submission['Response'].isin([0, 1]).all():
        logger.error("Invalid Response values! Must be binary 0/1")
        logger.error(f"Unique values: {submission['Response'].unique()}")
        sys.exit(1)
    else:
        logger.info("✓ Response values are binary (0/1)")
    
    # Check for duplicates
    if submission['Id'].duplicated().any():
        logger.error("Duplicate Ids found!")
        sys.exit(1)
    else:
        logger.info("✓ No duplicate Ids")
    
    # Print statistics
    logger.info(f"\nSubmission statistics:")
    logger.info(f"  Total rows: {len(submission):,}")
    logger.info(f"  Positive predictions (Response=1): {submission['Response'].sum():,}")
    logger.info(f"  Negative predictions (Response=0): {(submission['Response'] == 0).sum():,}")
    logger.info(f"  Positive rate: {submission['Response'].mean() * 100:.2f}%")
    
    # Print first 5 lines
    logger.info(f"\nFirst 5 lines:")
    logger.info("\n" + submission.head().to_string(index=False))
    
    logger.info("\n" + "=" * 60)
    logger.info("Checkpoint D: PASSED")
    logger.info("=" * 60)
    logger.info(f"\nSubmission is valid and ready for Kaggle!")
    logger.info(f"File: {SUBMISSION_PATH}")

