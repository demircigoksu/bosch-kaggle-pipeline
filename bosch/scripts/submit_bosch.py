#!/usr/bin/env python
"""
Generate submission CSV from trained model.
"""

import sys
from pathlib import Path
import yaml
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference import generate_submission_from_trained
from src.utils import setup_logging

# Setup paths
BOSCH_DIR = Path(__file__).parent.parent
CONFIG_PATH = BOSCH_DIR / "configs" / "kaggle_config.yaml"
MODEL_DIR = BOSCH_DIR / "outputs" / "models"
OUTPUT_DIR = BOSCH_DIR / "outputs" / "submissions"

if __name__ == "__main__":
    # Load config
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    setup_logging(BOSCH_DIR / "outputs" / "logs", level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Get data directory from config or command line
    if len(sys.argv) > 1:
        data_dir = Path(sys.argv[1])
    else:
        data_dir = Path(config['data_dir'])
    
    # Resolve relative paths
    if not data_dir.is_absolute():
        data_dir = BOSCH_DIR.parent / data_dir
    
    # Get model directory
    if len(sys.argv) > 2:
        model_dir = Path(sys.argv[2])
    else:
        model_dir = MODEL_DIR
    
    if not model_dir.is_absolute():
        model_dir = BOSCH_DIR / model_dir
    
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Model directory: {model_dir}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    
    # Generate submission
    submission = generate_submission_from_trained(
        data_dir=data_dir,
        model_dir=model_dir,
        output_dir=OUTPUT_DIR,
        config=config,
        submission_filename="submission.csv"
    )
    
    logger.info("\nSubmission generated successfully!")
    logger.info(f"Submission saved to: {OUTPUT_DIR / 'submission.csv'}")

