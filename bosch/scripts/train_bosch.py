#!/usr/bin/env python
"""
Main training script for Bosch Kaggle pipeline.
"""

import sys
from pathlib import Path
import yaml
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.train import train_bosch_pipeline
from src.utils import setup_logging

# Setup paths
BOSCH_DIR = Path(__file__).parent.parent
CONFIG_PATH = BOSCH_DIR / "configs" / "kaggle_config.yaml"
OUTPUT_DIR = BOSCH_DIR / "outputs"

if __name__ == "__main__":
    # Load config
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    setup_logging(OUTPUT_DIR / "logs", level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Get data directory from config or command line
    if len(sys.argv) > 1:
        data_dir = Path(sys.argv[1])
    else:
        data_dir = Path(config['data_dir'])
    
    # Resolve relative paths
    if not data_dir.is_absolute():
        data_dir = BOSCH_DIR.parent / data_dir
    
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    
    # Check for quick test flag
    quick_test = '--quick' in sys.argv or '-q' in sys.argv
    
    # Run training
    results = train_bosch_pipeline(
        data_dir=data_dir,
        output_dir=OUTPUT_DIR,
        config=config,
        n_folds=config.get('n_folds', 5),
        quick_test=quick_test
    )
    
    logger.info("\nTraining completed successfully!")
    logger.info(f"Results saved to: {OUTPUT_DIR}")

