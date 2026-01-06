#!/usr/bin/env python
"""
Checkpoint C: Model CV check - Run 1 fold quickly to validate pipeline.
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
    
    setup_logging(OUTPUT_DIR / "logs", level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("Checkpoint C: Model CV Check (Quick - 1 Fold)")
    logger.info("=" * 60)
    
    # Get data directory
    if len(sys.argv) > 1:
        data_dir = Path(sys.argv[1])
    else:
        data_dir = Path(config['data_dir'])
    
    if not data_dir.is_absolute():
        data_dir = BOSCH_DIR.parent / data_dir
    
    # Run training with quick_test=True (1 fold)
    logger.info("\nRunning quick CV test (1 fold)...")
    results = train_bosch_pipeline(
        data_dir=data_dir,
        output_dir=OUTPUT_DIR,
        config=config,
        n_folds=1,  # Only 1 fold for quick check
        quick_test=True
    )
    
    cv_results = results['cv_results']
    logger.info("\n" + "=" * 60)
    logger.info("CV Results Summary:")
    logger.info("=" * 60)
    logger.info(f"Fold MCCs: {cv_results['fold_mccs']}")
    logger.info(f"Mean MCC: {cv_results['mean_mcc']:.6f}")
    logger.info(f"Global threshold: {cv_results['global_threshold']:.6f}")
    logger.info(f"Global MCC: {cv_results['global_mcc']:.6f}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Checkpoint C: PASSED")
    logger.info("=" * 60)

