"""
Training pipeline with Stratified K-Fold CV and MCC optimization.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
import joblib

from .data_loader import load_kaggle_data
from .feature_engineering import FeatureEngineeringPipeline
from .threshold_optimizer import find_best_threshold_mcc
from .utils import (
    setup_logging,
    set_random_seeds,
    save_oof_predictions,
    save_feature_list,
    save_config_snapshot,
    save_model,
    save_threshold,
    save_metrics_report,
    save_cv_results,
    compute_metrics_report
)

logger = logging.getLogger(__name__)


def train_bosch_pipeline(
    data_dir: Path,
    output_dir: Path,
    config: Dict[str, Any],
    n_folds: int = 5,
    quick_test: bool = False
) -> Dict[str, Any]:
    """
    Main training pipeline with CV and MCC optimization.
    
    Args:
        data_dir: Directory containing Kaggle input files
        output_dir: Directory to save outputs
        config: Configuration dictionary
        n_folds: Number of CV folds
        quick_test: If True, run only 1 fold for quick validation
    
    Returns:
        Dictionary with training results
    """
    logger.info("=" * 60)
    logger.info("Bosch Kaggle Pipeline - Training")
    logger.info("=" * 60)
    
    # Setup
    set_random_seeds(config.get('random_seed', 42))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config snapshot
    save_config_snapshot(config, output_dir)
    
    # Load data
    logger.info("\n1. Loading data...")
    X_train, y_train, X_test, test_ids = load_kaggle_data(
        data_dir=data_dir,
        chunk_size=config.get('chunk_size', 50000),
        load_categorical=config.get('load_categorical', True),
        load_date=config.get('load_date', True)
    )
    
    logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    logger.info(f"Class distribution: {y_train.value_counts().to_dict()}")
    
    # Feature engineering
    logger.info("\n2. Feature engineering...")
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
    
    logger.info(f"After feature engineering - Train: {X_train_fe.shape}, Test: {X_test_fe.shape}")
    
    # Save feature list
    save_feature_list(feature_pipeline.feature_names_, output_dir)
    
    # Handle missing values (fill with 0 for tree models)
    logger.info("\n3. Handling missing values...")
    X_train_fe = X_train_fe.fillna(0.0)
    X_test_fe = X_test_fe.fillna(0.0)
    
    # Convert to float32 to save memory
    X_train_fe = X_train_fe.astype('float32')
    X_test_fe = X_test_fe.astype('float32')
    
    # Cross-validation
    logger.info(f"\n4. Cross-validation ({n_folds} folds)...")
    
    if quick_test:
        n_folds = 1
        logger.info("QUICK TEST MODE: Running only 1 fold")
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config.get('random_seed', 42))
    
    oof_predictions = np.zeros(len(X_train_fe))
    fold_thresholds = []
    fold_mccs = []
    fold_models = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_fe, y_train), 1):
        logger.info(f"\n  Fold {fold}/{n_folds}")
        logger.info(f"  Train: {len(train_idx)}, Val: {len(val_idx)}")
        
        X_fold_train = X_train_fe.iloc[train_idx]
        y_fold_train = y_train.iloc[train_idx]
        X_fold_val = X_train_fe.iloc[val_idx]
        y_fold_val = y_train.iloc[val_idx]
        
        # Calculate scale_pos_weight for imbalance
        scale_pos_weight = (y_fold_train == 0).sum() / (y_fold_train == 1).sum()
        logger.info(f"  scale_pos_weight: {scale_pos_weight:.2f}")
        
        # Train model
        model_params = config.get('model_params', {})
        model = XGBClassifier(
            n_estimators=model_params.get('n_estimators', 200),
            max_depth=model_params.get('max_depth', 6),
            learning_rate=model_params.get('learning_rate', 0.05),
            scale_pos_weight=scale_pos_weight,
            min_child_weight=model_params.get('min_child_weight', 3),
            subsample=model_params.get('subsample', 0.8),
            colsample_bytree=model_params.get('colsample_bytree', 0.8),
            eval_metric='auc',
            random_state=config.get('random_seed', 42),
            n_jobs=-1,
            verbosity=0
        )
        
        logger.info("  Training model...")
        model.fit(X_fold_train, y_fold_train)
        
        # Predict on validation fold
        y_fold_val_proba = model.predict_proba(X_fold_val)[:, 1]
        oof_predictions[val_idx] = y_fold_val_proba
        
        # Find best threshold for this fold
        fold_threshold, fold_mcc = find_best_threshold_mcc(y_fold_val, y_fold_val_proba)
        fold_thresholds.append(fold_threshold)
        fold_mccs.append(fold_mcc)
        fold_models.append(model)
        
        logger.info(f"  Fold {fold} - Threshold: {fold_threshold:.6f}, MCC: {fold_mcc:.6f}")
        
        # Save fold model (optional, can be disabled for memory)
        if config.get('save_fold_models', False):
            save_model(model, output_dir / 'models', f"fold_{fold}_model.pkl")
    
    # Aggregate OOF predictions and find global threshold
    logger.info("\n5. Finding optimal threshold on OOF predictions...")
    global_threshold, global_mcc = find_best_threshold_mcc(y_train, oof_predictions)
    
    logger.info(f"  Global threshold: {global_threshold:.6f}")
    logger.info(f"  Global MCC (OOF): {global_mcc:.6f}")
    
    # Evaluate on OOF with global threshold
    oof_pred_binary = (oof_predictions >= global_threshold).astype(int)
    oof_metrics = compute_metrics_report(y_train, oof_pred_binary, oof_predictions, global_threshold)
    
    logger.info("\n6. CV Results Summary:")
    logger.info(f"  Fold MCCs: {[f'{m:.6f}' for m in fold_mccs]}")
    logger.info(f"  Mean MCC: {np.mean(fold_mccs):.6f}")
    logger.info(f"  Std MCC: {np.std(fold_mccs):.6f}")
    logger.info(f"  Global threshold: {global_threshold:.6f}")
    logger.info(f"  Global MCC (OOF): {global_mcc:.6f}")
    
    # Save OOF predictions
    save_oof_predictions(oof_predictions, y_train.values, output_dir / 'oof_predictions')
    
    # Save CV results
    cv_results = {
        'fold_mccs': fold_mccs,
        'fold_thresholds': fold_thresholds,
        'mean_mcc': float(np.mean(fold_mccs)),
        'std_mcc': float(np.std(fold_mccs)),
        'global_threshold': float(global_threshold),
        'global_mcc': float(global_mcc),
        'oof_metrics': oof_metrics
    }
    save_cv_results(cv_results, output_dir)
    
    # Save metrics report
    save_metrics_report(oof_metrics, output_dir)
    
    # Train final model on full training data
    logger.info("\n7. Training final model on full training data...")
    final_scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    final_model = XGBClassifier(
        n_estimators=model_params.get('n_estimators', 200),
        max_depth=model_params.get('max_depth', 6),
        learning_rate=model_params.get('learning_rate', 0.05),
        scale_pos_weight=final_scale_pos_weight,
        min_child_weight=model_params.get('min_child_weight', 3),
        subsample=model_params.get('subsample', 0.8),
        colsample_bytree=model_params.get('colsample_bytree', 0.8),
        eval_metric='auc',
        random_state=config.get('random_seed', 42),
        n_jobs=-1,
        verbosity=0
    )
    
    final_model.fit(X_train_fe, y_train)
    
    # Predict on test set
    logger.info("8. Predicting on test set...")
    test_predictions_proba = final_model.predict_proba(X_test_fe)[:, 1]
    test_predictions_binary = (test_predictions_proba >= global_threshold).astype(int)
    
    logger.info(f"  Test predictions shape: {test_predictions_binary.shape}")
    logger.info(f"  Test positive predictions: {test_predictions_binary.sum()}")
    
    # Save final model, threshold, and feature pipeline
    save_model(final_model, output_dir / 'models', 'final_model.pkl')
    save_model(feature_pipeline, output_dir / 'models', 'feature_pipeline.pkl')
    save_threshold(global_threshold, output_dir)
    
    # Save test predictions (probabilities) for later use
    test_pred_df = pd.DataFrame({
        'Id': test_ids.values,
        'Response': test_predictions_binary,
        'Probability': test_predictions_proba
    })
    test_pred_df.to_csv(output_dir / 'test_predictions.csv', index=False)
    logger.info(f"  Saved test predictions to {output_dir / 'test_predictions.csv'}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)
    
    return {
        'cv_results': cv_results,
        'final_model': final_model,
        'feature_pipeline': feature_pipeline,
        'global_threshold': global_threshold,
        'test_predictions': test_predictions_binary,
        'test_ids': test_ids,
        'test_predictions_proba': test_predictions_proba
    }

