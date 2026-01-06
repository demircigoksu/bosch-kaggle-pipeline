"""
Memory-safe data loading for Bosch Kaggle competition.
Handles numeric, categorical, and date files with chunked reading.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging
import os

logger = logging.getLogger(__name__)

# Try to import psutil, but handle gracefully if not available
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not available, memory logging will be disabled")


def get_memory_usage_mb() -> float:
    """Get current memory usage in MB."""
    if PSUTIL_AVAILABLE:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    return 0.0


def log_memory(message: str = ""):
    """Log current memory usage."""
    mem_mb = get_memory_usage_mb()
    logger.info(f"{message} Memory: {mem_mb:.1f} MB")


def load_numeric_chunked(
    file_path: Path,
    chunk_size: int = 50000,
    usecols: Optional[list] = None,
    is_train: bool = True
) -> pd.DataFrame:
    """
    Load numeric CSV with chunked reading and dtype optimization.
    
    Args:
        file_path: Path to numeric CSV file
        chunk_size: Number of rows per chunk
        usecols: Optional list of columns to read
        is_train: Whether this is training data (has Response column)
    
    Returns:
        DataFrame with optimized dtypes
    """
    logger.info(f"Loading numeric data from {file_path}")
    log_memory("Before loading numeric")
    
    chunks = []
    total_rows = 0
    
    # First pass: determine dtypes from sample
    sample = pd.read_csv(file_path, nrows=10000, usecols=usecols)
    
    # Optimize dtypes
    dtype_dict = {}
    for col in sample.columns:
        if col == 'Id':
            dtype_dict[col] = 'int32'
        elif col == 'Response' and is_train:
            dtype_dict[col] = 'int8'
        else:
            # Check if column can be downcast
            col_data = sample[col].dropna()
            if len(col_data) > 0:
                if pd.api.types.is_integer_dtype(col_data):
                    dtype_dict[col] = 'float32'  # Use float32 to preserve NaNs
                else:
                    dtype_dict[col] = 'float32'
            else:
                dtype_dict[col] = 'float32'
    
    # Read in chunks and consolidate periodically to save memory
    # Consolidate every 10 chunks to limit memory usage
    consolidate_every = 10
    chunks = []
    df = None
    
    for chunk_num, chunk in enumerate(pd.read_csv(
        file_path,
        chunksize=chunk_size,
        dtype=dtype_dict,
        usecols=usecols
    ), 1):
        chunks.append(chunk)
        total_rows += len(chunk)
        
        # Periodically consolidate chunks to free memory
        if len(chunks) >= consolidate_every:
            consolidated = pd.concat(chunks, ignore_index=True)
            chunks = []  # Clear chunks list to free memory
            
            if df is None:
                df = consolidated
            else:
                df = pd.concat([df, consolidated], ignore_index=True)
            
            # Force garbage collection hint
            del consolidated
        
        if chunk_num % 10 == 0:
            logger.info(f"  Processed {total_rows:,} rows...")
            log_memory(f"  After chunk {chunk_num}")
    
    # Concatenate any remaining chunks
    if chunks:
        consolidated = pd.concat(chunks, ignore_index=True)
        chunks = []
        if df is None:
            df = consolidated
        else:
            df = pd.concat([df, consolidated], ignore_index=True)
        del consolidated
    
    logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    log_memory("After loading numeric")
    
    return df


def load_categorical_chunked(
    file_path: Path,
    chunk_size: int = 50000,
    usecols: Optional[list] = None
) -> pd.DataFrame:
    """
    Load categorical CSV with chunked reading.
    Reads as string to avoid huge object memory initially.
    
    Args:
        file_path: Path to categorical CSV file
        chunk_size: Number of rows per chunk
        usecols: Optional list of columns to read
    
    Returns:
        DataFrame with Id and categorical columns
    """
    logger.info(f"Loading categorical data from {file_path}")
    log_memory("Before loading categorical")
    
    total_rows = 0
    
    # Read Id as int32, rest as string
    dtype_dict = {'Id': 'int32'}
    
    # Read in chunks and consolidate periodically to save memory
    # Consolidate every 10 chunks to limit memory usage
    consolidate_every = 10
    chunks = []
    df = None
    
    for chunk_num, chunk in enumerate(pd.read_csv(
        file_path,
        chunksize=chunk_size,
        dtype=dtype_dict,
        usecols=usecols
    ), 1):
        # Convert categorical columns to string (they come as objects)
        for col in chunk.columns:
            if col != 'Id' and chunk[col].dtype == 'object':
                chunk[col] = chunk[col].astype('string')
        
        chunks.append(chunk)
        total_rows += len(chunk)
        
        # Periodically consolidate chunks to free memory
        if len(chunks) >= consolidate_every:
            consolidated = pd.concat(chunks, ignore_index=True)
            chunks = []  # Clear chunks list to free memory
            
            if df is None:
                df = consolidated
            else:
                df = pd.concat([df, consolidated], ignore_index=True)
            
            # Force garbage collection hint
            del consolidated
        
        if chunk_num % 10 == 0:
            logger.info(f"  Processed {total_rows:,} rows...")
            log_memory(f"  After chunk {chunk_num}")
    
    # Concatenate any remaining chunks
    if chunks:
        consolidated = pd.concat(chunks, ignore_index=True)
        chunks = []
        if df is None:
            df = consolidated
        else:
            df = pd.concat([df, consolidated], ignore_index=True)
        del consolidated
    
    logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    log_memory("After loading categorical")
    
    return df


def load_date_chunked(
    file_path: Path,
    chunk_size: int = 50000,
    usecols: Optional[list] = None
) -> pd.DataFrame:
    """
    Load date CSV with chunked reading and float32 dtype.
    
    Args:
        file_path: Path to date CSV file
        chunk_size: Number of rows per chunk
        usecols: Optional list of columns to read
    
    Returns:
        DataFrame with optimized dtypes
    """
    logger.info(f"Loading date data from {file_path}")
    log_memory("Before loading date")
    
    total_rows = 0
    
    # Use float32 for all date columns (preserves NaNs)
    dtype_dict = {'Id': 'int32'}
    
    # Read in chunks and consolidate periodically to save memory
    # Consolidate every 10 chunks to limit memory usage
    consolidate_every = 10
    chunks = []
    df = None
    
    for chunk_num, chunk in enumerate(pd.read_csv(
        file_path,
        chunksize=chunk_size,
        dtype=dtype_dict,
        usecols=usecols
    ), 1):
        # Convert date columns to float32
        for col in chunk.columns:
            if col != 'Id':
                chunk[col] = pd.to_numeric(chunk[col], errors='coerce').astype('float32')
        
        chunks.append(chunk)
        total_rows += len(chunk)
        
        # Periodically consolidate chunks to free memory
        if len(chunks) >= consolidate_every:
            consolidated = pd.concat(chunks, ignore_index=True)
            chunks = []  # Clear chunks list to free memory
            
            if df is None:
                df = consolidated
            else:
                df = pd.concat([df, consolidated], ignore_index=True)
            
            # Force garbage collection hint
            del consolidated
        
        if chunk_num % 10 == 0:
            logger.info(f"  Processed {total_rows:,} rows...")
            log_memory(f"  After chunk {chunk_num}")
    
    # Concatenate any remaining chunks
    if chunks:
        consolidated = pd.concat(chunks, ignore_index=True)
        chunks = []
        if df is None:
            df = consolidated
        else:
            df = pd.concat([df, consolidated], ignore_index=True)
        del consolidated
    
    logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    log_memory("After loading date")
    
    return df


def merge_by_id(
    dfs: list,
    on: str = 'Id',
    how: str = 'outer'
) -> pd.DataFrame:
    """
    Merge multiple dataframes on Id column safely.
    
    Args:
        dfs: List of dataframes to merge
        on: Column name to merge on
        how: Merge type ('inner', 'outer', 'left', 'right')
    
    Returns:
        Merged dataframe
    """
    logger.info(f"Merging {len(dfs)} dataframes on '{on}'")
    log_memory("Before merge")
    
    result = dfs[0]
    for i, df in enumerate(dfs[1:], 1):
        result = result.merge(df, on=on, how=how, suffixes=('', f'_dup{i}'))
        logger.info(f"  Merged dataframe {i+1}/{len(dfs)}")
        log_memory(f"  After merge {i+1}")
    
    logger.info(f"Merged shape: {result.shape}")
    log_memory("After merge")
    
    return result


def load_kaggle_data(
    data_dir: Path,
    chunk_size: int = 50000,
    load_categorical: bool = True,
    load_date: bool = True
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Main entry point: Load all Kaggle data files and return X_train, y_train, X_test, test_ids.
    
    Args:
        data_dir: Directory containing Kaggle input files
        chunk_size: Chunk size for reading
        load_categorical: Whether to load categorical files
        load_date: Whether to load date files
    
    Returns:
        Tuple of (X_train, y_train, X_test, test_ids)
    """
    logger.info("=" * 60)
    logger.info("Loading Kaggle Bosch Data")
    logger.info("=" * 60)
    
    # File paths
    train_numeric_path = data_dir / "train_numeric.csv"
    test_numeric_path = data_dir / "test_numeric.csv"
    train_categorical_path = data_dir / "train_categorical.csv"
    test_categorical_path = data_dir / "test_categorical.csv"
    train_date_path = data_dir / "train_date.csv"
    test_date_path = data_dir / "test_date.csv"
    
    # Check required files exist
    required_files = [train_numeric_path, test_numeric_path]
    for f in required_files:
        if not f.exists():
            raise FileNotFoundError(f"Required file not found: {f}")
    
    # Load numeric data (required)
    logger.info("\n1. Loading numeric data...")
    train_numeric = load_numeric_chunked(train_numeric_path, chunk_size, is_train=True)
    test_numeric = load_numeric_chunked(test_numeric_path, chunk_size, is_train=False)
    
    # Extract Id and Response from train
    train_ids = train_numeric['Id'].copy()
    y_train = train_numeric['Response'].copy() if 'Response' in train_numeric.columns else None
    train_numeric = train_numeric.drop(columns=['Id', 'Response'], errors='ignore')
    
    test_ids = test_numeric['Id'].copy()
    test_numeric = test_numeric.drop(columns=['Id'], errors='ignore')
    
    logger.info(f"Train numeric: {train_numeric.shape}, Test numeric: {test_numeric.shape}")
    
    # Load categorical data (optional)
    train_dfs = [train_numeric]
    test_dfs = [test_numeric]
    
    if load_categorical and train_categorical_path.exists() and test_categorical_path.exists():
        logger.info("\n2. Loading categorical data...")
        train_categorical = load_categorical_chunked(train_categorical_path, chunk_size)
        test_categorical = load_categorical_chunked(test_categorical_path, chunk_size)
        
        # Remove Id from categorical (will merge on it)
        train_cat_features = train_categorical.drop(columns=['Id'], errors='ignore')
        test_cat_features = test_categorical.drop(columns=['Id'], errors='ignore')
        
        train_dfs.append(train_cat_features)
        test_dfs.append(test_cat_features)
        logger.info(f"Train categorical: {train_cat_features.shape}, Test categorical: {test_cat_features.shape}")
    else:
        logger.info("\n2. Skipping categorical data (files not found or disabled)")
    
    # Load date data (optional)
    if load_date and train_date_path.exists() and test_date_path.exists():
        logger.info("\n3. Loading date data...")
        train_date = load_date_chunked(train_date_path, chunk_size)
        test_date = load_date_chunked(test_date_path, chunk_size)
        
        # Remove Id from date (will merge on it)
        train_date_features = train_date.drop(columns=['Id'], errors='ignore')
        test_date_features = test_date.drop(columns=['Id'], errors='ignore')
        
        train_dfs.append(train_date_features)
        test_dfs.append(test_date_features)
        logger.info(f"Train date: {train_date_features.shape}, Test date: {test_date_features.shape}")
    else:
        logger.info("\n3. Skipping date data (files not found or disabled)")
    
    # Merge all features
    logger.info("\n4. Merging features...")
    # Add Id back for merging
    train_dfs[0] = pd.concat([train_ids.to_frame(), train_dfs[0]], axis=1)
    test_dfs[0] = pd.concat([test_ids.to_frame(), test_dfs[0]], axis=1)
    
    X_train = merge_by_id(train_dfs, on='Id', how='inner')
    X_test = merge_by_id(test_dfs, on='Id', how='inner')
    
    # Remove Id from features (keep separate)
    train_ids_final = X_train['Id'].copy()
    test_ids_final = X_test['Id'].copy()
    X_train = X_train.drop(columns=['Id'], errors='ignore')
    X_test = X_test.drop(columns=['Id'], errors='ignore')
    
    logger.info("\n" + "=" * 60)
    logger.info("Data Loading Complete")
    logger.info("=" * 60)
    logger.info(f"X_train shape: {X_train.shape}")
    logger.info(f"X_test shape: {X_test.shape}")
    if y_train is not None:
        logger.info(f"y_train distribution:\n{y_train.value_counts().sort_index()}")
        logger.info(f"Class imbalance ratio: {(y_train == 0).sum() / (y_train == 1).sum():.1f}:1")
    log_memory("Final")
    
    return X_train, y_train, X_test, test_ids_final

