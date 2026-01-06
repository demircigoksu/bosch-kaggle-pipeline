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
import gc
import zipfile

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


def check_memory_available_mb() -> Optional[float]:
    """Check available system memory in MB. Returns None if psutil not available."""
    if PSUTIL_AVAILABLE:
        mem = psutil.virtual_memory()
        return mem.available / 1024 / 1024
    return None


def is_kaggle_environment() -> bool:
    """Check if running in Kaggle notebook environment."""
    import os
    return os.path.exists('/kaggle/input') or 'KAGGLE' in os.environ


def get_total_memory_mb() -> Optional[float]:
    """Get total system memory in MB."""
    if PSUTIL_AVAILABLE:
        mem = psutil.virtual_memory()
        return mem.total / 1024 / 1024
    return None


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
    
    # Check file size
    file_size_mb = file_path.stat().st_size / (1024 * 1024)
    logger.info(f"File size: {file_size_mb:.1f} MB")
    
    # Check available memory and environment
    available_mem = check_memory_available_mb()
    total_mem = get_total_memory_mb()
    is_kaggle = is_kaggle_environment()
    
    if available_mem is not None:
        logger.info(f"Available system memory: {available_mem:.1f} MB")
        if total_mem is not None:
            logger.info(f"Total system memory: {total_mem:.1f} MB")
        if is_kaggle:
            logger.info("Running in Kaggle environment - using optimized settings")
        if available_mem < 2000:
            logger.warning(f"Low available memory ({available_mem:.1f} MB). Consider reducing chunk_size.")
    
    log_memory("Before loading numeric")
    
    chunks = []
    total_rows = 0
    
    # First pass: determine dtypes from sample
    try:
        sample = pd.read_csv(file_path, nrows=10000, usecols=usecols)
    except Exception as e:
        logger.error(f"Failed to read sample from {file_path}: {str(e)}")
        raise
    
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
    
    # Clear sample to free memory
    del sample
    gc.collect()
    
    # Read in chunks and consolidate very frequently to save memory
    # Use smaller consolidate_every for large files to minimize peak memory
    # For files > 1GB, consolidate every 2 chunks for maximum memory efficiency
    if file_size_mb > 1000:
        consolidate_every = 2
        logger.info(f"Large file detected, using very aggressive consolidation (every {consolidate_every} chunks)")
    elif file_size_mb > 500:
        consolidate_every = 3
    else:
        consolidate_every = 5
    chunks = []
    df = None
    
    try:
        for chunk_num, chunk in enumerate(pd.read_csv(
            file_path,
            chunksize=chunk_size,
            dtype=dtype_dict,
            usecols=usecols
        ), 1):
            try:
                chunks.append(chunk)
                total_rows += len(chunk)
                
                # Periodically consolidate chunks to free memory (more frequently)
                if len(chunks) >= consolidate_every:
                    logger.debug(f"  Consolidating chunks at chunk {chunk_num}...")
                    consolidated = pd.concat(chunks, ignore_index=True)
                    chunks = []  # Clear chunks list to free memory
                    
                    if df is None:
                        df = consolidated
                    else:
                        # More memory-efficient concatenation: append in place when possible
                        # Use concat but immediately delete the old df to free memory
                        old_df = df
                        df = pd.concat([df, consolidated], ignore_index=True)
                        del old_df, consolidated
                        # Force garbage collection immediately to free old DataFrame memory
                        gc.collect()
                    
                    # Additional cleanup if not already done
                    if 'consolidated' in locals():
                        del consolidated
                        gc.collect()
                    
                    # Check memory after consolidation and force more aggressive cleanup if needed
                    current_mem = get_memory_usage_mb()
                    if current_mem > 3500:  # Warn if over 3.5GB
                        logger.warning(f"High memory usage: {current_mem:.1f} MB after chunk {chunk_num}")
                        # Force multiple garbage collection passes
                        for _ in range(2):
                            gc.collect()
                
                # Log progress more frequently for large files
                if chunk_num % 5 == 0:
                    logger.info(f"  Processed {total_rows:,} rows...")
                    log_memory(f"  After chunk {chunk_num}")
                    
            except Exception as e:
                logger.error(f"Error processing chunk {chunk_num}: {str(e)}")
                logger.exception("Full traceback:")
                raise
        
        # Concatenate any remaining chunks
        if chunks:
            logger.debug(f"  Consolidating remaining {len(chunks)} chunks...")
            consolidated = pd.concat(chunks, ignore_index=True)
            chunks = []
            if df is None:
                df = consolidated
            else:
                df = pd.concat([df, consolidated], ignore_index=True)
            del consolidated
            gc.collect()
        
        logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
        log_memory("After loading numeric")
        
        return df
        
    except Exception as e:
        logger.error(f"Fatal error in load_numeric_chunked: {str(e)}")
        logger.exception("Full traceback:")
        raise


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
    
    # Read in chunks and consolidate more frequently
    # Check file size for categorical data too
    file_size_mb = file_path.stat().st_size / (1024 * 1024)
    if file_size_mb > 2000:
        consolidate_every = 2
    elif file_size_mb > 1000:
        consolidate_every = 3
    else:
        consolidate_every = 5
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
            logger.debug(f"  Consolidating chunks at chunk {chunk_num}...")
            consolidated = pd.concat(chunks, ignore_index=True)
            chunks = []  # Clear chunks list to free memory
            
            if df is None:
                df = consolidated
            else:
                df = pd.concat([df, consolidated], ignore_index=True)
            
            # Force garbage collection
            del consolidated
            gc.collect()
        
        if chunk_num % 5 == 0:
            logger.info(f"  Processed {total_rows:,} rows...")
            log_memory(f"  After chunk {chunk_num}")
    
    # Concatenate any remaining chunks
    if chunks:
        logger.debug(f"  Consolidating remaining {len(chunks)} chunks...")
        consolidated = pd.concat(chunks, ignore_index=True)
        chunks = []
        if df is None:
            df = consolidated
        else:
            df = pd.concat([df, consolidated], ignore_index=True)
        del consolidated
        gc.collect()
    
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
    
    # Read in chunks and consolidate more frequently
    # Check file size for categorical data too
    file_size_mb = file_path.stat().st_size / (1024 * 1024)
    if file_size_mb > 2000:
        consolidate_every = 2
    elif file_size_mb > 1000:
        consolidate_every = 3
    else:
        consolidate_every = 5
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
            logger.debug(f"  Consolidating chunks at chunk {chunk_num}...")
            consolidated = pd.concat(chunks, ignore_index=True)
            chunks = []  # Clear chunks list to free memory
            
            if df is None:
                df = consolidated
            else:
                df = pd.concat([df, consolidated], ignore_index=True)
            
            # Force garbage collection
            del consolidated
            gc.collect()
        
        if chunk_num % 5 == 0:
            logger.info(f"  Processed {total_rows:,} rows...")
            log_memory(f"  After chunk {chunk_num}")
    
    # Concatenate any remaining chunks
    if chunks:
        logger.debug(f"  Consolidating remaining {len(chunks)} chunks...")
        consolidated = pd.concat(chunks, ignore_index=True)
        chunks = []
        if df is None:
            df = consolidated
        else:
            df = pd.concat([df, consolidated], ignore_index=True)
        del consolidated
        gc.collect()
    
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
        # Force garbage collection after each merge
        gc.collect()
    
    logger.info(f"Merged shape: {result.shape}")
    log_memory("After merge")
    
    return result


def _extract_zip_if_needed(data_dir: Path, csv_filename: str) -> Path:
    """
    Check if CSV file exists, if not check for zip file and extract it.
    Returns the path to the CSV file (either original or extracted).
    
    Args:
        data_dir: Directory containing data files
        csv_filename: Name of the CSV file to look for
    
    Returns:
        Path to the CSV file
    """
    csv_path = data_dir / csv_filename
    zip_path = data_dir / f"{csv_filename}.zip"
    
    # If CSV exists, return it directly
    if csv_path.exists():
        return csv_path
    
    # If CSV doesn't exist but zip does, extract it
    if zip_path.exists():
        logger.info(f"CSV file not found, extracting from {zip_path.name}...")
        
        # For Kaggle, /kaggle/input is read-only, so extract to /kaggle/working
        # For local, extract to data_dir/extracted
        if is_kaggle_environment():
            # Use /kaggle/working for extracted files
            extracted_dir = Path("/kaggle/working/extracted_data")
        else:
            # Use data_dir/extracted for local
            extracted_dir = data_dir / "extracted"
        
        extracted_dir.mkdir(parents=True, exist_ok=True)
        
        extracted_path = extracted_dir / csv_filename
        
        # Only extract if not already extracted
        if not extracted_path.exists():
            try:
                logger.info(f"  Extracting {zip_path.name} to {extracted_dir}...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extracted_dir)
                logger.info(f"  ✅ Extracted {csv_filename} successfully")
            except Exception as e:
                logger.error(f"Failed to extract {zip_path}: {str(e)}")
                raise
        
        return extracted_path
    
    # Neither CSV nor zip found
    return csv_path  # Return original path, will raise error later


def load_kaggle_data(
    data_dir: Path,
    chunk_size: int = 50000,
    load_categorical: bool = True,
    load_date: bool = True
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Main entry point: Load all Kaggle data files and return X_train, y_train, X_test, test_ids.
    
    Automatically handles both CSV files (local) and zip files (Kaggle).
    Automatically adjusts chunk_size based on available memory for large files.
    
    Args:
        data_dir: Directory containing Kaggle input files (CSV or zip)
        chunk_size: Chunk size for reading
        load_categorical: Whether to load categorical files
        load_date: Whether to load date files
    
    Returns:
        Tuple of (X_train, y_train, X_test, test_ids)
    """
    logger.info("=" * 60)
    logger.info("Loading Kaggle Bosch Data")
    logger.info("=" * 60)
    
    # Convert to Path if string
    data_dir = Path(data_dir)
    
    # Check if we need to extract zip files
    # File paths - try CSV first, then zip
    train_numeric_path = _extract_zip_if_needed(data_dir, "train_numeric.csv")
    test_numeric_path = _extract_zip_if_needed(data_dir, "test_numeric.csv")
    train_categorical_path = _extract_zip_if_needed(data_dir, "train_categorical.csv")
    test_categorical_path = _extract_zip_if_needed(data_dir, "test_categorical.csv")
    train_date_path = _extract_zip_if_needed(data_dir, "train_date.csv")
    test_date_path = _extract_zip_if_needed(data_dir, "test_date.csv")
    
    # Check required files exist
    required_files = [train_numeric_path, test_numeric_path]
    for f in required_files:
        if not f.exists():
            raise FileNotFoundError(f"Required file not found: {f}")
    
    # Auto-adjust chunk_size for large files to optimize memory usage
    train_file_size_mb = train_numeric_path.stat().st_size / (1024 * 1024)
    available_mem = check_memory_available_mb()
    total_mem = get_total_memory_mb()
    is_kaggle = is_kaggle_environment()
    
    # Kaggle notebooks have more memory (16-30GB), can use larger chunks
    if is_kaggle and total_mem and total_mem > 15000:
        # Kaggle environment with good memory - use larger chunks for efficiency
        if chunk_size < 50000:
            chunk_size = 50000
            logger.info(f"Kaggle environment detected ({total_mem:.1f} MB total), using chunk_size {chunk_size} for efficiency")
    elif train_file_size_mb > 2000:
        # Very large files (>2GB), use very small chunks
        if chunk_size > 20000:
            chunk_size = 20000
            logger.info(f"Very large file detected ({train_file_size_mb:.1f} MB), reducing chunk_size to {chunk_size} for memory optimization")
    elif available_mem is not None and train_file_size_mb > 1000:
        # For large files, use smaller chunks to reduce peak memory
        # Target: keep peak memory under 60% of available
        if chunk_size > 25000:
            chunk_size = 25000
            logger.info(f"Large file detected ({train_file_size_mb:.1f} MB), reducing chunk_size to {chunk_size} for memory optimization")
    
    # Load numeric data (required)
    logger.info("\n1. Loading numeric data...")
    try:
        train_numeric = load_numeric_chunked(train_numeric_path, chunk_size, is_train=True)
        logger.info("Train numeric loaded successfully")
        log_memory("After loading train numeric")
    except Exception as e:
        logger.error(f"Failed to load train numeric data: {str(e)}")
        logger.exception("Full traceback:")
        raise
    
    try:
        test_numeric = load_numeric_chunked(test_numeric_path, chunk_size, is_train=False)
        logger.info("Test numeric loaded successfully")
        log_memory("After loading test numeric")
    except Exception as e:
        logger.error(f"Failed to load test numeric data: {str(e)}")
        logger.exception("Full traceback:")
        raise
    
    # Extract Id and Response from train
    train_ids = train_numeric['Id'].copy()
    y_train = train_numeric['Response'].copy() if 'Response' in train_numeric.columns else None
    train_numeric = train_numeric.drop(columns=['Id', 'Response'], errors='ignore')
    
    test_ids = test_numeric['Id'].copy()
    test_numeric = test_numeric.drop(columns=['Id'], errors='ignore')
    
    logger.info(f"Train numeric: {train_numeric.shape}, Test numeric: {test_numeric.shape}")
    
    # Force garbage collection after dropping columns
    gc.collect()
    log_memory("After processing numeric data")
    
    # Load categorical data (optional)
    train_dfs = [train_numeric]
    test_dfs = [test_numeric]
    
    # Check if categorical files exist (already extracted if needed)
    if load_categorical and train_categorical_path.exists() and test_categorical_path.exists():
        logger.info("\n2. Loading categorical data...")
        train_categorical = load_categorical_chunked(train_categorical_path, chunk_size)
        log_memory("After loading train categorical")
        test_categorical = load_categorical_chunked(test_categorical_path, chunk_size)
        log_memory("After loading test categorical")
        
        # Remove Id from categorical (will merge on it)
        train_cat_features = train_categorical.drop(columns=['Id'], errors='ignore')
        test_cat_features = test_categorical.drop(columns=['Id'], errors='ignore')
        
        train_dfs.append(train_cat_features)
        test_dfs.append(test_cat_features)
        logger.info(f"Train categorical: {train_cat_features.shape}, Test categorical: {test_cat_features.shape}")
        
        # Free memory immediately
        del train_categorical, test_categorical
        gc.collect()
        log_memory("After processing categorical data")
    else:
        logger.info("\n2. Skipping categorical data (files not found or disabled)")
    
    # Load date data (optional)
    # Check if date files exist (already extracted if needed)
    if load_date and train_date_path.exists() and test_date_path.exists():
        logger.info("\n3. Loading date data...")
        train_date = load_date_chunked(train_date_path, chunk_size)
        log_memory("After loading train date")
        test_date = load_date_chunked(test_date_path, chunk_size)
        log_memory("After loading test date")
        
        # Remove Id from date (will merge on it)
        train_date_features = train_date.drop(columns=['Id'], errors='ignore')
        test_date_features = test_date.drop(columns=['Id'], errors='ignore')
        
        train_dfs.append(train_date_features)
        test_dfs.append(test_date_features)
        logger.info(f"Train date: {train_date_features.shape}, Test date: {test_date_features.shape}")
        
        # Free memory immediately
        del train_date, test_date
        gc.collect()
        log_memory("After processing date data")
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
