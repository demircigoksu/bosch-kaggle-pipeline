"""
Feature engineering for Bosch Kaggle competition.
Handles missingness, numeric aggregations, date features, and categorical encoding.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
import re
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)


def parse_column_name(col_name: str) -> Optional[Dict[str, str]]:
    """
    Parse column name patterns like L3_S36_F3939, L0_S0_D1.
    
    Returns dict with keys: 'line', 'station', 'feature_type', 'feature_num'
    """
    # Pattern: L{line}_S{station}_{F|D}{number}
    pattern = r'L(\d+)_S(\d+)_([FD])(\d+)'
    match = re.match(pattern, col_name)
    if match:
        return {
            'line': match.group(1),
            'station': match.group(2),
            'feature_type': match.group(3),  # F for feature, D for date
            'feature_num': match.group(4),
            'line_station': f"L{match.group(1)}_S{match.group(2)}"
        }
    return None


class MissingnessFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract missingness features per row and per station/line grouping."""
    
    def __init__(self, extract_station_features: bool = True):
        self.extract_station_features = extract_station_features
        self.station_groups_ = None
    
    def fit(self, X: pd.DataFrame, y=None):
        """Identify station groups from column names."""
        if self.extract_station_features:
            station_groups = {}
            for col in X.columns:
                parsed = parse_column_name(col)
                if parsed:
                    station = parsed['line_station']
                    if station not in station_groups:
                        station_groups[station] = []
                    station_groups[station].append(col)
            self.station_groups_ = station_groups
            logger.info(f"Identified {len(station_groups)} station groups")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Extract missingness features."""
        X_new = X.copy()
        
        # Row-level missingness counts
        X_new['missing_count_numeric'] = X.select_dtypes(include=[np.number]).isnull().sum(axis=1)
        X_new['missing_count_categorical'] = X.select_dtypes(include=['object', 'string']).isnull().sum(axis=1)
        X_new['missing_count_date'] = X.select_dtypes(include=[np.number]).isnull().sum(axis=1)  # Dates are float32
        X_new['missing_count_total'] = X.isnull().sum(axis=1)
        X_new['missing_pct_total'] = X.isnull().sum(axis=1) / len(X.columns)
        
        # Station-level missingness (if enabled)
        if self.extract_station_features and self.station_groups_:
            for station, cols in self.station_groups_.items():
                station_cols = [c for c in cols if c in X.columns]
                if station_cols:
                    X_new[f'{station}_missing_count'] = X[station_cols].isnull().sum(axis=1)
                    X_new[f'{station}_missing_pct'] = X[station_cols].isnull().sum(axis=1) / len(station_cols)
        
        return X_new


class NumericAggregator(BaseEstimator, TransformerMixin):
    """Extract row-level aggregations from numeric columns."""
    
    def __init__(self, ignore_nan: bool = True):
        self.ignore_nan = ignore_nan
    
    def fit(self, X: pd.DataFrame, y=None):
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Extract numeric aggregations per row."""
        X_new = X.copy()
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            logger.warning("No numeric columns found for aggregation")
            return X_new
        
        numeric_df = X[numeric_cols]
        
        # Row-level statistics
        X_new['numeric_mean'] = numeric_df.mean(axis=1, skipna=self.ignore_nan)
        X_new['numeric_std'] = numeric_df.std(axis=1, skipna=self.ignore_nan)
        X_new['numeric_min'] = numeric_df.min(axis=1, skipna=self.ignore_nan)
        X_new['numeric_max'] = numeric_df.max(axis=1, skipna=self.ignore_nan)
        X_new['numeric_median'] = numeric_df.median(axis=1, skipna=self.ignore_nan)
        X_new['numeric_sum'] = numeric_df.sum(axis=1, skipna=self.ignore_nan)
        X_new['numeric_nunique'] = numeric_df.nunique(axis=1)
        X_new['numeric_range'] = X_new['numeric_max'] - X_new['numeric_min']
        
        return X_new


class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract features from date columns."""
    
    def __init__(self):
        pass
    
    def fit(self, X: pd.DataFrame, y=None):
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Extract date features per row."""
        X_new = X.copy()
        
        # Identify date columns (they are float32, but we can identify by name pattern or separate them)
        # For now, we'll look for columns that might be dates based on naming
        # In practice, date columns might need to be passed separately
        # This is a simplified version - assumes date columns are already loaded
        
        # Try to identify date columns by name pattern (D in column name)
        date_cols = [col for col in X.columns if parse_column_name(col) and 
                     parse_column_name(col)['feature_type'] == 'D']
        
        if len(date_cols) == 0:
            # Fallback: use all numeric columns if no date pattern found
            # This is a heuristic - in practice date columns should be identified properly
            logger.warning("No date columns identified by pattern, using all numeric columns")
            date_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(date_cols) > 0:
            date_df = X[date_cols]
            
            # Row-level date statistics
            X_new['date_min'] = date_df.min(axis=1, skipna=True)
            X_new['date_max'] = date_df.max(axis=1, skipna=True)
            X_new['date_span'] = X_new['date_max'] - X_new['date_min']
            X_new['date_count'] = date_df.notna().sum(axis=1)
            X_new['date_mean'] = date_df.mean(axis=1, skipna=True)
            X_new['date_std'] = date_df.std(axis=1, skipna=True)
        else:
            # Fill with NaN if no date columns
            X_new['date_min'] = np.nan
            X_new['date_max'] = np.nan
            X_new['date_span'] = np.nan
            X_new['date_count'] = 0
            X_new['date_mean'] = np.nan
            X_new['date_std'] = np.nan
        
        return X_new


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """Frequency encoding for categorical columns (train-only fit to avoid leakage)."""
    
    def __init__(self, min_frequency: int = 2, top_k: Optional[int] = None):
        self.min_frequency = min_frequency
        self.top_k = top_k
        self.frequency_maps_ = {}
        self.columns_ = None
    
    def fit(self, X: pd.DataFrame, y=None):
        """Fit frequency encodings on training data only."""
        self.columns_ = X.select_dtypes(include=['object', 'string']).columns.tolist()
        self.frequency_maps_ = {}
        
        for col in self.columns_:
            value_counts = X[col].value_counts()
            # Filter by minimum frequency
            value_counts = value_counts[value_counts >= self.min_frequency]
            
            # Optionally keep only top-K
            if self.top_k:
                value_counts = value_counts.head(self.top_k)
            
            # Create frequency map (normalized)
            total = value_counts.sum()
            if total > 0:
                self.frequency_maps_[col] = (value_counts / total).to_dict()
            else:
                self.frequency_maps_[col] = {}
        
        logger.info(f"Fitted frequency encoding for {len(self.columns_)} categorical columns")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply frequency encoding."""
        X_new = X.copy()
        
        if self.columns_ is None:
            logger.warning("Encoder not fitted, returning original data")
            return X_new
        
        for col in self.columns_:
            if col in X.columns:
                # Map values to frequencies
                X_new[col] = X[col].map(self.frequency_maps_.get(col, {}))
                # Fill unseen values with 0 (or could use a small value)
                X_new[col] = X_new[col].fillna(0.0)
                # Convert to float32
                X_new[col] = X_new[col].astype('float32')
        
        return X_new


class FeatureSelector(BaseEstimator, TransformerMixin):
    """Drop constant columns and columns with high missingness."""
    
    def __init__(self, max_missing_pct: float = 0.99, drop_constant: bool = True):
        self.max_missing_pct = max_missing_pct
        self.drop_constant = drop_constant
        self.columns_to_drop_ = []
        self.columns_to_keep_ = None
    
    def fit(self, X: pd.DataFrame, y=None):
        """Identify columns to drop."""
        self.columns_to_drop_ = []
        
        # Drop constant columns
        if self.drop_constant:
            constant_cols = []
            for col in X.columns:
                if X[col].nunique() <= 1:
                    constant_cols.append(col)
            self.columns_to_drop_.extend(constant_cols)
            logger.info(f"Found {len(constant_cols)} constant columns to drop")
        
        # Drop high missingness columns
        missing_pct = X.isnull().sum() / len(X)
        high_missing_cols = missing_pct[missing_pct > self.max_missing_pct].index.tolist()
        self.columns_to_drop_.extend(high_missing_cols)
        logger.info(f"Found {len(high_missing_cols)} high missingness columns (>={self.max_missing_pct*100:.1f}%) to drop")
        
        # Remove duplicates
        self.columns_to_drop_ = list(set(self.columns_to_drop_))
        self.columns_to_keep_ = [col for col in X.columns if col not in self.columns_to_drop_]
        
        logger.info(f"Total columns to drop: {len(self.columns_to_drop_)}, keeping {len(self.columns_to_keep_)}")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Drop identified columns."""
        if self.columns_to_keep_ is None:
            logger.warning("Selector not fitted, returning original data")
            return X
        
        X_new = X[self.columns_to_keep_].copy()
        return X_new


class FeatureEngineeringPipeline:
    """Complete feature engineering pipeline."""
    
    def __init__(
        self,
        extract_missingness: bool = True,
        extract_numeric_agg: bool = True,
        extract_date_features: bool = True,
        encode_categorical: bool = True,
        max_missing_pct: float = 0.99,
        categorical_min_freq: int = 2,
        categorical_top_k: Optional[int] = None
    ):
        self.extract_missingness = extract_missingness
        self.extract_numeric_agg = extract_numeric_agg
        self.extract_date_features = extract_date_features
        self.encode_categorical = encode_categorical
        self.max_missing_pct = max_missing_pct
        self.categorical_min_freq = categorical_min_freq
        self.categorical_top_k = categorical_top_k
        
        # Initialize transformers
        self.transformers = []
        
        if self.extract_missingness:
            self.missingness_extractor = MissingnessFeatureExtractor(extract_station_features=True)
            self.transformers.append(('missingness', self.missingness_extractor))
        
        if self.extract_numeric_agg:
            self.numeric_aggregator = NumericAggregator(ignore_nan=True)
            self.transformers.append(('numeric_agg', self.numeric_aggregator))
        
        if self.extract_date_features:
            self.date_extractor = DateFeatureExtractor()
            self.transformers.append(('date_features', self.date_extractor))
        
        if self.encode_categorical:
            self.categorical_encoder = CategoricalEncoder(
                min_frequency=categorical_min_freq,
                top_k=categorical_top_k
            )
            self.transformers.append(('categorical', self.categorical_encoder))
        
        # Feature selector (always last)
        self.feature_selector = FeatureSelector(
            max_missing_pct=max_missing_pct,
            drop_constant=True
        )
        self.transformers.append(('selector', self.feature_selector))
    
    def fit(self, X: pd.DataFrame, y=None):
        """Fit all transformers on training data."""
        logger.info("Fitting feature engineering pipeline...")
        X_work = X.copy()
        
        for name, transformer in self.transformers:
            logger.info(f"  Fitting {name}...")
            transformer.fit(X_work, y)
            X_work = transformer.transform(X_work)
            logger.info(f"    Shape after {name}: {X_work.shape}")
        
        self.feature_names_ = X_work.columns.tolist()
        logger.info(f"Final feature count: {len(self.feature_names_)}")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data through pipeline."""
        X_work = X.copy()
        
        for name, transformer in self.transformers:
            X_work = transformer.transform(X_work)
        
        # Ensure same column order as training
        if hasattr(self, 'feature_names_'):
            # Add any new columns that weren't in training (shouldn't happen, but be safe)
            missing_cols = set(self.feature_names_) - set(X_work.columns)
            if missing_cols:
                logger.warning(f"Missing columns in transform: {missing_cols}, filling with 0")
                for col in missing_cols:
                    X_work[col] = 0.0
            
            # Reorder to match training
            X_work = X_work[self.feature_names_]
        
        return X_work
    
    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Fit and transform."""
        return self.fit(X, y).transform(X)

