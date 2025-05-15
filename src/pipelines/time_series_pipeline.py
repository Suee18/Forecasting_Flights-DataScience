"""
Time Series Pipeline for Flight Delay Prediction

This module contains the TimeSeriesPipeline class which implements 
preprocessing operations specialized for time series models.
"""

import pandas as pd
import numpy as np
from .base_pipeline import BasePipeline
import os
import sys

class TimeSeriesPipeline(BasePipeline):
    """Time series preprocessing pipeline for flight delay prediction."""
    
    def __init__(self, config=None):
        """
        Initialize the time series pipeline.
        
        Parameters:
        -----------
        config : dict, optional
            Configuration parameters for the pipeline.
            
        Additional Parameters:
        ---------------------
        timestamp_col : str
            Column to use as timestamp
        resample_freq : str
            Pandas frequency string for resampling (e.g. 'H', 'D')
        n_lags : int
            Number of lag features to create
        rolling_windows : list
            List of window sizes for rolling features
        """
        super().__init__(config)
        
        # Default config
        default_config = {
            'timestamp_col': 'FL_DATE',
            'resample_freq': 'D',  # Daily resampling
            'n_lags': 7,  # One week of lags
            'rolling_windows': [3, 7, 14],  # Rolling features windows
            'ts_test_size': 0.2,  # Time series test split ratio
            'use_time_split': True  # Use time-based split instead of random
        }
        
        # Update with user config
        if config is not None:
            default_config.update(config)
        
        self.config = default_config
        
    def create_datetime_features(self, df):
        """Extract datetime features from timestamp column."""
        print("Creating datetime features...")
        
        ts_col = self.config['timestamp_col']
        
        if ts_col in df.columns:
            # Ensure the column is datetime type
            if not pd.api.types.is_datetime64_any_dtype(df[ts_col]):
                df[ts_col] = pd.to_datetime(df[ts_col])
            
            # Extract datetime components
            df['year'] = df[ts_col].dt.year
            df['month'] = df[ts_col].dt.month
            df['day'] = df[ts_col].dt.day
            df['dayofweek'] = df[ts_col].dt.dayofweek
            df['quarter'] = df[ts_col].dt.quarter
            df['dayofyear'] = df[ts_col].dt.dayofyear
            df['weekofyear'] = df[ts_col].dt.isocalendar().week
            
            # Cyclical encoding of month, day of week (sine/cosine transformation)
            # This preserves the cyclical nature of these features
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
            df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
            df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
            
            # Is holiday/weekend feature
            if pd.api.types.is_datetime64_any_dtype(df[ts_col]):
                df['is_weekend'] = df[ts_col].dt.dayofweek >= 5
            
            # Handle hour of day if available
            if 'dep_hour' in df.columns:
                df['hour_sin'] = np.sin(2 * np.pi * df['dep_hour'] / 24)
                df['hour_cos'] = np.cos(2 * np.pi * df['dep_hour'] / 24)
        
        return df
    
    def resample_to_frequency(self, df, group_cols=None):
        """
        Resample data to specified frequency.
        
        This aggregates the data to a coarser time granularity, which is often
        necessary for time series forecasting.
        """
        print(f"Resampling to {self.config['resample_freq']} frequency...")
        
        ts_col = self.config['timestamp_col']
        
        # If we need to group by additional columns (e.g. by airport)
        if group_cols is not None:
            # Group by time and specified columns
            resampled_dfs = []
            for name, group in df.groupby(group_cols):
                # Convert to single group name if only one group column
                if not isinstance(name, tuple):
                    name = (name,)
                
                # Set timestamp as index for resampling
                group = group.set_index(ts_col)
                
                # Resample and aggregate
                resampled = group.resample(self.config['resample_freq']).agg({
                    self.target_column: 'mean',
                    'is_delayed': 'mean',  # Becomes the proportion of delayed flights
                    'year': 'first',
                    'month': 'first',
                    'dayofweek': 'first',
                    'DISTANCE': 'mean'
                })
                
                # Add back the group columns
                for i, col in enumerate(group_cols):
                    resampled[col] = name[i]
                
                resampled = resampled.reset_index()
                resampled_dfs.append(resampled)
                
            # Combine all resampled groups
            if resampled_dfs:
                df = pd.concat(resampled_dfs, axis=0)
            
        else:
            # Simple resampling without additional grouping
            df = df.set_index(ts_col)
            df = df.resample(self.config['resample_freq']).agg({
                self.target_column: 'mean',
                'is_delayed': 'mean',
                'year': 'first',
                'month': 'first',
                'dayofweek': 'first',
                'DISTANCE': 'mean'
            })
            df = df.reset_index()
            
        return df
    
    def generate_lag_features(self, df, group_cols=None):
        """
        Generate lagged features for time series forecasting.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe
            
        group_cols : list, optional
            Columns to group by before creating lags (e.g., airport)
        """
        print("Generating lag features...")
        
        target = self.target_column
        n_lags = self.config['n_lags']
        
        # Sort by timestamp
        ts_col = self.config['timestamp_col']
        df = df.sort_values(ts_col)
        
        # If we need to create lags within groups
        if group_cols is not None:
            for col in group_cols:
                if col not in df.columns:
                    raise ValueError(f"Group column {col} not found in dataframe")
            
            # Group and create lags
            for lag in range(1, n_lags + 1):
                lag_values = df.groupby(group_cols)[target].shift(lag)
                df[f"{target}_lag_{lag}"] = lag_values
                
                # Also create lags of the is_delayed feature if it exists
                if 'is_delayed' in df.columns:
                    delay_lag_values = df.groupby(group_cols)['is_delayed'].shift(lag)
                    df[f"is_delayed_lag_{lag}"] = delay_lag_values
                    
        else:
            # Create lags without grouping
            for lag in range(1, n_lags + 1):
                df[f"{target}_lag_{lag}"] = df[target].shift(lag)
                
                if 'is_delayed' in df.columns:
                    df[f"is_delayed_lag_{lag}"] = df['is_delayed'].shift(lag)
                    
        # Drop rows with NaN lag values (first n_lags rows)
        df = df.dropna(subset=[f"{target}_lag_{n_lags}"])
                    
        return df
    
    def create_rolling_features(self, df, group_cols=None):
        """
        Create rolling window features like moving averages.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe
            
        group_cols : list, optional
            Columns to group by before creating rolling features
        """
        print("Creating rolling features...")
        
        target = self.target_column
        windows = self.config['rolling_windows']
        
        # Sort by timestamp
        ts_col = self.config['timestamp_col']
        df = df.sort_values(ts_col)
        
        # If we need to create rolling features within groups
        if group_cols is not None:
            # Group and create rolling features
            for window in windows:
                # Rolling mean
                roll_means = df.groupby(group_cols)[target].transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                )
                df[f"{target}_roll_mean_{window}"] = roll_means
                
                # Rolling standard deviation
                roll_stds = df.groupby(group_cols)[target].transform(
                    lambda x: x.rolling(window, min_periods=1).std()
                )
                df[f"{target}_roll_std_{window}"] = roll_stds.fillna(0)
                
                # Rolling max
                roll_maxs = df.groupby(group_cols)[target].transform(
                    lambda x: x.rolling(window, min_periods=1).max()
                )
                df[f"{target}_roll_max_{window}"] = roll_maxs
                
        else:
            # Create rolling features without grouping
            for window in windows:
                df[f"{target}_roll_mean_{window}"] = df[target].rolling(window, min_periods=1).mean()
                df[f"{target}_roll_std_{window}"] = df[target].rolling(window, min_periods=1).std().fillna(0)
                df[f"{target}_roll_max_{window}"] = df[target].rolling(window, min_periods=1).max()
                
        return df
    
    def handle_seasonality(self, df):
        """
        Handle seasonality using decomposition or seasonal features.
        """
        print("Handling seasonality...")
        
        # Add seasonal indicators
        # For daily data: day of week indicators
        if 'dayofweek' in df.columns:
            for day in range(7):
                df[f'is_day_{day}'] = (df['dayofweek'] == day).astype(int)
                
        # For monthly data: month indicators  
        if 'month' in df.columns:
            for month in range(1, 13):
                df[f'is_month_{month}'] = (df['month'] == month).astype(int)
                
        # Add US holiday indicators if we have a library for it
        # This would require additional libraries like holidays
        
        return df
    
    def apply_time_based_split(self, df, test_size=None):
        """
        Split data based on time rather than randomly.
        
        For time series, we should always split chronologically to avoid
        data leakage, with the test set being the most recent data.
        """
        print("Applying time-based data split...")
        
        ts_col = self.config['timestamp_col']
        
        if test_size is None:
            test_size = self.config['ts_test_size']
            
        # Sort by time
        df = df.sort_values(ts_col)
        
        # Calculate split point
        split_idx = int(len(df) * (1 - test_size))
        val_idx = int(len(df) * (1 - 2 * test_size))
        
        # Split the data
        train = df.iloc[:val_idx].copy()
        val = df.iloc[val_idx:split_idx].copy()
        test = df.iloc[split_idx:].copy()
        
        print(f"Train: {len(train)} samples, Validation: {len(val)} samples, Test: {len(test)} samples")
        print(f"Train period: {train[ts_col].min()} to {train[ts_col].max()}")
        print(f"Val period: {val[ts_col].min()} to {val[ts_col].max()}")
        print(f"Test period: {test[ts_col].min()} to {test[ts_col].max()}")
        
        return train, val, test
    
    def run(self, data_path, group_cols=None):
        """Run the complete time series pipeline."""
        # Run base pipeline steps first (except splitting)
        df = self.load_data(data_path)
        df = self.clean_data(df)
        df = self.handle_missing_values(df)
        df = self.encode_categorical_variables(df)
        df = self.generate_basic_features(df)
        
        # Time series specific steps
        df = self.create_datetime_features(df)
        
        # Optional resampling
        if self.config.get('perform_resampling', False):
            df = self.resample_to_frequency(df, group_cols)
            
        df = self.generate_lag_features(df, group_cols)
        df = self.create_rolling_features(df, group_cols)
        df = self.handle_seasonality(df)
        
        # Time-based split
        if self.config.get('use_time_split', True):
            train, val, test = self.apply_time_based_split(df)
        else:
            train, val, test = self.split_data(df)
            
        return {
            'train': train,
            'validation': val, 
            'test': test,
            'full_data': df
        }
