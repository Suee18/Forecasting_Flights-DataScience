"""
Base Pipeline for Flight Delay Prediction

This module contains the BasePipeline class which implements common
preprocessing operations used across all model types.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
import os
import sys
import warnings

# Add path for imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(PROJECT_ROOT)
from src.data import processor

class BasePipeline:
    """Base preprocessing pipeline for flight delay prediction."""
    
    def __init__(self, config=None):
        """
        Initialize the base pipeline.
        
        Parameters:
        -----------
        config : dict, optional
            Configuration parameters for the pipeline.
        """
        self.config = config if config is not None else {}
        self.categorical_columns = [
            'OP_CARRIER', 'ORIGIN', 'DEST', 'OP_CARRIER_FL_NUM', 
            'ORIGIN_CITY', 'DEST_CITY'
        ]
        self.numerical_columns = [
            'DISTANCE', 'CRS_ELAPSED_TIME', 'CRS_DEP_TIME', 
            'CRS_ARR_TIME', 'CARRIER_DELAY', 'WEATHER_DELAY',
            'NAS_DELAY', 'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY'
        ]
        self.datetime_columns = ['FL_DATE', 'CRS_DEP_DATETIME', 'CRS_ARR_DATETIME']
        self.target_column = 'DEP_DELAY'
        
    def load_data(self, data_path):
        """Load flight data from CSV file."""
        print(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        
        # Convert date columns to datetime
        for col in self.datetime_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
                
        return df
    
    def clean_data(self, df):
        """Clean the dataframe by removing invalid entries."""
        print("Cleaning data...")
        orig_len = len(df)
        
        # Remove duplicate flights
        df = df.drop_duplicates()
        
        # Remove cancelled flights (these won't have delay information)
        if 'CANCELLED' in df.columns:
            df = df[df['CANCELLED'] == 0]
            
        # Handle negative delays (early departures) - clip or keep depending on goal
        if self.target_column in df.columns:
            if self.config.get('clip_negative_delays', True):
                df[self.target_column] = df[self.target_column].clip(lower=0)
        
        print(f"Removed {orig_len - len(df)} invalid records")
        return df
    
    def handle_missing_values(self, df):
        """Handle missing values in the dataframe."""
        print("Handling missing values...")
        
        # For target variable, we can only drop if missing
        if self.target_column in df.columns:
            df = df.dropna(subset=[self.target_column])
            
        # For numerical columns, impute with median
        for col in self.numerical_columns:
            if col in df.columns and df[col].isna().sum() > 0:
                median_value = df[col].median()
                df[col] = df[col].fillna(median_value)
                
        # For categorical columns, impute with mode
        for col in self.categorical_columns:
            if col in df.columns and df[col].isna().sum() > 0:
                mode_value = df[col].mode()[0]
                df[col] = df[col].fillna(mode_value)
                
        return df
    
    def encode_categorical_variables(self, df, fit=True):
        """
        Encode categorical variables using appropriate methods.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe
            
        fit : bool, default=True
            Whether to fit new encoders or use existing ones
        """
        print("Encoding categorical variables...")
        
        # High-cardinality categorical columns - use ordinal encoding to save memory
        high_card_cols = ['ORIGIN', 'DEST', 'ORIGIN_CITY', 'DEST_CITY']
        low_card_cols = [col for col in self.categorical_columns 
                         if col in df.columns and col not in high_card_cols]
        
        # Encode high-cardinality columns with ordinal encoding
        if fit:
            self.high_card_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', 
                                                   unknown_value=-1)
            high_card_cols_present = [col for col in high_card_cols if col in df.columns]
            if high_card_cols_present:
                self.high_card_encoder.fit(df[high_card_cols_present])
        
        high_card_cols_present = [col for col in high_card_cols if col in df.columns]
        if high_card_cols_present:
            encoded_vals = self.high_card_encoder.transform(df[high_card_cols_present])
            for i, col in enumerate(high_card_cols_present):
                df[f"{col}_encoded"] = encoded_vals[:, i]
        
        # One-hot encode low-cardinality columns
        if fit:
            self.low_card_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
            low_card_cols_present = [col for col in low_card_cols if col in df.columns]
            if low_card_cols_present:
                self.low_card_encoder.fit(df[low_card_cols_present])
        
        low_card_cols_present = [col for col in low_card_cols if col in df.columns]
        if low_card_cols_present:
            encoded_vals = self.low_card_encoder.transform(df[low_card_cols_present])
            encoded_df = pd.DataFrame(
                encoded_vals, 
                columns=self.low_card_encoder.get_feature_names_out(low_card_cols_present),
                index=df.index
            )
            df = pd.concat([df, encoded_df], axis=1)
            
        return df
    
    def generate_basic_features(self, df):
        """Generate basic features common to all models."""
        print("Generating basic features...")
        
        # Extract date components
        if 'FL_DATE' in df.columns:
            df['MONTH'] = df['FL_DATE'].dt.month
            df['DAY'] = df['FL_DATE'].dt.day
            df['DAY_OF_WEEK'] = df['FL_DATE'].dt.dayofweek
        
        # Time of day features from scheduled departure time
        if 'CRS_DEP_TIME' in df.columns:
            # Convert HHMM format to hour of day
            df['dep_hour'] = df['CRS_DEP_TIME'] // 100
            df['dep_minute'] = df['CRS_DEP_TIME'] % 100
            
        # Early morning, morning, afternoon, evening, night
        if 'dep_hour' in df.columns:
            df['time_of_day'] = pd.cut(
                df['dep_hour'], 
                bins=[-1, 5, 10, 15, 20, 24], 
                labels=['early_morning', 'morning', 'afternoon', 'evening', 'night']
            )
            
        # Delay binary target (for classification)
        if self.target_column in df.columns:
            df['is_delayed'] = (df[self.target_column] > 15).astype(int)
            
        return df
    
    def split_data(self, df, test_size=0.2, validation_size=0.2, random_state=42):
        """Split data into train, validation, and test sets."""
        print("Splitting data into train, validation, and test sets...")
        
        # First split off test data
        train_val, test = train_test_split(df, test_size=test_size, random_state=random_state)
        
        # Then split train data to get validation set
        val_split = validation_size / (1 - test_size)
        train, val = train_test_split(train_val, test_size=val_split, random_state=random_state)
        
        print(f"Train: {len(train)} samples, Validation: {len(val)} samples, Test: {len(test)} samples")
        return train, val, test
    
    def run(self, data_path):
        """Run the complete base pipeline."""
        df = self.load_data(data_path)
        df = self.clean_data(df)
        df = self.handle_missing_values(df)
        df = self.generate_basic_features(df)
        df = self.encode_categorical_variables(df)
        train, val, test = self.split_data(df)
        
        return {
            'train': train,
            'validation': val,
            'test': test,
            'full_data': df
        }
 