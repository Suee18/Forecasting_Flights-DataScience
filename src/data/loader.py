"""
Data loading utilities for flight delay prediction project.
This module handles the loading and initial inspection of flight data
 in chunks due to large dataset size. 
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, List, Tuple, Any


def peek_data(file_path: str, nrows: int = 5) -> pd.DataFrame:
    """
    Get a quick peek at the dataset by loading just the first few rows.
    
    Args:
        file_path: Path to the CSV file
        nrows: Number of rows to load
    
    Returns:
        DataFrame containing the first few rows
    """
    try:
        sample_df = pd.read_csv(file_path, nrows=nrows)
        return sample_df
    except Exception as e:
        print(f"Error peeking data: {e}")
        return pd.DataFrame()


def get_data_overview(file_path: str, chunksize: int = 100000) -> Dict[str, Any]:
    """
     overview of the dataset by processing it in chunks.
    
    Args:
        file_path: Path to the CSV file
        chunksize: Number of rows to process at a time
    
    Returns:
        Dictionary containing dataset overview information
    """
    try:
        total_rows = 0
        column_info = {}
        column_types = {}
        
        # Process data in chunks
        for chunk in pd.read_csv(file_path, chunksize=chunksize):
            # Count rows
            chunk_rows = len(chunk)
            total_rows += chunk_rows
            
            # Update column information
            for col in chunk.columns:
                if col not in column_info:
                    column_info[col] = {
                        'dtype': str(chunk[col].dtype),
                        'sample_values': chunk[col].dropna().sample(min(3, len(chunk[col].dropna()))).tolist()
                    }
                
                # Update data type counts for consistency check
                curr_type = str(chunk[col].dtype)
                if col not in column_types:
                    column_types[col] = {}
                if curr_type not in column_types[col]:
                    column_types[col][curr_type] = 1
                else:
                    column_types[col][curr_type] += 1
                    
        # Create DataFrame for column info
        cols_df = pd.DataFrame({
            'dtype': {col: info['dtype'] for col, info in column_info.items()},
            'sample_values': {col: str(info['sample_values']) for col, info in column_info.items()}
        })
        
        # Check for inconsistent data types
        for col, types in column_types.items():
            if len(types) > 1:
                cols_df.loc[col, 'note'] = f"Warning: inconsistent dtypes detected: {types}"
        
        return {
            'rows': total_rows,
            'columns': len(column_info),
            'column_info': cols_df
        }
        
    except Exception as e:
        print(f"Error generating data overview: {e}")
        return {'rows': 0, 'columns': 0, 'column_info': pd.DataFrame()}


def get_numeric_stats(file_path: str, chunksize: int = 100000) -> pd.DataFrame:
    """
    Calculate statistics for numeric columns by processing data in chunks.
    
    Args:
        file_path: Path to the CSV file
        chunksize: Number of rows to process at a time
    
    Returns:
        DataFrame containing statistics for numeric columns
    """
    try:
        # Initialize running statistics
        sum_values = {}
        sum_squares = {}
        min_values = {}
        max_values = {}
        count_values = {}
        
        # Process data in chunks
        for i, chunk in enumerate(pd.read_csv(file_path, chunksize=chunksize)):
            # On first chunk, identify numeric columns
            if i == 0:
                numeric_cols = chunk.select_dtypes(include=[np.number]).columns.tolist()
                for col in numeric_cols:
                    min_values[col] = float('inf')
                    max_values[col] = float('-inf')
                    sum_values[col] = 0
                    sum_squares[col] = 0
                    count_values[col] = 0
            
            # Update running statistics
            for col in numeric_cols:
                if col in chunk.columns:
                    # Skip non-numeric data in this chunk
                    if not pd.api.types.is_numeric_dtype(chunk[col]):
                        continue
                        
                    # Update statistics with non-NA values
                    valid_data = chunk[col].dropna()
                    n = len(valid_data)
                    if n > 0:
                        sum_values[col] += valid_data.sum()
                        sum_squares[col] += (valid_data ** 2).sum()
                        min_values[col] = min(min_values[col], valid_data.min())
                        max_values[col] = max(max_values[col], valid_data.max())
                        count_values[col] += n
        
        # Calculate final statistics
        stats = {}
        for col in numeric_cols:
            if count_values[col] > 0:
                mean = sum_values[col] / count_values[col]
                # Variance calculation using the computational formula
                var = (sum_squares[col] / count_values[col]) - (mean ** 2)
                std = np.sqrt(max(0, var))  # Ensure non-negative variance due to floating point errors
                
                stats[col] = {
                    'count': count_values[col],
                    'mean': mean,
                    'std': std,
                    'min': min_values[col],
                    'max': max_values[col],
                    '25%': None,  # Exact quantiles can't be computed in chunks accurately
                    '50%': None,
                    '75%': None
                }
        
        # Create DataFrame from stats dictionary
        stats_df = pd.DataFrame(stats).T
        
        # Approximate quantiles by sampling
        sample = pd.read_csv(file_path, 
                             usecols=numeric_cols, 
                             skiprows=lambda i: i > 0 and np.random.random() > 0.1)  # 10% sample
        
        # Update quantile values in the stats DataFrame
        for col in numeric_cols:
            if col in sample.columns:
                try:
                    quantiles = sample[col].quantile([0.25, 0.5, 0.75]).values
                    stats_df.loc[col, '25%'] = quantiles[0]
                    stats_df.loc[col, '50%'] = quantiles[1]
                    stats_df.loc[col, '75%'] = quantiles[2]
                except:
                    pass
                    
        return stats_df
        
    except Exception as e:
        print(f"Error calculating numeric statistics: {e}")
        return pd.DataFrame()


def get_categorical_stats(file_path: str, chunksize: int = 100000, 
                          max_categories: int = 50) -> Dict[str, pd.Series]:
    """
    Calculate value counts for categorical columns by processing data in chunks.
    
    Args:
        file_path: Path to the CSV file
        chunksize: Number of rows to process at a time
        max_categories: Maximum number of unique categories to track per column
        
    Returns:
        Dictionary containing value count Series for categorical columns
    """
    try:
        # Initialize tracking of categorical columns and value counts
        categorical_cols = None
        value_counts = {}
        
        # Process data in chunks
        for i, chunk in enumerate(pd.read_csv(file_path, chunksize=chunksize)):
            # On first chunk, identify categorical columns
            if i == 0:
                # Get non-numeric columns or columns with few unique values
                numeric_cols = chunk.select_dtypes(include=[np.number])
                potentially_categorical = []
                
                for col in numeric_cols:
                    if chunk[col].nunique() < 30:  # Numeric with few unique values could be categorical
                        potentially_categorical.append(col)
                
                # Add string/object columns
                string_cols = chunk.select_dtypes(include=['object', 'string', 'category']).columns
                categorical_cols = list(string_cols) + potentially_categorical
                
                # Initialize value_counts dictionary
                for col in categorical_cols:
                    value_counts[col] = pd.Series(dtype='int64')
            
            # Update value counts for each categorical column
            for col in categorical_cols:
                if col in chunk.columns:
                    # Skip if the column isn't a valid categorical column in this chunk
                    if not pd.api.types.is_object_dtype(chunk[col]) and not pd.api.types.is_categorical_dtype(chunk[col]) \
                       and not (pd.api.types.is_numeric_dtype(chunk[col]) and chunk[col].nunique() < 30):
                        continue
                        
                    # Update value counts
                    chunk_counts = chunk[col].value_counts()
                    value_counts[col] = value_counts[col].add(chunk_counts, fill_value=0)
                    
                    # Keep only top categories to avoid memory issues
                    if len(value_counts[col]) > max_categories:
                        value_counts[col] = value_counts[col].nlargest(max_categories)
        
        # Sort value counts for each column
        for col in value_counts:
            value_counts[col] = value_counts[col].sort_values(ascending=False)
            
        return value_counts
        
    except Exception as e:
        print(f"Error calculating categorical statistics: {e}")
        return {}


def chunk_iterator(file_path: str, chunksize: int = 100000):
    """
    Generator function that yields chunks of data from the CSV file.
    
    Args:
        file_path: Path to the CSV file
        chunksize: Number of rows to yield at a time
        
    Yields:
        DataFrame chunks of the CSV file
    """
    try:
        for chunk in pd.read_csv(file_path, chunksize=chunksize):
            yield chunk
    except Exception as e:
        print(f"Error in chunk iterator: {e}")
        yield pd.DataFrame()
