"""
Deep Learning Pipeline for Flight Delay Prediction

This module contains the DeepLearningPipeline class which implements 
preprocessing operations specialized for deep learning models.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from .base_pipeline import BasePipeline

class DeepLearningPipeline(BasePipeline):
    """Deep learning preprocessing pipeline for flight delay prediction."""
    
    def __init__(self, config=None):
        """
        Initialize the deep learning pipeline.
        
        Parameters:
        -----------
        config : dict, optional
            Configuration parameters for the pipeline.
            
        Additional Parameters:
        ---------------------
        batch_size : int
            Batch size for mini-batch training
        sequence_length : int
            Length of sequences for recurrent models
        embedding_dims : dict
            Dictionary mapping categorical columns to embedding dimensions
        scaler_type : str
            Type of scaling to apply ('standard', 'minmax')
        """
        super().__init__(config)
        
        # Default config
        default_config = {
            'batch_size': 64,
            'sequence_length': 10,  # For sequential models
            'embedding_dims': {
                'OP_CARRIER': 4,
                'ORIGIN': 8,
                'DEST': 8
            },
            'scaler_type': 'standard',  # 'standard' or 'minmax'
            'create_sequences': False,  # Whether to create sequences for RNNs
            'categorical_embed_method': 'embedding'  # 'embedding' or 'onehot'
        }
        
        # Update with user config
        if config is not None:
            default_config.update(config)
            
        self.config = default_config
        
    def normalize_inputs(self, df, fit=True):
        """
        Normalize numerical inputs for neural networks.
        
        Neural networks generally train better with normalized inputs.
        """
        print("Normalizing inputs...")
        
        # Get numeric columns
        numeric_cols = [col for col in df.columns if 
                       col in self.numerical_columns or 
                       (col.startswith(self.target_column) and col != self.target_column) or
                       col.startswith('roll_') or
                       col.startswith('lag_')]
        
        # Add target to be normalized
        if self.target_column in df.columns:
            numeric_cols.append(self.target_column)
        
        # Create scaler if fitting
        if fit:
            if self.config['scaler_type'] == 'standard':
                self.scaler = StandardScaler()
            else:
                self.scaler = MinMaxScaler()
            
            # Fit scaler
            self.scaler.fit(df[numeric_cols])
            
        # Transform data
        normalized_data = self.scaler.transform(df[numeric_cols])
        
        # Replace original columns with normalized values
        for i, col in enumerate(numeric_cols):
            df[col] = normalized_data[:, i]
            
        return df
    
    def create_embeddings(self, df):
        """
        Prepare categorical variables for embedding layers.
        
        For deep learning, we need to convert categories to integer indices
        which will be inputs to embedding layers.
        """
        print("Preparing embeddings for categorical variables...")
        
        # Get embedding configuration
        embedding_dims = self.config.get('embedding_dims', {})
        categorical_columns = list(embedding_dims.keys())
        
        # Only process columns that exist in the data
        for col in categorical_columns:
            if col not in df.columns:
                print(f"Warning: Embedding column {col} not found in data")
                continue
                
            # Create category mapping if not already created
            if not hasattr(self, f'{col}_mapping'):
                # Get unique categories and assign indices
                categories = df[col].unique()
                mapping = {cat: idx for idx, cat in enumerate(categories)}
                setattr(self, f'{col}_mapping', mapping)
                
                # Store vocab size for embedding layer configuration
                setattr(self, f'{col}_vocab_size', len(mapping) + 1)  # +1 for unknown
                
            # Apply mapping
            mapping = getattr(self, f'{col}_mapping')
            df[f'{col}_idx'] = df[col].map(mapping).fillna(len(mapping)).astype(int)
            
        return df
    
    def sequence_preparation(self, df):
        """
        Prepare sequential data for RNNs, LSTMs or Transformer models.
        
        Creates sequences of data points for each group (e.g. airport, route).
        """
        if not self.config.get('create_sequences', False):
            return df
            
        print("Preparing sequences for recurrent models...")
        
        ts_col = self.config.get('timestamp_col', 'FL_DATE')
        seq_len = self.config.get('sequence_length', 10)
        
        # Sort by time
        df = df.sort_values(ts_col)
        
        # Define features to include in sequences
        feature_cols = [col for col in df.columns if 
                       col not in [ts_col, self.target_column] and
                       not col.endswith('_idx')]
        
        # Add embedding indices
        feature_cols.extend([col for col in df.columns if col.endswith('_idx')])
        
        # Create sequences
        sequences = []
        targets = []
        
        # Get group columns from config
        group_cols = self.config.get('group_cols', [])
        
        if group_cols:
            # Create sequences for each group
            for _, group in df.groupby(group_cols):
                # Skip groups with too few samples
                if len(group) < seq_len + 1:
                    continue
                    
                # Extract features and target
                features = group[feature_cols].values
                target = group[self.target_column].values
                
                # Create sequences
                for i in range(len(group) - seq_len):
                    sequences.append(features[i:i+seq_len])
                    targets.append(target[i+seq_len])
        else:
            # Create sequences without grouping
            features = df[feature_cols].values
            target = df[self.target_column].values
            
            for i in range(len(df) - seq_len):
                sequences.append(features[i:i+seq_len])
                targets.append(target[i+seq_len])
                
        # Convert to numpy arrays
        X = np.array(sequences)
        y = np.array(targets)
        
        print(f"Created {len(X)} sequences with shape {X.shape}")
        
        return X, y
    
    def batch_preparation(self, X, y=None):
        """
        Prepare data batches for training or inference.
        
        Parameters:
        -----------
        X : array-like
            Input features or sequences
            
        y : array-like, optional
            Target variable
            
        Returns:
        --------
        dataset : tf.data.Dataset or similar
            Dataset ready for neural network training
        """
        if not isinstance(X, np.ndarray):
            print("Warning: batch_preparation expects numpy arrays, skipping")
            return X, y
            
        print("Preparing batches for training...")
        
        batch_size = self.config.get('batch_size', 64)
        
        # This implementation depends on the deep learning framework being used
        # Here we'll just return the arrays with a note about batching
        print(f"Data ready for batching with batch_size={batch_size}")
        
        # If using TensorFlow:
        # import tensorflow as tf
        # dataset = tf.data.Dataset.from_tensor_slices((X, y))
        # dataset = dataset.batch(batch_size)
        # dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        # return dataset
        
        # If using PyTorch:
        # from torch.utils.data import DataLoader, TensorDataset
        # import torch
        # tensor_x = torch.Tensor(X)
        # tensor_y = torch.Tensor(y)
        # dataset = TensorDataset(tensor_x, tensor_y)
        # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        # return dataloader
        
        return X, y
    
    def run(self, data_path):
        """Run the complete deep learning pipeline."""
        # Run base pipeline steps first
        df = self.load_data(data_path)
        df = self.clean_data(df)
        df = self.handle_missing_values(df)
        df = self.generate_basic_features(df)
        
        # For deep learning, we might want to use embeddings instead of one-hot encoding
        if self.config.get('categorical_embed_method', 'embedding') == 'embedding':
            df = self.create_embeddings(df)
        else:
            df = self.encode_categorical_variables(df)
            
        # Normalize inputs
        df = self.normalize_inputs(df)
        
        # Split the data
        train_df, val_df, test_df = self.split_data(df)
        
        # Create sequences if using RNN/LSTM
        if self.config.get('create_sequences', False):
            X_train, y_train = self.sequence_preparation(train_df)
            X_val, y_val = self.sequence_preparation(val_df)
            X_test, y_test = self.sequence_preparation(test_df)
            
            # Prepare batches
            train_batches = self.batch_preparation(X_train, y_train)
            val_batches = self.batch_preparation(X_val, y_val)
            test_batches = self.batch_preparation(X_test, y_test)
            
            return {
                'train': train_batches,
                'validation': val_batches,
                'test': test_batches,
                'train_df': train_df,
                'val_df': val_df,
                'test_df': test_df,
                'full_data': df
            }
        else:
            # For non-sequential models
            return {
                'train': train_df,
                'validation': val_df,
                'test': test_df, 
                'full_data': df
            }
