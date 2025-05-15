"""
Machine Learning Pipeline for Flight Delay Prediction

This module contains the MLPipeline class which implements 
preprocessing operations specialized for traditional machine learning models.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA
from .base_pipeline import BasePipeline

class MLPipeline(BasePipeline):
    """Traditional machine learning preprocessing pipeline for flight delay prediction."""
    
    def __init__(self, config=None):
        """
        Initialize the machine learning pipeline.
        
        Parameters:
        -----------
        config : dict, optional
            Configuration parameters for the pipeline.
            
        Additional Parameters:
        ---------------------
        feature_selection : str
            Method to use for feature selection ('none', 'kbest', 'pca')
        k_features : int
            Number of features to select if using k-best
        n_components : int or float
            Number of components for PCA
        scaler_type : str
            Type of scaling to apply ('standard', 'robust', 'minmax')
        """
        super().__init__(config)
        
        # Default config
        default_config = {
            'feature_selection': 'none',  # 'none', 'kbest', 'pca'
            'k_features': 20,  # Number of features for k-best
            'n_components': 0.95,  # Variance explained for PCA
            'scaler_type': 'robust',  # 'standard', 'robust', 'minmax'
            'handle_outliers': True
        }
        
        # Update with user config
        if config is not None:
            default_config.update(config)
            
        self.config = default_config
        
    def feature_engineering(self, df):
        """
        Create domain-specific features for machine learning models.
        
        This extends the basic feature generation with more complex
        engineered features specifically for ML models.
        """
        print("Performing advanced feature engineering...")
        
        # Generate interaction terms between important features
        
        # Time-based features
        if 'dep_hour' in df.columns and 'dayofweek' in df.columns:
            # Busy travel times (rush hours on weekdays)
            morning_rush = ((df['dep_hour'] >= 6) & (df['dep_hour'] <= 9) & 
                           (df['dayofweek'] < 5))  # Weekdays only
            evening_rush = ((df['dep_hour'] >= 16) & (df['dep_hour'] <= 19) & 
                           (df['dayofweek'] < 5))  # Weekdays only
            
            df['is_morning_rush'] = morning_rush.astype(int)
            df['is_evening_rush'] = evening_rush.astype(int)
            
        # Distance-based features
        if 'DISTANCE' in df.columns:
            # Flight duration categories
            df['distance_category'] = pd.cut(
                df['DISTANCE'],
                bins=[0, 500, 1000, 2000, np.inf],
                labels=['short', 'medium', 'long', 'very_long']
            )
            
        # Create polynomial features for highly correlated variables
        if 'DISTANCE' in df.columns:
            df['DISTANCE_squared'] = df['DISTANCE'] ** 2
            
        # Day type (weekday/weekend)
        if 'dayofweek' in df.columns:
            df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
            
        # Month quarter (beginning, middle, end)
        if 'day' in df.columns and 'month' in df.columns:
            df['month_part'] = pd.cut(
                df['day'], 
                bins=[0, 10, 20, 32], 
                labels=['beginning', 'middle', 'end']
            )
            
        # Season
        if 'month' in df.columns:
            df['season'] = pd.cut(
                df['month'],
                bins=[0, 3, 6, 9, 13],
                labels=['winter', 'spring', 'summer', 'fall']
            )
            
        return df
    
    def feature_selection(self, X, y=None):
        """
        Select most relevant features to reduce dimensionality.
        
        Parameters:
        -----------
        X : pandas.DataFrame or numpy.ndarray
            Input features
            
        y : pandas.Series or numpy.ndarray, optional
            Target variable for supervised feature selection
            
        Returns:
        --------
        X_selected : pandas.DataFrame or numpy.ndarray
            Selected features
        """
        feature_selection = self.config['feature_selection']
        
        if feature_selection == 'none':
            return X
            
        print(f"Performing feature selection using {feature_selection}...")
        
        if feature_selection == 'kbest':
            k = min(self.config['k_features'], X.shape[1])
            
            if y is not None:
                # Supervised feature selection
                selector = SelectKBest(f_regression, k=k)
                X_selected = selector.fit_transform(X, y)
                
                # If X is a DataFrame, keep column names
                if isinstance(X, pd.DataFrame):
                    selected_features = X.columns[selector.get_support()]
                    X_selected = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
                    
                self.feature_selector = selector
                
                print(f"Selected {k} best features")
                return X_selected
            else:
                print("Warning: y is required for kbest feature selection, skipping")
                return X
        
        elif feature_selection == 'pca':
            # Unsupervised dimensionality reduction with PCA
            n_components = self.config['n_components']
            
            # If n_components is a float, it represents variance to be explained
            if isinstance(n_components, float):
                pca = PCA(n_components=n_components, random_state=42)
            else:
                # Otherwise, use the specified number of components
                pca = PCA(n_components=min(n_components, X.shape[1]), random_state=42)
                
            X_pca = pca.fit_transform(X)
            
            # If X is a DataFrame, create a new DataFrame with PCA components
            if isinstance(X, pd.DataFrame):
                X_pca = pd.DataFrame(
                    X_pca,
                    columns=[f'PC{i+1}' for i in range(X_pca.shape[1])],
                    index=X.index
                )
                
            self.pca = pca
            
            print(f"Reduced to {X_pca.shape[1]} components explaining {pca.explained_variance_ratio_.sum():.2%} variance")
            return X_pca
            
        else:
            print(f"Warning: Unknown feature selection method {feature_selection}, skipping")
            return X
    
    def outlier_handling(self, df):
        """
        Handle outliers in the data.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe
            
        Returns:
        --------
        df : pandas.DataFrame
            Dataframe with outliers handled
        """
        if not self.config.get('handle_outliers', True):
            return df
            
        print("Handling outliers...")
        
        # Define columns to check for outliers
        numeric_cols = [col for col in df.columns if 
                      pd.api.types.is_numeric_dtype(df[col]) and 
                      col != self.target_column]  # Don't clip target
                      
        # We'll use a robust approach - winsorizing (clipping) extreme values
        for col in numeric_cols:
            # Calculate Q1, Q3 and IQR
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define bounds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Clip outliers
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
            
        return df
    
    def scaling(self, df, fit=True):
        """
        Scale numerical features.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe
            
        fit : bool, default=True
            Whether to fit a new scaler or use existing one
            
        Returns:
        --------
        df : pandas.DataFrame
            Dataframe with scaled features
        """
        print("Scaling numerical features...")
        
        # Get numeric columns to scale
        numeric_cols = [col for col in df.columns if 
                       pd.api.types.is_numeric_dtype(df[col])]
        
        # Skip scaling if no numeric columns
        if not numeric_cols:
            return df
            
        # Choose and fit scaler if needed
        if fit:
            if self.config['scaler_type'] == 'robust':
                self.scaler = RobustScaler()
            elif self.config['scaler_type'] == 'minmax':
                self.scaler = MinMaxScaler()
            else:  # default to standard
                self.scaler = StandardScaler()
                
            self.scaler.fit(df[numeric_cols])
            
        # Transform the data
        scaled_data = self.scaler.transform(df[numeric_cols])
        
        # Update the dataframe
        df_scaled = df.copy()
        df_scaled[numeric_cols] = scaled_data
        
        return df_scaled
    
    def run(self, data_path):
        """Run the complete machine learning pipeline."""
        # Run base pipeline steps
        df = self.load_data(data_path)
        df = self.clean_data(df)
        df = self.handle_missing_values(df)
        df = self.generate_basic_features(df)
        df = self.encode_categorical_variables(df)
        
        # ML specific steps
        df = self.feature_engineering(df)
        df = self.outlier_handling(df)
        
        # Split before scaling to prevent data leakage
        train_df, val_df, test_df = self.split_data(df)
        
        # Scale the data
        train_df_scaled = self.scaling(train_df, fit=True)
        val_df_scaled = self.scaling(val_df, fit=False)
        test_df_scaled = self.scaling(test_df, fit=False)
        
        # Separate features and target
        X_train = train_df_scaled.drop(columns=[self.target_column])
        y_train = train_df_scaled[self.target_column]
        
        X_val = val_df_scaled.drop(columns=[self.target_column])
        y_val = val_df_scaled[self.target_column]
        
        X_test = test_df_scaled.drop(columns=[self.target_column])
        y_test = test_df_scaled[self.target_column]
        
        # Apply feature selection if configured
        X_train = self.feature_selection(X_train, y_train)
        X_val = self.feature_selection(X_val)
        X_test = self.feature_selection(X_test)
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'train_df': train_df,
            'val_df': val_df,
            'test_df': test_df,
            'full_data': df
        }
