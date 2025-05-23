"""
Functions for handling missing values in flight delay data
"""

def handle_missing_values(df):
    """
    Handle missing values with smart imputation logic:
    - ARR_DELAY: Use DEP_DELAY from same record, or median
    - DEP_DELAY: Use median by carrier/origin/day_of_week
    - Numeric columns: Use median by relevant grouping
    - Categorical columns: Use mode
    """
    # Make a copy to avoid modifying the original data
    df_imputed = df.copy()
    
    # 1. Handle ARR_DELAY using DEP_DELAY when possible
    if 'ARR_DELAY' in df_imputed.columns and 'DEP_DELAY' in df_imputed.columns:
        mask = df_imputed['ARR_DELAY'].isna() & df_imputed['DEP_DELAY'].notna()
        df_imputed.loc[mask, 'ARR_DELAY'] = df_imputed.loc[mask, 'DEP_DELAY']
    
    # 2. Group columns by type for appropriate imputation
    numeric_cols = df_imputed.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = df_imputed.select_dtypes(include=['object', 'category']).columns
    
    # 3. Handle remaining numeric columns
    for col in numeric_cols:
        if df_imputed[col].isna().any():
            # Try carrier/origin/day_of_week grouping first
            if all(x in df_imputed.columns for x in ['OP_CARRIER', 'ORIGIN', 'day_of_week']):
                grouped_median = df_imputed.groupby(['OP_CARRIER', 'ORIGIN', 'day_of_week'])[col].transform('median')
                df_imputed[col].fillna(grouped_median, inplace=True)
            
            # If still has missing values, try simpler groupings
            if df_imputed[col].isna().any() and 'OP_CARRIER' in df_imputed.columns:
                carrier_median = df_imputed.groupby('OP_CARRIER')[col].transform('median')
                df_imputed[col].fillna(carrier_median, inplace=True)
            
            # Finally, use overall median if still missing
            df_imputed[col].fillna(df_imputed[col].median(), inplace=True)
    
    # 4. Handle categorical columns with mode
    for col in categorical_cols:
        if df_imputed[col].isna().any():
            df_imputed[col].fillna(df_imputed[col].mode()[0], inplace=True)
    
    # 5. Verify no missing values remain
    remaining_nulls = df_imputed.isnull().sum()
    if remaining_nulls.any():
        print("Warning: Some missing values could not be imputed:")
        print(remaining_nulls[remaining_nulls > 0])
    
    return df_imputed
