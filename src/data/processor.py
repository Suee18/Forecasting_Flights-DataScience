"""
Data processing utilities for flight delay prediction project.
This module handles data processing operations in chunks for large flight datasets.
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, List, Tuple, Any
from datetime import datetime


def analyze_missing_values(file_path: str, chunksize: int = 100000) -> pd.DataFrame:
    """
    Analyze missing values in the dataset by processing it in chunks.
    
    Args:
        file_path: Path to the CSV file
        chunksize: Number of rows to process at a time
    
    Returns:
        DataFrame containing missing value statistics for each column
    """
    try:
        # Initialize counters
        total_rows = 0
        missing_counts = {}
        
        # Process data in chunks
        for chunk in pd.read_csv(file_path, chunksize=chunksize):
            # Count rows
            chunk_rows = len(chunk)
            total_rows += chunk_rows
            
            # Count missing values in this chunk
            chunk_missing = chunk.isnull().sum()
            
            # Update missing counts
            for col in chunk_missing.index:
                if col not in missing_counts:
                    missing_counts[col] = 0
                missing_counts[col] += chunk_missing[col]
        
        # Create DataFrame with missing value statistics
        missing_df = pd.DataFrame({
            'missing_count': missing_counts,
            'missing_percent': {col: count / total_rows * 100 for col, count in missing_counts.items()}
        }).sort_values('missing_percent', ascending=False)
        
        return missing_df
        
    except Exception as e:
        print(f"Error analyzing missing values: {e}")
        return pd.DataFrame()


def analyze_delays(file_path: str, chunksize: int = 100000) -> Dict[str, Any]:
    """
    Analyze delay statistics in the dataset by processing it in chunks.
    
    Args:
        file_path: Path to the CSV file
        chunksize: Number of rows to process at a time
    
    Returns:
        Dictionary containing delay analysis results
    """
    try:
        # Initialize storage for statistics
        delay_columns = ['DEP_DELAY', 'ARR_DELAY', 'CARRIER_DELAY', 'WEATHER_DELAY', 
                         'NAS_DELAY', 'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY']
        
        all_delays = {col: [] for col in delay_columns}
        delay_bins = {col: {} for col in delay_columns}
        
        # Create bins for delay distributions
        bins = [-float('inf'), -15, 0, 15, 30, 60, 120, float('inf')]
        bin_labels = ['very early', 'early', 'on-time', 'slight delay', 
                     'moderate delay', 'significant delay', 'severe delay']
        
        # Process data in chunks
        for chunk in pd.read_csv(file_path, chunksize=chunksize):
            for col in delay_columns:
                if col in chunk.columns:
                    # Sample delays for distribution analysis (to avoid memory issues)
                    if len(chunk) > 1000:
                        delay_sample = chunk[col].dropna().sample(1000).tolist()
                    else:
                        delay_sample = chunk[col].dropna().tolist()
                    
                    all_delays[col].extend(delay_sample)
                    
                    # Bin the delays
                    binned_data = pd.cut(chunk[col].dropna(), bins=bins, labels=bin_labels)
                    binned_counts = binned_data.value_counts()
                    
                    # Update bin counts
                    for label, count in binned_counts.items():
                        if label not in delay_bins[col]:
                            delay_bins[col][label] = 0
                        delay_bins[col][label] += count
        
        # Convert bin counts to DataFrames
        for col in delay_columns:
            delay_bins[col] = pd.Series(delay_bins[col]).fillna(0)
            
            # Limit sample size for memory management
            if len(all_delays[col]) > 100000:
                all_delays[col] = np.random.choice(all_delays[col], 100000, replace=False).tolist()
        
        return {
            'delay_samples': all_delays,
            'delay_distributions': delay_bins
        }
        
    except Exception as e:
        print(f"Error analyzing delays: {e}")
        return {'delay_samples': {}, 'delay_distributions': {}}


def calculate_delay_frequency(file_path: str, chunksize: int = 100000, 
                             delay_threshold: int = 15) -> Dict[str, Any]:
    """
    Calculate frequency of delays by different metrics.
    
    Args:
        file_path: Path to the CSV file
        chunksize: Number of rows to process at a time
        delay_threshold: Minutes of arrival delay to consider a flight delayed
    
    Returns:
        Dictionary containing delay frequency analysis
    """
    try:
        # Initialize counters
        total_flights = 0
        delayed_flights = 0
        delay_by_carrier = {}
        delay_by_origin = {}
        delay_by_dest = {}
        delay_by_month = {}
        delay_by_day = {}
        delay_by_hour = {}
        
        # Process data in chunks
        for chunk in pd.read_csv(file_path, chunksize=chunksize):
            # Count flights
            chunk_flights = len(chunk)
            total_flights += chunk_flights
            
            # Handle column standardization
            arr_delay_col = 'ARR_DELAY' if 'ARR_DELAY' in chunk.columns else 'ARRIVAL_DELAY'
            
            # Mark delayed flights
            chunk['IS_DELAYED'] = (chunk[arr_delay_col] > delay_threshold).astype(int)
            delayed_flights += chunk['IS_DELAYED'].sum()
            
            # Delay by carrier
            if 'AIRLINE' in chunk.columns or 'CARRIER' in chunk.columns or 'OP_CARRIER' in chunk.columns:
                carrier_col = [col for col in ['OP_CARRIER', 'CARRIER', 'AIRLINE'] if col in chunk.columns][0]
                carrier_delays = chunk.groupby(carrier_col)['IS_DELAYED'].agg(['sum', 'count']).reset_index()
                
                for _, row in carrier_delays.iterrows():
                    carrier = row[carrier_col]
                    if carrier not in delay_by_carrier:
                        delay_by_carrier[carrier] = {'delayed': 0, 'total': 0}
                    
                    delay_by_carrier[carrier]['delayed'] += row['sum']
                    delay_by_carrier[carrier]['total'] += row['count']
            
            # Delay by origin airport
            if 'ORIGIN' in chunk.columns:
                origin_delays = chunk.groupby('ORIGIN')['IS_DELAYED'].agg(['sum', 'count']).reset_index()
                
                for _, row in origin_delays.iterrows():
                    origin = row['ORIGIN']
                    if origin not in delay_by_origin:
                        delay_by_origin[origin] = {'delayed': 0, 'total': 0}
                    
                    delay_by_origin[origin]['delayed'] += row['sum']
                    delay_by_origin[origin]['total'] += row['count']
            
            # Delay by destination airport
            if 'DEST' in chunk.columns:
                dest_delays = chunk.groupby('DEST')['IS_DELAYED'].agg(['sum', 'count']).reset_index()
                
                for _, row in dest_delays.iterrows():
                    dest = row['DEST']
                    if dest not in delay_by_dest:
                        delay_by_dest[dest] = {'delayed': 0, 'total': 0}
                    
                    delay_by_dest[dest]['delayed'] += row['sum']
                    delay_by_dest[dest]['total'] += row['count']
            
            # Parse date and time information
            if 'FL_DATE' in chunk.columns and pd.api.types.is_string_dtype(chunk['FL_DATE']):
                chunk['FL_DATE'] = pd.to_datetime(chunk['FL_DATE'])
                
                # Delay by month
                month_delays = chunk.groupby(chunk['FL_DATE'].dt.month)['IS_DELAYED'].agg(['sum', 'count']).reset_index()
                
                for _, row in month_delays.iterrows():
                    month = row['FL_DATE']
                    if month not in delay_by_month:
                        delay_by_month[month] = {'delayed': 0, 'total': 0}
                    
                    delay_by_month[month]['delayed'] += row['sum']
                    delay_by_month[month]['total'] += row['count']
                
                # Delay by day of week
                day_delays = chunk.groupby(chunk['FL_DATE'].dt.dayofweek)['IS_DELAYED'].agg(['sum', 'count']).reset_index()
                
                for _, row in day_delays.iterrows():
                    day = row['FL_DATE']
                    if day not in delay_by_day:
                        delay_by_day[day] = {'delayed': 0, 'total': 0}
                    
                    delay_by_day[day]['delayed'] += row['sum']
                    delay_by_day[day]['total'] += row['count']
            
            # Delay by hour
            if 'CRS_DEP_TIME' in chunk.columns:
                # Convert departure time to hour
                chunk['DEP_HOUR'] = chunk['CRS_DEP_TIME'].apply(lambda x: int(str(int(x)).zfill(4)[:2]) if not pd.isna(x) else np.nan)
                hour_delays = chunk.groupby('DEP_HOUR')['IS_DELAYED'].agg(['sum', 'count']).reset_index()
                
                for _, row in hour_delays.iterrows():
                    hour = row['DEP_HOUR']
                    if pd.isna(hour):
                        continue
                        
                    if hour not in delay_by_hour:
                        delay_by_hour[hour] = {'delayed': 0, 'total': 0}
                    
                    delay_by_hour[hour]['delayed'] += row['sum']
                    delay_by_hour[hour]['total'] += row['count']
        
        # Calculate overall delay rate
        overall_delay_rate = delayed_flights / total_flights if total_flights > 0 else 0
        
        # Calculate delay rates for each carrier, airport, etc.
        carrier_delay_rates = {carrier: data['delayed'] / data['total'] if data['total'] > 0 else 0 
                             for carrier, data in delay_by_carrier.items()}
        
        origin_delay_rates = {origin: data['delayed'] / data['total'] if data['total'] > 0 else 0 
                            for origin, data in delay_by_origin.items()}
        
        dest_delay_rates = {dest: data['delayed'] / data['total'] if data['total'] > 0 else 0 
                          for dest, data in delay_by_dest.items()}
        
        month_delay_rates = {month: data['delayed'] / data['total'] if data['total'] > 0 else 0 
                           for month, data in delay_by_month.items()}
        
        day_delay_rates = {day: data['delayed'] / data['total'] if data['total'] > 0 else 0 
                         for day, data in delay_by_day.items()}
        
        hour_delay_rates = {hour: data['delayed'] / data['total'] if data['total'] > 0 else 0 
                          for hour, data in delay_by_hour.items()}
        
        return {
            'overall_delay_rate': overall_delay_rate,
            'carrier_delay_rates': pd.Series(carrier_delay_rates).sort_values(ascending=False),
            'origin_delay_rates': pd.Series(origin_delay_rates).sort_values(ascending=False),
            'dest_delay_rates': pd.Series(dest_delay_rates).sort_values(ascending=False),
            'month_delay_rates': pd.Series(month_delay_rates),
            'day_delay_rates': pd.Series(day_delay_rates),
            'hour_delay_rates': pd.Series(hour_delay_rates)
        }
        
    except Exception as e:
        print(f"Error calculating delay frequency: {e}")
        return {}


def analyze_temporal_patterns(file_path: str, chunksize: int = 100000) -> Dict[str, pd.DataFrame]:
    """
    Analyze temporal patterns of flight delays.
    
    Args:
        file_path: Path to the CSV file
        chunksize: Number of rows to process at a time
    
    Returns:
        Dictionary containing DataFrames for different temporal patterns
    """
    try:
        # Initialize storage for hourly, daily, and monthly metrics
        hourly_metrics = {hour: {'flights': 0, 'delay_sum': 0, 'delay_count': 0} for hour in range(24)}
        daily_metrics = {day: {'flights': 0, 'delay_sum': 0, 'delay_count': 0} for day in range(7)}
        monthly_metrics = {month: {'flights': 0, 'delay_sum': 0, 'delay_count': 0} for month in range(1, 13)}
        
        # Process data in chunks
        for chunk in pd.read_csv(file_path, chunksize=chunksize):
            # Ensure we have the right columns
            arr_delay_col = 'ARR_DELAY' if 'ARR_DELAY' in chunk.columns else 'ARRIVAL_DELAY'
            
            # Handle date conversion
            if 'FL_DATE' in chunk.columns and pd.api.types.is_string_dtype(chunk['FL_DATE']):
                chunk['FL_DATE'] = pd.to_datetime(chunk['FL_DATE'])
                
                # Monthly analysis
                month_groups = chunk.groupby(chunk['FL_DATE'].dt.month).agg({
                    arr_delay_col: ['count', 'sum', lambda x: x[x > 0].count()]
                })
                
                for month, (total, delay_sum, delay_count) in zip(
                    month_groups.index, 
                    month_groups[arr_delay_col].values
                ):
                    monthly_metrics[month]['flights'] += total
                    monthly_metrics[month]['delay_sum'] += delay_sum
                    monthly_metrics[month]['delay_count'] += delay_count
                
                # Daily analysis
                day_groups = chunk.groupby(chunk['FL_DATE'].dt.dayofweek).agg({
                    arr_delay_col: ['count', 'sum', lambda x: x[x > 0].count()]
                })
                
                for day, (total, delay_sum, delay_count) in zip(
                    day_groups.index, 
                    day_groups[arr_delay_col].values
                ):
                    daily_metrics[day]['flights'] += total
                    daily_metrics[day]['delay_sum'] += delay_sum
                    daily_metrics[day]['delay_count'] += delay_count
            
            # Hourly analysis
            if 'CRS_DEP_TIME' in chunk.columns:
                # Convert departure time to hour
                chunk['DEP_HOUR'] = chunk['CRS_DEP_TIME'].apply(lambda x: int(str(int(x)).zfill(4)[:2]) if not pd.isna(x) else np.nan)
                
                hour_groups = chunk.groupby('DEP_HOUR').agg({
                    arr_delay_col: ['count', 'sum', lambda x: x[x > 0].count()]
                })
                
                for hour, (total, delay_sum, delay_count) in zip(
                    hour_groups.index, 
                    hour_groups[arr_delay_col].values
                ):
                    if pd.isna(hour) or hour >= 24:
                        continue
                    
                    hourly_metrics[hour]['flights'] += total
                    hourly_metrics[hour]['delay_sum'] += delay_sum
                    hourly_metrics[hour]['delay_count'] += delay_count
        
        # Create DataFrames from the metrics dictionaries
        hourly_df = pd.DataFrame({
            'flights': [data['flights'] for data in hourly_metrics.values()],
            'delay_sum': [data['delay_sum'] for data in hourly_metrics.values()],
            'delay_count': [data['delay_count'] for data in hourly_metrics.values()]
        }, index=hourly_metrics.keys())
        
        daily_df = pd.DataFrame({
            'flights': [data['flights'] for data in daily_metrics.values()],
            'delay_sum': [data['delay_sum'] for data in daily_metrics.values()],
            'delay_count': [data['delay_count'] for data in daily_metrics.values()]
        }, index=daily_metrics.keys())
        
        monthly_df = pd.DataFrame({
            'flights': [data['flights'] for data in monthly_metrics.values()],
            'delay_sum': [data['delay_sum'] for data in monthly_metrics.values()],
            'delay_count': [data['delay_count'] for data in monthly_metrics.values()]
        }, index=monthly_metrics.keys())
        
        # Calculate derived metrics
        for df in [hourly_df, daily_df, monthly_df]:
            df['delay_rate'] = df['delay_count'] / df['flights']
            df['avg_delay'] = df['delay_sum'] / df['delay_count']
            df['avg_delay'].fillna(0, inplace=True)
        
        # Map day numbers to names
        day_names = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 
                     3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
        daily_df = daily_df.rename(index=day_names)
        
        # Map month numbers to names
        month_names = {1: 'January', 2: 'February', 3: 'March', 4: 'April',
                       5: 'May', 6: 'June', 7: 'July', 8: 'August',
                       9: 'September', 10: 'October', 11: 'November', 12: 'December'}
        monthly_df = monthly_df.rename(index=month_names)
        
        return {
            'hourly': hourly_df,
            'daily': daily_df,
            'monthly': monthly_df
        }
        
    except Exception as e:
        print(f"Error analyzing temporal patterns: {e}")
        return {'hourly': pd.DataFrame(), 'daily': pd.DataFrame(), 'monthly': pd.DataFrame()}


def analyze_geographic_patterns(file_path: str, chunksize: int = 100000) -> Dict[str, pd.DataFrame]:
    """
    Analyze geographic patterns of flight delays by airports and routes.
    
    Args:
        file_path: Path to the CSV file
        chunksize: Number of rows to process at a time
    
    Returns:
        Dictionary containing DataFrames for airport and route delay patterns
    """
    try:
        # Initialize storage for airport and route metrics
        airport_metrics = {}  # {airport_code: {'departures': 0, 'arrivals': 0, 'dep_delay_sum': 0, 'arr_delay_sum': 0, ...}}
        route_metrics = {}    # {origin-dest: {'flights': 0, 'delay_sum': 0, ...}}
        
        # Process data in chunks
        for chunk in pd.read_csv(file_path, chunksize=chunksize):
            # Ensure we have the right columns
            arr_delay_col = 'ARR_DELAY' if 'ARR_DELAY' in chunk.columns else 'ARRIVAL_DELAY'
            dep_delay_col = 'DEP_DELAY' if 'DEP_DELAY' in chunk.columns else 'DEPARTURE_DELAY'
            
            if 'ORIGIN' in chunk.columns and 'DEST' in chunk.columns:
                # Process origin airports
                origin_groups = chunk.groupby('ORIGIN').agg({
                    dep_delay_col: ['count', 'sum', lambda x: x[x > 0].count()]
                })
                
                for origin, (total, delay_sum, delay_count) in zip(
                    origin_groups.index, 
                    origin_groups[dep_delay_col].values
                ):
                    if origin not in airport_metrics:
                        airport_metrics[origin] = {
                            'departures': 0, 'arrivals': 0, 
                            'dep_delay_sum': 0, 'arr_delay_sum': 0,
                            'dep_delay_count': 0, 'arr_delay_count': 0
                        }
                    
                    airport_metrics[origin]['departures'] += total
                    airport_metrics[origin]['dep_delay_sum'] += delay_sum
                    airport_metrics[origin]['dep_delay_count'] += delay_count
                
                # Process destination airports
                dest_groups = chunk.groupby('DEST').agg({
                    arr_delay_col: ['count', 'sum', lambda x: x[x > 0].count()]
                })
                
                for dest, (total, delay_sum, delay_count) in zip(
                    dest_groups.index, 
                    dest_groups[arr_delay_col].values
                ):
                    if dest not in airport_metrics:
                        airport_metrics[dest] = {
                            'departures': 0, 'arrivals': 0, 
                            'dep_delay_sum': 0, 'arr_delay_sum': 0,
                            'dep_delay_count': 0, 'arr_delay_count': 0
                        }
                    
                    airport_metrics[dest]['arrivals'] += total
                    airport_metrics[dest]['arr_delay_sum'] += delay_sum
                    airport_metrics[dest]['arr_delay_count'] += delay_count
                
                # Process routes
                chunk['ROUTE'] = chunk['ORIGIN'] + '-' + chunk['DEST']
                route_groups = chunk.groupby('ROUTE').agg({
                    arr_delay_col: ['count', 'sum', lambda x: x[x > 0].count()]
                })
                
                for route, (total, delay_sum, delay_count) in zip(
                    route_groups.index, 
                    route_groups[arr_delay_col].values
                ):
                    if route not in route_metrics:
                        route_metrics[route] = {
                            'flights': 0, 
                            'delay_sum': 0,
                            'delay_count': 0
                        }
                    
                    route_metrics[route]['flights'] += total
                    route_metrics[route]['delay_sum'] += delay_sum
                    route_metrics[route]['delay_count'] += delay_count
        
        # Create DataFrames from the metrics dictionaries
        airports_df = pd.DataFrame({
            'departures': [data['departures'] for data in airport_metrics.values()],
            'arrivals': [data['arrivals'] for data in airport_metrics.values()],
            'total_flights': [data['departures'] + data['arrivals'] for data in airport_metrics.values()],
            'dep_delay_sum': [data['dep_delay_sum'] for data in airport_metrics.values()],
            'arr_delay_sum': [data['arr_delay_sum'] for data in airport_metrics.values()],
            'total_delay_sum': [data['dep_delay_sum'] + data['arr_delay_sum'] for data in airport_metrics.values()],
            'dep_delay_count': [data['dep_delay_count'] for data in airport_metrics.values()],
            'arr_delay_count': [data['arr_delay_count'] for data in airport_metrics.values()],
            'total_delay_count': [data['dep_delay_count'] + data['arr_delay_count'] for data in airport_metrics.values()]
        }, index=airport_metrics.keys())
        
        routes_df = pd.DataFrame({
            'flights': [data['flights'] for data in route_metrics.values()],
            'delay_sum': [data['delay_sum'] for data in route_metrics.values()],
            'delay_count': [data['delay_count'] for data in route_metrics.values()]
        }, index=route_metrics.keys())
        
        # Calculate derived metrics
        # For airports
        airports_df['dep_delay_rate'] = airports_df['dep_delay_count'] / airports_df['departures']
        airports_df['arr_delay_rate'] = airports_df['arr_delay_count'] / airports_df['arrivals']
        airports_df['avg_dep_delay'] = airports_df['dep_delay_sum'] / airports_df['dep_delay_count']
        airports_df['avg_arr_delay'] = airports_df['arr_delay_sum'] / airports_df['arr_delay_count']
        airports_df['delay_rate'] = airports_df['total_delay_count'] / airports_df['total_flights']
        airports_df['avg_delay'] = airports_df['total_delay_sum'] / airports_df['total_delay_count']
        
        # Fill NaN values
        airports_df.fillna(0, inplace=True)
        
        # For routes
        routes_df['delay_rate'] = routes_df['delay_count'] / routes_df['flights']
        routes_df['avg_delay'] = routes_df['delay_sum'] / routes_df['delay_count']
        routes_df.fillna(0, inplace=True)
        
        # Filter out routes with too few flights for statistical significance
        routes_df = routes_df[routes_df['flights'] >= 100]
        
        return {
            'airports': airports_df,
            'routes': routes_df
        }
        
    except Exception as e:
        print(f"Error analyzing geographic patterns: {e}")
        return {'airports': pd.DataFrame(), 'routes': pd.DataFrame()}


def analyze_carrier_delays(file_path: str, chunksize: int = 100000) -> pd.DataFrame:
    """
    Analyze flight delays by airline carrier.
    
    Args:
        file_path: Path to the CSV file
        chunksize: Number of rows to process at a time
    
    Returns:
        DataFrame containing carrier delay statistics
    """
    try:
        # Initialize storage for carrier metrics
        carrier_metrics = {}  # {carrier_code: {'flights': 0, 'delay_sum': 0, 'delay_count': 0, ...}}
        
        # Process data in chunks
        for chunk in pd.read_csv(file_path, chunksize=chunksize):
            # Ensure we have the right columns
            arr_delay_col = 'ARR_DELAY' if 'ARR_DELAY' in chunk.columns else 'ARRIVAL_DELAY'
            
            # Determine carrier column
            carrier_col = None
            for col in ['OP_CARRIER', 'CARRIER', 'AIRLINE', 'UNIQUE_CARRIER']:
                if col in chunk.columns:
                    carrier_col = col
                    break
            
            if carrier_col is not None:
                # Process carriers
                carrier_groups = chunk.groupby(carrier_col).agg({
                    arr_delay_col: ['count', 'sum', lambda x: x[x > 0].count(), 'mean']
                })
                
                for carrier, (total, delay_sum, delay_count, delay_mean) in zip(
                    carrier_groups.index, 
                    carrier_groups[arr_delay_col].values
                ):
                    if carrier not in carrier_metrics:
                        carrier_metrics[carrier] = {
                            'flights': 0, 
                            'delay_sum': 0,
                            'delay_count': 0,
                            'delay_total': 0  # For weighted average calculation
                        }
                    
                    carrier_metrics[carrier]['flights'] += total
                    carrier_metrics[carrier]['delay_sum'] += delay_sum
                    carrier_metrics[carrier]['delay_count'] += delay_count
                    carrier_metrics[carrier]['delay_total'] += delay_mean * total  # Weighted by flights
        
        # Create DataFrame from the metrics dictionary
        carriers_df = pd.DataFrame({
            'flights': [data['flights'] for data in carrier_metrics.values()],
            'delay_sum': [data['delay_sum'] for data in carrier_metrics.values()],
            'delay_count': [data['delay_count'] for data in carrier_metrics.values()],
            'delay_total': [data['delay_total'] for data in carrier_metrics.values()]
        }, index=carrier_metrics.keys())
        
        # Calculate derived metrics
        carriers_df['delay_rate'] = carriers_df['delay_count'] / carriers_df['flights']
        carriers_df['avg_delay'] = carriers_df['delay_sum'] / carriers_df['delay_count']
        carriers_df['weighted_avg_delay'] = carriers_df['delay_total'] / carriers_df['flights']
        
        # Fill NaN values
        carriers_df.fillna(0, inplace=True)
        
        return carriers_df
        
    except Exception as e:
        print(f"Error analyzing carrier delays: {e}")
        return pd.DataFrame()


def analyze_distance_vs_delay(file_path: str, chunksize: int = 100000) -> pd.DataFrame:
    """
    Analyze relationship between flight distance and delays.
    
    Args:
        file_path: Path to the CSV file
        chunksize: Number of rows to process at a time
    
    Returns:
        DataFrame containing distance vs. delay analysis
    """
    try:
        # Initialize storage for distance bins
        distance_bins = [0, 250, 500, 750, 1000, 1500, 2000, 3000, float('inf')]
        bin_labels = ['0-250', '250-500', '500-750', '750-1000', '1000-1500', '1500-2000', '2000-3000', '3000+']
        distance_metrics = {label: {'flights': 0, 'delay_sum': 0, 'delay_count': 0} for label in bin_labels}
        
        # Process data in chunks
        for chunk in pd.read_csv(file_path, chunksize=chunksize):
            # Ensure we have the right columns
            arr_delay_col = 'ARR_DELAY' if 'ARR_DELAY' in chunk.columns else 'ARRIVAL_DELAY'
            distance_col = None
            
            for col in ['DISTANCE', 'DISTANCE_GROUP']:
                if col in chunk.columns:
                    distance_col = col
                    break
            
            if distance_col is not None and arr_delay_col in chunk.columns:
                # Create distance bins
                chunk['DISTANCE_BIN'] = pd.cut(
                    chunk[distance_col], 
                    bins=distance_bins,
                    labels=bin_labels
                )
                
                # Group by distance bin
                distance_groups = chunk.groupby('DISTANCE_BIN').agg({
                    arr_delay_col: ['count', 'sum', lambda x: x[x > 0].count()]
                })
                
                for bin_label, (total, delay_sum, delay_count) in zip(
                    distance_groups.index, 
                    distance_groups[arr_delay_col].values
                ):
                    if pd.isna(bin_label):
                        continue
                        
                    distance_metrics[bin_label]['flights'] += total
                    distance_metrics[bin_label]['delay_sum'] += delay_sum
                    distance_metrics[bin_label]['delay_count'] += delay_count
        
        # Create DataFrame from the metrics dictionary
        distance_df = pd.DataFrame({
            'flights': [data['flights'] for data in distance_metrics.values()],
            'delay_sum': [data['delay_sum'] for data in distance_metrics.values()],
            'delay_count': [data['delay_count'] for data in distance_metrics.values()]
        }, index=distance_metrics.keys())
        
        # Calculate derived metrics
        distance_df['delay_rate'] = distance_df['delay_count'] / distance_df['flights']
        distance_df['avg_delay'] = distance_df['delay_sum'] / distance_df['delay_count']
        
        # Fill NaN values
        distance_df.fillna(0, inplace=True)
        
        return distance_df
        
    except Exception as e:
        print(f"Error analyzing distance vs delay: {e}")
        return pd.DataFrame()


def calculate_correlations(file_path: str, chunksize: int = 100000, 
                           max_samples: int = 100000) -> pd.DataFrame:
    """
    Calculate correlations between numeric variables in chunks to manage memory.
    
    Args:
        file_path: Path to the CSV file
        chunksize: Number of rows to process at a time
        max_samples: Maximum number of samples to use for correlation calculation
    
    Returns:
        DataFrame containing correlation matrix
    """
    try:
        # Sample the data to avoid memory issues
        sampled_chunks = []
        total_rows = 0
        
        for chunk in pd.read_csv(file_path, chunksize=chunksize):
            # Skip if we already have enough samples
            if total_rows >= max_samples:
                break
                
            # Sample this chunk if needed
            if len(chunk) > (max_samples - total_rows):
                sample_size = max_samples - total_rows
                chunk = chunk.sample(sample_size)
            
            # Keep only numeric columns
            numeric_chunk = chunk.select_dtypes(include=[np.number])
            sampled_chunks.append(numeric_chunk)
            total_rows += len(numeric_chunk)
            
            if total_rows >= max_samples:
                break
        
        # Combine the samples
        combined_df = pd.concat(sampled_chunks)
        
        # Calculate correlations
        corr_matrix = combined_df.corr()
        
        return corr_matrix
        
    except Exception as e:
        print(f"Error calculating correlations: {e}")
        return pd.DataFrame()


def save_insights_report(insights: Dict[str, Any], report_path: str) -> None:
    """
    Save exploratory data analysis insights to a Markdown report.
    
    Args:
        insights: Dictionary containing analysis insights
        report_path: Path to save the report file
    
    Returns:
        None
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        # Generate report content
        report_content = f"""# Flight Delay EDA Summary Report
Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Key Findings

### Overall Statistics
- Overall flight delay rate: {insights['delay_rate']:.2%}

### Temporal Patterns
- Month with highest average delay: {insights['worst_month']}
- Day of week with highest average delay: {insights['worst_day']} 
- Hour of day with highest average delay: {insights['worst_hour']:02d}:00

### Geographic Patterns
- Airport with highest average delay: {insights['worst_airport']}
- Route with highest average delay: {insights['worst_route']}

### Carrier Performance
- Carrier with highest average delay: {insights['worst_carrier']}

### Key Correlations
Top correlations with delay:
"""        # Add correlation information
        for (var1, var2), corr_value in insights['top_correlations'].items():
            report_content += f"- {var1} vs {var2}: {corr_value:.3f}\n"
        
        # Write the report to file
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        print(f"Report saved to {report_path}")
        
    except Exception as e:
        print(f"Error saving insights report: {e}")
