"""
Visualization utilities for flight delay prediction project.
This module provides functions for creating exploratory visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any
import os


def save_fig(fig, filename, reports_dir=None, dpi=300, bbox_inches='tight', subfolder='3m'):
    """
    Helper function to save figures to the reports/figures/subfolder directory.
    
    Args:
        fig: Matplotlib figure object to save
        filename: Name of the file to save
        reports_dir: Path to reports directory (defaults to project root + reports)
        dpi: Resolution for saving the figure
        bbox_inches: Bounding box parameter for saving
        subfolder: Subfolder name within figures directory (default: '3m')
    """
    if reports_dir is None:
        current_dir = os.getcwd()
        if 'notebooks' in current_dir:
            project_root = os.path.abspath(os.path.join(current_dir, '../..'))
        else:
            project_root = os.path.abspath(os.path.join(current_dir, '..'))
        
        # Include subfolder in the path
        reports_dir = os.path.join(project_root, 'reports', 'figures')
        if subfolder:
            reports_dir = os.path.join(reports_dir, subfolder)
    
    # Create the directory if it doesn't exist
    os.makedirs(reports_dir, exist_ok=True)
    
    # Save the figure
    fig_path = os.path.join(reports_dir, filename)
    fig.savefig(fig_path, dpi=dpi, bbox_inches=bbox_inches)
    plt.close(fig)
    
    print(f"Figure saved to {fig_path}")


def plot_missing_values(missing_df: pd.DataFrame):
    """
    Plot missing values in the dataset.
    
    Args:
        missing_df: DataFrame containing missing value statistics
    
    Returns:
        Matplotlib figure object
    """
    # Select top columns with missing values for visualization
    top_missing = missing_df[missing_df['missing_percent'] > 0].sort_values('missing_percent', ascending=False).head(20)
    
    if len(top_missing) == 0:
        print("No missing values found")
        return None
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create bar chart
    bars = ax.barh(top_missing.index, top_missing['missing_percent'], color='skyblue')
    
    # Add data labels
    for bar in bars:
        width = bar.get_width()
        label_x_pos = width + 0.5
        ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', 
                va='center', fontsize=9)
    
    ax.set_xlabel('Missing Values (%)')
    ax.set_title('Percentage of Missing Values by Column')
    ax.set_xlim([0, 100])  # Percentage scale
    

    plt.tight_layout()
    
    save_fig(fig, 'missing_values.png')
    
    return fig


def plot_delay_distributions(delay_stats: Dict[str, Any]):
    """
    Plot delay distributions for different delay types.
    
    Args:
        delay_stats: Dictionary containing delay samples and distributions
    
    Returns:
        Matplotlib figure objects
    """
    figures = []
    delay_samples = delay_stats['delay_samples']
    delay_bins = delay_stats['delay_distributions']
    
    # Plot histograms for main delay columns
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    axs = axs.flatten()
    
    main_delays = ['DEP_DELAY', 'ARR_DELAY']
    for i, delay_type in enumerate(main_delays):
        if delay_type in delay_samples and len(delay_samples[delay_type]) > 0:
            # Filter extreme values for better visualization
            filtered_data = [d for d in delay_samples[delay_type] if d > -60 and d < 180]
            
            sns.histplot(filtered_data, kde=True, ax=axs[i], color='skyblue')
            axs[i].set_title(f'{delay_type} Distribution')
            axs[i].set_xlabel('Delay (minutes)')
            axs[i].set_xlim([-60, 180])
            
            # Add vertical line at zero
            axs[i].axvline(x=0, color='red', linestyle='--')
            
            # Add text indicating percentage of delays
            positive_pct = sum(d > 0 for d in delay_samples[delay_type]) / len(delay_samples[delay_type])
            axs[i].text(100, axs[i].get_ylim()[1] * 0.9, f'Delayed: {positive_pct:.1%}', 
                    fontsize=12, ha='center', va='center',
                    bbox=dict(facecolor='white', alpha=0.7))
    
    # For specific delay reason types (if available)
    specific_delays = ['CARRIER_DELAY', 'WEATHER_DELAY', 'NAS_DELAY', 'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY']
    delay_data = []
    delay_names = []
    
    for delay_type in specific_delays:
        if delay_type in delay_samples and len(delay_samples[delay_type]) > 0:
            # Filter out zeros and extreme values
            filtered_data = [d for d in delay_samples[delay_type] if d > 0 and d < 180]
            if filtered_data:
                delay_data.append(filtered_data)
                delay_names.append(delay_type.replace('_DELAY', ''))
    
    if delay_data:
        # Plot violin plot for delay reasons
        axs[2].violinplot(delay_data, showmedians=True)
        axs[2].set_xticks(range(1, len(delay_names) + 1))
        axs[2].set_xticklabels(delay_names, rotation=45)
        axs[2].set_title('Distribution of Delay Reasons (When Delayed)')
        axs[2].set_ylabel('Delay (minutes)')
        axs[2].set_ylim([0, 180])  # Limit y-axis for better visualization
    
    # Plot delay bin distribution
    if 'ARR_DELAY' in delay_bins:
        delay_bins_df = pd.DataFrame(delay_bins['ARR_DELAY']).reset_index()
        delay_bins_df.columns = ['category', 'count']
        # Sort by delay category in a sensible order
        order = ['very early', 'early', 'on-time', 'slight delay', 'moderate delay', 'significant delay', 'severe delay']
        delay_bins_df['category'] = pd.Categorical(delay_bins_df['category'], categories=order, ordered=True)
        delay_bins_df = delay_bins_df.sort_values('category')
        
        sns.barplot(x='category', y='count', data=delay_bins_df, ax=axs[3], palette='viridis')
        axs[3].set_title('Arrival Delay Categories')
        axs[3].set_xticklabels(axs[3].get_xticklabels(), rotation=45, ha='right')
        axs[3].set_ylabel('Count')
    
    plt.tight_layout()
    figures.append(fig)
    save_fig(fig, 'delay_distributions.png')
    
    return figures


def plot_hourly_patterns(hourly_df: pd.DataFrame):
    """
    Plot hourly delay patterns.
    
    Args:
        hourly_df: DataFrame containing hourly delay metrics
    
    Returns:
        Matplotlib figure object
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot number of flights by hour
    ax1.bar(hourly_df.index, hourly_df['flights'], alpha=0.3, color='gray', label='Flights')
    ax1.set_xlabel('Hour of Day (24-hour)')
    ax1.set_ylabel('Number of Flights')
    ax1.tick_params(axis='y')
    
    # Create second y-axis for delay metrics
    ax2 = ax1.twinx()
    ax2.plot(hourly_df.index, hourly_df['delay_rate'], 'r-', label='Delay Rate')
    ax2.plot(hourly_df.index, hourly_df['avg_delay'] / 60, 'b-', label='Avg Delay (hours)')
    ax2.set_ylabel('Delay Rate / Avg Delay (hours)')
    ax2.tick_params(axis='y')
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.title('Flight Delays by Hour of Day')
    
    # Set x-axis limits and ticks
    ax1.set_xlim([-0.5, 23.5])
    ax1.set_xticks(range(0, 24))
    ax1.set_xticklabels([f'{i:02d}:00' for i in range(24)])
    

    plt.tight_layout()
    
    save_fig(fig, 'hourly_patterns.png')
    
    return fig


def plot_daily_patterns(daily_df: pd.DataFrame):
    """
    Plot daily delay patterns.
    
    Args:
        daily_df: DataFrame containing daily delay metrics
    
    Returns:
        Matplotlib figure object
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot number of flights by day
    ax1.bar(daily_df.index, daily_df['flights'], alpha=0.3, color='gray', label='Flights')
    ax1.set_xlabel('Day of Week')
    ax1.set_ylabel('Number of Flights')
    ax1.tick_params(axis='y')
    
    # Create second y-axis for delay metrics
    ax2 = ax1.twinx()
    ax2.plot(daily_df.index, daily_df['delay_rate'], 'r-', marker='o', label='Delay Rate')
    ax2.plot(daily_df.index, daily_df['avg_delay'], 'b-', marker='s', label='Avg Delay (min)')
    ax2.set_ylabel('Delay Rate / Avg Delay (min)')
    ax2.tick_params(axis='y')
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.title('Flight Delays by Day of Week')
    

    plt.tight_layout()
    
    save_fig(fig, 'daily_patterns.png')
    
    return fig


def plot_monthly_patterns(monthly_df: pd.DataFrame):
    """
    Plot monthly delay patterns.
    
    Args:
        monthly_df: DataFrame containing monthly delay metrics
    
    Returns:
        Matplotlib figure object
    """
    fig, ax1 = plt.subplots(figsize=(14, 6))
    
    # Plot number of flights by month
    ax1.bar(monthly_df.index, monthly_df['flights'], alpha=0.3, color='gray', label='Flights')
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Number of Flights')
    ax1.tick_params(axis='y')
    
    # Create second y-axis for delay metrics
    ax2 = ax1.twinx()
    ax2.plot(monthly_df.index, monthly_df['delay_rate'], 'r-', marker='o', label='Delay Rate')
    ax2.plot(monthly_df.index, monthly_df['avg_delay'], 'b-', marker='s', label='Avg Delay (min)')
    ax2.set_ylabel('Delay Rate / Avg Delay (min)')
    ax2.tick_params(axis='y')
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.title('Flight Delays by Month')
    

    plt.tight_layout()
    
    save_fig(fig, 'monthly_patterns.png')
    
    return fig


def plot_airport_delays(airports_df: pd.DataFrame, top_n: int = 15):
    """
    Plot airports with highest delay frequencies.
    
    Args:
        airports_df: DataFrame containing airport delay metrics
        top_n: Number of top airports to display
    
    Returns:
        Matplotlib figure object
    """
    # Select top airports by delay rate (minimum flight threshold)
    busy_airports = airports_df[airports_df['total_flights'] > 1000]
    top_delay_airports = busy_airports.sort_values('delay_rate', ascending=False).head(top_n)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create horizontal bar chart
    bars = ax.barh(top_delay_airports.index, top_delay_airports['delay_rate'] * 100, color='skyblue')
    
    # Add data labels
    for bar in bars:
        width = bar.get_width()
        label_x_pos = width + 0.5
        ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', 
                va='center')
    
    # Add flight count as text next to airport code
    for i, airport in enumerate(top_delay_airports.index):
        flights = top_delay_airports.loc[airport, 'total_flights']
        avg_delay = top_delay_airports.loc[airport, 'avg_delay']
        ax.text(-5, i, f"  {flights:,} flights, {avg_delay:.1f} min avg", ha='right', va='center', fontsize=8)
    
    ax.set_xlabel('Delay Rate (%)')
    ax.set_title(f'Top {top_n} Airports with Highest Delay Rates')
    

    plt.tight_layout()
    
    save_fig(fig, 'airport_delays.png')
    
    return fig


def plot_route_delays(routes_df: pd.DataFrame, top_n: int = 15):
    """
    Plot routes with highest delay frequencies.
    
    Args:
        routes_df: DataFrame containing route delay metrics
        top_n: Number of top routes to display
    
    Returns:
        Matplotlib figure object
    """
    # Select top routes by delay rate (minimum flight threshold)
    busy_routes = routes_df[routes_df['flights'] > 100]
    top_delay_routes = busy_routes.sort_values('delay_rate', ascending=False).head(top_n)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create horizontal bar chart
    bars = ax.barh(top_delay_routes.index, top_delay_routes['delay_rate'] * 100, color='lightgreen')
    
    # Add data labels
    for bar in bars:
        width = bar.get_width()
        label_x_pos = width + 0.5
        ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', 
                va='center')
    
    # Add flight count as text next to route code
    for i, route in enumerate(top_delay_routes.index):
        flights = top_delay_routes.loc[route, 'flights']
        avg_delay = top_delay_routes.loc[route, 'avg_delay']
        ax.text(-5, i, f"  {flights:,} flights, {avg_delay:.1f} min avg", ha='right', va='center', fontsize=8)
    
    ax.set_xlabel('Delay Rate (%)')
    ax.set_title(f'Top {top_n} Routes with Highest Delay Rates')
    

    plt.tight_layout()
    
    save_fig(fig, 'route_delays.png')
    
    return fig


def plot_correlation_heatmap(corr_matrix: pd.DataFrame):
    """
    Plot correlation heatmap between numeric variables.
    
    Args:
        corr_matrix: DataFrame containing correlation matrix
    
    Returns:
        Matplotlib figure object
    """
    # Select delay-related columns and other important features
    delay_cols = [col for col in corr_matrix.columns if 'DELAY' in col]
    important_cols = ['DISTANCE', 'AIR_TIME', 'TAXI_OUT', 'TAXI_IN', 'CRS_ELAPSED_TIME']
    selected_cols = []
    
    # Add available important columns
    for col in delay_cols + important_cols:
        if col in corr_matrix.columns:
            selected_cols.append(col)
    
    # Add a few more columns if needed to get a good selection
    if len(selected_cols) < 10:
        remaining_cols = [col for col in corr_matrix.columns if col not in selected_cols]
        selected_cols.extend(remaining_cols[:10 - len(selected_cols)])
    
    # Select the correlation submatrix
    if selected_cols:
        selected_corr = corr_matrix.loc[selected_cols, selected_cols]
    else:
        selected_corr = corr_matrix
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Create heatmap
    mask = np.triu(np.ones_like(selected_corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    fig = plt.figure(figsize=(12, 10))
    sns.heatmap(selected_corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                annot=True, fmt='.2f', square=True, linewidths=.5, cbar_kws={"shrink": .5})
    
    plt.title('Correlation Matrix of Flight Variables')
    
    plt.tight_layout()
    

    save_fig(fig, 'correlation_heatmap.png')
    
    return fig


def plot_carrier_delays(carrier_df: pd.DataFrame):
    """
    Plot carrier delay comparison.
    
    Args:
        carrier_df: DataFrame containing carrier delay metrics
    
    Returns:
        Matplotlib figure object
    """
    # Sort by delay rate and select top carriers (with minimum flight threshold)
    busy_carriers = carrier_df[carrier_df['flights'] > 1000]
    sorted_carriers = busy_carriers.sort_values('delay_rate', ascending=False)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create scatter plot
    scatter = ax.scatter(sorted_carriers['delay_rate'] * 100, 
                         sorted_carriers['avg_delay'],
                         s=sorted_carriers['flights'] / 1000,  # Size proportional to flights
                         alpha=0.6,
                         c=sorted_carriers['flights'],
                         cmap='viridis')
    
    # Add carrier labels
    for idx, carrier in enumerate(sorted_carriers.index):
        ax.annotate(carrier, 
                   (sorted_carriers['delay_rate'][idx] * 100, sorted_carriers['avg_delay'][idx]),
                   xytext=(5, 5), textcoords='offset points')
    
    # Add a colorbar for flights
    cbar = plt.colorbar(scatter)
    cbar.set_label('Number of Flights')
    
    ax.set_xlabel('Delay Rate (%)')
    ax.set_ylabel('Average Delay (minutes)')
    ax.set_title('Carrier Delay Performance')
    
    ax.grid(True, linestyle='--', alpha=0.7)
    

    plt.tight_layout()
    

    save_fig(fig, 'carrier_delays.png')
    
    return fig


def plot_distance_vs_delay(distance_df: pd.DataFrame):
    """
    Plot relationship between flight distance and delays.
    
    Args:
        distance_df: DataFrame containing distance vs. delay metrics
    
    Returns:
        Matplotlib figure object
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot number of flights by distance bin
    ax1.bar(distance_df.index, distance_df['flights'], alpha=0.3, color='gray', label='Flights')
    ax1.set_xlabel('Distance (miles)')
    ax1.set_ylabel('Number of Flights')
    ax1.tick_params(axis='y')
    
    # Create second y-axis for delay metrics
    ax2 = ax1.twinx()
    ax2.plot(distance_df.index, distance_df['delay_rate'], 'r-', marker='o', label='Delay Rate')
    ax2.plot(distance_df.index, distance_df['avg_delay'], 'b-', marker='s', label='Avg Delay (min)')
    ax2.set_ylabel('Delay Rate / Avg Delay (min)')
    ax2.tick_params(axis='y')
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    plt.title('Flight Delays by Distance')
    plt.xticks(rotation=45) 
    plt.tight_layout()  
    save_fig(fig, 'distance_vs_delay.png')
    
    return fig
