"""
Baseball StatCast data extraction and loading.

Provides utilities to fetch and format MLB StatCast data for spatio-temporal analysis
of hit clustering patterns.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings

try:
    from pybaseball import statcast
except ImportError:
    warnings.warn(
        "pybaseball not installed. Run: pip install pybaseball\n"
        "Alternatively, download CSV directly from https://baseballsavant.com/statcast"
    )


def fetch_statcast_data(year, min_exit_velocity=50, hit_types=None):
    """
    Fetch StatCast data for a given season and filter by criteria.
    
    Parameters
    ----------
    year : int
        MLB season year (2015-present)
    min_exit_velocity : float, optional
        Minimum exit velocity in mph (default: 50)
    hit_types : list, optional
        Hit types to include, e.g., ['single', 'double', 'triple', 'home_run']
        Default includes all in-play hits
    
    Returns
    -------
    pd.DataFrame
        Filtered StatCast data with columns: [pitcher_id, game_date, hc_x, hc_y, 
                                              exit_velocity, launch_angle, distance, hit_type]
    
    Notes
    -----
    Requires pybaseball: pip install pybaseball
    
    Examples
    --------
    >>> data = fetch_statcast_data(2023, min_exit_velocity=50)
    >>> print(data.shape)
    (50000, 8)
    """
    print(f"Fetching StatCast data for {year}...")
    df = statcast(year)
    
    if df is None or df.empty:
        raise ValueError(f"No StatCast data available for year {year}")
    
    # Filter: balls in play only
    df = df[df['inplay_run_expectancy'].notna()].copy()
    
    # Filter: exit velocity threshold
    df = df[df['launch_speed'] >= min_exit_velocity].copy()
    
    # Filter: hit types if specified
    if hit_types is not None:
        df = df[df['type'].isin(hit_types)].copy()
    
    # Standardize column names and keep relevant columns
    df_filtered = pd.DataFrame({
        'pitcher_id': df['pitcher'].astype('Int64'),
        'game_date': pd.to_datetime(df['game_date']),
        'hc_x': df['hc_x'],  # Hit coordinates
        'hc_y': df['hc_y'],
        'exit_velocity': df['launch_speed'],
        'launch_angle': df['launch_angle'],
        'distance': df['estimated_ba_using_speedangle'],
        'hit_type': df['type'],
        'batter_id': df['batter'].astype('Int64'),
    })
    
    # Remove rows with missing coordinate data
    df_filtered = df_filtered.dropna(subset=['hc_x', 'hc_y']).reset_index(drop=True)
    
    print(f"  Loaded {len(df_filtered)} hits from {year} season")
    
    return df_filtered


def load_statcast_csv(csv_path, min_exit_velocity=50, hit_types=None):
    """
    Load StatCast data from downloaded CSV file.
    
    Parameters
    ----------
    csv_path : str
        Path to StatCast CSV (download from https://baseballsavant.com/statcast)
    min_exit_velocity : float, optional
        Minimum exit velocity in mph (default: 50)
    hit_types : list, optional
        Hit types to include
    
    Returns
    -------
    pd.DataFrame
        Filtered StatCast data
    
    Examples
    --------
    >>> data = load_statcast_csv('data/raw/statcast_2023.csv', min_exit_velocity=50)
    """
    df = pd.read_csv(csv_path)
    
    # Rename columns to match standard StatCast naming
    rename_map = {
        'pitcher': 'pitcher_id',
        'batter': 'batter_id',
        'launch_speed': 'exit_velocity',
        'estimated_woba_using_speedangle': 'distance',
        'game_date': 'game_date',
    }
    df = df.rename(columns=rename_map)
    
    # Filter: balls in play
    if 'type' in df.columns:
        df = df[df['type'] == 'X'].copy()
    
    # Filter: exit velocity
    df = df[df['exit_velocity'] >= min_exit_velocity].copy()
    
    # Filter: hit types
    if hit_types is not None and 'events' in df.columns:
        df = df[df['events'].isin(hit_types)].copy()
    
    # Keep relevant columns
    cols = ['pitcher_id', 'game_date', 'hc_x', 'hc_y', 'exit_velocity', 
            'launch_angle', 'distance', 'batter_id']
    df_filtered = df[[c for c in cols if c in df.columns]].copy()
    df_filtered = df_filtered.dropna(subset=['hc_x', 'hc_y']).reset_index(drop=True)
    
    print(f"Loaded {len(df_filtered)} hits from CSV")
    return df_filtered


def prepare_hits_spatiotemporal(hits_df, reference_date=None):
    """
    Convert hit catalog to spatio-temporal point process format.
    
    Transforms hit coordinates and game dates into the format needed for 
    spatio-temporal clustering analysis.
    
    Parameters
    ----------
    hits_df : pd.DataFrame
        Hit data with columns: [hc_x, hc_y, game_date, ...]
    reference_date : datetime, optional
        Reference start date. If None, uses first game date.
    
    Returns
    -------
    tuple
        (SimPoisson, Tmax) where:
        - SimPoisson: ndarray of shape (N, 3) with columns [x, y, time_days]
        - Tmax: float, total time span in days
    
    Examples
    --------
    >>> data_spatiotemporal, tmax = prepare_hits_spatiotemporal(hits_df)
    >>> print(data_spatiotemporal.shape)
    (5000, 3)
    >>> print(f"Season duration: {tmax:.1f} days")
    Season duration: 180.0 days
    """
    hits_copy = hits_df.copy()
    hits_copy['game_date'] = pd.to_datetime(hits_copy['game_date'])
    
    if reference_date is None:
        reference_date = hits_copy['game_date'].min()
    
    max_date = hits_copy['game_date'].max()
    
    # Convert dates to days since reference
    time_days = (hits_copy['game_date'] - reference_date).dt.total_seconds() / 86400
    
    # Build spatio-temporal point pattern
    SimPoisson = hits_copy[['hc_x', 'hc_y']].to_numpy()
    SimPoisson = np.hstack((SimPoisson, time_days.values.reshape(-1, 1)))
    
    Tmax = (max_date - reference_date).total_seconds() / 86400
    
    return SimPoisson, Tmax


def assign_temporal_partition(SimPoisson, pT, Tmax):
    """
    Assign each hit to a temporal partition.
    
    Useful for analyzing time-varying hit patterns across a season.
    
    Parameters
    ----------
    SimPoisson : ndarray
        Spatio-temporal data (N x 3): [x, y, time_days]
    pT : int
        Number of temporal partitions (e.g., innings, weeks)
    Tmax : float
        Maximum time (days)
    
    Returns
    -------
    ndarray
        Data with partition column appended (N x 4): [x, y, time, partition_id]
    
    Examples
    --------
    >>> data_with_partition = assign_temporal_partition(data, pT=4, Tmax=180)
    >>> print(f"Partition assignments: {data_with_partition[:, 3]}")
    """
    CutPoint = np.arange(0, pT + 1) * Tmax / pT
    SimPoisson_partitioned = np.hstack((SimPoisson, np.zeros((len(SimPoisson), 1))))
    
    for i in range(pT):
        mask = (CutPoint[i] < SimPoisson_partitioned[:, 2]) & (
            SimPoisson_partitioned[:, 2] <= CutPoint[i + 1]
        )
        SimPoisson_partitioned[mask, 3] = i
    
    return SimPoisson_partitioned


def quick_load_season(year, min_exit_velocity=50, reference_date=None, pT=4):
    """
    Quick one-liner to load and prepare a full season of hit data.
    
    Parameters
    ----------
    year : int
        MLB season to load
    min_exit_velocity : float, optional
        Minimum exit velocity threshold
    reference_date : datetime, optional
        Reference date for temporal conversion
    pT : int, optional
        Number of temporal partitions
    
    Returns
    -------
    dict
        Dictionary with keys: ['hits', 'SimPoisson', 'Tmax', 'hits_partitioned']
    
    Examples
    --------
    >>> data = quick_load_season(2023, min_exit_velocity=50, pT=4)
    >>> print(f"Total hits: {len(data['hits'])}")
    >>> print(f"Season length: {data['Tmax']:.1f} days")
    """
    print(f"\n{'='*60}")
    print(f"Loading MLB {year} season hit data...")
    print(f"{'='*60}")
    
    # Fetch data
    hits = fetch_statcast_data(year, min_exit_velocity=min_exit_velocity)
    
    # Prepare spatio-temporal format
    SimPoisson, Tmax = prepare_hits_spatiotemporal(hits, reference_date=reference_date)
    
    # Add temporal partitions
    hits_partitioned = assign_temporal_partition(SimPoisson, pT=pT, Tmax=Tmax)
    
    print(f"\n{'='*60}")
    print(f"Data Summary:")
    print(f"  Total hits: {len(hits)}")
    print(f"  Season duration: {Tmax:.1f} days")
    print(f"  Temporal partitions: {pT}")
    print(f"  Coordinate range (x): [{SimPoisson[:, 0].min():.1f}, {SimPoisson[:, 0].max():.1f}]")
    print(f"  Coordinate range (y): [{SimPoisson[:, 1].min():.1f}, {SimPoisson[:, 1].max():.1f}]")
    print(f"{'='*60}\n")
    
    return {
        'hits': hits,
        'SimPoisson': SimPoisson,
        'Tmax': Tmax,
        'hits_partitioned': hits_partitioned,
    }
