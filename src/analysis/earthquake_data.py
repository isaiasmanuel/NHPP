"""
Baseball hit data loading and preprocessing.

Handles hit catalog filtering, temporal binning, and visualization.
Uses StatCast hit coordinate data (x, y) and game date information.
"""

import pandas as pd
import numpy as np
from datetime import datetime


def load_and_filter_hits(
    csv_path,
    xlim,
    Xlim,
    ylim,
    Ylim,
    exit_velocity_min=50,
    launch_angle_min=None,
    launch_angle_max=None,
    exclude_region=None,
):
    """
    Load and filter baseball hit catalog.
    
    Parameters
    ----------
    csv_path : str
        Path to hit catalog CSV file (StatCast format)
    xlim, Xlim, ylim, Ylim : float
        Field coordinate domain bounds (feet)
    exit_velocity_min : float, optional
        Minimum exit velocity in mph (default: 50)
    launch_angle_min : float, optional
        Minimum launch angle in degrees
    launch_angle_max : float, optional
        Maximum launch angle in degrees
    exclude_region : tuple, optional
        Region to exclude: ((x_min, x_max), (y_min, y_max))
    
    Returns
    -------
    pd.DataFrame
        Filtered hit catalog
    """
    df = pd.read_csv(csv_path)

    # Spatial filter: field coordinates
    df = df[(df["hc_x"] < Xlim) & (df["hc_x"] > xlim)]
    df = df[(df["hc_y"] < Ylim) & (df["hc_y"] > ylim)]

    # Exit velocity filter
    df = df[df["exit_velocity"] >= exit_velocity_min]

    # Launch angle filters
    if launch_angle_min is not None:
        df = df[df["launch_angle"] >= launch_angle_min]
    if launch_angle_max is not None:
        df = df[df["launch_angle"] <= launch_angle_max]

    # Exclude region if specified
    if exclude_region is not None:
        (x_min, x_max), (y_min, y_max) = exclude_region
        df = df[~((df["hc_x"] > x_min) & (df["hc_x"] < x_max) & 
                  (df["hc_y"] > y_min) & (df["hc_y"] < y_max))]

    df = df.reset_index(drop=True)
    return df


def hits_to_spatiotemporal(hits, min_date, max_date):
    """
    Convert hit catalog to spatio-temporal point process format.
    
    Parameters
    ----------
    hits : pd.DataFrame
        Hit catalog with game_date column and hc_x, hc_y coordinates
    min_date : datetime
        Reference start time
    max_date : datetime
        Reference end time
    
    Returns
    -------
    tuple
        (SimPoisson, Tmax) where SimPoisson has columns [x, y, time_days]
    """
    # Convert game_date to datetime if not already
    if not isinstance(hits['game_date'].dtype, 'datetime64'):
        dates = pd.to_datetime(hits['game_date'])
    else:
        dates = hits['game_date']

    # Convert to days since min_date
    seconds_per_day = 86400
    time_diffs = np.array([(d - min_date).total_seconds() for d in dates])
    time_diffs = time_diffs / seconds_per_day

    # Time span
    Tmax = (max_date - min_date).total_seconds() / seconds_per_day

    SimPoisson = hits[["hc_x", "hc_y"]].to_numpy()
    SimPoisson = np.hstack((SimPoisson, time_diffs.reshape(-1, 1)))

    return SimPoisson, Tmax


def assign_time_partition(SimPoisson, pT, Tmax):
    """
    Assign each hit to a temporal partition.
    
    Parameters
    ----------
    SimPoisson : ndarray
        Spatio-temporal data (N x 3): [x, y, time]
    pT : int
        Number of temporal partitions
    Tmax : float
        Maximum time
    
    Returns
    -------
    ndarray
        Data with partition column appended (N x 4)
    """
    CutPoint = np.arange(0, pT + 1) * Tmax / pT
    SimPoisson = np.hstack((SimPoisson, np.zeros((len(SimPoisson), 1))))

    for i in range(pT):
        mask = (CutPoint[i] < SimPoisson[:, 2]) & (SimPoisson[:, 2] <= CutPoint[i + 1])
        SimPoisson[mask, 3] = i

    return SimPoisson
