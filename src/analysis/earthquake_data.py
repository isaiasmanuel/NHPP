"""
Real earthquake data loading and preprocessing.

Handles earthquake catalog filtering, temporal binning, and visualization.
"""

import pandas as pd
import numpy as np
from datetime import datetime


def load_and_filter_earthquakes(
    csv_path,
    xlim,
    Xlim,
    ylim,
    Ylim,
    depth_max=100,
    magnitude_min=4.0,
    year_min=2000,
    exclude_region=None,
):
    """
    Load and filter earthquake catalog.
    
    Parameters
    ----------
    csv_path : str
        Path to earthquake CSV file
    xlim, Xlim, ylim, Ylim : float
        Geographic domain bounds (longitude, latitude)
    depth_max : float
        Maximum depth filter
    magnitude_min : float
        Minimum magnitude filter
    year_min : int
        Minimum year filter
    exclude_region : tuple, optional
        Region to exclude: ((lon_min, lon_max), (lat_min, lat_max))
    
    Returns
    -------
    pd.DataFrame
        Filtered earthquake catalog
    """
    df = pd.read_csv(csv_path)

    # Spatial filter
    df = df[(df["Longitude"] < Xlim) & (df["Longitude"] > xlim)]
    df = df[(df["Latitude"] < Ylim) & (df["Latitude"] > ylim)]

    # Depth filter
    df = df[df["Depth"] <= depth_max]

    # Magnitude filter
    df = df[df["Mw*"] >= magnitude_min]

    # Temporal filter
    df = df[df["Year"] >= year_min]

    # Exclude region
    if exclude_region is not None:
        (lon_min, lon_max), (lat_min, lat_max) = exclude_region
        df = df[~((df["Longitude"] > lon_min) & (df["Latitude"] > lat_min))]

    df = df.reset_index(drop=True)
    return df


def earthquakes_to_spatiotemporal(earthquakes, min_date, max_date):
    """
    Convert earthquake catalog to spatio-temporal point process format.
    
    Parameters
    ----------
    earthquakes : pd.DataFrame
        Earthquake catalog with Year, Month, Day, Hour, Minute, Second columns
    min_date : datetime
        Reference start time
    max_date : datetime
        Reference end time
    
    Returns
    -------
    tuple
        (SimPoisson, Tmax) where SimPoisson has columns [lon, lat, time_days]
    """
    # Build datetime for each event
    fechas = []
    for i, row in earthquakes.iterrows():
        dt = datetime(
            int(row["Year"]),
            int(row["Month"]),
            int(row["Day"]),
            int(row["Hour"]),
            int(row["Minute"]),
            int(row["Second"]),
        )
        fechas.append(dt)

    # Convert to days since min_date
    segundos_por_dia = 86400
    diferencias = np.array([(f - min_date).total_seconds() for f in fechas])
    diferencias = diferencias / segundos_por_dia

    # Time span
    Tmax = (max_date - min_date).total_seconds() / segundos_por_dia

    SimPoisson = earthquakes[["Longitude", "Latitude"]].to_numpy()
    SimPoisson = np.hstack((SimPoisson, diferencias.reshape(-1, 1)))

    return SimPoisson, Tmax


def assign_time_partition(SimPoisson, pT, Tmax):
    """
    Assign each event to a temporal partition.
    
    Parameters
    ----------
    SimPoisson : ndarray
        Spatio-temporal data (N x 3): [lon, lat, time]
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
        mask = (CutPoint[i] < SimPoisson[:, 2]) & (SimPoisson[:, 2] < CutPoint[i + 1])
        SimPoisson[mask, 3] = i

    return SimPoisson
