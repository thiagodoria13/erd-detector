"""
Utility functions for data loading and manipulation.

This module provides functions to load and process OpenBMI Motor Imagery data.
"""

import numpy as np
from scipy.io import loadmat
from pathlib import Path
import mne


def load_openbmi_data(subject_id, session, data_type='train', data_dir=None):
    """
    Load OpenBMI Motor Imagery data for a specific subject and session.
    
    The OpenBMI dataset contains EEG data in MATLAB .mat format with the following structure:
    - x: Continuous EEG signals (samples × channels), raw voltage in µV
    - t: Stimulus onset times for each trial (in sample indices)
    - fs: Sampling rate (1000 Hz for all subjects)
    - y_dec: Class labels as integers (1 = left hand, 2 = right hand)
    - chan: Channel names (62 EEG channels following 10-20 system)
    
    Args:
        subject_id (int): Subject number from 1 to 54
        session (int): Session number, either 1 or 2
        data_type (str): Either 'train' or 'test'
        data_dir (str, optional): Base directory containing OpenBMI data
    
    Returns:
        dict: Dictionary containing:
            - 'data': Continuous EEG data (samples × channels) in µV
            - 'events': Event times in samples (array of integers)
            - 'labels': Class labels (1=left, 2=right)
            - 'fs': Sampling rate in Hz
            - 'channels': List of channel names
            - 'subject_id': Subject ID
            - 'session': Session number
    
    Raises:
        FileNotFoundError: If data file doesn't exist
        ValueError: If subject_id or session are out of range
    """
    # Validate inputs
    if not (1 <= subject_id <= 54):
        raise ValueError(f"subject_id must be between 1 and 54, got {subject_id}")
    
    if session not in [1, 2]:
        raise ValueError(f"session must be 1 or 2, got {session}")
    
    if data_type not in ['train', 'test']:
        raise ValueError(f"data_type must be 'train' or 'test', got {data_type}")
    
    # Default data directory
    if data_dir is None:
        data_dir = "/mnt/data/openbmi_raw"
    
    # Construct file path
    # File structure: /mnt/data/openbmi_raw/s01/sess01/EEG_MI_train.mat
    data_path = Path(data_dir) / f"s{subject_id:02d}" / f"sess{session:02d}" / f"EEG_MI_{data_type}.mat"
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # Load MATLAB file
    mat = loadmat(str(data_path))
    
    # Extract data components
    # x: Continuous EEG (samples × channels)
    data = mat['x'].astype(np.float64)
    
    # t: Event times (1D array of sample indices)
    events = mat['t'].flatten().astype(np.int32)
    
    # fs: Sampling rate (scalar, typically 1000 Hz)
    fs = int(mat['fs'].flatten()[0])
    
    # y_dec: Class labels (1=left hand, 2=right hand)
    labels = mat['y_dec'].flatten().astype(np.int32)
    
    # chan: Channel names (extract from MATLAB nested structure)
    channels = []
    for ch in mat['chan'].flatten():
        if isinstance(ch, np.ndarray):
            ch_name = str(ch[0]) if len(ch) > 0 else ''
        else:
            ch_name = str(ch)
        channels.append(ch_name.strip())
    
    # Create return dictionary
    result = {
        'data': data,           # (samples × channels) EEG voltage in µV
        'events': events,       # (n_trials,) event onset times in samples
        'labels': labels,       # (n_trials,) class labels (1=left, 2=right)
        'fs': fs,              # Sampling rate in Hz
        'channels': channels,   # List of channel names
        'subject_id': subject_id,
        'session': session,
        'data_type': data_type
    }
    
    return result


def extract_trial(data, event_time, fs, window=(-3.0, 4.0)):
    """
    Extract single trial from continuous EEG data.
    
    Given continuous EEG data and an event time, extract a window of data
    around that event. Default window is [-3s, +4s]:
    - [-3s, -1s]: Baseline period (no movement preparation)
    - [-1s, 0s]: Pre-movement period (ERD typically starts here)
    - [0s, +4s]: Task period (motor imagery or execution)
    
    Args:
        data (ndarray): Continuous EEG data (samples × channels)
        event_time (int): Event onset time in samples
        fs (int): Sampling rate in Hz
        window (tuple): Time window in seconds (start, end) relative to event
    
    Returns:
        ndarray: Trial data (samples × channels) for the specified window
    """
    # Convert time window to samples
    start_offset = int(window[0] * fs)  # e.g., -3.0 * 1000 = -3000 samples
    end_offset = int(window[1] * fs)    # e.g., 4.0 * 1000 = 4000 samples
    
    # Calculate absolute sample indices
    start_sample = event_time + start_offset
    end_sample = event_time + end_offset
    
    # Boundary checking
    if start_sample < 0:
        raise ValueError(f"Trial window starts before data begins (start_sample={start_sample})")
    
    if end_sample > data.shape[0]:
        raise ValueError(f"Trial window extends past data end (end_sample={end_sample}, data_length={data.shape[0]})")
    
    # Extract trial data
    trial_data = data[start_sample:end_sample, :]
    
    return trial_data


def get_channel_indices(channel_names, target_channels):
    """
    Get indices of target channels from list of channel names.
    
    Args:
        channel_names (list): List of all channel names in data
        target_channels (list): List of desired channel names
    
    Returns:
        list: Indices of target channels in the full channel array
    
    Raises:
        ValueError: If any target channel is not found
    """
    indices = []
    
    for target in target_channels:
        try:
            idx = channel_names.index(target)
            indices.append(idx)
        except ValueError:
            raise ValueError(f"Channel '{target}' not found in data. Available channels: {channel_names}")
    
    return indices


def create_mne_raw(data, channel_names, fs):
    """
    Create MNE Raw object from numpy array.
    
    MNE-Python provides tools for EEG analysis including filtering, artifact
    removal, and visualization.
    
    Args:
        data (ndarray): EEG data (samples × channels) in µV
        channel_names (list): List of channel names
        fs (int): Sampling rate in Hz
    
    Returns:
        mne.io.Raw: MNE Raw object with standard 10-20 montage
    """
    # MNE expects data in format (channels × samples), so transpose
    data_T = data.T
    
    # Create MNE Info object with channel information
    info = mne.create_info(
        ch_names=channel_names,
        sfreq=fs,
        ch_types='eeg',
        verbose=False
    )
    
    # Create Raw object (MNE expects data in Volts, OpenBMI is in µV)
    data_volts = data_T * 1e-6
    raw = mne.io.RawArray(data_volts, info, verbose=False)
    
    # Set standard 10-20 electrode positions
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage, on_missing='ignore', verbose=False)
    
    return raw
