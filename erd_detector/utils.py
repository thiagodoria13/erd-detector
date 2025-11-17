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
        data_type (str): Either 'train' or 'test' (legacy, determines which mat key to use)
        data_dir (str, optional): Base directory containing OpenBMI data
                                 Default: 'data/openbmi_sample'

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

    File Naming Formats Supported:
        1. GigaDB format: sess{session:02d}_subj{subject:02d}_EEG_MI.mat
        2. Nested format: s{subject:02d}/sess{session:02d}/EEG_MI_{data_type}.mat
        3. Flat format: EEG_MI_{data_type}.mat (with subject/session dirs)
    """
    # Validate inputs
    if not (1 <= subject_id <= 54):
        raise ValueError(f"subject_id must be between 1 and 54, got {subject_id}")

    if session not in [1, 2]:
        raise ValueError(f"session must be 1 or 2, got {session}")

    if data_type not in ['train', 'test']:
        raise ValueError(f"data_type must be 'train' or 'test', got {data_type}")

    # Default data directory for local testing
    if data_dir is None:
        data_dir = "data/openbmi_sample"

    data_dir = Path(data_dir)

    # Try multiple file path patterns for flexibility
    possible_paths = [
        # Pattern 1: GigaDB flat format (actual download format)
        # sess01_subj01_EEG_MI.mat
        data_dir / f"sess{session:02d}_subj{subject_id:02d}_EEG_MI.mat",

        # Pattern 2: Nested directory structure
        # s01/sess01/EEG_MI_train.mat
        data_dir / f"s{subject_id:02d}" / f"sess{session:02d}" / f"EEG_MI_{data_type}.mat",

        # Pattern 3: Session-based directory structure
        # session1/s01/sess01_subj01_EEG_MI.mat
        data_dir / f"session{session}" / f"s{subject_id}" / f"sess{session:02d}_subj{subject_id:02d}_EEG_MI.mat",

        # Pattern 4: Simple nested by subject
        # s01/sess01_subj01_EEG_MI.mat
        data_dir / f"s{subject_id:02d}" / f"sess{session:02d}_subj{subject_id:02d}_EEG_MI.mat",
    ]

    # Find the first existing path
    data_path = None
    for path in possible_paths:
        if path.exists():
            data_path = path
            break

    if data_path is None:
        # Provide helpful error message with all attempted paths
        attempted = "\n  - ".join(str(p) for p in possible_paths)
        raise FileNotFoundError(
            f"Data file not found for subject {subject_id}, session {session}.\n"
            f"Tried:\n  - {attempted}"
        )
    
    # Load MATLAB file
    mat = loadmat(str(data_path))

    # OpenBMI files have a structure named EEG_MI_train or EEG_MI_test
    # containing x, t, fs, y_dec, and chan fields
    key = f"EEG_MI_{data_type}"

    # Check if structured format (EEG_MI_train/test) or flat format
    if key in mat:
        # Structured format: extract from EEG_MI_train or EEG_MI_test
        eeg_data = mat[key][0, 0]

        # x: Continuous EEG (samples × channels)
        data = eeg_data['x'].astype(np.float64)

        # t: Event times (1D array of sample indices)
        events = eeg_data['t'].flatten().astype(np.int32)

        # fs: Sampling rate (scalar, typically 1000 Hz)
        fs = int(eeg_data['fs'].flatten()[0])

        # y_dec: Class labels (1=left hand, 2=right hand)
        labels = eeg_data['y_dec'].flatten().astype(np.int32)

        # chan: Channel names
        chan_data = eeg_data['chan']
    else:
        # Flat format: data at top level (backward compatibility)
        data = mat['x'].astype(np.float64)
        events = mat['t'].flatten().astype(np.int32)
        fs = int(mat['fs'].flatten()[0])
        labels = mat['y_dec'].flatten().astype(np.int32)
        chan_data = mat['chan']

    # Extract channel names (same nested structure in both formats)
    # OpenBMI stores channels as nested arrays: array([['Fp1']], dtype='<U3')
    # We need to recursively extract until we get the actual string
    channels = []
    for ch in chan_data.flatten():
        # Recursively extract from nested arrays
        while isinstance(ch, np.ndarray):
            if len(ch) == 0:
                ch = ''
                break
            ch = ch[0]
        # Now ch should be a string or convertible to string
        ch_name = str(ch) if ch else ''
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
