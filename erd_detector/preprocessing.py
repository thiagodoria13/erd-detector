"""
Preprocessing Module for ERD Detection

This module implements the preprocessing pipeline for EEG signal preparation:
1. Bandpass filtering (8-30 Hz) to isolate mu and beta rhythms
2. Surface Laplacian filtering for spatial enhancement
3. Artifact rejection using amplitude thresholding

Neurophysiological Justification:
- Mu rhythm (8-13 Hz): Sensorimotor cortex, shows ERD during motor imagery
- Beta rhythm (14-30 Hz): Motor cortex, desynchronizes during movement preparation
- Surface Laplacian: Enhances local cortical activity, reduces volume conduction
- Artifact threshold (+/-100 microV): Removes eye blinks, muscle artifacts, electrode noise

Author: Lucas Pereira da Fonseca, Thiago Anversa Sampaio Doria
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt
from typing import Dict, List, Tuple, Optional
import warnings


def bandpass_filter(
    data: np.ndarray,
    fs: float,
    lowcut: float = 8.0,
    highcut: float = 30.0,
    order: int = 5
) -> np.ndarray:
    """
    Apply bandpass Butterworth filter to isolate mu (8-13 Hz) and beta (14-30 Hz) rhythms.

    Mathematical formulation:
    Transfer function H(s) of nth-order Butterworth filter:

    $$|H(j\omega)|^2 = \frac{1}{1 + (\omega/\omega_c)^{2n}}$$

    where omega_c is the cutoff frequency and n is the filter order.

    For bandpass: cascade of highpass (lowcut) and lowpass (highcut).

    Implementation uses Second-Order Sections (SOS) for numerical stability.
    Zero-phase filtering via forward-backward pass (sosfiltfilt).

    Args:
        data: EEG data, shape (n_channels, n_samples) or (n_samples,)
        fs: Sampling frequency in Hz
        lowcut: Lower cutoff frequency in Hz (default: 8 Hz for mu rhythm)
        highcut: Upper cutoff frequency in Hz (default: 30 Hz for beta rhythm)
        order: Filter order (default: 5 for -30 dB/decade rolloff)

    Returns:
        Filtered data with same shape as input

    Neurophysiological Note:
        - Mu rhythm (8-13 Hz): Reflects idle state of sensorimotor cortex
        - Beta rhythm (14-30 Hz): Associated with active motor control
        - ERD = desynchronization (power decrease) during motor imagery
        - Frequencies outside 8-30 Hz contain minimal motor-related information
    """
    # Store original shape for later restoration
    original_shape = data.shape

    # Handle both 1D (single channel) and 2D (multi-channel) inputs
    if data.ndim == 1:
        # Reshape 1D to 2D: (n_samples,) -> (1, n_samples)
        data = data.reshape(1, -1)

    # Validate filter parameters
    nyquist = fs / 2.0  # Nyquist frequency = half the sampling rate
    if lowcut >= nyquist or highcut >= nyquist:
        raise ValueError(f"Cutoff frequencies must be < Nyquist frequency ({nyquist} Hz)")
    if lowcut >= highcut:
        raise ValueError(f"lowcut ({lowcut}) must be < highcut ({highcut})")

    # Normalize cutoff frequencies to Nyquist frequency (required by scipy)
    low = lowcut / nyquist
    high = highcut / nyquist

    # Design Butterworth bandpass filter in SOS (Second-Order Sections) format
    # SOS format is numerically more stable than transfer function (b, a) format
    # Avoids numerical errors from high-order polynomial representations
    sos = butter(order, [low, high], btype='band', output='sos')

    # Apply zero-phase filter to each channel
    # sosfiltfilt = forward-backward filtering -> zero phase distortion
    # This preserves the timing of ERD events (critical for latency measurement)
    filtered_data = np.zeros_like(data)
    for ch_idx in range(data.shape[0]):
        # Filter one channel at a time
        filtered_data[ch_idx, :] = sosfiltfilt(sos, data[ch_idx, :])

    # Restore original shape if input was 1D
    if len(original_shape) == 1:
        filtered_data = filtered_data.flatten()

    return filtered_data


def laplacian_filter(
    data: np.ndarray,
    channels: List[str],
    target_channels: List[str] = ['C3', 'C4']
) -> Tuple[np.ndarray, List[str]]:
    """
    Apply Surface Laplacian filter to motor cortex channels (C3, C4).

    Mathematical formulation:
    Surface Laplacian approximation using nearest neighbors:

    $$V_{Laplacian}(C) = V(C) - \frac{1}{N}\sum_{i=1}^{N} V(N_i)$$

    where:
    - V(C) = potential at central electrode
    - V(N_i) = potential at ith neighbor electrode
    - N = number of neighbors (typically 4 for 10-20 system)

    Neighbor definitions (10-20 system):
    - C3: [FC1, FC5, CP1, CP5] (left motor cortex)
    - C4: [FC2, FC6, CP2, CP6] (right motor cortex)

    Args:
        data: EEG data, shape (n_channels, n_samples)
        channels: List of channel names corresponding to data rows
        target_channels: Channels to apply Laplacian filter (default: C3, C4)

    Returns:
        Tuple of (laplacian_data, laplacian_channels)
        - laplacian_data: Filtered data, shape (len(target_channels), n_samples)
        - laplacian_channels: Names of Laplacian channels (e.g., ['C3_lap', 'C4_lap'])

    Neurophysiological Note:
        - C3: Left motor cortex (contralateral to right hand movement)
        - C4: Right motor cortex (contralateral to left hand movement)
        - Laplacian enhances local cortical activity by subtracting volume conduction
        - Reduces common-mode artifacts (eye blinks, powerline noise)
        - Spatial high-pass filter: emphasizes radial current sources

    Raises:
        ValueError: If target channels or their neighbors are not found in data
    """
    # Define neighbor electrodes for each motor channel (10-20 system layout)
    # These are the 4 nearest electrodes surrounding C3/C4
    neighbor_map = {
        'C3': ['FC1', 'FC5', 'CP1', 'CP5'],  # Left hemisphere
        'C4': ['FC2', 'FC6', 'CP2', 'CP6']   # Right hemisphere
    }

    # Convert channel list to dictionary for fast lookup
    # Maps channel name -> row index in data array
    channel_indices = {ch: idx for idx, ch in enumerate(channels)}

    # Store Laplacian-filtered channels
    laplacian_data = []
    laplacian_channels = []

    # Process each target channel (typically C3 and C4)
    for target_ch in target_channels:
        # Check if target channel exists in data
        if target_ch not in channel_indices:
            warnings.warn(f"Target channel {target_ch} not found in data. Skipping.")
            continue

        # Check if neighbor map exists for this channel
        if target_ch not in neighbor_map:
            warnings.warn(f"No neighbor definition for {target_ch}. Skipping.")
            continue

        # Get neighbor channel names
        neighbors = neighbor_map[target_ch]

        # Find indices of available neighbors
        neighbor_indices = []
        missing_neighbors = []
        for neighbor_ch in neighbors:
            if neighbor_ch in channel_indices:
                neighbor_indices.append(channel_indices[neighbor_ch])
            else:
                missing_neighbors.append(neighbor_ch)

        # Check if we have enough neighbors (need at least 2 for meaningful Laplacian)
        if len(neighbor_indices) < 2:
            warnings.warn(
                f"Insufficient neighbors for {target_ch}. "
                f"Found {len(neighbor_indices)}, need e2. Missing: {missing_neighbors}. Skipping."
            )
            continue

        # Warn if some neighbors are missing (but still compute Laplacian)
        if missing_neighbors:
            warnings.warn(
                f"Some neighbors missing for {target_ch}: {missing_neighbors}. "
                f"Using {len(neighbor_indices)} available neighbors."
            )

        # Get central electrode signal
        central_signal = data[channel_indices[target_ch], :]

        # Compute average of neighbor signals
        # Stack neighbor signals: (n_neighbors, n_samples)
        neighbor_signals = np.array([data[idx, :] for idx in neighbor_indices])
        # Average across neighbors: (n_samples,)
        neighbor_average = np.mean(neighbor_signals, axis=0)

        # Apply Laplacian formula: V_lap = V_center - mean(V_neighbors)
        # This is the surface Laplacian approximation
        # Equivalent to second spatial derivative (�V) on the scalp surface
        laplacian_signal = central_signal - neighbor_average

        # Store result
        laplacian_data.append(laplacian_signal)
        laplacian_channels.append(f"{target_ch}_lap")

    # Check if any channels were successfully processed
    if len(laplacian_data) == 0:
        raise ValueError(
            f"Could not compute Laplacian for any target channels. "
            f"Targets: {target_channels}, Available: {channels}"
        )

    # Convert list to numpy array: (n_laplacian_channels, n_samples)
    laplacian_data = np.array(laplacian_data)

    return laplacian_data, laplacian_channels


def reject_artifacts(
    data: np.ndarray,
    threshold: float = 100.0
) -> Tuple[bool, np.ndarray]:
    """
    Detect artifacts using amplitude thresholding.

    Mathematical formulation:
    Artifact detected if:

    $$\max_t |x(t)| > T$$

    where:
    - x(t) = EEG signal (any channel)
    - T = threshold (default 100 microV)

    Args:
        data: EEG data in microV, shape (n_channels, n_samples) or (n_samples,)
        threshold: Amplitude threshold in microV (default: 100 microV)

    Returns:
        Tuple of (is_clean, data)
        - is_clean: True if no artifacts detected, False otherwise
        - data: Original data (unchanged, for pipeline consistency)

    Artifact Sources:
        - Eye blinks: 100-300 microV (frontal channels)
        - Muscle tension: 50-200 microV (temporal channels)
        - Electrode noise: >100 microV (poor contact)
        - Movement artifacts: Variable, often >100 microV

    Neurophysiological Note:
        - Normal EEG: 10-50 microV (posterior alpha, central mu/beta)
        - Motor imagery ERD: ~20-40% power reduction from baseline
        - Threshold of 100 microV is conservative but effective
        - False rejection rate: ~5-10% for typical motor imagery tasks

    Note:
        This function only DETECTS artifacts, does not remove them.
        The calling function decides whether to reject the trial.
        More sophisticated methods (ICA, ASR) can be added later.
    """
    # Handle both 1D and 2D inputs
    if data.ndim == 1:
        # Single channel: check max absolute value
        max_amplitude = np.max(np.abs(data))
    else:
        # Multiple channels: check max across all channels
        max_amplitude = np.max(np.abs(data))

    # Artifact detected if any sample exceeds threshold
    is_clean = max_amplitude <= threshold

    # Return flag and original data
    # Note: We don't modify the data here, just flag it
    # The calling function (e.g., process_trial) decides what to do with rejected trials
    return is_clean, data


def preprocess_trial(
    trial_data: np.ndarray,
    channels: List[str],
    fs: float,
    motor_channels: List[str] = ['C3', 'C4'],
    reference_channels: List[str] = ['O1', 'O2', 'Fz'],
    bandpass_params: Optional[Dict] = None,
    artifact_threshold: float = 100.0,
    apply_laplacian: bool = True
) -> Dict:
    """
    Complete preprocessing pipeline for a single trial.

    Pipeline stages:
    1. Artifact rejection (raw data, all channels)
    2. Bandpass filter (8-30 Hz, all channels)
    3. Surface Laplacian filter (motor channels only)
    4. Extract motor and reference channels

    Args:
        trial_data: Raw EEG data, shape (n_channels, n_samples)
        channels: List of channel names
        fs: Sampling frequency in Hz
        motor_channels: Channels for ERD detection (default: C3, C4)
        reference_channels: Channels for baseline (default: O1, O2, Fz)
        bandpass_params: Optional dict with 'lowcut', 'highcut', 'order'
                        (default: 8-30 Hz, order 5)
        artifact_threshold: Amplitude threshold in microV (default: 100)
        apply_laplacian: Whether to apply Laplacian to motor channels (default: True)

    Returns:
        Dictionary containing:
        - 'is_clean': bool, True if no artifacts detected
        - 'motor_data': Preprocessed motor channel data, shape (n_motor, n_samples)
        - 'motor_channels': List of motor channel names
        - 'reference_data': Preprocessed reference data, shape (n_ref, n_samples)
        - 'reference_channels': List of reference channel names
        - 'fs': Sampling frequency (passed through)
        - 'artifact_max': Maximum amplitude detected (for logging)

    Processing Flow:
        Raw EEG (all channels)
        �
        [Artifact Detection] � is_clean flag
        �
        [Bandpass 8-30 Hz] � Isolate mu/beta
        �
        � [Laplacian C3/C4] � Motor channels (enhanced local activity)
        � [Extract O1/O2/Fz] � Reference channels (stable baseline)

    Example:
        >>> result = preprocess_trial(trial_data, channels, fs=1000)
        >>> if result['is_clean']:
        ...     motor_data = result['motor_data']  # Shape: (2, n_samples) for C3, C4
        ...     ref_data = result['reference_data']  # Shape: (3, n_samples) for O1, O2, Fz

    Raises:
        ValueError: If required channels are not found in data
    """
    # Set default bandpass parameters if not provided
    if bandpass_params is None:
        bandpass_params = {
            'lowcut': 8.0,   # Mu rhythm lower bound
            'highcut': 30.0,  # Beta rhythm upper bound
            'order': 5        # Standard filter order
        }

    # STAGE 1: Artifact Detection (on raw data)
    # Check raw data before any processing to avoid wasting computation
    is_clean, _ = reject_artifacts(trial_data, threshold=artifact_threshold)
    max_amplitude = np.max(np.abs(trial_data))  # Store for logging

    # Note: We continue processing even if artifacts detected
    # The calling function decides whether to use this trial
    # This allows for optional artifact correction methods later

    # STAGE 2: Bandpass Filtering (all channels)
    # Apply to all channels to maintain consistency
    filtered_data = bandpass_filter(
        trial_data,
        fs,
        lowcut=bandpass_params['lowcut'],
        highcut=bandpass_params['highcut'],
        order=bandpass_params['order']
    )

    # STAGE 3: Extract and Process Motor Channels
    if apply_laplacian:
        # Apply Surface Laplacian to motor channels
        motor_data, motor_ch_names = laplacian_filter(
            filtered_data,
            channels,
            target_channels=motor_channels
        )
    else:
        # Extract motor channels without Laplacian (for comparison studies)
        channel_indices = {ch: idx for idx, ch in enumerate(channels)}
        motor_indices = []
        motor_ch_names = []
        for ch in motor_channels:
            if ch in channel_indices:
                motor_indices.append(channel_indices[ch])
                motor_ch_names.append(ch)
            else:
                warnings.warn(f"Motor channel {ch} not found. Skipping.")

        if len(motor_indices) == 0:
            raise ValueError(f"No motor channels found. Requested: {motor_channels}")

        motor_data = filtered_data[motor_indices, :]

    # STAGE 4: Extract Reference Channels
    # These provide baseline for ERD normalization
    # Should be channels unaffected by motor imagery (occipital, frontal midline)
    channel_indices = {ch: idx for idx, ch in enumerate(channels)}
    reference_indices = []
    reference_ch_names = []
    for ch in reference_channels:
        if ch in channel_indices:
            reference_indices.append(channel_indices[ch])
            reference_ch_names.append(ch)
        else:
            warnings.warn(f"Reference channel {ch} not found. Skipping.")

    if len(reference_indices) == 0:
        raise ValueError(f"No reference channels found. Requested: {reference_channels}")

    # Extract reference channel data (already bandpass filtered)
    reference_data = filtered_data[reference_indices, :]

    # Return comprehensive result dictionary
    return {
        'is_clean': is_clean,
        'motor_data': motor_data,
        'motor_channels': motor_ch_names,
        'reference_data': reference_data,
        'reference_channels': reference_ch_names,
        'fs': fs,
        'artifact_max': max_amplitude
    }
