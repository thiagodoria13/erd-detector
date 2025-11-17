"""
ERD Detection Module

This module implements the ERD detection algorithm using -2 sigma threshold:
1. Baseline calculation from reference channels (O1, O2, Fz)
2. Sliding window analysis of motor channels (C3, C4)
3. Normalization: z = (P - mu) / sigma
4. Detection: z <= -2 sigma in >= 2 motor channels

Neurophysiological Justification:
- Reference channels (O1, O2, Fz) provide stable baseline (unaffected by motor imagery)
- Motor channels (C3, C4) show ERD during motor imagery
- -2 sigma threshold: statistically significant deviation (p < 0.05, one-tailed)
- Require >= 2 motor channels: reduces false positives

Author: Lucas Pereira da Fonseca, Thiago Anversa Sampaio Doria
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from . import preprocessing
from . import hht


def calculate_baseline(
    reference_data: np.ndarray,
    fs: float,
    baseline_window: Tuple[float, float] = (-3.0, -1.0),
    trial_start_time: float = -3.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate baseline statistics from reference channels.

    Baseline is computed from reference channels (O1, O2, Fz) during
    pre-stimulus period (default: -3s to -1s before stimulus onset).

    Args:
        reference_data: Reference channel data after HHT, shape (n_ref_channels, n_samples)
        fs: Sampling frequency in Hz
        baseline_window: Time window for baseline (start, end) relative to cue (s)
                         Default: (-3.0, -1.0) = 2 seconds before stimulus
        trial_start_time: Time of sample index 0 relative to cue (s). Default -3.0

    Returns:
        Tuple of (baseline_mean, baseline_std)
        - baseline_mean: Mean power per channel (array shape: n_channels,)
        - baseline_std: Standard deviation per channel (array shape: n_channels,)

    Neurophysiological Note:
        - Reference channels should not show motor-related activity
        - O1, O2: Occipital (visual cortex)
        - Fz: Frontal midline (executive function)
        - Baseline period: far enough from stimulus to avoid anticipation effects
    """
    n_samples = reference_data.shape[1]

    # Convert desired time window (relative to cue) into indices of this trial
    # Index 0 corresponds to trial_start_time seconds relative to cue.
    start_idx = int((baseline_window[0] - trial_start_time) * fs)
    end_idx = int((baseline_window[1] - trial_start_time) * fs)

    start_idx = max(0, start_idx)
    end_idx = min(n_samples, end_idx)

    if start_idx >= end_idx:
        raise ValueError(
            f"Invalid baseline window: {baseline_window}. "
            f"Samples: {start_idx} to {end_idx}"
        )

    # Extract baseline period
    baseline_data = reference_data[:, start_idx:end_idx]

    # Compute mean/std per channel (not pooled across channels)
    baseline_mean = np.mean(baseline_data, axis=1)
    baseline_std = np.std(baseline_data, axis=1)

    # Avoid division by zero per channel
    baseline_std[baseline_std == 0] = 1.0

    return baseline_mean, baseline_std


def detect_erd_sliding_window(
    motor_power: np.ndarray,
    baseline_mean: np.ndarray,
    baseline_std: np.ndarray,
    fs: float,
    window_size: float = 0.2,
    step_size: float = 0.05,
    threshold_sigma: float = -2.0,
    min_channels: int = 2,
    task_window: Tuple[float, float] = (0.0, 4.0),
    trial_start_time: float = -3.0,
    use_temporal_weight: bool = True,
    penalty_center: float = 0.6,
    penalty_sigma: float = 0.25,
    penalty_floor: float = 0.3
) -> Dict:
    """
    Detect ERD using sliding window with -2 sigma threshold.

    Algorithm:
    1. Slide window through task period (default: 0 to +4s after stimulus)
    2. For each window position:
       a. Compute mean power in window for each motor channel
       b. Normalize: z = (P - baseline_mean) / baseline_std
       c. Count channels where z <= threshold (-2 sigma)
       d. Detect ERD if count >= min_channels (default: 2)
    3. Return first detection time (onset) and all detections

    Args:
        motor_power: Instantaneous power from motor channels, shape (n_motor, n_samples)
        baseline_mean: Baseline mean from calculate_baseline()
        baseline_std: Baseline std from calculate_baseline()
        fs: Sampling frequency in Hz
        window_size: Sliding window size in seconds (default: 0.2s = 200ms)
        step_size: Step size for sliding window (default: 0.05s = 50ms)
        threshold_sigma: Detection threshold in sigma units (default: -2.0)
        min_channels: Minimum number of channels for detection (default: 2)
        task_window: Time window for detection (start, end) relative to cue (seconds)
                     Default: (0.0, 4.0) = 4 seconds after stimulus
        trial_start_time: Time of sample index 0 relative to cue (seconds)
        use_temporal_weight: If True, apply Gaussian time weighting
        penalty_center: Center of Gaussian weight (seconds relative to cue)
        penalty_sigma: Sigma of Gaussian weight (seconds)
        penalty_floor: Minimum weight value (to avoid zeroing far windows)

    Returns:
        Dictionary containing:
        - 'detected': bool, True if ERD detected
        - 'onset_time': float, time of first detection in seconds (None if not detected)
        - 'onset_sample': int, sample index of first detection (None if not detected)
        - 'latency': float, latency from stimulus onset in seconds (None if not detected)
        - 'detection_times': list of all detection times
        - 'z_scores': array of z-scores over time, shape (n_motor, n_windows)
        - 'detection_count': array of channel count per window, shape (n_windows,)

    Neurophysiological Note:
        - Window size 200ms: balances temporal precision vs statistical power
        - Step size 50ms: 75% overlap, smooth detection curve
        - Threshold -2 sigma: p < 0.05 (one-tailed), statistically significant
        - Min 2 channels: C3 + C4 must both show ERD (bilateral motor cortex)
    """
    n_motor_channels, n_samples = motor_power.shape
    if baseline_mean.shape[0] != n_motor_channels or baseline_std.shape[0] != n_motor_channels:
        raise ValueError(
            f"Baseline arrays must match motor channels: got baseline_mean {baseline_mean.shape}, "
            f"baseline_std {baseline_std.shape}, motor channels {n_motor_channels}"
        )

    # Convert window parameters to samples
    window_samples = int(window_size * fs)
    step_samples = int(step_size * fs)

    # Convert task window (relative to cue) to sample indices in trial
    start_idx = int((task_window[0] - trial_start_time) * fs)
    end_idx = int((task_window[1] - trial_start_time) * fs)

    start_idx = max(0, start_idx)
    end_idx = min(n_samples, end_idx)

    if start_idx >= end_idx:
        raise ValueError(
            f"Invalid task window: {task_window}. "
            f"Samples: {start_idx} to {end_idx}"
        )

    # Calculate number of windows
    n_windows = (end_idx - start_idx - window_samples) // step_samples + 1

    if n_windows <= 0:
        raise ValueError(
            f"Not enough samples for sliding window. "
            f"Need at least {window_samples} samples, have {end_idx - start_idx}"
        )

    # Initialize storage for results
    z_scores = np.zeros((n_motor_channels, n_windows))
    detection_count = np.zeros(n_windows, dtype=float)
    window_times = np.zeros(n_windows)
    detection_times = []

    # Precompute temporal weights (Gaussian around expected ERD timing)
    weights = np.ones(n_windows)
    if use_temporal_weight:
        weights = np.exp(-0.5 * ((window_times - penalty_center) / penalty_sigma) ** 2)
        weights = np.clip(weights, penalty_floor, 1.0)

    # Slide window through task period
    for win_idx in range(n_windows):
        # Calculate window start and end indices
        win_start = start_idx + win_idx * step_samples
        win_end = win_start + window_samples

        # Window center time relative to cue (seconds)
        window_times[win_idx] = trial_start_time + (win_start + window_samples / 2) / fs

        # Update weight now that window_times is set
        if use_temporal_weight:
            weights[win_idx] = np.exp(-0.5 * ((window_times[win_idx] - penalty_center) / penalty_sigma) ** 2)
            weights[win_idx] = np.clip(weights[win_idx], penalty_floor, 1.0)
        else:
            weights[win_idx] = 1.0

        # Process each motor channel
        for ch_idx in range(n_motor_channels):
            # Extract power in current window
            window_power = motor_power[ch_idx, win_start:win_end]

            # Compute mean power in window
            mean_power = np.mean(window_power)

            # Normalize to z-score
            z = (mean_power - baseline_mean[ch_idx]) / baseline_std[ch_idx]
            z_scores[ch_idx, win_idx] = z

            # Check if this channel shows ERD (z <= threshold)
            if z <= threshold_sigma:
                detection_count[win_idx] += weights[win_idx]

        # Check if ERD detected in this window (>= min_channels)
        if detection_count[win_idx] >= min_channels:
            detection_times.append(window_times[win_idx])

    # Determine if ERD was detected
    detected = len(detection_times) > 0

    # Get onset time (first detection)
    if detected:
        onset_time = detection_times[0]
        onset_sample = int((onset_time - trial_start_time) * fs)
        latency = onset_time  # already relative to cue (time 0)
    else:
        onset_time = None
        onset_sample = None
        latency = None

    # Return comprehensive results
    return {
        'detected': detected,
        'onset_time': onset_time,
        'onset_sample': onset_sample,
        'latency': latency,
        'detection_times': detection_times,
        'z_scores': z_scores,
        'detection_count': detection_count,
        'window_times': window_times,
        'window_weights': weights
    }


class ERDDetector:
    """
    Main ERD Detector class integrating preprocessing, HHT, and detection.

    This class implements the complete pipeline from raw EEG to ERD detection:
    1. Preprocessing: Bandpass filter, Laplacian, artifact rejection
    2. HHT: EMD + IMF selection + Hilbert transform + power
    3. Detection: Baseline calculation + sliding window + -2 sigma threshold

    Example:
        >>> detector = ERDDetector(
        ...     motor_channels=['C3', 'C4'],
        ...     reference_channels=['O1', 'O2', 'Fz'],
        ...     threshold_sigma=-2.0,
        ...     min_channels=2
        ... )
        >>> result = detector.process_trial(trial_data, channels, fs=1000)
        >>> if result['detected']:
        ...     print(f"ERD detected at {result['onset_time']:.3f}s")
    """

    def __init__(
        self,
        motor_channels: List[str] = ['C3', 'C4'],
        reference_channels: List[str] = ['O1', 'O2', 'Fz'],
        threshold_sigma: float = -2.0,
        min_channels: int = 2,
        baseline_window: Tuple[float, float] = (-1.0, 0.0),
        task_window: Tuple[float, float] = (0.0, 4.0),
        detection_window_size: float = 0.2,
        detection_step_size: float = 0.05,
        trial_start_time: float = -3.0,
        bandpass_params: Optional[Dict] = None,
        artifact_threshold: float = 100.0,
        hht_freq_band: Tuple[float, float] = (8.0, 30.0),
        hht_power_threshold: float = 0.6,
        use_temporal_weight: bool = True,
        penalty_center: float = 0.6,
        penalty_sigma: float = 0.25,
        penalty_floor: float = 0.3
    ):
        """
        Initialize ERD Detector.

        Args:
            motor_channels: Channels for ERD detection (default: ['C3', 'C4'])
            reference_channels: Channels for baseline (default: ['O1', 'O2', 'Fz'])
            threshold_sigma: Detection threshold (default: -2.0)
            min_channels: Minimum channels for detection (default: 2)
            baseline_window: Baseline time window (default: (-3.0, -1.0))
            task_window: Task time window (default: (0.0, 4.0))
            detection_window_size: Sliding window size in s (default: 0.2)
            detection_step_size: Sliding window step in s (default: 0.05)
            trial_start_time: Time of first sample relative to cue (default: -3.0s)
            bandpass_params: Bandpass filter parameters (default: 8-30 Hz, order 5)
            artifact_threshold: Artifact threshold in microV (default: 100)
            hht_freq_band: HHT frequency band (default: (8.0, 30.0))
            hht_power_threshold: HHT IMF selection threshold (default: 0.6)
        """
        self.motor_channels = motor_channels
        self.reference_channels = reference_channels
        self.threshold_sigma = threshold_sigma
        self.min_channels = min_channels
        self.baseline_window = baseline_window
        self.task_window = task_window
        self.detection_window_size = detection_window_size
        self.detection_step_size = detection_step_size
        self.trial_start_time = trial_start_time
        self.artifact_threshold = artifact_threshold
        self.hht_freq_band = hht_freq_band
        self.hht_power_threshold = hht_power_threshold
        self.use_temporal_weight = use_temporal_weight
        self.penalty_center = penalty_center
        self.penalty_sigma = penalty_sigma
        self.penalty_floor = penalty_floor

        # Set bandpass parameters
        if bandpass_params is None:
            self.bandpass_params = {
                'lowcut': 8.0,
                'highcut': 30.0,
                'order': 5
            }
        else:
            self.bandpass_params = bandpass_params

    def process_trial(
        self,
        trial_data: np.ndarray,
        channels: List[str],
        fs: float,
        trial_start_time: Optional[float] = None
    ) -> Dict:
        """
        Process a single trial and detect ERD.

        Complete pipeline:
        1. Preprocessing (bandpass, Laplacian, artifacts)
        2. HHT for motor and reference channels
        3. Baseline calculation from reference channels
        4. ERD detection with sliding window

        Args:
            trial_data: Raw EEG data, shape (n_channels, n_samples)
            channels: List of channel names
            fs: Sampling frequency in Hz
            trial_start_time: Time (s) of first sample relative to cue. Defaults to
                               detector setting (e.g., -3.0s for [-3,+4] trials).

        Returns:
            Dictionary containing:
            - All fields from detect_erd_sliding_window()
            - 'is_clean': bool, artifact status
            - 'artifact_max': float, maximum amplitude
            - 'motor_channels_used': list of motor channel names
            - 'reference_channels_used': list of reference channel names
            - 'baseline_mean': np.ndarray (per-channel baseline mean)
            - 'baseline_std': np.ndarray (per-channel baseline std)

        Raises:
            ValueError: If required channels not found or processing fails
        """
        # STAGE 1: Preprocessing
        preproc_result = preprocessing.preprocess_trial(
            trial_data,
            channels,
            fs,
            motor_channels=self.motor_channels,
            reference_channels=self.reference_channels,
            bandpass_params=self.bandpass_params,
            artifact_threshold=self.artifact_threshold,
            apply_laplacian=True
        )

        # Extract preprocessed data
        motor_data = preproc_result['motor_data']
        reference_data = preproc_result['reference_data']
        is_clean = preproc_result['is_clean']
        artifact_max = preproc_result['artifact_max']

        # STAGE 2: HHT for motor channels
        n_motor = motor_data.shape[0]
        motor_power = []
        motor_imfs = []

        for ch_idx in range(n_motor):
            hht_result = hht.process_channel_hht(
                motor_data[ch_idx, :],
                fs,
                freq_band=self.hht_freq_band,
                power_threshold=self.hht_power_threshold
            )
            motor_power.append(hht_result['power'])
            motor_imfs.append(hht_result.get('imfs', np.empty((0, motor_data.shape[1]))))

        motor_power = np.array(motor_power)  # Shape: (n_motor, n_samples)

        # STAGE 3: HHT for reference channels
        n_ref = reference_data.shape[0]
        reference_power = []

        for ch_idx in range(n_ref):
            hht_result = hht.process_channel_hht(
                reference_data[ch_idx, :],
                fs,
                freq_band=self.hht_freq_band,
                power_threshold=self.hht_power_threshold
            )
            reference_power.append(hht_result['power'])

        reference_power = np.array(reference_power)  # Shape: (n_ref, n_samples)

        # STAGE 4: Calculate baseline from motor channels (self-baseline)
        # Determine actual trial start (allows overriding per call)
        trial_start = self.trial_start_time if trial_start_time is None else trial_start_time

        baseline_mean, baseline_std = calculate_baseline(
            motor_power,
            fs,
            baseline_window=self.baseline_window,
            trial_start_time=trial_start
        )

        # STAGE 5: Detect ERD in motor channels
        detection_result = detect_erd_sliding_window(
            motor_power,
            baseline_mean,
            baseline_std,
            fs,
            window_size=self.detection_window_size,
            step_size=self.detection_step_size,
            threshold_sigma=self.threshold_sigma,
            min_channels=self.min_channels,
            task_window=self.task_window,
            trial_start_time=trial_start,
            use_temporal_weight=self.use_temporal_weight,
            penalty_center=self.penalty_center,
            penalty_sigma=self.penalty_sigma,
            penalty_floor=self.penalty_floor
        )

        # Add preprocessing info to result
        detection_result.update({
            'is_clean': is_clean,
            'artifact_max': artifact_max,
            'motor_channels_used': preproc_result['motor_channels'],
            'reference_channels_used': preproc_result['reference_channels'],
            'baseline_mean': baseline_mean,
            'baseline_std': baseline_std,
            'trial_start_time': trial_start,
            'motor_power': motor_power,
            'motor_imfs': motor_imfs,
            'baseline_window': self.baseline_window,
            'threshold_sigma': self.threshold_sigma
        })

        return detection_result
