"""
Hilbert-Huang Transform (HHT) Module for ERD Detection

This module implements the HHT pipeline for extracting instantaneous power:
1. Empirical Mode Decomposition (EMD) - Adaptive signal decomposition
2. Intrinsic Mode Function (IMF) selection - Spectral criterion (60% power in 8-30 Hz)
3. Hilbert Transform - Extract instantaneous amplitude and frequency
4. Instantaneous power calculation - Power = amplitude^2

Neurophysiological Justification:
- HHT is adaptive: handles non-linear, non-stationary EEG signals
- IMFs represent different oscillatory modes in the signal
- Spectral selection focuses on mu (8-13 Hz) and beta (14-30 Hz) rhythms
- Instantaneous power captures rapid ERD dynamics (important for low latency)

Author: Lucas Pereira da Fonseca, Thiago Anversa Sampaio Doria
"""

import numpy as np
from scipy.signal import hilbert, welch
# EMD-signal publishes the implementation under the `pyemd` package but
# original docs reference `PyEMD`. Support both import paths gracefully.
try:
    from PyEMD import EMD  # type: ignore
except ImportError:  # pragma: no cover - fallback for lowercase package name
    try:
        from pyemd import EMD  # type: ignore
    except ImportError as exc:  # pragma: no cover - improved error message
        raise ImportError(
            "PyEMD/EMD-signal is required. Install via `pip install EMD-signal`."
        ) from exc
from typing import List, Tuple, Dict, Optional
import warnings


def empirical_mode_decomposition(
    signal: np.ndarray,
    max_imf: int = -1
) -> np.ndarray:
    """
    Decompose signal into Intrinsic Mode Functions (IMFs) using EMD.

    EMD algorithm (sifting process):
    1. Identify all local extrema (maxima and minima)
    2. Interpolate maxima and minima with cubic splines
    3. Compute mean envelope: m = (upper_envelope + lower_envelope) / 2
    4. Extract candidate IMF: h = signal - m
    5. Repeat until h satisfies IMF criteria
    6. Subtract IMF from signal, repeat for residual

    Args:
        signal: Input signal, shape (n_samples,)
        max_imf: Maximum number of IMFs to extract (default: -1 = all)

    Returns:
        IMFs array, shape (n_imfs, n_samples)

    Neurophysiological Note:
        - IMF1 (highest frequency): Often noise or muscle artifacts
        - IMF2-4 (mid frequency): Typically contain beta (14-30 Hz) and mu (8-13 Hz)
        - IMF5+ (low frequency): Slow cortical potentials, drift
        - Number of IMFs is data-driven, typically 6-10 for EEG
    """
    # Validate input
    if signal.ndim != 1:
        raise ValueError(f"Expected 1D signal, got shape {signal.shape}")

    if len(signal) < 10:
        raise ValueError(f"Signal too short for EMD: {len(signal)} samples (need >= 10)")

    # Initialize EMD object
    emd = EMD()

    # Set max_imf if specified
    if max_imf > 0:
        emd.FIXE_H = max_imf

    # Perform EMD decomposition
    try:
        imfs = emd.emd(signal)
    except Exception as e:
        warnings.warn(f"EMD failed: {e}. Returning original signal as single IMF.")
        return signal.reshape(1, -1)

    # Handle case where EMD returns 1D array (single IMF)
    if imfs.ndim == 1:
        imfs = imfs.reshape(1, -1)

    # PyEMD returns shape (n_samples, n_imfs), transpose to (n_imfs, n_samples)
    if imfs.shape[0] > imfs.shape[1]:
        imfs = imfs.T

    return imfs


def select_imfs_spectral(
    imfs: np.ndarray,
    fs: float,
    freq_band: Tuple[float, float] = (8.0, 30.0),
    power_threshold: float = 0.6
) -> Tuple[np.ndarray, List[int]]:
    """
    Select IMFs with significant power in the target frequency band.

    Spectral criterion:
    For each IMF, compute power spectral density (PSD) using Welch's method.
    Select IMF if: P_band / P_total >= threshold

    where:
    - P_band = power in freq_band (8-30 Hz for mu/beta)
    - P_total = total power across all frequencies
    - threshold = 0.6 (60%, empirically determined)

    Args:
        imfs: IMFs from EMD, shape (n_imfs, n_samples)
        fs: Sampling frequency in Hz
        freq_band: Target frequency band (default: 8-30 Hz)
        power_threshold: Minimum fraction of power in band (default: 0.6)

    Returns:
        Tuple of (selected_imfs, selected_indices)
        - selected_imfs: Selected IMFs, shape (n_selected, n_samples)
        - selected_indices: Indices of selected IMFs in original array

    Neurophysiological Note:
        - 60% threshold balances specificity vs sensitivity
        - Lower threshold: more IMFs selected, more noise
        - Higher threshold: fewer IMFs, may miss weak ERD
        - Typical result: 2-4 IMFs selected for motor imagery EEG
    """
    if imfs.ndim != 2:
        raise ValueError(f"Expected 2D IMFs array, got shape {imfs.shape}")

    n_imfs = imfs.shape[0]
    selected_indices = []

    # Process each IMF
    for imf_idx in range(n_imfs):
        imf = imfs[imf_idx, :]

        # Compute power spectral density using Welch's method
        # nperseg = 256 samples ~ 256ms at 1000 Hz (good frequency resolution)
        nperseg = min(256, len(imf) // 4)  # At least 4 segments

        try:
            freqs, psd = welch(imf, fs=fs, nperseg=nperseg)
        except Exception as e:
            warnings.warn(f"PSD computation failed for IMF {imf_idx}: {e}. Skipping.")
            continue

        # Compute total power (integrate PSD over all frequencies)
        total_power = np.trapz(psd, freqs)

        if total_power == 0:
            continue

        # Find frequency indices within target band
        band_mask = (freqs >= freq_band[0]) & (freqs <= freq_band[1])

        if not np.any(band_mask):
            warnings.warn(f"No frequencies in target band {freq_band} Hz for IMF {imf_idx}")
            continue

        # Compute power in target band
        band_power = np.trapz(psd[band_mask], freqs[band_mask])

        # Compute power ratio
        power_ratio = band_power / total_power

        # Select IMF if power ratio exceeds threshold
        if power_ratio >= power_threshold:
            selected_indices.append(imf_idx)

    # Check if any IMFs were selected
    if len(selected_indices) == 0:
        warnings.warn(
            f"No IMFs met spectral criterion (>= {power_threshold*100}% power in {freq_band} Hz). "
            f"Returning empty array."
        )
        return np.array([]).reshape(0, imfs.shape[1]), []

    # Extract selected IMFs
    selected_imfs = imfs[selected_indices, :]

    return selected_imfs, selected_indices


def hilbert_transform(
    signal: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Hilbert Transform to extract instantaneous amplitude and frequency.

    Mathematical formulation:
    For real signal x(t), the analytic signal is:
    z(t) = x(t) + j * H[x(t)]

    where H[x(t)] is the Hilbert transform of x(t).

    In polar form: z(t) = A(t) * exp(j * phi(t))

    where:
    - A(t) = |z(t)| = sqrt(x^2 + H[x]^2) is instantaneous amplitude
    - phi(t) = angle(z(t)) is instantaneous phase
    - f(t) = (1/2pi) * d(phi)/dt is instantaneous frequency

    Args:
        signal: Input signal, shape (n_samples,) or (n_channels, n_samples)

    Returns:
        Tuple of (amplitude, frequency)
        - amplitude: Instantaneous amplitude, same shape as input
        - frequency: Instantaneous frequency in Hz, same shape as input

    Neurophysiological Note:
        - Instantaneous amplitude tracks ERD dynamics in real-time
        - Traditional Fourier: fixed time window, poor time resolution
        - HHT + Hilbert: adaptive, captures transient ERD changes
        - Critical for low-latency BCI (<200ms detection)
    """
    # Store original shape
    original_shape = signal.shape

    # Handle both 1D and 2D inputs
    if signal.ndim == 1:
        signal = signal.reshape(1, -1)

    n_channels, n_samples = signal.shape

    # Initialize output arrays
    amplitude = np.zeros_like(signal)
    frequency = np.zeros_like(signal)

    # Process each channel/IMF
    for ch_idx in range(n_channels):
        # Compute analytic signal using Hilbert transform
        analytic_signal = hilbert(signal[ch_idx, :])

        # Extract instantaneous amplitude: |z(t)|
        amplitude[ch_idx, :] = np.abs(analytic_signal)

        # Extract instantaneous phase: angle(z(t))
        phase = np.unwrap(np.angle(analytic_signal))

        # Compute instantaneous frequency: (1/2pi) * d(phase)/dt
        phase_derivative = np.gradient(phase)
        frequency[ch_idx, :] = phase_derivative / (2.0 * np.pi)

    # Restore original shape if input was 1D
    if len(original_shape) == 1:
        amplitude = amplitude.flatten()
        frequency = frequency.flatten()

    return amplitude, frequency


def instantaneous_power(
    amplitude: np.ndarray
) -> np.ndarray:
    """
    Compute instantaneous power from instantaneous amplitude.

    Mathematical formulation:
    P(t) = A(t)^2

    where:
    - P(t) = instantaneous power
    - A(t) = instantaneous amplitude from Hilbert transform

    Args:
        amplitude: Instantaneous amplitude, shape (n_samples,) or (n_channels, n_samples)

    Returns:
        Instantaneous power, same shape as input

    Neurophysiological Note:
        - Power (not amplitude) is standard measure of oscillatory activity
        - Power ~ neural population synchronization
        - ERD = power decrease relative to baseline
        - Typical ERD: 20-40% power reduction during motor imagery
    """
    # Simply square the amplitude
    power = amplitude ** 2

    return power


def process_channel_hht(
    signal: np.ndarray,
    fs: float,
    freq_band: Tuple[float, float] = (8.0, 30.0),
    power_threshold: float = 0.6,
    max_imf: int = -1,
    return_frequency: bool = False
) -> Dict:
    """
    Complete HHT pipeline for a single channel.

    Pipeline stages:
    1. EMD: Decompose signal into IMFs
    2. IMF selection: Select IMFs with >= 60% power in freq_band
    3. Hilbert Transform: Extract instantaneous amplitude/frequency
    4. Power calculation: Compute instantaneous power
    5. Aggregation: Sum power across selected IMFs

    Args:
        signal: Input signal, shape (n_samples,)
        fs: Sampling frequency in Hz
        freq_band: Target frequency band (default: 8-30 Hz)
        power_threshold: IMF selection threshold (default: 0.6)
        max_imf: Maximum number of IMFs (default: -1 = all)
        return_frequency: Whether to return instantaneous frequency (default: False)

    Returns:
        Dictionary containing:
        - 'power': Instantaneous power time series, shape (n_samples,)
        - 'imfs': Selected IMFs, shape (n_selected, n_samples)
        - 'n_imfs_total': Total number of IMFs from EMD
        - 'n_imfs_selected': Number of IMFs selected
        - 'selected_indices': Indices of selected IMFs
        - 'frequency': Instantaneous frequency (if return_frequency=True)

    Example:
        >>> result = process_channel_hht(signal, fs=1000)
        >>> power = result['power']  # Shape: (n_samples,)
        >>> print(f"Selected {result['n_imfs_selected']} / {result['n_imfs_total']} IMFs")
    """
    # Validate input
    if signal.ndim != 1:
        raise ValueError(f"Expected 1D signal, got shape {signal.shape}")

    # STAGE 1: Empirical Mode Decomposition
    imfs = empirical_mode_decomposition(signal, max_imf=max_imf)
    n_imfs_total = imfs.shape[0]

    # STAGE 2: IMF Selection (spectral criterion)
    selected_imfs, selected_indices = select_imfs_spectral(
        imfs,
        fs,
        freq_band=freq_band,
        power_threshold=power_threshold
    )
    n_imfs_selected = len(selected_indices)

    # Check if any IMFs were selected
    if n_imfs_selected == 0:
        warnings.warn("No IMFs selected. Returning zero power.")
        return {
            'power': np.zeros_like(signal),
            'imfs': selected_imfs,
            'n_imfs_total': n_imfs_total,
            'n_imfs_selected': 0,
            'selected_indices': [],
            'frequency': np.zeros_like(signal) if return_frequency else None
        }

    # STAGE 3: Hilbert Transform
    amplitude, frequency_all = hilbert_transform(selected_imfs)

    # STAGE 4: Instantaneous Power
    power_per_imf = instantaneous_power(amplitude)

    # STAGE 5: Aggregate power across selected IMFs
    if power_per_imf.ndim == 1:
        # Single IMF selected
        total_power = power_per_imf
        mean_frequency = frequency_all
    else:
        # Multiple IMFs selected, sum across IMFs
        total_power = np.sum(power_per_imf, axis=0)
        # For frequency, take weighted average (weight by amplitude)
        total_amplitude = np.sum(amplitude, axis=0)
        mean_frequency = np.sum(amplitude * frequency_all, axis=0) / (total_amplitude + 1e-10)

    # Prepare result dictionary
    result = {
        'power': total_power,
        'imfs': selected_imfs,
        'n_imfs_total': n_imfs_total,
        'n_imfs_selected': n_imfs_selected,
        'selected_indices': selected_indices
    }

    # Add frequency if requested
    if return_frequency:
        result['frequency'] = mean_frequency
    else:
        result['frequency'] = None

    return result
