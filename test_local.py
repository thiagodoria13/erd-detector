#!/usr/bin/env python
"""
Local test script for ERD detector using synthetic EEG data.

This script generates synthetic EEG signals with simulated ERD events
to test the complete pipeline without requiring the full OpenBMI dataset.

Usage:
    python test_local.py

Author: Lucas Pereira da Fonseca, Thiago Anversa Sampaio Doria
"""

import numpy as np
import matplotlib.pyplot as plt
from erd_detector import ERDDetector

# Set random seed for reproducibility
np.random.seed(42)


def generate_synthetic_eeg(
    fs: int = 1000,
    duration: float = 7.0,
    n_channels: int = 62,
    erd_onset: float = 1.0,
    erd_duration: float = 2.0,
    erd_strength: float = 0.6
):
    """
    Generate synthetic EEG data with simulated ERD event.

    Args:
        fs: Sampling frequency in Hz
        duration: Trial duration in seconds
        n_channels: Number of EEG channels
        erd_onset: Time of ERD onset relative to cue (seconds)
        erd_duration: Duration of ERD event (seconds)
        erd_strength: ERD strength (0-1, fraction of power reduction)

    Returns:
        Tuple of (data, channels, fs)
        - data: Synthetic EEG, shape (n_samples, n_channels)
        - channels: List of channel names
        - fs: Sampling frequency
    """
    n_samples = int(duration * fs)
    t = np.linspace(0, duration, n_samples)

    # Standard 10-20 channel layout (subset of 62 channels)
    # Focus on motor and reference channels
    channels = [
        'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
        'FC5', 'FC1', 'FC2', 'FC6',
        'T7', 'C3', 'Cz', 'C4', 'T8',
        'CP5', 'CP1', 'CP2', 'CP6',
        'P7', 'P3', 'Pz', 'P4', 'P8',
        'O1', 'Oz', 'O2'
    ]

    # Pad with additional channels if needed
    while len(channels) < n_channels:
        channels.append(f'Ch{len(channels)+1}')

    channels = channels[:n_channels]

    # Initialize data array
    data = np.zeros((n_samples, n_channels))

    # Define motor channels indices (C3, C4) for ERD simulation
    motor_indices = [channels.index('C3'), channels.index('C4')]

    # Generate baseline oscillations for all channels
    for ch_idx in range(n_channels):
        # Mu rhythm (8-13 Hz) - dominant in motor cortex
        mu_freq = np.random.uniform(9, 12)
        mu_amplitude = 15.0 if ch_idx in motor_indices else 10.0

        # Beta rhythm (14-30 Hz)
        beta_freq = np.random.uniform(18, 25)
        beta_amplitude = 10.0 if ch_idx in motor_indices else 5.0

        # Alpha rhythm (8-13 Hz) - stronger in occipital
        alpha_amplitude = 20.0 if 'O' in channels[ch_idx] else 5.0

        # Generate rhythms
        data[:, ch_idx] = (
            mu_amplitude * np.sin(2 * np.pi * mu_freq * t) +
            beta_amplitude * np.sin(2 * np.pi * beta_freq * t) +
            alpha_amplitude * np.sin(2 * np.pi * 10 * t)
        )

        # Add pink noise (1/f noise typical in EEG)
        noise = np.random.randn(n_samples) * 5.0
        data[:, ch_idx] += noise

    # Simulate ERD event in motor channels (power reduction)
    # ERD = Event-Related Desynchronization = amplitude decrease
    trial_start_time = -3.0  # Trial starts at -3s relative to cue (time 0)
    erd_start_sample = int((erd_onset - trial_start_time) * fs)
    erd_end_sample = int((erd_onset + erd_duration - trial_start_time) * fs)

    for ch_idx in motor_indices:
        # Create smooth ERD envelope (gradual onset/offset)
        envelope = np.ones(n_samples)

        # Ramp down (ERD onset)
        ramp_samples = int(0.5 * fs)  # 500ms ramp
        for i in range(erd_start_sample, min(erd_start_sample + ramp_samples, n_samples)):
            progress = (i - erd_start_sample) / ramp_samples
            envelope[i] = 1.0 - erd_strength * progress

        # Sustained ERD
        envelope[erd_start_sample + ramp_samples:erd_end_sample] = 1.0 - erd_strength

        # Ramp up (ERD offset)
        for i in range(erd_end_sample, min(erd_end_sample + ramp_samples, n_samples)):
            progress = (i - erd_end_sample) / ramp_samples
            envelope[i] = 1.0 - erd_strength * (1.0 - progress)

        # Apply envelope to reduce amplitude (simulates desynchronization)
        data[:, ch_idx] *= envelope

    return data, channels, fs


def plot_results(data, channels, result, fs=1000):
    """Plot test results."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Get motor channel indices
    c3_idx = channels.index('C3')
    c4_idx = channels.index('C4')

    # Time vector (relative to cue at t=0)
    trial_start_time = result.get('trial_start_time', -3.0)
    t = np.arange(data.shape[0]) / fs + trial_start_time

    # Plot 1: Raw EEG from motor channels
    axes[0].plot(t, data[:, c3_idx], label='C3', alpha=0.7)
    axes[0].plot(t, data[:, c4_idx], label='C4', alpha=0.7)
    axes[0].axvline(0, color='k', linestyle='--', alpha=0.5, label='Cue onset')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude (µV)')
    axes[0].set_title('Raw EEG - Motor Channels (C3, C4)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Z-scores over time
    if 'z_scores' in result and result['z_scores'].size > 0:
        window_times = result['window_times']
        z_scores = result['z_scores']

        for ch_idx in range(z_scores.shape[0]):
            ch_name = result['motor_channels_used'][ch_idx]
            axes[1].plot(window_times, z_scores[ch_idx, :], label=ch_name, alpha=0.7)

        axes[1].axhline(result.get('threshold_sigma', -2.0),
                       color='r', linestyle='--', label='Detection threshold')
        axes[1].axvline(0, color='k', linestyle='--', alpha=0.5)

        if result['detected']:
            axes[1].axvline(result['onset_time'], color='g',
                          linestyle='-', linewidth=2, label='ERD onset')

        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Z-score')
        axes[1].set_title('Normalized Power (Z-scores) - Sliding Window Analysis')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    # Plot 3: Detection summary
    axes[2].text(0.1, 0.8, 'Detection Results:', fontsize=14, fontweight='bold')

    if result['detected']:
        axes[2].text(0.1, 0.6, f"✓ ERD DETECTED", fontsize=12, color='green', fontweight='bold')
        axes[2].text(0.1, 0.45, f"  Onset time: {result['onset_time']:.3f} s", fontsize=11)
        axes[2].text(0.1, 0.35, f"  Latency from cue: {result['latency']:.3f} s ({result['latency']*1000:.0f} ms)",
                    fontsize=11)
        axes[2].text(0.1, 0.25, f"  Channels: {', '.join(result['motor_channels_used'])}", fontsize=11)
    else:
        axes[2].text(0.1, 0.6, f"✗ No ERD detected", fontsize=12, color='red')

    axes[2].text(0.1, 0.1, f"  Artifact status: {'Clean' if result['is_clean'] else 'Artifacts detected'}",
                fontsize=11)
    axes[2].text(0.1, 0.0, f"  Max amplitude: {result['artifact_max']:.1f} µV", fontsize=11)

    axes[2].set_xlim(0, 1)
    axes[2].set_ylim(0, 1)
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig('test_results.png', dpi=150, bbox_inches='tight')
    print("\n[Saved] test_results.png")
    plt.show()


def main():
    """Main test function."""
    print("=" * 70)
    print("ERD Detector - Local Test with Synthetic Data")
    print("=" * 70)

    # Generate synthetic EEG data
    print("\n[1/4] Generating synthetic EEG data...")
    print("  - Duration: 7 seconds (trial from -3s to +4s)")
    print("  - Sampling rate: 1000 Hz")
    print("  - Simulated ERD: onset at +1.0s, duration 2.0s, strength 60%")

    data, channels, fs = generate_synthetic_eeg(
        fs=1000,
        duration=7.0,  # -3s to +4s
        n_channels=62,
        erd_onset=1.0,  # ERD starts 1s after cue
        erd_duration=2.0,
        erd_strength=0.6  # 60% power reduction
    )

    print(f"  - Data shape: {data.shape} (samples x channels)")
    print(f"  - Channels: {len(channels)} total")
    print(f"  - Motor channels: C3, C4")
    print(f"  - Reference channels: O1, O2, Fz")

    # Initialize ERD detector
    print("\n[2/4] Initializing ERD detector...")
    detector = ERDDetector(
        motor_channels=['C3', 'C4'],
        reference_channels=['O1', 'O2', 'Fz'],
        threshold_sigma=-2.0,
        min_channels=2,
        baseline_window=(-3.0, -1.0),  # 2s baseline before cue
        task_window=(0.0, 4.0),  # 4s task period after cue
        trial_start_time=-3.0  # Trial starts at -3s
    )
    print("  - Threshold: -2.0 sigma")
    print("  - Min channels: 2 (both C3 and C4)")
    print("  - Baseline window: -3.0s to -1.0s")
    print("  - Task window: 0.0s to +4.0s")

    # Process trial
    print("\n[3/4] Processing trial through complete pipeline...")
    print("  Stage 1: Preprocessing (bandpass, Laplacian, artifacts)")
    print("  Stage 2: HHT analysis (EMD, IMF selection, Hilbert)")
    print("  Stage 3: Baseline calculation from reference channels")
    print("  Stage 4: ERD detection with sliding window")

    try:
        result = detector.process_trial(
            trial_data=data,
            channels=channels,
            fs=fs,
            trial_start_time=-3.0
        )

        print("\n  ✓ Pipeline completed successfully!")

    except Exception as e:
        print(f"\n  ✗ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return

    # Display results
    print("\n[4/4] Results:")
    print("-" * 70)

    if result['detected']:
        print(f"  ✓ ERD DETECTED")
        print(f"    - Onset time: {result['onset_time']:.3f} s (relative to cue)")
        print(f"    - Latency: {result['latency']:.3f} s ({result['latency']*1000:.0f} ms)")
        print(f"    - Detection sample: {result['onset_sample']}")
        print(f"    - Number of detections: {len(result['detection_times'])}")
    else:
        print(f"  ✗ No ERD detected")

    print(f"\n  Preprocessing:")
    print(f"    - Artifact status: {'Clean' if result['is_clean'] else 'Artifacts detected'}")
    print(f"    - Max amplitude: {result['artifact_max']:.1f} µV")
    print(f"    - Motor channels used: {', '.join(result['motor_channels_used'])}")
    print(f"    - Reference channels used: {', '.join(result['reference_channels_used'])}")

    print(f"\n  Baseline statistics:")
    print(f"    - Mean power: {result['baseline_mean']:.2f}")
    print(f"    - Std deviation: {result['baseline_std']:.2f}")

    if 'z_scores' in result and result['z_scores'].size > 0:
        print(f"\n  Detection windows:")
        print(f"    - Number of windows: {len(result['window_times'])}")
        print(f"    - Time range: {result['window_times'][0]:.2f}s to {result['window_times'][-1]:.2f}s")
        min_z = np.min(result['z_scores'])
        print(f"    - Minimum z-score: {min_z:.2f} (threshold: {detector.threshold_sigma:.2f})")

    print("-" * 70)

    # Plot results
    print("\n[Plotting] Generating visualization...")
    try:
        plot_results(data, channels, result, fs)
    except Exception as e:
        print(f"  Warning: Could not generate plot: {e}")

    print("\n" + "=" * 70)
    print("Test completed successfully!")
    print("=" * 70)

    return result


if __name__ == '__main__':
    result = main()
