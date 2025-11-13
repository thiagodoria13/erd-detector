#!/usr/bin/env python
"""
Local test script for ERD detector using real OpenBMI data.

This script processes real EEG trials from the OpenBMI dataset to test
the complete ERD detection pipeline.

Prerequisites:
    - Run download_sample_data.py first to get the data files
    - Or manually download 1-2 subject files from https://gigadb.org/dataset/100542

Usage:
    python test_local.py

Author: Lucas Pereira da Fonseca, Thiago Anversa Sampaio Doria
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

from erd_detector import ERDDetector
from erd_detector.utils import load_openbmi_data, extract_trial


def find_data_files(data_dir='data/openbmi_sample'):
    """Find downloaded OpenBMI .mat files."""
    data_path = Path(data_dir)

    if not data_path.exists():
        return []

    mat_files = list(data_path.glob("*.mat"))
    return sorted(mat_files)


def plot_results(trial_data, channels, result, fs=1000, trial_idx=0):
    """Plot test results for a single trial."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Get motor channel indices
    try:
        c3_idx = channels.index('C3')
        c4_idx = channels.index('C4')
    except ValueError:
        print("Warning: C3 or C4 not found in channels, using first 2 channels")
        c3_idx = 0
        c4_idx = 1

    # Time vector (relative to cue at t=0)
    trial_start_time = result.get('trial_start_time', -3.0)
    t = np.arange(trial_data.shape[0]) / fs + trial_start_time

    # Plot 1: Raw EEG from motor channels
    axes[0].plot(t, trial_data[:, c3_idx], label='C3', alpha=0.7, linewidth=0.5)
    axes[0].plot(t, trial_data[:, c4_idx], label='C4', alpha=0.7, linewidth=0.5)
    axes[0].axvline(0, color='k', linestyle='--', alpha=0.5, label='Cue onset')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude (µV)')
    axes[0].set_title(f'Raw EEG - Trial {trial_idx+1} - Motor Channels (C3, C4)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([trial_start_time, trial_start_time + trial_data.shape[0]/fs])

    # Plot 2: Z-scores over time
    if 'z_scores' in result and result['z_scores'].size > 0:
        window_times = result['window_times']
        z_scores = result['z_scores']

        for ch_idx in range(z_scores.shape[0]):
            ch_name = result['motor_channels_used'][ch_idx]
            axes[1].plot(window_times, z_scores[ch_idx, :], label=ch_name, alpha=0.7)

        threshold = result.get('threshold_sigma', -2.0)
        axes[1].axhline(threshold, color='r', linestyle='--', label=f'Threshold ({threshold}σ)')
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
    axes[2].text(0.1, 0.8, f'Trial {trial_idx+1} Detection Results:', fontsize=14, fontweight='bold')

    if result['detected']:
        axes[2].text(0.1, 0.6, "✓ ERD DETECTED", fontsize=12, color='green', fontweight='bold')
        axes[2].text(0.1, 0.45, f"  Onset time: {result['onset_time']:.3f} s", fontsize=11)
        axes[2].text(0.1, 0.35, f"  Latency from cue: {result['latency']:.3f} s ({result['latency']*1000:.0f} ms)",
                    fontsize=11)
        axes[2].text(0.1, 0.25, f"  Channels: {', '.join(result['motor_channels_used'])}", fontsize=11)
    else:
        axes[2].text(0.1, 0.6, "✗ No ERD detected", fontsize=12, color='red')

    axes[2].text(0.1, 0.1, f"  Artifact status: {'Clean' if result['is_clean'] else 'Artifacts detected'}",
                fontsize=11)
    axes[2].text(0.1, 0.0, f"  Max amplitude: {result['artifact_max']:.1f} µV", fontsize=11)

    axes[2].set_xlim(0, 1)
    axes[2].set_ylim(0, 1)
    axes[2].axis('off')

    plt.tight_layout()
    output_file = f'test_results_trial_{trial_idx+1}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  [Saved] {output_file}")
    plt.close()


def main():
    """Main test function."""
    print("=" * 70)
    print("ERD Detector - Local Test with Real OpenBMI Data")
    print("=" * 70)
    print()

    # Check for downloaded data
    print("[1/5] Checking for OpenBMI data files...")
    data_files = find_data_files()

    if not data_files:
        print()
        print("✗ No data files found!")
        print()
        print("Please download sample data first:")
        print("  python download_sample_data.py")
        print()
        print("Or manually download from: https://gigadb.org/dataset/100542")
        print("  - sess01_subj01_EEG_MI.mat")
        print("  - sess02_subj01_EEG_MI.mat")
        print()
        print("Save files to: data/openbmi_sample/")
        return

    print(f"  ✓ Found {len(data_files)} data file(s):")
    for f in data_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"    - {f.name} ({size_mb:.1f} MB)")
    print()

    # Load data from first file
    print("[2/5] Loading EEG data...")
    try:
        # Try to parse subject and session from filename
        # Format: sess01_subj01_EEG_MI.mat or sess02_subj01_EEG_MI.mat
        first_file = data_files[0].name
        if 'sess01' in first_file:
            session = 1
        elif 'sess02' in first_file:
            session = 2
        else:
            session = 1

        # Extract subject number
        import re
        match = re.search(r'subj(\d+)', first_file)
        if match:
            subject_id = int(match.group(1))
        else:
            subject_id = 1

        # Determine data type from session
        data_type = 'train' if session == 1 else 'test'

        print(f"  - Subject: {subject_id}")
        print(f"  - Session: {session}")
        print(f"  - Data type: {data_type}")

        # Load data
        data = load_openbmi_data(
            subject_id=subject_id,
            session=session,
            data_type=data_type,
            data_dir='data/openbmi_sample'
        )

        print(f"  ✓ Loaded {len(data['events'])} trials")
        print(f"  - Channels: {len(data['channels'])} total")
        print(f"  - Sampling rate: {data['fs']} Hz")
        print(f"  - Data shape: {data['data'].shape}")

    except Exception as e:
        print(f"  ✗ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return

    print()

    # Initialize ERD detector
    print("[3/5] Initializing ERD detector...")
    detector = ERDDetector(
        motor_channels=['C3', 'C4'],
        reference_channels=['O1', 'O2', 'Fz'],
        threshold_sigma=-2.0,
        min_channels=2,
        baseline_window=(-3.0, -1.0),
        task_window=(0.0, 4.0),
        trial_start_time=-3.0
    )
    print("  - Threshold: -2.0 sigma")
    print("  - Min channels: 2 (both C3 and C4)")
    print("  - Baseline window: -3.0s to -1.0s")
    print("  - Task window: 0.0s to +4.0s")
    print()

    # Process multiple trials
    print("[4/5] Processing trials...")
    num_trials_to_test = min(5, len(data['events']))
    print(f"  Testing first {num_trials_to_test} trials")
    print()

    results = []
    for trial_idx in range(num_trials_to_test):
        print(f"  Trial {trial_idx+1}/{num_trials_to_test}:")

        # Extract trial
        event_time = data['events'][trial_idx]
        trial_data = extract_trial(data['data'], event_time, data['fs'])

        print(f"    - Trial shape: {trial_data.shape}")

        # Process trial
        try:
            result = detector.process_trial(
                trial_data=trial_data,
                channels=data['channels'],
                fs=data['fs'],
                trial_start_time=-3.0
            )

            results.append(result)

            # Show result
            if result['detected']:
                print(f"    ✓ ERD detected at {result['onset_time']:.3f}s (latency: {result['latency']*1000:.0f}ms)")
            else:
                print(f"    ✗ No ERD detected")

            print(f"    - Artifact status: {'Clean' if result['is_clean'] else 'Artifacts'}")

        except Exception as e:
            print(f"    ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            results.append(None)

        print()

    # Summary statistics
    print("[5/5] Summary:")
    print("-" * 70)

    successful_trials = [r for r in results if r is not None]
    detected_trials = [r for r in successful_trials if r['detected']]

    print(f"  Trials processed: {len(successful_trials)}/{num_trials_to_test}")
    print(f"  ERD detected: {len(detected_trials)}/{len(successful_trials)} ({100*len(detected_trials)/max(1,len(successful_trials)):.1f}%)")

    if detected_trials:
        latencies = [r['latency']*1000 for r in detected_trials]
        print(f"  Average latency: {np.mean(latencies):.0f} ms (±{np.std(latencies):.0f} ms)")
        print(f"  Latency range: {np.min(latencies):.0f} - {np.max(latencies):.0f} ms")

    clean_trials = [r for r in successful_trials if r['is_clean']]
    print(f"  Clean trials: {len(clean_trials)}/{len(successful_trials)}")

    print("-" * 70)
    print()

    # Plot first trial
    print("[Plotting] Generating visualization for first trial...")
    if successful_trials:
        try:
            plot_results(
                trial_data=extract_trial(data['data'], data['events'][0], data['fs']),
                channels=data['channels'],
                result=successful_trials[0],
                fs=data['fs'],
                trial_idx=0
            )
        except Exception as e:
            print(f"  Warning: Could not generate plot: {e}")

    print()
    print("=" * 70)
    print("Test completed successfully!")
    print("=" * 70)
    print()

    return results


if __name__ == '__main__':
    results = main()
