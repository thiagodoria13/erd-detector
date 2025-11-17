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
    """Plot a simple, step-by-step view for a single trial."""
    fig, axes = plt.subplots(5, 1, figsize=(12, 13), sharex=True)
    # Always show x-axis tick labels on every panel
    for ax in axes:
        ax.tick_params(axis='x', labelbottom=True)

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
    baseline_window = result.get('baseline_window', (-3.0, -2.0))
    baseline_start, baseline_end = baseline_window
    t = np.arange(trial_data.shape[0]) / fs + trial_start_time

    # Plot 1: Raw EEG (motor channels only)
    axes[0].plot(t, trial_data[:, c3_idx], label='C3', alpha=0.8, linewidth=0.8)
    axes[0].plot(t, trial_data[:, c4_idx], label='C4', alpha=0.8, linewidth=0.8)
    axes[0].axvspan(baseline_start, baseline_end, color='gray', alpha=0.1, label='Baseline')
    axes[0].axvspan(0.0, 4.0, color='yellow', alpha=0.1, label='Task')
    axes[0].axvline(0, color='k', linestyle='--', alpha=0.6, label='Cue (0 s)')
    if result.get('detected') and result.get('onset_time') is not None:
        axes[0].axvline(result['onset_time'], color='g', linestyle='-', alpha=0.8, label='Detection')
    axes[0].set_ylabel('µV')
    axes[0].set_title(f'Trial {trial_idx+1}: Raw C3/C4')
    axes[0].legend(loc='upper right', fontsize=8)
    axes[0].grid(True, alpha=0.3)
    end_time = trial_start_time + trial_data.shape[0]/fs
    axes[0].set_xlim([trial_start_time, end_time])
    axes[0].set_xlabel('Time (s) relative to cue')

    # Plot 2: Z-scores over time
    if 'z_scores' in result and result['z_scores'].size > 0:
        window_times = result['window_times']
        z_scores = result['z_scores']

        for ch_idx in range(z_scores.shape[0]):
            ch_name = result['motor_channels_used'][ch_idx]
            axes[1].plot(window_times, z_scores[ch_idx, :], label=ch_name, alpha=0.8)

        threshold = result.get('threshold_sigma', -2.0)
        axes[1].axhline(threshold, color='r', linestyle='--', label=f'Threshold ({threshold}σ)')
        axes[1].axvline(0, color='k', linestyle='--', alpha=0.5, label='Cue (0 s)')
        if result['detected'] and result.get('onset_time') is not None:
            axes[1].axvline(result['onset_time'], color='g', linestyle='-', linewidth=2, label='Detection')

        axes[1].axvspan(baseline_start, baseline_end, color='gray', alpha=0.1)
        axes[1].axvspan(0.0, 4.0, color='yellow', alpha=0.1)

        axes[1].set_ylabel('Z-score')
        axes[1].set_title('Sliding-window Z (baseline gray, task yellow)')
        axes[1].legend(loc='upper right', fontsize=8)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlabel('Time (s) relative to cue')

    # Plot 3: IMF waveforms (first motor channel)
    if 'motor_imfs' in result and len(result.get('motor_imfs', [])) > 0:
        # Plot selected IMFs for each motor channel (up to 2 IMFs per channel to reduce clutter)
        for ch_idx, imfs in enumerate(result['motor_imfs']):
            if imfs is None or np.size(imfs) == 0:
                continue
            n_plot = min(imfs.shape[0], 2)
            for k in range(n_plot):
                axes[2].plot(t, imfs[k, :], label=f"{result['motor_channels_used'][ch_idx]} IMF{k+1}", alpha=0.7, linewidth=0.7)
        axes[2].axvspan(baseline_start, baseline_end, color='gray', alpha=0.1)
        axes[2].axvspan(0.0, 4.0, color='yellow', alpha=0.1)
        axes[2].axvline(0, color='k', linestyle='--', alpha=0.6)
        if result.get('detected') and result.get('onset_time') is not None:
            axes[2].axvline(result['onset_time'], color='g', linestyle='-', alpha=0.8)
        axes[2].set_ylabel('µV')
        axes[2].set_title('Selected IMFs (motor channels)')
        axes[2].legend(loc='upper right', fontsize=8)
        axes[2].grid(True, alpha=0.3)
        axes[2].set_xlabel('Time (s) relative to cue')
    else:
        axes[2].text(0.1, 0.5, "No IMFs available", fontsize=12)
        axes[2].axis('off')

    # Plot 4: Instantaneous power (HHT sum of selected IMFs)
    if 'motor_power' in result:
        mpower = result['motor_power']
        axes[3].plot(t, mpower[0, :], label='C3 power', alpha=0.8, linewidth=0.8)
        if mpower.shape[0] > 1:
            axes[3].plot(t, mpower[1, :], label='C4 power', alpha=0.8, linewidth=0.8)

        baseline_req = result.get('baseline_mean', None)
        baseline_std = result.get('baseline_std', None)
        thr_sigma = result.get('threshold_sigma', -0.5)
        if baseline_req is not None and baseline_std is not None:
            baseline_req = np.asarray(baseline_req).flatten()
            baseline_std = np.asarray(baseline_std).flatten()
            for ch_idx in range(min(mpower.shape[0], baseline_req.shape[0])):
                ch_label = result['motor_channels_used'][ch_idx]
                axes[3].axhline(
                    baseline_req[ch_idx],
                    color='k',
                    linestyle='--',
                    alpha=0.6 if ch_idx == 0 else 0.3,
                    label=f'{ch_label} baseline' if ch_idx == 0 else None
                )
                axes[3].axhline(
                    baseline_req[ch_idx] + thr_sigma * baseline_std[ch_idx],
                    color='r',
                    linestyle='--',
                    alpha=0.6 if ch_idx == 0 else 0.3,
                    label=f'{ch_label} threshold' if ch_idx == 0 else None
                )

        axes[3].axvspan(baseline_start, baseline_end, color='gray', alpha=0.1)
        axes[3].axvspan(0.0, 4.0, color='yellow', alpha=0.1)
        axes[3].axvline(0, color='k', linestyle='--', alpha=0.6)
        if result.get('detected') and result.get('onset_time') is not None:
            axes[3].axvline(result['onset_time'], color='g', linestyle='-', alpha=0.8)

        axes[3].set_ylabel('Power (µV²)')
        axes[3].set_title('Instantaneous power (HHT)')
        axes[3].legend(loc='upper right', fontsize=8)
        axes[3].grid(True, alpha=0.3)
        axes[3].set_xlabel('Time (s) relative to cue')

    # Plot 5: Detection summary
    axes[4].text(0.1, 0.8, f"Trial {trial_idx+1} Detection Results:", fontsize=14, fontweight="bold")

    if result["detected"]:
        axes[4].text(0.1, 0.6, "ERD DETECTED", fontsize=12, color="green", fontweight="bold")
        axes[4].text(0.1, 0.45, f"  Onset time: {result['onset_time']:.3f} s", fontsize=11)
        axes[4].text(0.1, 0.35, f"  Latency from cue: {result['latency']:.3f} s ({result['latency']*1000:.0f} ms)", fontsize=11)
        axes[4].text(0.1, 0.25, f"  Channels: {', '.join(result['motor_channels_used'])}", fontsize=11)
    else:
        axes[4].text(0.1, 0.6, "No ERD detected", fontsize=12, color="red")
    axes[4].text(0.1, 0.1, f"  Artifact status: {'Clean' if result['is_clean'] else 'Artifacts detected'}",
                fontsize=11)
    axes[4].text(0.1, 0.0, f"  Max amplitude: {result['artifact_max']:.1f} µV", fontsize=11)

    axes[4].set_xlim(0, 1)
    axes[4].set_ylim(0, 1)
    axes[4].axis('off')
    axes[4].set_title('Decision summary')

    # Set x-label and consistent tick marks (1 s step) for all axes except summary
    tick_vals = np.arange(np.floor(trial_start_time), np.ceil(end_time) + 1, 1.0)
    for ax in axes[:-1]:
        ax.set_xticks(tick_vals)
        ax.set_xlabel('Time (s) relative to cue')
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
        threshold_sigma=-0.60,             # tuned threshold
        min_channels=1,                    # allow single motor channel
        baseline_window=(-2.0, -1.0),      # tuned baseline
        task_window=(0.0, 4.0),            # detect only post-cue window
        detection_step_size=0.005,         # 5 ms step
        trial_start_time=-3.0,
        artifact_threshold=750.0,          # raised artifact gate
        hht_power_threshold=0.35           # IMF selection for mu/beta
    )
    # Print the actual parameters used (keeps logs and images consistent)
    print(f"  - Threshold: {detector.threshold_sigma} sigma")
    print(f"  - Min channels: {detector.min_channels}")
    print(f"  - Baseline window: {detector.baseline_window[0]}s to {detector.baseline_window[1]}s (motor self-baseline)")
    print(f"  - Task window: {detector.task_window[0]}s to {detector.task_window[1]}s (detection window)")
    print(f"  - Artifact threshold: {detector.artifact_threshold:.0f} µV")
    print(f"  - IMF power threshold: {detector.hht_power_threshold*100:.0f}% in 8-30 Hz")
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

    # Plot all processed trials
    print("[Plotting] Generating visualizations for processed trials...")
    for idx, r in enumerate(successful_trials):
        try:
            trial_data = extract_trial(data['data'], data['events'][idx], data['fs'])
            plot_results(
                trial_data=trial_data,
                channels=data['channels'],
                result=r,
                fs=data['fs'],
                trial_idx=idx
            )
        except Exception as e:
            print(f"  Warning: Could not generate plot for trial {idx+1}: {e}")

    print()
    print("=" * 70)
    print("Test completed successfully!")
    print("=" * 70)
    print()

    return results


if __name__ == '__main__':
    results = main()
