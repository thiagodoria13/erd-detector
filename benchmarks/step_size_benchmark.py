#!/usr/bin/env python3
"""
Benchmark de tempo para step_size do sliding window (50 ms vs 5 ms).

Carrega o OpenBMI sample (subj1 sess1, train), seleciona 10 ensaios aleatórios
e mede o tempo médio por ensaio para step_size=0.05 e step_size=0.005.
"""

import random
import time
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from erd_detector import ERDDetector  # noqa: E402
from erd_detector.utils import load_openbmi_data, extract_trial  # noqa: E402


def make_detector(step_size: float):
    return ERDDetector(
        motor_channels=["C3", "C4"],
        reference_channels=["O1", "O2", "Fz"],
        threshold_sigma=-0.75,
        min_channels=1,
        baseline_window=(-1.0, 0.0),
        task_window=(0.0, 4.0),
        detection_window_size=0.2,
        detection_step_size=step_size,
        trial_start_time=-3.0,
        artifact_threshold=750.0,
        hht_power_threshold=0.35,
    )


def run_trials(detector, data, trial_indices):
    t0 = time.perf_counter()
    results = []
    for idx in trial_indices:
        trial_data = extract_trial(data["data"], data["events"][idx], data["fs"])
        r = detector.process_trial(
            trial_data=trial_data,
            channels=data["channels"],
            fs=data["fs"],
            trial_start_time=-3.0,
        )
        results.append(r)
    elapsed = time.perf_counter() - t0
    return elapsed, results


def main():
    random.seed(42)
    data_dir = Path("data/openbmi_sample")
    data = load_openbmi_data(subject_id=1, session=1, data_type="train", data_dir=str(data_dir))
    n_trials = len(data["events"])
    trial_indices = random.sample(range(n_trials), k=10)
    print(f"Usando 10 ensaios aleatórios de {n_trials} disponíveis: {trial_indices}")

    # Step 50 ms
    det50 = make_detector(0.05)
    t50, res50 = run_trials(det50, data, trial_indices)
    # Step 5 ms
    det5 = make_detector(0.005)
    t5, res5 = run_trials(det5, data, trial_indices)

    print("\n=== Benchmark step_size ===")
    print(f"50 ms: {t50:.3f} s para 10 ensaios -> {t50/10:.3f} s/ensaio")
    print(f"  Deteções: {sum(r['detected'] for r in res50)}/10")
    print(f"5 ms:  {t5:.3f} s para 10 ensaios -> {t5/10:.3f} s/ensaio")
    print(f"  Deteções: {sum(r['detected'] for r in res5)}/10")
    slowdown = t5 / t50 if t50 > 0 else float("inf")
    print(f"Fator de tempo (5ms vs 50ms): {slowdown:.2f}x")


if __name__ == "__main__":
    main()
