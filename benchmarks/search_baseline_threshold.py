#!/usr/bin/env python3
"""
Busca eficiente de baseline/limiar usando TODOS os trials (400) sem recalcular HHT a cada combo.
Pré-processa cada trial uma única vez (bandpass+Laplacian+HHT) e aplica 10 combinações de:
 - baseline: 3 janelas ([-2,-1], [-1.5,-0.5], [-1,0])
 - limiar sigma: -1.0, -0.85, -0.7, -0.6 (quatro valores) -> total 10 combos selecionados
Usa peso temporal gaussiano padrão (center=0.6s, sigma=0.25s, floor=0.3), step 5 ms.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from erd_detector import detection, preprocessing, hht  # noqa: E402
from erd_detector.utils import load_openbmi_data, extract_trial  # noqa: E402

DATA_DIR = Path("data/openbmi_sample")


def preprocess_trial_once(trial_data, channels, fs):
    # Stage 1: preprocess
    pre = preprocessing.preprocess_trial(
        trial_data,
        channels,
        fs,
        motor_channels=["C3", "C4"],
        reference_channels=["O1", "O2", "Fz"],
        bandpass_params={"lowcut": 8.0, "highcut": 30.0, "order": 5},
        artifact_threshold=750.0,
        apply_laplacian=True,
    )
    motor_data = pre["motor_data"]
    is_clean = pre["is_clean"]
    artifact_max = pre["artifact_max"]

    # Stage 2: HHT per motor channel
    mpower = []
    imfs = []
    for ch in range(motor_data.shape[0]):
        res = hht.process_channel_hht(
            motor_data[ch, :],
            fs,
            freq_band=(8.0, 30.0),
            power_threshold=0.35,
        )
        mpower.append(res["power"])
        imfs.append(res.get("imfs", np.empty((0, motor_data.shape[1]))))
    mpower = np.array(mpower)

    return {
        "motor_power": mpower,
        "is_clean": is_clean,
        "artifact_max": artifact_max,
    }


def load_all_trials():
    files = sorted(DATA_DIR.glob("*.mat"))
    trials = []
    for f in files:
        name = f.stem
        sess = int(name[4:6])
        subj = int(name[11:13])
        data = load_openbmi_data(subject_id=subj, session=sess, data_type="train", data_dir=str(DATA_DIR))
        for idx, event_time in enumerate(data["events"]):
            trial = extract_trial(data["data"], event_time, data["fs"])
            proc = preprocess_trial_once(trial, data["channels"], data["fs"])
            trials.append(
                {
                    "subject": subj,
                    "session": sess,
                    "trial_idx": idx + 1,
                    "fs": data["fs"],
                    "motor_power": proc["motor_power"],
                    "is_clean": proc["is_clean"],
                    "artifact_max": proc["artifact_max"],
                }
            )
    return trials


def evaluate(trials, baseline_window, threshold_sigma):
    detected = 0
    total = len(trials)
    for t in trials:
        bw_mean, bw_std = detection.calculate_baseline(
            t["motor_power"],
            t["fs"],
            baseline_window=baseline_window,
            trial_start_time=-3.0,
        )
        res = detection.detect_erd_sliding_window(
            t["motor_power"],
            bw_mean,
            bw_std,
            t["fs"],
            window_size=0.2,
            step_size=0.005,
            threshold_sigma=threshold_sigma,
            min_channels=1,
            task_window=(0.0, 4.0),
            trial_start_time=-3.0,
            use_temporal_weight=True,
            penalty_center=0.6,
            penalty_sigma=0.25,
            penalty_floor=0.3,
        )
        if res["detected"]:
            detected += 1
    rate = detected / total * 100.0
    return rate, detected, total


def main():
    print("Pré-processando todos os 400 trials uma única vez...")
    trials = load_all_trials()
    print("OK. Iniciando avaliação das 10 combinações...\n")

    combos = [
        ((-2.0, -1.0), -1.0),
        ((-2.0, -1.0), -0.85),
        ((-2.0, -1.0), -0.70),
        ((-2.0, -1.0), -0.60),
        ((-1.5, -0.5), -1.0),
        ((-1.5, -0.5), -0.85),
        ((-1.5, -0.5), -0.70),
        ((-1.5, -0.5), -0.60),
        ((-1.0, 0.0), -0.85),
        ((-1.0, 0.0), -0.60),
    ]

    results = []
    for bw, thr in combos:
        rate, det, total = evaluate(trials, bw, thr)
        results.append((rate, det, total, bw, thr))
        print(f"Baseline {bw}, limiar {thr:+.2f}σ -> {det}/{total} ({rate:.1f}%)")

    best = max(results, key=lambda x: x[0])
    rate, det, total, bw, thr = best
    print("\nMELHOR COMBINAÇÃO:")
    print(f"Baseline {bw}, limiar {thr:+.2f}σ -> {det}/{total} ({rate:.1f}%)")


if __name__ == "__main__":
    main()
