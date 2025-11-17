#!/usr/bin/env python3
"""
Gera 20 figuras de ensaios "médios" (latência mais próxima da mediana)
com detecção de ERD, usando os mesmos parâmetros do relatório local.

Saídas:
  - results/trial_figures/trial_subjXX_sessYY_idxZZ.png
Requisitos:
  - results/local_report_trials.csv (criado por run_full_report.py)
  - dados em data/openbmi_sample/
"""

import csv
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from erd_detector import ERDDetector  # noqa: E402
from erd_detector import preprocessing  # noqa: E402
from erd_detector.utils import load_openbmi_data, extract_trial  # noqa: E402


DATA_DIR = Path("data/openbmi_sample")
CSV_PATH = Path("results/local_report_trials.csv")
FIG_DIR = Path("results/trial_figures")


def load_rows():
    rows = []
    with CSV_PATH.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            r["detected"] = r["detected"].lower() in ("true", "1", "yes")
            r["latency_s"] = float(r["latency_s"]) if r["latency_s"] else float("nan")
            r["is_clean"] = r["is_clean"].lower() in ("true", "1", "yes")
            r["subject"] = int(r["subject"])
            r["session"] = int(r["session"])
            r["trial"] = int(r["trial"])
            rows.append(r)
    return rows


def pick_median_trials(rows, n_pick=20):
    detected = [r for r in rows if r["detected"] and not np.isnan(r["latency_s"])]
    if not detected:
        return []
    latencies = np.array([r["latency_s"] for r in detected])
    med = np.median(latencies)
    # Ordena por proximidade da mediana (absoluto)
    detected.sort(key=lambda r: abs(r["latency_s"] - med))
    return detected[:n_pick]


def plot_trial(trial_data, channels, result, trial_idx, out_path, detector):
    """
    Gera 8 gráficos:
      1) EEG bruto C3/C4
      2) EEG filtrado C3/C4
      3) IMF principal C3
      4) IMF principal C4
      5) Z-score C3 (baseline 0, limiar)
      6) Z-score C4 (baseline 0, limiar)
      7) Potência HHT C3 (baseline/limiar)
      8) Potência HHT C4 (baseline/limiar)
    Sempre com eixo em segundos, linha do cue, janela de baseline e detecção.
    """
    fig, axes = plt.subplots(8, 1, figsize=(12, 18), sharex=False)

    fs = result.get("fs", 1000)
    trial_start_time = result.get("trial_start_time", -3.0)
    baseline_window = result.get("baseline_window", (-1.0, 0.0))
    baseline_start, baseline_end = baseline_window
    end_time = trial_start_time + trial_data.shape[0] / fs
    t = np.arange(trial_data.shape[0]) / fs + trial_start_time

    # índices de canais
    try:
        c3_idx = channels.index("C3")
        c4_idx = channels.index("C4")
    except ValueError:
        c3_idx, c4_idx = 0, 1

    # Pré-processamento para pegar filtrado
    pre = preprocessing.preprocess_trial(
        trial_data,
        channels,
        fs,
        motor_channels=detector.motor_channels,
        reference_channels=detector.reference_channels,
        bandpass_params=detector.bandpass_params,
        artifact_threshold=detector.artifact_threshold,
        apply_laplacian=True,
    )
    motor_filt = pre["motor_data"]  # shape (n_motor, samples)

    # Conveniências
    mpower = result.get("motor_power", None)
    imfs = result.get("motor_imfs", [])
    z_scores = result.get("z_scores", None)
    window_times = result.get("window_times", None)
    bmean = np.asarray(result.get("baseline_mean", [])) if result.get("baseline_mean", None) is not None else None
    bstd = np.asarray(result.get("baseline_std", [])) if result.get("baseline_std", None) is not None else None
    thr_sigma = result.get("threshold_sigma", detector.threshold_sigma)
    onset = result.get("onset_time")

    # Helper para marcar janelas
    def add_spans(ax):
        ax.axvspan(baseline_start, baseline_end, color="gray", alpha=0.12)
        ax.axvspan(0.0, 4.0, color="yellow", alpha=0.12)
        ax.axvline(0, color="k", linestyle="--", alpha=0.8)
        if result.get("detected") and onset is not None:
            ax.axvline(onset, color="g", linestyle="-", linewidth=1.6)
        ax.grid(True, alpha=0.25)
        tick_vals = np.arange(np.floor(trial_start_time), np.ceil(end_time) + 1, 1.0)
        ax.set_xticks(tick_vals)
        ax.set_xlabel("Tempo (s) relativo ao cue")

    # 1) EEG bruto C3/C4
    axes[0].plot(t, trial_data[:, c3_idx], label="C3", linewidth=0.8)
    axes[0].plot(t, trial_data[:, c4_idx], label="C4", linewidth=0.8)
    axes[0].set_ylabel("µV")
    axes[0].set_title(f"Ensaio {trial_idx}: EEG bruto")
    add_spans(axes[0])
    axes[0].legend(loc="upper right", fontsize=8)

    # 2) EEG filtrado (bandpass + Laplacian)
    axes[1].plot(t, motor_filt[0, :], label="C3 filtrado", linewidth=0.8)
    if motor_filt.shape[0] > 1:
        axes[1].plot(t, motor_filt[1, :], label="C4 filtrado", linewidth=0.8)
    axes[1].set_ylabel("µV")
    axes[1].set_title("EEG filtrado (8–30 Hz + Laplacian)")
    add_spans(axes[1])
    axes[1].legend(loc="upper right", fontsize=8)

    # 3) IMF principal C3
    if imfs and len(imfs) > 0 and imfs[0].size > 0:
        axes[2].plot(t, imfs[0][0, :], color="C0", linewidth=0.7)
        axes[2].set_title("IMF principal C3")
    axes[2].set_ylabel("µV")
    add_spans(axes[2])

    # 4) IMF principal C4
    if imfs and len(imfs) > 1 and imfs[1].size > 0:
        axes[3].plot(t, imfs[1][0, :], color="C1", linewidth=0.7)
        axes[3].set_title("IMF principal C4")
    axes[3].set_ylabel("µV")
    add_spans(axes[3])

    # 5) Z-score C3
    if z_scores is not None and window_times is not None and z_scores.shape[0] >= 1:
        axes[4].plot(window_times, z_scores[0, :], color="C0", linewidth=0.9, label="Z-score C3")
        axes[4].axhline(0, color="k", linestyle="--", alpha=0.6, label="Baseline (0)")
        axes[4].axhline(thr_sigma, color="r", linestyle="--", alpha=0.8, label="Limiar")
        # Gaussian weight overlay (0-1)
        if "window_weights" in result:
            ax2 = axes[4].twinx()
            ax2.fill_between(window_times, result["window_weights"], color="gray", alpha=0.15, step="mid")
            ax2.set_ylim(0, 1.05)
            ax2.set_yticks([0, 0.5, 1.0])
            ax2.set_ylabel("Peso temporal (gaussiana)", fontsize=8)
        axes[4].set_ylabel("Z (C3)")
        axes[4].set_title("Z-score C3 (baseline 0, limiar vermelho)")
        add_spans(axes[4])
        axes[4].legend(loc="upper right", fontsize=8)

    # 6) Z-score C4
    if z_scores is not None and window_times is not None and z_scores.shape[0] >= 2:
        axes[5].plot(window_times, z_scores[1, :], color="C1", linewidth=0.9, label="Z-score C4")
        axes[5].axhline(0, color="k", linestyle="--", alpha=0.6, label="Baseline (0)")
        axes[5].axhline(thr_sigma, color="r", linestyle="--", alpha=0.8, label="Limiar")
        # Gaussian weight overlay
        if "window_weights" in result:
            ax2b = axes[5].twinx()
            ax2b.fill_between(window_times, result["window_weights"], color="gray", alpha=0.15, step="mid")
            ax2b.set_ylim(0, 1.05)
            ax2b.set_yticks([0, 0.5, 1.0])
            ax2b.set_ylabel("Peso temporal (gaussiana)", fontsize=8)
        axes[5].set_ylabel("Z (C4)")
        axes[5].set_title("Z-score C4 (baseline 0, limiar vermelho)")
        add_spans(axes[5])
        axes[5].legend(loc="upper right", fontsize=8)

    # 7) Potência HHT C3
    if mpower is not None and mpower.shape[0] >= 1:
        axes[6].plot(t, mpower[0, :], color="C0", linewidth=0.8, label="Potência C3")
        if bmean is not None and bstd is not None and bmean.size > 0:
            axes[6].axhline(bmean[0], color="k", linestyle="--", alpha=0.7, label="Baseline")
            axes[6].axhline(bmean[0] + thr_sigma * bstd[0], color="r", linestyle="--", alpha=0.7, label="Limiar")
        axes[6].set_ylabel("Power (µV²)")
        axes[6].set_title("Potência HHT C3 (baseline/limiar)")
        add_spans(axes[6])
        axes[6].legend(loc="upper right", fontsize=8)

    # 8) Potência HHT C4
    if mpower is not None and mpower.shape[0] >= 2:
        axes[7].plot(t, mpower[1, :], color="C1", linewidth=0.8, label="Potência C4")
        if bmean is not None and bstd is not None and bmean.size > 1:
            axes[7].axhline(bmean[1], color="k", linestyle="--", alpha=0.7, label="Baseline")
            axes[7].axhline(bmean[1] + thr_sigma * bstd[1], color="r", linestyle="--", alpha=0.7, label="Limiar")
        axes[7].set_ylabel("Power (µV²)")
        axes[7].set_title("Potência HHT C4 (baseline/limiar)")
        add_spans(axes[7])
        axes[7].legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    rows = load_rows()
    picked = pick_median_trials(rows, n_pick=20)
    if not picked:
        print("Nenhum ensaio detectado para gerar figuras.")
        return

    # Configura detector igual ao relatório
    detector = ERDDetector(
        motor_channels=["C3", "C4"],
        reference_channels=["O1", "O2", "Fz"],
        threshold_sigma=-0.75,
        min_channels=1,
        baseline_window=(-1.0, 0.0),
        task_window=(0.0, 4.0),
        trial_start_time=-3.0,
        artifact_threshold=750.0,
        hht_power_threshold=0.35,
    )

    # Carregar dados por arquivo apenas uma vez
    cache = {}
    for r in picked:
        key = r["file"]
        if key not in cache:
            subject = r["subject"]
            session = r["session"]
            data = load_openbmi_data(subject_id=subject, session=session, data_type="train", data_dir=str(DATA_DIR))
            cache[key] = data
        data = cache[key]
        event_idx = r["trial"] - 1
        trial_data = extract_trial(data["data"], data["events"][event_idx], data["fs"])
        result = detector.process_trial(
            trial_data=trial_data,
            channels=data["channels"],
            fs=data["fs"],
            trial_start_time=-3.0,
        )
        result["min_channels"] = detector.min_channels
        result["threshold_sigma"] = detector.threshold_sigma
        result["fs"] = data["fs"]
        out_name = f"trial_subj{r['subject']:02d}_sess{r['session']:02d}_idx{r['trial']:03d}.png"
        out_path = FIG_DIR / out_name
        plot_trial(
            trial_data,
            data["channels"],
            result,
            r["trial"],
            out_path,
            detector,
        )
        print(f"Figura salva: {out_path}")


if __name__ == "__main__":
    main()
