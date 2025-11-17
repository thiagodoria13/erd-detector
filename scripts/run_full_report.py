#!/usr/bin/env python3
"""
Executa a detecção em todos os arquivos locais do OpenBMI sample
e gera um relatório em LaTeX com métricas agregadas.

Saídas geradas:
  - results/local_report_trials.csv   (todas as tentativas)
  - results/local_report_summary.txt  (resumo rápido)
  - results/local_latency_hist.png    (histograma de latências)
  - results/local_detection_bar.png   (taxa de detecção por sujeito-sessão)
  - docs/local_report.tex             (relatório em LaTeX, português)
"""

import re
import sys
import csv
from pathlib import Path
from statistics import mean, median, pstdev
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from erd_detector import ERDDetector
from erd_detector.utils import load_openbmi_data, extract_trial  # noqa: E402


DATA_DIR = Path("data/openbmi_sample")
RESULTS_DIR = Path("results")
DOCS_DIR = Path("docs")


def parse_fname(fname: str):
    """
    Extrai subject/session do padrão sessXX_subjYY_EEG_MI.mat.
    """
    m = re.search(r"sess(\d+)_subj(\d+)_EEG_MI", fname)
    if not m:
        raise ValueError(f"Nome de arquivo inesperado: {fname}")
    session = int(m.group(1))
    subject = int(m.group(2))
    return subject, session


def ensure_dirs():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)


def collect_results(detector: ERDDetector):
    rows = []
    file_list = sorted(DATA_DIR.glob("*.mat"))
    if not file_list:
        raise FileNotFoundError(f"Nenhum .mat encontrado em {DATA_DIR}")

    for mat_file in file_list:
        subject, session = parse_fname(mat_file.name)
        print(f"[Arquivo] {mat_file.name} (Subj {subject}, Sess {session})")
        data = load_openbmi_data(subject_id=subject, session=session, data_type="train", data_dir=str(DATA_DIR))

        for idx, event_time in enumerate(data["events"]):
            trial_data = extract_trial(data["data"], event_time, data["fs"])
            result = detector.process_trial(
                trial_data=trial_data,
                channels=data["channels"],
                fs=data["fs"],
                trial_start_time=-3.0,
            )

            rows.append(
                {
                    "file": mat_file.name,
                    "subject": subject,
                    "session": session,
                    "trial": idx + 1,
                    "detected": bool(result["detected"]),
                    "latency_s": result["latency"] if result["latency"] is not None else np.nan,
                    "is_clean": bool(result["is_clean"]),
                    "artifact_max_uV": result["artifact_max"],
                }
            )
        print(f"  -> {len(data['events'])} ensaios processados\n")
    return rows


def save_csv(rows):
    out_csv = RESULTS_DIR / "local_report_trials.csv"
    fieldnames = ["file", "subject", "session", "trial", "detected", "latency_s", "is_clean", "artifact_max_uV"]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return out_csv


def compute_stats(rows):
    total = len(rows)
    detected_rows = [r for r in rows if r["detected"]]
    clean_rows = [r for r in rows if r["is_clean"]]
    detected_rate = len(detected_rows) / total * 100 if total else 0.0

    latencies = [r["latency_s"] for r in detected_rows if not np.isnan(r["latency_s"])]
    stats = {
        "total_trials": total,
        "detected": len(detected_rows),
        "detected_rate": detected_rate,
        "clean_trials": len(clean_rows),
        "lat_mean": mean(latencies) if latencies else np.nan,
        "lat_median": median(latencies) if latencies else np.nan,
        "lat_std": pstdev(latencies) if len(latencies) > 1 else 0.0,
        "lat_min": min(latencies) if latencies else np.nan,
        "lat_max": max(latencies) if latencies else np.nan,
    }
    return stats, detected_rows


def plot_latency(latencies):
    out_png = RESULTS_DIR / "local_latency_hist.png"
    if not latencies:
        return None
    plt.figure(figsize=(6, 4))
    plt.hist(np.array(latencies) * 1000, bins=20, color="skyblue", edgecolor="black")
    plt.xlabel("Latência (ms)")
    plt.ylabel("Frequência")
    plt.title("Distribuição de latências (todas as detecções)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    return out_png


def plot_detection_rate(rows):
    out_png = RESULTS_DIR / "local_detection_bar.png"
    by_group = {}
    for r in rows:
        key = f"S{r['subject']:02d}-sess{r['session']:02d}"
        by_group.setdefault(key, {"detected": 0, "total": 0})
        by_group[key]["total"] += 1
        by_group[key]["detected"] += int(r["detected"])

    labels = sorted(by_group.keys())
    rates = [by_group[k]["detected"] / by_group[k]["total"] * 100 for k in labels]

    plt.figure(figsize=(7, 4))
    plt.bar(labels, rates, color="mediumseagreen", edgecolor="black")
    plt.xticks(rotation=45)
    plt.ylabel("Taxa de detecção (%)")
    plt.title("Detecção por sujeito-sessão")
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    return out_png


def write_summary_txt(stats, summary_path):
    lines = [
        "Resumo rápido (todas as tentativas locais)",
        f"Total de tentativas: {stats['total_trials']}",
        f"Detecções: {stats['detected']} ({stats['detected_rate']:.1f}%)",
        f"Ensaios limpos: {stats['clean_trials']}",
        f"Latência média: {stats['lat_mean']*1000:.0f} ms",
        f"Latência mediana: {stats['lat_median']*1000:.0f} ms",
        f"Latência desvio-padrão: {stats['lat_std']*1000:.0f} ms",
        f"Latência mín-máx: {stats['lat_min']*1000:.0f} – {stats['lat_max']*1000:.0f} ms",
    ]
    summary_path.write_text("\n".join(lines), encoding="utf-8")


def write_latex(stats, lat_png, det_png):
    tex_path = DOCS_DIR / "local_report.tex"
    # Tentar incluir até 2 exemplos de figuras de ensaio, se existirem
    trial_figs = sorted((RESULTS_DIR / "trial_figures").glob("*.png"))
    ex1 = trial_figs[0].name if len(trial_figs) > 0 else ""
    ex2 = trial_figs[1].name if len(trial_figs) > 1 else ""

    tex = r"""\documentclass[12pt]{article}
\usepackage[utf8]{inputenc}
\usepackage[brazil]{babel}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{geometry}
\geometry{margin=2cm}
\title{Relatório Local de Detecção de ERD (OpenBMI Sample)}
\author{Pipeline ERD Detector}
\date{\today}
\begin{document}
\maketitle
\section{Configuração}
\begin{itemize}
  \item Baseline: $[-1.0, 0.0]$ s (auto-referência C3/C4)
  \item Janela de detecção: $[0, 4]$ s
  \item Limiar: $-0{,}75\,\sigma$
  \item Mínimo de canais: 1 (C3 ou C4)
  \item Artefatos: limiar 750 $\mu$V
  \item Seleção de IMF: potência $\geq 35$ por cento em 8--30 Hz
\end{itemize}
\section{Resultados Globais}
\begin{tabular}{ll}
\toprule
Total de tentativas & %(total)d \\
Detecções & %(det)d ( %(rate).1f\text{ por cento} ) \\
Ensaios limpos & %(clean)d \\
Latência média & %(latmean).0f ms \\
Latência mediana & %(latmed).0f ms \\
Latência desvio-padrão & %(latstd).0f ms \\
Latência mín--máx & %(latmin).0f -- %(latmax).0f ms \\
\bottomrule
\end{tabular}
\section{Figuras}
\subsection{Distribuição de latências}
\begin{center}
\includegraphics[width=0.9\linewidth]{%(lat_png)s}
\end{center}
\subsection{Taxa de detecção por sujeito-sessão}
\begin{center}
\includegraphics[width=0.9\linewidth]{%(det_png)s}
\end{center}
\section{Interpretação Rápida}
\subsection{Exemplos de detecção em ensaios reais}
\begin{center}
\includegraphics[width=0.9\linewidth]{%(ex1)s}\\[6pt]
\includegraphics[width=0.9\linewidth]{%(ex2)s}
\end{center}
\medskip
Cada figura traz: EEG bruto e filtrado (C3/C4), principal IMF por canal, Z-score por canal com baseline/limiar, e potência HHT por canal com baseline/limiar. Faixas cinza = baseline; amarelo = janela de detecção; linha tracejada preta = cue (0 s); linha verde = detecção.
\section{Interpretação Rápida}
\begin{itemize}
  \item A taxa de detecção reflete a configuração sensível (1 canal, -0{,}75$\sigma$), favorecendo alta sensibilidade.
  \item A proximidade do baseline ([-1,0] s) reduz a super-detecção precoce, mas ainda captura quedas de potência pós-cue.
  \item Ensaios com artefatos (picos $>$ 750 $\mu$V) permanecem contabilizados; considere filtragem adicional para análise final.
\end{itemize}
\end{document}
"""
    tex_filled = tex % {
        "total": stats["total_trials"],
        "det": stats["detected"],
        "rate": stats["detected_rate"],
        "clean": stats["clean_trials"],
        "latmean": stats["lat_mean"] * 1000 if stats["lat_mean"] is not None else 0,
        "latmed": stats["lat_median"] * 1000 if stats["lat_median"] is not None else 0,
        "latstd": stats["lat_std"] * 1000 if stats["lat_std"] is not None else 0,
        "latmin": stats["lat_min"] * 1000 if stats["lat_min"] is not None else 0,
        "latmax": stats["lat_max"] * 1000 if stats["lat_max"] is not None else 0,
        "lat_png": lat_png.name if lat_png else "",
        "det_png": det_png.name if det_png else "",
        "ex1": ex1,
        "ex2": ex2,
    }
    tex_path.write_text(tex_filled, encoding="utf-8")
    return tex_path


def main():
    ensure_dirs()

    detector = ERDDetector(
        motor_channels=["C3", "C4"],
        reference_channels=["O1", "O2", "Fz"],
        threshold_sigma=-0.60,
        min_channels=1,
        baseline_window=(-2.0, -1.0),
        task_window=(0.0, 4.0),
        trial_start_time=-3.0,
        artifact_threshold=750.0,
        hht_power_threshold=0.35,
    )

    print("Iniciando processamento completo...\n")
    rows = collect_results(detector)

    out_csv = save_csv(rows)
    stats, detected_rows = compute_stats(rows)
    latencies = [r["latency_s"] for r in detected_rows if not np.isnan(r["latency_s"])]
    lat_png = plot_latency(latencies)
    det_png = plot_detection_rate(rows)

    summary_path = RESULTS_DIR / "local_report_summary.txt"
    write_summary_txt(stats, summary_path)

    tex_path = write_latex(stats, lat_png, det_png)

    print("Resumo salvo em:", summary_path)
    print("CSV salvo em:", out_csv)
    if lat_png:
        print("Histograma salvo em:", lat_png)
    if det_png:
        print("Detecção por sujeito-sessão salva em:", det_png)
    print("Relatório LaTeX salvo em:", tex_path)
    print("\nPara compilar (se tiver LaTeX instalado):")
    print(f"  cd docs && pdflatex {tex_path.name}")


if __name__ == "__main__":
    main()
