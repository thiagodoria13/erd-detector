"""
ERD Detector: Single-trial Event-Related Desynchronization Detection

This package implements a Hilbert-Huang Transform (HHT) based approach for
detecting ERD in EEG signals without machine learning calibration.

Core Components:
- preprocessing: Signal filtering and artifact control
- hht: Empirical Mode Decomposition and Hilbert Transform
- detection: ERD detection using -2σ threshold
- metrics: Performance evaluation (accuracy, latency, ROC)
- visualization: ERD topographic maps and time-frequency plots

Citation:
If you use this package, please cite:
Fonseca, L.P. & Doria, T.A.S. (2025). Processamento de sinais de EEG para
detecção de ERD em baixa latência para controle de interfaces cérebro-computador (BCI).
Universidade de São Paulo.

And cite the OpenBMI dataset:
Lee, M.H., et al. (2019). EEG dataset and OpenBMI toolbox for three BCI
paradigms: an investigation into BCI illiteracy. GigaScience, 8(5).
"""

__version__ = "1.0.0"
__author__ = "Lucas Pereira da Fonseca, Thiago Anversa Sampaio Doria"

# Import implemented modules
from . import preprocessing
from . import hht
from . import detection

# Import main class for convenience
from .detection import ERDDetector

__all__ = ['preprocessing', 'hht', 'detection', 'ERDDetector']
