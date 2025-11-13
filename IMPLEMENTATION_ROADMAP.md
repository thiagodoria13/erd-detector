# Implementation Roadmap

## Codex Review - All Issues Fixed ✅

1. ✅ Fixed channel parsing bug in utils.py (recursive array extraction)
2. ✅ Added MIT LICENSE file
3. ✅ Updated README to reflect actual status (no overpromising)
4. ✅ Committed all pending files
5. ✅ Documented that core modules need implementation

## Current Status

**Working**: utils.py (data loading, fully functional)
**Next**: Implement core algorithm modules

## Implementation Priority

### 1. preprocessing.py (START HERE)
- bandpass_filter() - 8-30 Hz Butterworth
- laplacian_filter() - Surface Laplacian for C3/C4  
- reject_artifacts() - Amplitude thresholding
- preprocess_trial() - Complete pipeline

### 2. hht.py
- empirical_mode_decomposition() - PyEMD
- select_imfs_spectral() - 60% power in 8-30 Hz
- hilbert_transform() - Instantaneous amplitude
- instantaneous_power() - Power = amplitude²
- process_channel_hht() - Complete HHT pipeline

### 3. detection.py
- calculate_baseline() - μ and σ from reference channels
- detect_erd_sliding_window() - -2σ threshold detection
- ERDDetector class - Complete detection pipeline

### 4. Scripts (after core modules work)
- download_data.py
- process_all.py  
- analyze.py
- visualize.py

### 5. Supporting (parallel with scripts)
- metrics.py
- visualization.py
- tests/

## Quick Start

All detailed implementation code is in our conversation history.
Start with preprocessing.py using the template provided in NEXT_STEPS.md.

Current directory: /c/Users/Thiago Doria/Desktop/EEG/erd-detector
