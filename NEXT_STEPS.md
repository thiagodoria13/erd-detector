# Next Steps - ERD Detector Implementation

## ✅ What's Been Completed

### Foundation (100%)
1. **Project Structure** - Complete directory layout created
2. **Git Repository** - Initialized with proper .gitignore
3. **Documentation**:
   - README.md - Comprehensive project overview
   - CLOUD_SETUP.md - Complete GCP deployment guide
   - requirements.txt - All dependencies
   - setup.py - Package configuration
4. **Utils Module** - FULLY IMPLEMENTED with:
   - `load_openbmi_data()` - Load OpenBMI .mat files
   - `extract_trial()` - Extract trial windows
   - `get_channel_indices()` - Channel name lookup
   - `create_mne_raw()` - Convert to MNE format

### Module Skeletons Created
- preprocessing.py
- hht.py
- detection.py
- metrics.py
- visualization.py

---

## 🔨 What Needs To Be Done

### Priority 1: Core Algorithm Modules

#### 1. preprocessing.py
Implement 4 functions with line-by-line comments:
- `bandpass_filter()` - 8-30 Hz Butterworth filter
- `laplacian_filter()` - Surface Laplacian for C3/C4
- `reject_artifacts()` - Amplitude thresholding
- `preprocess_trial()` - Complete preprocessing pipeline

**Key Details:**
- Use scipy.signal.butter + sosfiltfilt
- C3_lap = C3 - mean(FC1, FC5, CP1, CP5)
- C4_lap = C4 - mean(FC2, FC6, CP2, CP6)
- Threshold: ±100µV

#### 2. hht.py
Implement 5 functions with full mathematical documentation:
- `empirical_mode_decomposition()` - Use PyEMD library
- `select_imfs_spectral()` - Select IMFs with 60% power in 8-30 Hz
- `hilbert_transform()` - Compute instantaneous amplitude
- `instantaneous_power()` - Power = amplitude²
- `process_channel_hht()` - Complete HHT pipeline

**Key Details:**
- from PyEMD import EMD
- from scipy.signal import hilbert
- Spectral selection: peak freq in [8,30] Hz AND ≥60% power in band
- Document mathematical formulas in LaTeX

#### 3. detection.py
Implement ERDDetector class and 2 functions:
- `calculate_baseline()` - μ and σ from reference channels [-3s, -1s]
- `detect_erd_sliding_window()` - Sliding window with -2σ threshold
- `ERDDetector` class - Complete pipeline

**Key Details:**
- Window: 200ms, step: 50ms
- Detection: z = (P - μ) / σ ≤ -2.0
- Confirmation: ≥2 motor channels (C3 AND C4)
- Return: onset_time, latency, z_scores

---

### Priority 2: Batch Processing Scripts

#### scripts/download_data.py
Download OpenBMI from GigaDB:
- URL: http://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100542/
- Files: 54 subjects × 2 sessions × 2 files = 216 files
- Size: ~209 GB
- Features: Resume support, checksum verification, progress bars

#### scripts/process_all.py
Batch process all subjects:
- Stratified split: 43 calibration, 11 validation (seed=42)
- Parallel processing with joblib
- Save: calibration_results.csv, validation_results.csv
- Progress tracking with tqdm

#### scripts/analyze.py
Compute metrics and statistics:
- Detection rate per subject
- Mean latency, std, percentiles
- Compare with CSP+LDA (71.1%)
- Statistical significance tests
- Parameter adjustment decision

#### scripts/visualize.py
Generate all figures:
- Time-frequency analysis (motor vs reference)
- ROC curves
- Latency distributions
- Per-subject comparisons

---

### Priority 3: Metrics & Visualization

#### metrics.py
- `compute_binary_accuracy()` - Detection rate
- `compute_latency_statistics()` - Latency metrics
- `compute_per_subject_metrics()` - Identify BCI illiteracy

#### visualization.py
- `plot_time_frequency_analysis()` - Motor vs reference power
- `plot_roc_curve()` - ROC with AUC
- `plot_latency_distribution()` - Histogram + box plot
- `plot_per_subject_comparison()` - Bar chart

---

## 🚀 Recommended Workflow

### Week 1: Local Development
**Days 1-2:** Implement preprocessing.py + hht.py
**Day 3:** Implement detection.py
**Day 4:** Test on sample data (1-2 subjects)

### Week 2: Cloud Processing
**Day 1:** Set up GCP VM + download dataset
**Days 2-3:** Create and run batch processing scripts
**Day 4:** Analyze results

### Week 3: Finalization
**Days 1-2:** Create all visualizations
**Day 3:** Write technical report
**Day 4:** Final review and documentation

---

## 📋 Detailed Implementation Guide

### Step 1: Implement Preprocessing (Start Here!)

```bash
cd "/c/Users/Thiago Doria/Desktop/EEG/erd-detector"
# Edit: erd_detector/preprocessing.py
```

Use this template structure:

```python
import numpy as np
from scipy.signal import butter, sosfiltfilt
from .utils import get_channel_indices

def bandpass_filter(data, fs, lowcut=8.0, highcut=30.0, order=5):
    """
    Apply bandpass filter to isolate µ (8-13 Hz) and β (14-30 Hz) rhythms.
    
    [Add detailed docstring explaining:
     - Why 8-30 Hz range
     - Why Butterworth filter
     - Mathematical transfer function
     - Zero-phase filtering explanation]
    
    Args:
        data (ndarray): EEG data (samples,) or (samples, channels)
        fs (int): Sampling rate in Hz
        lowcut (float): Low cutoff (default 8.0 Hz)
        highcut (float): High cutoff (default 30.0 Hz)
        order (int): Filter order (default 5)
    
    Returns:
        ndarray: Filtered data, same shape as input
    """
    # Step 1: Design Butterworth filter in SOS format
    # SOS (Second-Order Sections) provides numerical stability
    sos = butter(
        N=order,                    # Filter order
        Wn=[lowcut, highcut],      # Passband edges
        btype='bandpass',          # Bandpass filter
        analog=False,              # Digital filter
        fs=fs,                     # Sampling frequency
        output='sos'               # SOS format
    )
    
    # Step 2: Apply forward-backward filter (zero phase)
    # sosfiltfilt applies filter forwards then backwards
    # This doubles effective order and eliminates phase distortion
    if data.ndim == 1:
        # Single channel
        filtered = sosfiltfilt(sos, data, axis=0)
    else:
        # Multi-channel (samples × channels)
        filtered = sosfiltfilt(sos, data, axis=0)
    
    return filtered
```

---

## 📊 Success Criteria

### Module Completion
- [ ] All functions have comprehensive docstrings
- [ ] Every line has explanatory comments
- [ ] Mathematical formulas in LaTeX format
- [ ] Unit tests pass (when created)

### Algorithm Performance
- [ ] Detection rate ≥80%
- [ ] Mean latency ≤200ms
- [ ] Outperforms CSP+LDA (71.1%)

### Documentation
- [ ] README updated with results
- [ ] Technical report completed
- [ ] All figures generated
- [ ] Code fully commented

---

## 💻 Quick Commands

```bash
# Navigate to project
cd "/c/Users/Thiago Doria/Desktop/EEG/erd-detector"

# Create virtual environment (when ready to test)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .

# Test imports (after implementation)
python -c "from erd_detector import load_openbmi_data; print('✓ Utils working')"
```

---

## 📝 Notes

- **Start with preprocessing.py** - It's the foundation
- **Test incrementally** - Implement one function, test it, move on
- **Use the detailed plan** - All implementation details are in our conversation history
- **Reference the thesis** - Your PDF has the neurophysiological justifications
- **Ask for help** - If anything is unclear, reference the detailed plan we created

---

**Created**: 2025-01-13
**Last Updated**: 2025-01-13
**Current Location**: `/c/Users/Thiago Doria/Desktop/EEG/erd-detector/`
