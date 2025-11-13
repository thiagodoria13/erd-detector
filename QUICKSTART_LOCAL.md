# Quick Start Guide - Local Testing

This guide will help you test the ERD detector locally using synthetic data (no dataset download required).

## Prerequisites

- Python 3.10 or higher
- Windows, Linux, or macOS
- ~500 MB free disk space (for dependencies)

## Step-by-Step Instructions

### 1. Run Setup Script

**Windows (Command Prompt):**
```cmd
setup_local_test.bat
```

**Git Bash / Linux / Mac:**
```bash
bash setup_local_test.sh
```

This will:
- Create a virtual environment (`.venv/`)
- Install all required dependencies
- Take ~2-5 minutes depending on your internet connection

### 2. Activate Virtual Environment

**Windows CMD:**
```cmd
.venv\Scripts\activate
```

**Git Bash (Windows):**
```bash
source .venv/Scripts/activate
```

**Linux/Mac:**
```bash
source .venv/bin/activate
```

You should see `(.venv)` appear in your command prompt.

### 3. Run Test Script

```bash
python test_local.py
```

### 4. Expected Output

The script will display progress through 4 stages:

```
======================================================================
ERD Detector - Local Test with Synthetic Data
======================================================================

[1/4] Generating synthetic EEG data...
  - Duration: 7 seconds (trial from -3s to +4s)
  - Sampling rate: 1000 Hz
  - Simulated ERD: onset at +1.0s, duration 2.0s, strength 60%
  - Data shape: (7000, 62) (samples x channels)

[2/4] Initializing ERD detector...
  - Threshold: -2.0 sigma
  - Min channels: 2 (both C3 and C4)

[3/4] Processing trial through complete pipeline...
  Stage 1: Preprocessing (bandpass, Laplacian, artifacts)
  Stage 2: HHT analysis (EMD, IMF selection, Hilbert)
  Stage 3: Baseline calculation from reference channels
  Stage 4: ERD detection with sliding window

  ✓ Pipeline completed successfully!

[4/4] Results:
----------------------------------------------------------------------
  ✓ ERD DETECTED
    - Onset time: 1.xxx s (relative to cue)
    - Latency: 1.xxx s (xxx ms)
    - Detection sample: xxxx

  Preprocessing:
    - Artifact status: Clean
    - Max amplitude: xx.x µV

  Baseline statistics:
    - Mean power: xxx.xx
    - Std deviation: xxx.xx
----------------------------------------------------------------------

[Plotting] Generating visualization...
[Saved] test_results.png

======================================================================
Test completed successfully!
======================================================================
```

### 5. View Results

A file named `test_results.png` will be created in the current directory showing:
- Raw EEG signals from C3 and C4
- Z-score evolution over time
- Detection summary

## Troubleshooting

### Issue: "Python not found"
**Solution:** Install Python 3.10+ from [python.org](https://www.python.org/downloads/)

### Issue: "pip install failed"
**Solution:** Try updating pip first:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Issue: "EMD-signal installation failed"
**Solution:** This package requires numpy. Install numpy first:
```bash
pip install numpy
pip install EMD-signal
```

### Issue: "No module named 'erd_detector'"
**Solution:** Make sure you're in the project directory and virtual environment is activated:
```bash
cd /c/Users/Thiago\ Doria/Desktop/EEG/erd-detector
source .venv/Scripts/activate  # or appropriate activation command
```

### Issue: "No ERD detected" (unexpected)
This can happen due to random variations in synthetic data generation. The test uses random noise, so occasionally the simulated ERD might not be strong enough. Simply run the test again:
```bash
python test_local.py
```

### Issue: Plot window doesn't appear
**Solution:** The plot is saved to `test_results.png` even if the window doesn't display. Check for this file in your current directory.

## What Happens During Testing?

The test script:

1. **Generates synthetic EEG** (7 seconds, 62 channels)
   - Realistic mu (8-13 Hz) and beta (14-30 Hz) rhythms
   - Simulated ERD event in C3/C4 starting 1s after cue
   - Pink noise typical of EEG recordings

2. **Runs preprocessing**
   - Bandpass filter (8-30 Hz)
   - Surface Laplacian on motor channels
   - Artifact detection

3. **Applies HHT analysis**
   - Empirical Mode Decomposition (EMD)
   - IMF selection (60% power in 8-30 Hz band)
   - Hilbert transform for instantaneous power

4. **Detects ERD**
   - Calculates baseline from reference channels (O1, O2, Fz)
   - Sliding window analysis (200ms windows, 50ms steps)
   - Applies -2σ threshold criterion

5. **Displays results**
   - Console output with detailed metrics
   - Visualization plot saved as PNG

## Next Steps

Once local testing works:

1. **Download OpenBMI dataset** (requires ~209 GB storage)
   - Use cloud VM recommended (see README.md)

2. **Run batch processing** (to be implemented)
   - Process all 54 subjects
   - Generate performance metrics
   - Create result visualizations

3. **Analyze results**
   - Compare with thesis targets (≥80% accuracy, ≤200ms latency)
   - Generate ROC curves and statistical analysis

## Dependencies Installed

The setup script installs:
- `numpy` - Numerical computing
- `scipy` - Signal processing
- `pandas` - Data manipulation
- `mne` - EEG processing utilities
- `EMD-signal` - Empirical Mode Decomposition
- `scikit-learn` - Machine learning metrics
- `matplotlib` - Plotting
- `seaborn` - Statistical visualization
- `tqdm` - Progress bars
- `joblib` - Parallel processing
- `psutil` - System utilities

Total installation size: ~400-500 MB

## Deactivating Virtual Environment

When done testing:
```bash
deactivate
```

## Need Help?

- Check `README.md` for detailed documentation
- Review error messages carefully
- Ensure Python 3.10+ is installed
- Make sure virtual environment is activated
