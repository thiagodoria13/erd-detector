# Quick Start Guide - Local Testing with Real Data

This guide will help you test the ERD detector locally using real OpenBMI EEG data (small sample, ~4-8 GB).

## Prerequisites

- Python 3.10 or higher
- Windows, Linux, or macOS
- ~10 GB free disk space (for dependencies + data)
- Internet connection for downloading data

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
- Install all required dependencies (~500 MB)
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

### 3. Download Sample OpenBMI Data

```bash
python download_sample_data.py
```

This script will display download instructions for the OpenBMI dataset.

**Option A: Manual Download (Recommended)**

1. Visit: https://gigadb.org/dataset/100542
2. Click the "Files" tab
3. Download these files:
   - `sess01_subj01_EEG_MI.mat` (~2 GB)
   - `sess02_subj01_EEG_MI.mat` (~2 GB)
4. Save to folder: `data/openbmi_sample/`

**Option B: Command Line (Linux/Mac)**

```bash
mkdir -p data/openbmi_sample
cd data/openbmi_sample
wget ftp://penguin.genomics.cn/pub/10.5524/100001_101000/100542/sess01_subj01_EEG_MI.mat
wget ftp://penguin.genomics.cn/pub/10.5524/100001_101000/100542/sess02_subj01_EEG_MI.mat
cd ../..
```

**Expected files:**
```
data/openbmi_sample/
├── sess01_subj01_EEG_MI.mat  (~2 GB)
└── sess02_subj01_EEG_MI.mat  (~2 GB)
```

### 4. Run Test Script

```bash
python test_local.py
```

### 5. Expected Output

The script will process 5 real EEG trials and display results:

```
======================================================================
ERD Detector - Local Test with Real OpenBMI Data
======================================================================

[1/5] Checking for OpenBMI data files...
  ✓ Found 2 data file(s):
    - sess01_subj01_EEG_MI.mat (2048.5 MB)
    - sess02_subj01_EEG_MI.mat (2048.5 MB)

[2/5] Loading EEG data...
  - Subject: 1
  - Session: 1
  - Data type: train
  ✓ Loaded 100 trials
  - Channels: 62 total
  - Sampling rate: 1000 Hz
  - Data shape: (7000, 62)

[3/5] Initializing ERD detector...
  - Threshold: -2.0 sigma
  - Min channels: 2 (both C3 and C4)
  - Baseline window: -3.0s to -1.0s
  - Task window: 0.0s to +4.0s

[4/5] Processing trials...
  Testing first 5 trials

  Trial 1/5:
    - Trial shape: (7000, 62)
    ✓ ERD detected at 1.234s (latency: 1234ms)
    - Artifact status: Clean

  Trial 2/5:
    - Trial shape: (7000, 62)
    ✗ No ERD detected
    - Artifact status: Clean

  Trial 3/5:
    - Trial shape: (7000, 62)
    ✓ ERD detected at 0.987s (latency: 987ms)
    - Artifact status: Clean

  Trial 4/5:
    - Trial shape: (7000, 62)
    ✓ ERD detected at 1.456s (latency: 1456ms)
    - Artifact status: Artifacts

  Trial 5/5:
    - Trial shape: (7000, 62)
    ✓ ERD detected at 1.123s (latency: 1123ms)
    - Artifact status: Clean

[5/5] Summary:
----------------------------------------------------------------------
  Trials processed: 5/5
  ERD detected: 4/5 (80.0%)
  Average latency: 1200 ms (±183 ms)
  Latency range: 987 - 1456 ms
  Clean trials: 4/5
----------------------------------------------------------------------

[Plotting] Generating visualization for first trial...
  [Saved] test_results_trial_1.png

======================================================================
Test completed successfully!
======================================================================
```

### 6. View Results

A file named `test_results_trial_1.png` will be created showing:
- **Plot 1:** Raw EEG signals from C3 and C4 motor channels
- **Plot 2:** Z-score evolution over time (normalized power)
- **Plot 3:** Detection summary with timing information

## Understanding the Results

### Detection Rate
- **Expected:** ~40-70% for motor imagery tasks
- Motor imagery is challenging - not all subjects/trials show clear ERD
- Detection rate varies by subject (BCI illiteracy phenomenon)

### Latency
- **Target:** ≤200ms from task onset
- **Typical:** 500-2000ms for motor imagery ERD
- Real ERD starts after imagery begins (not at cue)
- Our algorithm detects when power reduction crosses -2σ threshold

### Clean vs Artifacts
- **Clean:** Max amplitude ≤100 µV
- **Artifacts:** Eye blinks, muscle tension, electrode noise
- Artifact rejection improves detection specificity

## Troubleshooting

### Issue: "No data files found"
**Solution:** Make sure you downloaded the .mat files to `data/openbmi_sample/`

### Issue: "Error loading data"
**Solution:**
1. Verify files are complete (each should be ~2 GB)
2. Check you have scipy installed: `pip install scipy`
3. Re-download corrupted files

### Issue: "No ERD detected" for all trials
**Possible causes:**
1. **Normal variation** - Not all subjects show strong ERD
2. **Threshold too strict** - Try adjusting to -1.5σ in detector initialization
3. **Wrong channels** - Verify C3, C4, O1, O2, Fz are in the data

**To adjust threshold:**
```python
detector = ERDDetector(
    threshold_sigma=-1.5,  # Less strict (was -2.0)
    min_channels=1         # Only one channel needed (was 2)
)
```

### Issue: EMD warnings
You may see warnings like:
```
Warning: EMD failed: ... Returning original signal.
```

This is normal for some trials. EMD can fail on very noisy or short segments.
The algorithm handles this gracefully.

### Issue: Slow processing
- **Expected:** ~10-20 seconds per trial
- EMD is computationally intensive
- Normal for single-core processing
- Batch processing will use parallelization

## What Each Stage Does

1. **Preprocessing** (~2s per trial)
   - Bandpass filter: Remove frequencies outside 8-30 Hz
   - Laplacian filter: Enhance local motor cortex activity
   - Artifact detection: Flag trials with large amplitude spikes

2. **HHT Analysis** (~8-15s per trial)
   - EMD: Decompose signal into oscillatory components
   - IMF selection: Keep components with 60% power in 8-30 Hz
   - Hilbert transform: Extract instantaneous amplitude
   - Power calculation: Compute time-varying power

3. **Baseline Calculation** (<1s)
   - Use reference channels (O1, O2, Fz) from -3s to -1s
   - Compute mean and standard deviation
   - These channels should not show motor-related activity

4. **ERD Detection** (<1s)
   - Sliding window: 200ms windows, 50ms steps
   - Z-score normalization: (power - baseline_mean) / baseline_std
   - Threshold: Detect when z ≤ -2σ in ≥2 motor channels
   - Onset: First detection time

## Next Steps

Once local testing works:

1. **Adjust parameters** based on your needs
   - Detection threshold (-1.5σ to -2.5σ)
   - Window size (100-300ms)
   - Frequency band (8-30 Hz or subsets)

2. **Process more subjects**
   - Download additional subjects from GigaDB
   - Compare detection rates across subjects
   - Identify "good" vs "poor" performers

3. **Batch processing** (to be implemented)
   - Process all 54 subjects
   - Generate comprehensive metrics
   - Create statistical analysis

4. **Cloud deployment**
   - Set up GCP VM for full dataset
   - Run batch processing on all subjects
   - Generate thesis results

## Deactivating Virtual Environment

When done testing:
```bash
deactivate
```

## Additional Resources

- **OpenBMI Dataset:** https://gigadb.org/dataset/100542
- **Paper:** Lee et al. (2019), GigaScience 8(5)
- **README.md:** Full project documentation
- **CLOUD_SETUP.md:** Instructions for cloud processing

## Need Help?

- Verify Python 3.10+ is installed
- Ensure virtual environment is activated
- Check all dependencies installed: `pip list`
- Review error messages carefully
- Make sure data files are complete (~2 GB each)
