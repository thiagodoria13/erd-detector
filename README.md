# ERD Detector: Single-Trial Event-Related Desynchronization Detection

Single-trial ERD detection using Hilbert-Huang Transform for Brain-Computer Interface applications.

## Overview

This package implements the methodology described in your thesis for detecting Event-Related Desynchronization (ERD) in EEG signals **without machine learning calibration**.

### Algorithm Components

1. **Preprocessing**
   - Bandpass filter (8-30 Hz) to isolate µ and β rhythms
   - Laplacian spatial filter for C3/C4 to enhance local activity
   - Artifact rejection using amplitude thresholding (±100µV)

2. **Hilbert-Huang Transform (HHT)**
   - Empirical Mode Decomposition (EMD) → Intrinsic Mode Functions (IMFs)
   - Spectral IMF selection (60% power in 8-30 Hz band)
   - Hilbert Transform → Instantaneous amplitude
   - Power calculation (amplitude²)

3. **ERD Detection**
   - Baseline calculation from reference channels (O1, O2, Fz) during [-3s, -1s]
   - Sliding window (200ms, 50ms step) through task period [0s, +4s]
   - Normalize motor channels (C3, C4): z = (P - μ) / σ
   - Detect ERD when z ≤ -2σ in ≥2 motor channels

### Key Features

✅ **No Calibration** - Uses neurophysiological thresholds  
✅ **Single-Trial** - Works on individual trials  
✅ **Low Latency** - Target ≤200ms processing time  
✅ **High Accuracy** - Target ≥80% detection rate  

## Project Structure

```
erd-detector/
├── erd_detector/           # Main package
│   ├── __init__.py        # Package initialization
│   ├── utils.py           # Data loading (OpenBMI)
│   ├── preprocessing.py   # Filtering & artifacts
│   ├── hht.py            # EMD + Hilbert Transform
│   ├── detection.py      # ERD detection algorithm
│   ├── metrics.py        # Performance evaluation
│   └── visualization.py  # Plotting functions
├── scripts/               # Batch processing
│   ├── download_data.py  # Download from GigaDB
│   ├── process_all.py    # Process 54 subjects
│   ├── analyze.py        # Compute metrics
│   └── visualize.py      # Generate figures
├── tests/                 # Unit tests
├── docs/                  # Documentation
├── examples/              # Usage examples
├── results/               # Output directory
│   ├── figures/          # Generated plots
│   ├── data/             # Processed results
│   └── reports/          # Analysis summaries
├── requirements.txt       # Python dependencies
├── setup.py              # Package installation
└── README.md             # This file
```

## Installation

### Local Setup (For Testing)

```bash
cd /c/Users/Thiago\ Doria/Desktop/EEG/erd-detector
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

### Cloud Setup (For Full Processing)

The dataset is ~209 GB, so cloud processing is required.

**GCP Setup (Recommended - Hong Kong region closest to GigaDB):**

```bash
# 1. Create GCP project
gcloud projects create erd-research-2025

# 2. Enable Compute Engine
gcloud services enable compute.googleapis.com

# 3. Create VM
gcloud compute instances create erd-vm \
    --project=erd-research-2025 \
    --zone=asia-east2-a \
    --machine-type=n1-highmem-16 \
    --boot-disk-size=50GB \
    --boot-disk-type=pd-ssd

# 4. Create data disk
gcloud compute disks create erd-data \
    --project=erd-research-2025 \
    --zone=asia-east2-a \
    --size=500GB \
    --type=pd-ssd

# 5. Attach disk
gcloud compute instances attach-disk erd-vm \
    --disk=erd-data \
    --zone=asia-east2-a

# 6. SSH and setup
gcloud compute ssh erd-vm --zone=asia-east2-a
```

## Usage

### Quick Start

```python
from erd_detector import ERDDetector
from erd_detector.utils import load_openbmi_data, extract_trial

# Initialize detector with thesis defaults
detector = ERDDetector(
    motor_channels=['C3', 'C4'],
    reference_channels=['O1', 'O2', 'Fz'],
    threshold_sigma=-2.0,       # Thesis default: -2σ
    min_channels=2              # Require both C3 and C4
)

# Load data
data = load_openbmi_data(subject_id=1, session=1, data_type='train')

# Process single trial
event_time = data['events'][0]
trial_data = extract_trial(data['data'], event_time, data['fs'])

# Detect ERD
result = detector.process_trial(
    trial_data,
    data['channels'],
    data['fs']
)

# Check result
if result['detected']:
    print(f"✓ ERD detected at {result['onset_time']:.3f}s")
    print(f"  Latency: {result['latency']*1000:.1f}ms")
else:
    print("✗ No ERD detected")
```

### Batch Processing

```bash
# Download OpenBMI dataset
python scripts/download_data.py

# Process all 54 subjects
python scripts/process_all.py

# Analyze results
python scripts/analyze.py

# Generate figures
python scripts/visualize.py
```

## Dataset

**OpenBMI Motor Imagery Dataset**
- 54 subjects × 2 sessions
- 200 trials per session (100 train, 100 test)
- 62 EEG channels @ 1000 Hz
- Motor imagery: left hand vs right hand
- Total size: ~209 GB

**Citation:**
Lee, M.H., et al. (2019). EEG dataset and OpenBMI toolbox for three BCI paradigms: an investigation into BCI illiteracy. GigaScience, 8(5).

**Download:** https://gigadb.org/dataset/100542

## Results (To Be Completed)

Targets from thesis:
- Detection rate: ≥80%
- Latency: ≤200ms
- Comparison: CSP+LDA baseline = 71.1%

Results will be populated after processing all subjects.

## Implementation Status

- [x] Project structure
- [x] Requirements and dependencies
- [x] Utils module (data loading)
- [ ] Preprocessing module (in progress)
- [ ] HHT module (in progress)
- [ ] Detection module (in progress)
- [ ] Metrics module
- [ ] Visualization module
- [ ] Batch processing scripts
- [ ] Analysis scripts
- [ ] Documentation
- [ ] Tests

## Next Steps

1. **Complete Core Modules** - Finish implementing preprocessing, HHT, detection
2. **Local Testing** - Test on 1-2 subjects locally
3. **Cloud Deployment** - Set up GCP VM and download full dataset
4. **Batch Processing** - Process all 54 subjects
5. **Analysis** - Compute metrics and compare with baseline
6. **Visualization** - Generate all figures for thesis
7. **Documentation** - Write technical report

## Development

### Running Tests

```bash
pytest tests/ -v
```

### Code Style

All code includes:
- Line-by-line comments explaining logic
- Docstrings with mathematical formulas (LaTeX)
- Type hints for clarity
- Neurophysiological justifications

## Citation

If you use this package, please cite your thesis:

```bibtex
@mastersthesis{fonseca2025erd,
  title={Processamento de sinais de EEG para detecção de ERD em baixa latência para controle de interfaces cérebro-computador (BCI)},
  author={Fonseca, Lucas Pereira da and Doria, Thiago Anversa Sampaio},
  year={2025},
  school={Universidade de São Paulo}
}
```

## License

MIT License - See LICENSE file

## Contact

- Lucas Pereira da Fonseca
- Thiago Anversa Sampaio Doria
- Advisor: Prof. Arturo Forner-Cordero
- Institution: Escola Politécnica, Universidade de São Paulo

## Acknowledgments

- Prof. Arturo Forner-Cordero for guidance and introducing this research topic
- Bruna and André for research support and valuable insights
- OpenBMI team for providing the dataset
