#!/usr/bin/env python
"""
Download a small sample of OpenBMI dataset for local testing.

This script downloads data for 1-2 subjects (~4-8 GB) instead of the full
209 GB dataset, allowing for local testing with real EEG data.

OpenBMI Motor Imagery Dataset:
- 54 subjects total
- 2 sessions per subject
- Each session: ~2 GB (100 trials)
- Source: https://gigadb.org/dataset/100542

Usage:
    python download_sample_data.py

Author: Lucas Pereira da Fonseca, Thiago Anversa Sampaio Doria
"""

import os
import urllib.request
import sys
from pathlib import Path
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    """Progress bar for urllib downloads."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    """Download file with progress bar."""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path.name) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def download_openbmi_sample(
    subjects=[1, 2],
    sessions=[1, 2],
    data_dir='data/openbmi_sample'
):
    """
    Download sample OpenBMI data for local testing.

    Args:
        subjects: List of subject IDs to download (default: [1, 2])
        sessions: List of session numbers (default: [1, 2])
        data_dir: Directory to save data (default: 'data/openbmi_sample')

    OpenBMI File Structure:
        Session 1 (Training): sess01_subj{:02d}_EEG_MI.mat
        Session 2 (Test): sess02_subj{:02d}_EEG_MI.mat

    Each file contains:
        - EEG data: 62 channels @ 1000 Hz
        - Events: Trial markers (left hand vs right hand imagery)
        - Channel names
        - Sampling rate
    """
    # Base URL for OpenBMI dataset
    # Note: GigaDB uses FTP. We'll provide instructions for manual download
    base_url = "ftp://penguin.genomics.cn/pub/10.5524/100001_101000/100542"

    # Create data directory
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("OpenBMI Dataset - Sample Download")
    print("=" * 70)
    print()
    print(f"Downloading data for {len(subjects)} subject(s), {len(sessions)} session(s) each")
    print(f"Estimated total size: ~{len(subjects) * len(sessions) * 2} GB")
    print(f"Destination: {data_path.absolute()}")
    print()

    # FTP downloads can be tricky, provide manual instructions
    print("=" * 70)
    print("DOWNLOAD INSTRUCTIONS")
    print("=" * 70)
    print()
    print("Due to FTP access requirements, please download files manually:")
    print()
    print("Option 1: Direct Browser Download (Recommended)")
    print("-" * 70)
    print("1. Visit: https://gigadb.org/dataset/100542")
    print("2. Click 'Files' tab")
    print("3. Download these files:")
    print()

    files_to_download = []
    for subj in subjects:
        for sess in sessions:
            if sess == 1:
                filename = f"sess01_subj{subj:02d}_EEG_MI.mat"
            else:
                filename = f"sess02_subj{subj:02d}_EEG_MI.mat"
            files_to_download.append(filename)
            print(f"   - {filename}")

    print()
    print(f"4. Save files to: {data_path.absolute()}")
    print()
    print("Option 2: Command Line Download (Linux/Mac)")
    print("-" * 70)
    print("Use wget or curl to download from FTP:")
    print()

    for filename in files_to_download:
        ftp_url = f"{base_url}/{filename}"
        print(f"wget {ftp_url} -P {data_dir}")

    print()
    print("Option 3: Python Script Download")
    print("-" * 70)
    print("Run this script with ftplib:")
    print()

    # Create a simple FTP download helper
    ftp_script = data_path / "download_ftp.py"
    with open(ftp_script, 'w') as f:
        f.write('''#!/usr/bin/env python
"""FTP download helper for OpenBMI dataset."""
from ftplib import FTP
import sys

def download_openbmi_ftp(files, output_dir="."):
    """Download files from OpenBMI FTP server."""
    ftp = FTP("penguin.genomics.cn")
    ftp.login()  # Anonymous login
    ftp.cwd("pub/10.5524/100001_101000/100542")

    for filename in files:
        print(f"Downloading {filename}...")
        local_path = f"{output_dir}/{filename}"
        with open(local_path, 'wb') as f:
            ftp.retrbinary(f'RETR {filename}', f.write)
        print(f"  Saved to {local_path}")

    ftp.quit()
    print("All downloads complete!")

if __name__ == "__main__":
    files = [
''')
        for filename in files_to_download:
            f.write(f'        "{filename}",\n')
        f.write(f'''    ]
    download_openbmi_ftp(files, "{data_dir}")
''')

    print(f"python {ftp_script}")
    print()
    print("=" * 70)
    print()

    # Check if files already exist
    existing_files = []
    missing_files = []
    for filename in files_to_download:
        file_path = data_path / filename
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            existing_files.append((filename, size_mb))
        else:
            missing_files.append(filename)

    if existing_files:
        print("✓ Found existing files:")
        for filename, size_mb in existing_files:
            print(f"  - {filename} ({size_mb:.1f} MB)")
        print()

    if missing_files:
        print("✗ Missing files:")
        for filename in missing_files:
            print(f"  - {filename}")
        print()
        print(f"Please download these files to: {data_path.absolute()}")
    else:
        print("=" * 70)
        print("✓ All required files are present!")
        print("=" * 70)
        print()
        print("You can now run:")
        print("  python test_local.py")

    return data_path


def verify_downloaded_data(data_dir='data/openbmi_sample'):
    """Verify downloaded .mat files are valid."""
    import scipy.io as sio

    data_path = Path(data_dir)
    mat_files = list(data_path.glob("*.mat"))

    if not mat_files:
        print(f"No .mat files found in {data_path}")
        return False

    print()
    print("=" * 70)
    print("Verifying downloaded files...")
    print("=" * 70)
    print()

    all_valid = True
    for mat_file in mat_files:
        try:
            print(f"Checking {mat_file.name}...")
            data = sio.loadmat(mat_file)

            # Check for required fields
            required_keys = ['EEG_MI_train', 'EEG_MI_test']
            has_data = any(key in data for key in required_keys)

            if has_data:
                print(f"  ✓ Valid OpenBMI file")
                # Show file info
                size_mb = mat_file.stat().st_size / (1024 * 1024)
                print(f"  - Size: {size_mb:.1f} MB")
            else:
                print(f"  ✗ Missing required data fields")
                all_valid = False

        except Exception as e:
            print(f"  ✗ Error reading file: {e}")
            all_valid = False

        print()

    if all_valid:
        print("=" * 70)
        print("✓ All files verified successfully!")
        print("=" * 70)
    else:
        print("=" * 70)
        print("✗ Some files failed verification")
        print("=" * 70)

    return all_valid


def main():
    """Main function."""
    print()
    print("OpenBMI Dataset - Sample Downloader")
    print("For local testing with real EEG data")
    print()

    # Default: Download Subject 1 and 2, both sessions
    subjects = [1, 2]
    sessions = [1, 2]
    data_dir = 'data/openbmi_sample'

    # Download (provides instructions)
    data_path = download_openbmi_sample(
        subjects=subjects,
        sessions=sessions,
        data_dir=data_dir
    )

    # If scipy is available, verify files
    try:
        verify_downloaded_data(data_dir)
    except ImportError:
        print("Note: Install scipy to verify downloaded files:")
        print("  pip install scipy")


if __name__ == '__main__':
    main()
