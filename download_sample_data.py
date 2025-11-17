#!/usr/bin/env python
"""
Download OpenBMI sample data for local testing using Wasabi URLs.

This script downloads 1-2 subjects from the OpenBMI dataset using fast
Wasabi S3 URLs with resume support and progress bars.

OpenBMI Motor Imagery Dataset:
- 54 subjects total
- 2 sessions per subject
- Each file: ~2 GB (100 trials)
- Source: https://gigadb.org/dataset/100542

Usage:
    python download_sample_data.py [--subjects 1,2] [--sessions 1,2]

Author: Lucas Pereira da Fonseca, Thiago Anversa Sampaio Doria
"""

import argparse
import urllib.request
import urllib.error
import sys
from pathlib import Path
from tqdm import tqdm
import time


class DownloadProgressBar(tqdm):
    """Progress bar for urllib downloads with resume support."""

    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_with_resume(url, output_path, max_retries=3, timeout=300):
    """
    Download file with resume support and retries.

    Args:
        url: URL to download from
        output_path: Path to save file
        max_retries: Maximum number of retry attempts
        timeout: Timeout in seconds for each attempt

    Returns:
        bool: True if successful, False otherwise
    """
    output_path = Path(output_path)
    temp_path = output_path.with_suffix(output_path.suffix + '.part')

    # Check if complete file exists
    if output_path.exists():
        print(f"  [Exists] {output_path.name}")
        return True

    # Get file size from server
    try:
        req = urllib.request.Request(url, method='HEAD')
        with urllib.request.urlopen(req, timeout=30) as response:
            total_size = int(response.headers.get('Content-Length', 0))
    except Exception as e:
        print(f"  [Error] Could not get file size: {e}")
        return False

    # Check if partial download exists
    resume_pos = 0
    if temp_path.exists():
        resume_pos = temp_path.stat().st_size
        if resume_pos >= total_size:
            # Partial file is complete, rename it
            temp_path.rename(output_path)
            print(f"  [Complete] {output_path.name}")
            return True
        print(f"  [Resume] from {resume_pos / (1024**3):.2f} GB")

    # Download with retries
    for attempt in range(max_retries):
        try:
            # Prepare request with resume header
            req = urllib.request.Request(url)
            if resume_pos > 0:
                req.add_header('Range', f'bytes={resume_pos}-')

            with urllib.request.urlopen(req, timeout=timeout) as response:
                # Open file in append mode if resuming
                mode = 'ab' if resume_pos > 0 else 'wb'

                with open(temp_path, mode) as f:
                    with DownloadProgressBar(
                        unit='B',
                        unit_scale=True,
                        miniters=1,
                        desc=output_path.name,
                        total=total_size,
                        initial=resume_pos
                    ) as pbar:
                        chunk_size = 1024 * 1024  # 1 MB chunks
                        while True:
                            chunk = response.read(chunk_size)
                            if not chunk:
                                break
                            f.write(chunk)
                            pbar.update(len(chunk))

            # Download complete, rename temp file
            temp_path.rename(output_path)
            print(f"  [Success] {output_path.name}")
            return True

        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            print(f"  [Attempt {attempt + 1}/{max_retries}] Failed: {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"  [Wait] Retrying in {wait_time}s...")
                time.sleep(wait_time)
                # Update resume position
                if temp_path.exists():
                    resume_pos = temp_path.stat().st_size
            else:
                print(f"  [Error] Max retries reached")
                return False
        except Exception as e:
            print(f"  [Error] Unexpected error: {e}")
            return False

    return False


def download_openbmi_sample(
    subjects=[1, 2],
    sessions=[1, 2],
    data_dir='data/openbmi_sample'
):
    """
    Download sample OpenBMI data from Wasabi CDN.

    Args:
        subjects: List of subject IDs to download (default: [1, 2])
        sessions: List of session numbers (default: [1, 2])
        data_dir: Directory to save data (default: 'data/openbmi_sample')

    Returns:
        bool: True if all downloads successful
    """
    # Wasabi S3 base URL (faster than GigaDB FTP)
    base_url = "https://s3.us-west-1.wasabisys.com/gigadb-datasets/live/pub/10.5524/100001_101000/100542"

    # Create data directory
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("OpenBMI Dataset - Sample Download (Wasabi CDN)")
    print("=" * 70)
    print()
    print(f"Downloading {len(subjects)} subject(s) × {len(sessions)} session(s)")
    print(f"Estimated total: ~{len(subjects) * len(sessions) * 2} GB")
    print(f"Destination: {data_path.absolute()}")
    print()

    # Build download list
    downloads = []
    for subj in subjects:
        for sess in sessions:
            filename = f"sess{sess:02d}_subj{subj:02d}_EEG_MI.mat"
            url = f"{base_url}/{filename}"
            output = data_path / filename
            downloads.append((url, output, filename))

    # Download files
    print(f"Starting downloads ({len(downloads)} files)...")
    print("-" * 70)

    successful = 0
    failed = []

    for url, output, filename in downloads:
        print(f"\nDownloading: {filename}")
        if download_with_resume(url, output, max_retries=3, timeout=300):
            successful += 1
        else:
            failed.append(filename)

    # Summary
    print()
    print("=" * 70)
    print("Download Summary")
    print("=" * 70)
    print(f"Successful: {successful}/{len(downloads)}")

    if failed:
        print(f"Failed: {len(failed)}")
        for f in failed:
            print(f"  - {f}")
        print()
        print("To retry failed downloads, run this script again.")
        print("Partial downloads will resume automatically.")
        return False
    else:
        print()
        print("✓ All downloads complete!")
        print()
        print("You can now run:")
        print("  python test_local.py")
        return True


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Download OpenBMI sample data for local testing'
    )
    parser.add_argument(
        '--subjects',
        type=str,
        default='1,2',
        help='Comma-separated subject IDs (default: 1,2)'
    )
    parser.add_argument(
        '--sessions',
        type=str,
        default='1,2',
        help='Comma-separated session numbers (default: 1,2)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/openbmi_sample',
        help='Output directory (default: data/openbmi_sample)'
    )

    args = parser.parse_args()

    # Parse subjects and sessions
    try:
        subjects = [int(s.strip()) for s in args.subjects.split(',')]
        sessions = [int(s.strip()) for s in args.sessions.split(',')]
    except ValueError as e:
        print(f"Error parsing subjects/sessions: {e}")
        return 1

    # Validate inputs
    for subj in subjects:
        if not (1 <= subj <= 54):
            print(f"Error: Subject {subj} out of range (1-54)")
            return 1

    for sess in sessions:
        if sess not in [1, 2]:
            print(f"Error: Session {sess} invalid (must be 1 or 2)")
            return 1

    # Download
    success = download_openbmi_sample(
        subjects=subjects,
        sessions=sessions,
        data_dir=args.output_dir
    )

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
