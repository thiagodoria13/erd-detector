# Cloud Infrastructure Setup Guide

## Option 1: Google Cloud Platform (Recommended)

### Why GCP asia-east2?
- GigaDB servers are in Hong Kong
- Minimal network latency
- No egress charges within same region
- Cost: ~$200 for 1 week ($300 free credit available)

### Step-by-Step Setup

#### 1. Create GCP Account
- Visit: https://console.cloud.google.com
- Sign up (get $300 free credit)
- Enable billing

#### 2. Install gcloud CLI
```bash
# Download from: https://cloud.google.com/sdk/docs/install
# Or use cloud shell (built-in)
```

#### 3. Create Project
```bash
gcloud projects create erd-research-2025 --name="EEG ERD Research"
gcloud config set project erd-research-2025
gcloud services enable compute.googleapis.com
```

#### 4. Create VM
```bash
gcloud compute instances create erd-vm \
    --zone=asia-east2-a \
    --machine-type=n1-highmem-16 \
    --boot-disk-size=50GB \
    --boot-disk-type=pd-ssd \
    --image-family=ubuntu-2004-lts \
    --image-project=ubuntu-os-cloud
```

#### 5. Create and Attach Data Disk
```bash
# Create 500GB SSD for dataset
gcloud compute disks create erd-data \
    --zone=asia-east2-a \
    --size=500GB \
    --type=pd-ssd

# Attach to VM
gcloud compute instances attach-disk erd-vm \
    --disk=erd-data \
    --zone=asia-east2-a
```

#### 6. SSH and Setup Environment
```bash
# Connect
gcloud compute ssh erd-vm --zone=asia-east2-a

# Format and mount data disk
sudo mkfs.ext4 -m 0 -F /dev/sdb
sudo mkdir -p /mnt/data
sudo mount /dev/sdb /mnt/data
sudo chmod 777 /mnt/data

# Make permanent
echo "/dev/sdb /mnt/data ext4 defaults 0 0" | sudo tee -a /etc/fstab

# Install Python
sudo apt-get update
sudo apt-get install -y python3.10 python3.10-venv python3-pip git build-essential

# Create project directory
mkdir -p /mnt/data/{openbmi_raw,openbmi_processed,results}
cd /home/$USER
git clone <your-repo-url> erd-detector
cd erd-detector

# Setup Python environment
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

#### 7. Download Dataset
```bash
python scripts/download_data.py
# This will download ~209 GB to /mnt/data/openbmi_raw
# Expected time: 4-6 hours depending on network
```

#### 8. Process Data
```bash
# Process all 54 subjects
python scripts/process_all.py

# Analyze results
python scripts/analyze.py

# Generate figures
python scripts/visualize.py
```

#### 9. Download Results
```bash
# From your local machine:
gcloud compute scp --recurse erd-vm:/mnt/data/results ./results --zone=asia-east2-a
```

#### 10. Cleanup (When Done)
```bash
# Stop VM (keeps data, stops billing for compute)
gcloud compute instances stop erd-vm --zone=asia-east2-a

# Delete VM (when completely done)
gcloud compute instances delete erd-vm --zone=asia-east2-a

# Delete disk (WARNING: loses all data)
gcloud compute disks delete erd-data --zone=asia-east2-a
```

## Cost Breakdown

**GCP asia-east2 (1 week):**
- n1-highmem-16: $0.95/hr × 168 hrs = $160
- 500GB SSD: $0.17/GB/mo = $20
- Network egress: 209GB download = $25
- **Total: ~$205** (covered by $300 free credit)

## Alternative: AWS

If you prefer AWS:

```bash
# Similar setup in ap-east-1 (Hong Kong)
# EC2 instance: r5.4xlarge (16 vCPU, 128GB RAM)
# EBS: 500GB gp3 SSD
# Cost: Similar to GCP (~$200/week)
```

## Monitoring

### Check VM Status
```bash
gcloud compute instances list
```

### Check Disk Usage
```bash
df -h /mnt/data
```

### Monitor Processing
```bash
tail -f /mnt/data/processing.log
```

## Troubleshooting

### Disk Full
```bash
# Check space
df -h

# Clean up if needed
rm -rf /mnt/data/openbmi_raw  # After processing complete
```

### Out of Memory
```bash
# Check memory
free -h

# If OOM, upgrade to larger instance:
gcloud compute instances set-machine-type erd-vm \
    --machine-type=n1-highmem-32 \
    --zone=asia-east2-a
```

### Network Issues
```bash
# Test connection to GigaDB
ping ftp.cngb.org

# Resume interrupted download
python scripts/download_data.py  # Script has resume support
```
