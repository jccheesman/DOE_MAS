#!/usr/bin/env bash
# setup_ollama.sh — Provision Ollama on a Jetstream2 VM with an attached volume
# Run as root (or with sudo): sudo bash setup_ollama.sh
#
# This script:
#   1. Detects and mounts the attached volume
#   2. Installs Ollama
#   3. Configures Ollama to store models on the volume (not root disk)
#   4. Cleans up old model data from root disk
#   5. Pulls llama3.1:70b
#   6. Starts Ollama bound to 0.0.0.0 for remote access

set -euo pipefail

MOUNT_POINT="/mnt/ollama_volume"
OLLAMA_DATA_DIR="${MOUNT_POINT}/ollama/models"
MODEL="llama3.1:70b"

# ---------- helpers ----------
info()  { echo -e "\n===> $*"; }
error() { echo -e "\nERROR: $*" >&2; exit 1; }

# Must run as root
[[ $EUID -eq 0 ]] || error "Please run as root: sudo bash $0"

# =========================================================
# Step 1: Detect and mount the attached volume
# =========================================================
info "Detecting attached volume..."

# Find block devices that are disks (not partitions) and not the root device
ROOT_DEV=$(findmnt -n -o SOURCE / | sed 's/[0-9]*$//' | sed 's/p$//')
ATTACHED_DEV=""

for dev in /dev/vdb /dev/vdc /dev/sdb /dev/sdc; do
    if [[ -b "$dev" ]] && [[ "$dev" != "$ROOT_DEV" ]]; then
        ATTACHED_DEV="$dev"
        break
    fi
done

if [[ -z "$ATTACHED_DEV" ]]; then
    echo "No attached volume found automatically."
    echo "Available block devices:"
    lsblk -o NAME,SIZE,TYPE,MOUNTPOINT
    read -rp "Enter the device path for your volume (e.g. /dev/vdb): " ATTACHED_DEV
    [[ -b "$ATTACHED_DEV" ]] || error "Device $ATTACHED_DEV does not exist"
fi

echo "Using volume: $ATTACHED_DEV"

# Check if already mounted
if mountpoint -q "$MOUNT_POINT" 2>/dev/null; then
    info "Volume already mounted at $MOUNT_POINT"
else
    # Format if no filesystem exists
    if ! blkid -s TYPE -o value "$ATTACHED_DEV" &>/dev/null; then
        info "No filesystem found on $ATTACHED_DEV — formatting as ext4..."
        mkfs.ext4 -L ollama_vol "$ATTACHED_DEV"
    else
        echo "Existing filesystem detected: $(blkid -s TYPE -o value "$ATTACHED_DEV")"
    fi

    # Mount
    mkdir -p "$MOUNT_POINT"
    mount "$ATTACHED_DEV" "$MOUNT_POINT"
    info "Mounted $ATTACHED_DEV at $MOUNT_POINT"

    # Add to fstab for persistence across reboots
    UUID=$(blkid -s UUID -o value "$ATTACHED_DEV")
    if ! grep -q "$UUID" /etc/fstab; then
        echo "UUID=$UUID  $MOUNT_POINT  ext4  defaults,nofail  0  2" >> /etc/fstab
        echo "Added fstab entry (UUID=$UUID)"
    fi
fi

# Verify
df -h "$MOUNT_POINT"

# =========================================================
# Step 2: Create Ollama data directory on the volume
# =========================================================
info "Creating Ollama data directory on volume..."
mkdir -p "$OLLAMA_DATA_DIR"

# =========================================================
# Step 3: Install Ollama
# =========================================================
if command -v ollama &>/dev/null; then
    info "Ollama already installed: $(ollama --version)"
else
    info "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
fi

# =========================================================
# Step 4: Clean up old model data from root disk
# =========================================================
if [[ -d "$HOME/.ollama" ]]; then
    OLD_SIZE=$(du -sh "$HOME/.ollama" 2>/dev/null | cut -f1)
    info "Removing old Ollama data from root disk ($HOME/.ollama — $OLD_SIZE)..."
    rm -rf "$HOME/.ollama"
    echo "Freed ~$OLD_SIZE on root disk"
fi

# Also check /usr/share/ollama/.ollama (systemd user)
if [[ -d /usr/share/ollama/.ollama ]]; then
    OLD_SIZE=$(du -sh /usr/share/ollama/.ollama 2>/dev/null | cut -f1)
    info "Removing old Ollama data from /usr/share/ollama/.ollama ($OLD_SIZE)..."
    rm -rf /usr/share/ollama/.ollama
    echo "Freed ~$OLD_SIZE on root disk"
fi

# =========================================================
# Step 5: Configure systemd service to use volume + bind 0.0.0.0
# =========================================================
info "Configuring Ollama systemd service..."

mkdir -p /etc/systemd/system/ollama.service.d

cat > /etc/systemd/system/ollama.service.d/override.conf <<EOF
[Service]
Environment="OLLAMA_MODELS=${OLLAMA_DATA_DIR}"
Environment="OLLAMA_HOST=0.0.0.0:11434"
EOF

systemctl daemon-reload
systemctl enable ollama
systemctl restart ollama

echo "Ollama service configured:"
echo "  OLLAMA_MODELS = $OLLAMA_DATA_DIR"
echo "  OLLAMA_HOST   = 0.0.0.0:11434"

# Wait for service to be ready
info "Waiting for Ollama to start..."
for i in {1..15}; do
    if curl -s http://localhost:11434/api/tags &>/dev/null; then
        echo "Ollama is running."
        break
    fi
    sleep 2
done

# =========================================================
# Step 6: Pull the model
# =========================================================
info "Pulling $MODEL (this may take a while)..."
ollama pull "$MODEL"

# =========================================================
# Step 7: Verify and print client instructions
# =========================================================
info "Setup complete!"
echo ""
echo "Verification:"
echo "  Volume:  $(df -h "$MOUNT_POINT" | tail -1)"
echo "  Models:  $(ls "$OLLAMA_DATA_DIR" 2>/dev/null || echo '(check subdirectories)')"
echo "  Ollama:  $(ollama list 2>/dev/null)"
echo ""
echo "============================================"
echo "  On your CLIENT machine, run:"
echo ""
VM_IP=$(hostname -I | awk '{print $1}')
echo "  export OLLAMA_API_BASE=http://${VM_IP}:11434"
echo "  export OLLAMA_MODEL=${MODEL}"
echo "  python run_graph.py"
echo ""
echo "  Make sure your VM's firewall allows TCP 11434"
echo "============================================"
