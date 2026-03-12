# Jetstream2 VM Setup Guide — DOE MAS Project

---

## A. Setting Up a New Instance

### 1. Create the VM (Jetstream2 Web UI)

- **Image**: Ubuntu (latest LTS)
- **GPU**: NVIDIA A100
- **Root Disk**: 150 GB
- Allow a few minutes for the instance to build

### 2. Attach Storage Volume (Jetstream2 Web UI)

- Navigate to your instance on the Jetstream2 dashboard
- Attach your existing volume (or create a new one, 200 GB+ recommended)
- Note: the volume persists your code, venv, Ollama models, and data between instances

### 3. Open Terminal (Guacamole Desktop)

```bash
# Load conda/mamba environment manager
module load miniforge
```

### 4. Navigate to Volume

```bash
# IMPORTANT: cd to volume BEFORE setting up the virtual environment
cd /media/volume/<your-volume-name>
```

### 5. Set Up Python Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

### 6. Transfer Project Files

- **Drag and drop** files from your local machine into the Guacamole terminal
  - They land in `/home/exouser/`
- **Move them to the volume:**

```bash
mv /home/exouser/*.py /media/volume/<your-volume-name>/
mv /home/exouser/*.txt /media/volume/<your-volume-name>/
mv /home/exouser/*.sh /media/volume/<your-volume-name>/
mv /home/exouser/*.csv /media/volume/<your-volume-name>/
mv /home/exouser/*.zip /media/volume/<your-volume-name>/
```

### 7. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 8. Run Additional Installations

```bash
chmod +x installations.txt
./installations.txt
```

### 9. Set Up Ollama (LLM Server)

```bash
sudo bash setup_ollama.sh
```

This script will:
- Detect and mount the attached volume
- Install Ollama
- Store models on the volume (not root disk)
- Pull `llama3.1:70b` (~40 GB)
- Bind Ollama to `0.0.0.0:11434` for access

### 10. Run the Project

**Important**: Always run inside `tmux` so the pipeline survives if your Guacamole session disconnects:

```bash
# Start a named tmux session
tmux new -s pipeline

# Run the pipeline
python run_graph.py

# To detach (process keeps running): press Ctrl+B, then D
# To reattach later:  tmux attach -t pipeline
```

If the pipeline crashes mid-run, just re-run `python run_graph.py` — it will skip completed steps automatically (checkpoint stored in `outputs/.checkpoint`). To force a full re-run, delete that file first.

---

## B. Reopening an Existing Instance

### 1. Open Guacamole Desktop

- Log in through the Jetstream2 web UI and launch the Guacamole console

### 2. Activate Environment

```bash
module load miniforge
cd /media/volume/<your-volume-name>
source venv/bin/activate
```

### 3. (If needed) Restart Ollama

Ollama should auto-start via systemd, but if not:

```bash
sudo systemctl start ollama
```

Verify it's running:

```bash
curl http://localhost:11434/api/tags
```

### 4. Run the Project

```bash
# Always use tmux so disconnects don't kill the pipeline
tmux new -s pipeline
python run_graph.py
```

---

## C. Quick Reference

| Item | Value |
|---|---|
| GPU | NVIDIA A100 |
| Root Disk | 150 GB |
| Volume | 200 GB+ (attached, persists between instances) |
| Python venv | `/media/volume/<name>/venv` |
| Ollama models | `/mnt/ollama_volume/ollama/models` |
| Ollama endpoint | `http://localhost:11434` |
| LLM model | `llama3.1:70b` |
| Key env vars | `OLLAMA_API_BASE`, `OLLAMA_MODEL` |

---

## D. Troubleshooting

- **LiteLLM timeout error**: The pipeline uses a 30-minute timeout per LLM request. If you still hit timeouts, increase `timeout` in `pipeline.py:get_llm()`
- **Pipeline crashed mid-run**: Just re-run `python run_graph.py` — it resumes from the last completed step. Delete `outputs/.checkpoint` to force a fresh start
- **Ollama not responding**: `sudo systemctl restart ollama` then check `journalctl -u ollama -f`
- **Disk full on root**: Models should be on the volume — check with `df -h`
- **Volume not mounted after reboot**: `sudo mount -a` (fstab entry is set by setup_ollama.sh)
- **Permission denied on scripts**: `chmod +x <script-name>`
