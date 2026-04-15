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

### 9. Configure the LLM backend

The pipeline routes agent calls through **OpenRouter** by default (Claude
Haiku 4.5 for fast agents, Claude Sonnet 4.6 for reasoning/writing agents).
Export your API key before running:

```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
# Persist across sessions by appending to ~/.bashrc or the venv activate script
```

**Optional Ollama fallback:** If you prefer the local `llama3.1:70b`
backend, install it with:

```bash
sudo bash setup_ollama.sh      # mounts volume, installs Ollama, pulls llama3.1:70b (~40 GB)
export LLM_PROVIDER=ollama     # switch the pipeline to use Ollama
```

### 10. Run the Project

**Important**: Always run inside `tmux` so the pipeline survives if your Guacamole session disconnects:

```bash
# Start a named tmux session
tmux new -s pipeline

# Sanity-check the LLM backend before the full run
python -c "import pipeline; pipeline.check_llm()"

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

### 3. (If needed) Re-export the LLM credentials

If `OPENROUTER_API_KEY` isn't already in the environment, re-export it:

```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
python -c "import pipeline; pipeline.check_llm()"   # should print "OpenRouter health check passed"
```

If you're using the Ollama fallback (`LLM_PROVIDER=ollama`), make sure the
server is running:

```bash
sudo systemctl start ollama
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
| Default LLM provider | OpenRouter (Claude Haiku 4.5 + Sonnet 4.6 per-agent) |
| Required env var | `OPENROUTER_API_KEY` |
| Optional env var | `LLM_PROVIDER=ollama` (use local Ollama fallback instead) |
| Ollama models (fallback) | `/mnt/ollama_volume/ollama/models` |
| Ollama endpoint (fallback) | `http://localhost:11434` |
| Ollama model (fallback) | `llama3.1:70b` |
| Ollama env vars (fallback) | `OLLAMA_API_BASE`, `OLLAMA_MODEL` |

---

## D. Troubleshooting

- **LiteLLM timeout error**: The pipeline uses a 30-minute timeout per LLM request. If you still hit timeouts, increase `timeout` in `pipeline.py:get_llm()`
- **Pipeline crashed mid-run**: Just re-run `python run_graph.py` — it resumes from the last completed step. Delete `outputs/.checkpoint` to force a fresh start
- **Ollama not responding**: `sudo systemctl restart ollama` then check `journalctl -u ollama -f`
- **Disk full on root**: Models should be on the volume — check with `df -h`
- **Volume not mounted after reboot**: `sudo mount -a` (fstab entry is set by setup_ollama.sh)
- **Permission denied on scripts**: `chmod +x <script-name>`
