# DOE_MAS
This repository contains Multi-Agent tools to analyze fuel delivery complexities in Alaska.

## LLM Provider: Ollama (Llama 3.1:70b)

This project uses [Ollama](https://ollama.com/) to serve the Llama 3.1:70b model, hosted on a Jetstream VM. All LLM configuration is centralized in `pipeline.py`.

### Jetstream / Ollama Setup

**On the Jetstream VM (server):**

1. Provision a VM with GPU support (~40GB VRAM for the 70b model, e.g. A100)
2. Install Ollama:
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ```
3. Pull the model:
   ```bash
   ollama pull llama3.1:70b
   ```
4. Start the server, binding to all interfaces for remote access:
   ```bash
   OLLAMA_HOST=0.0.0.0:11434 ollama serve
   ```
5. Ensure the VM's firewall/security group allows inbound TCP on port 11434

**On the client machine (where this code runs):**

```bash
export OLLAMA_API_BASE=http://<jetstream-ip>:11434
export OLLAMA_MODEL=llama3.1:70b    # optional, this is the default
python run_graph.py
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_API_BASE` | `http://localhost:11434` | URL of the Ollama server |
| `OLLAMA_MODEL` | `llama3.1:70b` | Model to use |

## Files

**Data:**
- Alaska_Energy_Authority_Library/
- Utilities_Bulk_Fuel_Inventory.csv

**Installation:**
- installations.txt
- requirements.txt

**Core module:**
- pipeline.py — shared utilities, LLM configuration, file I/O, DuckDB connections

**Before Graph Database (legacy):**
- broad_overview_agent_discussion.py
- regionalization.py
- run.py
- tsp_model.py

**After Graph Database (current):**
- market_cost_analysis.py
- regionalization_graph.py
- run_graph.py
- tsp_model_graph.py
