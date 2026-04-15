# DOE_MAS
This repository contains Multi-Agent tools to analyze fuel delivery complexities in Alaska.

Agent LLM calls are routed through OpenRouter by default, with per-agent
Claude Haiku 4.5 / Sonnet 4.6 assignment (see `pipeline.get_llm()` and
`RUNNING_GUIDE.md`). An Ollama fallback is available via `LLM_PROVIDER=ollama`.

Files:

Data:
- Alaksa_Energy_Authority_library
- Utilities_Bulk_Fuel_Inventory.csv

Installation text files:
- installations.txt
- requirements.txt

Pipeline (graph database):
- run_graph.py — orchestration entry point
- regionalization_graph.py
- market_cost_analysis.py
- friction_surface.py, friction_agents.py, friction_config.py
- tsp_model_graph.py
- pipeline.py — shared helpers (LLM factory, DuckDB connector, logging)
