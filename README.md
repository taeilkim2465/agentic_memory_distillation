# Agentic Memory Distillation

This repository contains the code for our paper on agentic memory distillation — a framework for building and leveraging memory from teacher agent trajectories to improve student agent performance across tool-use benchmarks.

## Overview

We evaluate our memory distillation approach on three benchmarks:

| Benchmark | Directory | Description |
|-----------|-----------|-------------|
| [AppWorld](https://github.com/ace-agent/ace-appworld) | `appworld/` | Multi-app task completion benchmark |
| [BFCL](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard) | `bfcl/` | Berkeley Function-Calling Leaderboard |
| [ToolSandbox](https://github.com/apple/ToolSandbox) | `toolsandbox/` | Stateful conversational tool-use benchmark |

## Memory Methods

Each benchmark directory contains implementations of three memory strategies:

- **MEMP** (Memory Experience Memory Pool): Stores proceduralized experience memories from teacher trajectories
- **RB** (Reasoning Bank): Builds a bank of successful reasoning chains for retrieval
- **SASM** (Subtask-Aware Semantic Memory): Decomposes tasks into subtasks and stores subtask-level memories

## Repository Structure

Each benchmark directory is organized by memory method (`memp/`, `rb/`, `sasm/`), with shared code in `common/`.

```
agentic_memory_distillation/
├── appworld/
│   ├── memp/
│   │   ├── experiments/code/     # MEMP agent implementation
│   │   ├── experiments/configs/  # Experiment configs (.jsonnet)
│   │   ├── experiments/prompts/  # Prompt templates
│   │   ├── scripts/              # Run scripts (teacher & student)
│   │   └── generate_memp_memories.py
│   ├── rb/
│   │   ├── experiments/code/
│   │   ├── experiments/configs/
│   │   ├── experiments/prompts/
│   │   └── scripts/
│   └── sasm/
│       ├── experiments/code/
│       ├── experiments/configs/
│       ├── experiments/prompts/
│       ├── scripts/
│       └── build_sasm_memory_offline.py
├── bfcl/
│   ├── common/
│   │   ├── memory/           # Shared memory module (builder, injection, store)
│   │   ├── model_handler/    # Modified base_handler.py
│   │   └── scripts/          # Baseline & evaluation scripts
│   ├── memp/scripts/
│   ├── rb/scripts/
│   └── sasm/scripts/
└── toolsandbox/
    ├── common/
    │   ├── memory/           # Shared memory module
    │   ├── roles/            # memory_augmented_agent.py
    │   └── utils/            # Utility scripts
    ├── baseline/scripts/     # Baseline & workflow-only variants
    ├── memp/scripts/
    ├── rb/scripts/
    └── sasm/scripts/
```

## Setup

Each benchmark requires its own environment setup. Please refer to the original benchmark repositories for installation instructions, then apply the code from this repository on top.

### AppWorld

```bash
git clone https://github.com/ace-agent/ace-appworld.git
cd ace-appworld
# Follow setup instructions in the original repo
# Then copy appworld/ contents into the cloned repo
```

### BFCL

```bash
git clone https://github.com/ShishirPatil/gorilla.git
cd gorilla/berkeley-function-call-leaderboard
# Follow setup instructions in the original repo
# Then copy bfcl/memory/ into bfcl_eval/memory/
# And bfcl/model_handler/base_handler.py into bfcl_eval/model_handler/
```

### ToolSandbox

```bash
git clone https://github.com/apple/ToolSandbox.git
cd ToolSandbox
# Follow setup instructions in the original repo
# Then copy toolsandbox/memory/ into tool_sandbox/memory/
# And toolsandbox/roles/memory_augmented_agent.py into tool_sandbox/roles/
```

## Running Experiments

See the `scripts/` directory within each benchmark folder for experiment run scripts.

## Citation

If you use this code, please cite our paper:

```bibtex
@article{,
  title={},
  author={},
  year={2025}
}
```
