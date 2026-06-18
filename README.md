# Agentic Memory Distillation

This repository contains the official code for our papers on agentic memory distillation — a framework for building and leveraging memory distilled from teacher agent trajectories to improve student agent performance across tool-use benchmarks.

## Papers

| Method | Paper |
|--------|-------|
| **SASM** | [Structurally Aligned Subtask-Level Memory for Software Engineering Agents](#) |
| **MEMP** | [MEMP: Exploring Agent Procedural Memory](#) |
| **RB** | [ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory](#) |

## Benchmarks

We evaluate all three methods on three tool-use benchmarks:

| Benchmark | Directory | Description |
|-----------|-----------|-------------|
| [AppWorld](https://github.com/ace-agent/ace-appworld) | `appworld/` | Multi-app task completion benchmark |
| [BFCL](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard) | `bfcl/` | Berkeley Function-Calling Leaderboard |
| [ToolSandbox](https://github.com/apple/ToolSandbox) | `toolsandbox/` | Stateful conversational tool-use benchmark |

## Memory Methods

- **SASM** — *Structurally Aligned Subtask-Level Memory for Software Engineering Agents*: Decomposes tasks into subtasks and stores structurally aligned subtask-level memories distilled from teacher trajectories.
- **MEMP** — *MEMP: Exploring Agent Procedural Memory*: Distills proceduralized experience memories from teacher agent trajectories for retrieval during student inference.
- **ReasoningBank (RB)** — *ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory*: Builds a self-evolving reasoning memory bank from successful teacher trajectories, enabling agents to scale with accumulated reasoning experience.

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

Each benchmark requires its own environment setup. Clone the original repository first, then overlay the code from this repo.

### AppWorld

```bash
git clone https://github.com/ace-agent/ace-appworld.git
cd ace-appworld
# Follow the setup instructions in the original repo, then:
cp -r /path/to/this/repo/appworld/memp  ./memp
cp -r /path/to/this/repo/appworld/rb    ./rb
cp -r /path/to/this/repo/appworld/sasm  ./sasm
```

### BFCL

```bash
git clone https://github.com/ShishirPatil/gorilla.git
cd gorilla/berkeley-function-call-leaderboard
# Follow the setup instructions in the original repo, then:
cp -r /path/to/this/repo/bfcl/common/memory        ./bfcl_eval/memory
cp    /path/to/this/repo/bfcl/common/model_handler/base_handler.py \
                                                    ./bfcl_eval/model_handler/base_handler.py
```

### ToolSandbox

```bash
git clone https://github.com/apple/ToolSandbox.git
cd ToolSandbox
# Follow the setup instructions in the original repo, then:
cp -r /path/to/this/repo/toolsandbox/common/memory  ./tool_sandbox/memory
cp    /path/to/this/repo/toolsandbox/common/roles/memory_augmented_agent.py \
                                                    ./tool_sandbox/roles/memory_augmented_agent.py
# Set the repo path for run scripts:
export TOOLSANDBOX_ROOT=$(pwd)
```

## Running Experiments

### AppWorld

```bash
# 1. Build teacher memory
bash appworld/memp/scripts/memp_run_teacher.sh

# 2. Run student inference with memory
bash appworld/memp/scripts/memp_run_student.sh
```

Run scripts for RB and SASM follow the same `*_teacher` → `*_student` pattern.

### BFCL

```bash
# Start vLLM server (example for Qwen3-4B)
bash bfcl/common/scripts/run_vllm_qwen3_4b.sh

# Build teacher memory
bash bfcl/common/scripts/teacher_build_memory_multi_turn.sh

# Run student inference
bash bfcl/memp/scripts/run_memp_qwen3_4b.sh   # MEMP
bash bfcl/rb/scripts/run_rb_qwen3_4b.sh        # RB
bash bfcl/sasm/scripts/run_sasm_qwen3_4b.sh    # SASM
```

### ToolSandbox

```bash
export TOOLSANDBOX_ROOT=/path/to/ToolSandbox

# Build teacher memory and run student inference
bash toolsandbox/memp/scripts/run_memp_teacher.sh
bash toolsandbox/memp/scripts/run_memp_student.sh

bash toolsandbox/rb/scripts/run_rb_teacher.sh
bash toolsandbox/rb/scripts/run_rb_student.sh

bash toolsandbox/sasm/scripts/run_sasm_teacher.sh
bash toolsandbox/sasm/scripts/run_sasm_from_existing.sh
```

## Citation

If you use this code, please cite our papers:

```bibtex
@article{sasm2025,
  title={Structurally Aligned Subtask-Level Memory for Software Engineering Agents},
  author={},
  year={2025}
}

@article{memp2025,
  title={MEMP: Exploring Agent Procedural Memory},
  author={},
  year={2025}
}

@article{rb2025,
  title={ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory},
  author={},
  year={2025}
}
```
