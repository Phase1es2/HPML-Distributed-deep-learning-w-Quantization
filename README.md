# HPML HW4 — Distributed Deep Learning

## Setup

### Install uv

```bash
pip install uv
```

If `uv` is not found after installation, add pip's user bin directory to your PATH:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

To make this permanent, add the line above to your `~/.bashrc` or `~/.zshrc` and reload:

```bash
source ~/.bashrc   # or source ~/.zshrc
```

### Configure uv environment

```bash
# Pin to Python 3.13 (required for PyTorch compatibility)
uv python pin 3.13

# Install all dependencies (PyTorch cu124 + torchvision)
uv sync
```

---

## Running Experiments

### Q1 — Computational Efficiency w.r.t Batch Size (single GPU)

Sweeps batch sizes 32 * factor of 4 on a single GPU and reports data I/O time, training time, and total time per epoch.

```bash
uv run q1.py
```

---

### Q2 — Speedup Measurement (multi-GPU DDP)

Runs the same batch size sweep on 1, 2 or 4 GPUs using DistributedDataParallel and reports timing for speedup analysis.

```bash
# 2 GPUs
uv run torchrun --nproc_per_node=2 q2.py

# 4 GPUs
uv run torchrun --nproc_per_node=4 q2.py
```

---

### Q3 — Computation vs Communication (multi-GPU DDP)

Reports compute time and communication (all-reduce) time separately per batch size, plus bandwidth utilization.

```bash
# 2 GPUs
uv run torchrun --nproc_per_node=2 q3.py

# 4 GPUs
uv run torchrun --nproc_per_node=4 q3.py
```

---

### Q4 — Large Batch Training (5 epochs, 4 GPUs)

Trains for 5 epochs using the largest batch size per GPU found in Q1 on 4 GPUs and reports loss and accuracy at epoch 5.

```bash
# Replace <batch_size> with the largest batch size from Q1 (e.g. 1024)
uv run torchrun --nproc_per_node=4 q4.py --batch-size <batch_size>
```
