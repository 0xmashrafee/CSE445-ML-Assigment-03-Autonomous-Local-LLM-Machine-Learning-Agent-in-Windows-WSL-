# CSE445-ML-Assigment-03-Autonomous-Local-LLM-Machine-Learning-Agent-in-Windows-WSL-
Developed an Autonomous Local LLM Machine Learning Agent for CSE445 using WSL2, Ollama, Python, PyTorch, and Scikit-Learn. Implemented a ReAct-based AI workflow that enables automated reasoning, ML tool selection, model training, evaluation, and data analysis through a privacy-preserving local AI system.

This repo covers all three tasks from the assignment: a local (Ollama-served)
LLM ReAct agent that drives Scikit-Learn and PyTorch tools.

## 0. What's in here

| File | Purpose |
|---|---|
| `ml_tools.py` | All 6 tools: 3 baseline (Task 1) + 3 advanced (Task 2) |
| `react_agent.py` | ReAct loop + Ollama client + self-correction (Tasks 1 & 3) |
| `benchmark_runner.py` | 3-algorithm x 2-dataset benchmark, ground truth + agent trace (Task 3) |
| `requirements.txt` | Python dependencies |
| `RUN-*.txt` | Saved execution traces |
| `CSE445_Assignment_3_Technical_Report.pdf` / `.docx` | Write-up |

## 1. One-time environment setup (run inside WSL2 Ubuntu, NOT Windows PowerShell)

Open **Windows PowerShell as Administrator** first, just to install WSL itself:

```powershell
wsl --install -d Ubuntu-22.04
wsl --set-default-version 2
```

Reboot if prompted, then open the **Ubuntu app** from the Start menu (this drops
you into a real Linux shell — everything below runs there, not in PowerShell).

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv curl build-essential git

mkdir -p ~/cse445_agent && cd ~/cse445_agent
python3 -m venv venv
source venv/bin/activate
```

Copy the project files into `~/cse445_agent/` — e.g. `cp` them from
`/mnt/c/Users/<you>/Downloads/` if you downloaded them on Windows first
(WSL mounts your Windows drives under `/mnt/c/`).

## 2. Install and serve the local LLM

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &                 # leave this running in the background
ollama pull llama3.2:3b        # ~2 GB download, one-time
```

Sanity check it's actually up before touching the agent:

```bash
curl http://127.0.0.1:11434/api/generate -d '{
  "model": "llama3.2:3b",
  "prompt": "Say OK.",
  "stream": false
}'
```

You should get back a JSON blob with a `"response"` field. If you get
`Connection refused`, `ollama serve` isn't running — open a second WSL
terminal tab and run it there.

## 3. Install Python dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
# NVIDIA GPU + CUDA instead of CPU-only torch:
# pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## 4. Run it

**Quick single-task smoke test:**
```bash
python react_agent.py --backend ollama --model llama3.2:3b \
  --task "Train a decision tree and a random forest on the wine dataset and compare them."
```

**The required execution log traces (5 built-in scenarios, one per file):**
```bash
python react_agent.py --backend ollama --model llama3.2:3b --demo --log-dir logs
```
This already writes a timestamped `.log` per scenario into `logs/` on its
own — no need to pipe it through `tee`. Rename/copy the ones you want into
your submission (that's where the `RUN-*.txt` files here came from).

**Full Task 3 benchmark (3 algorithms x 2 datasets + CV + Markdown table):**
```bash
python benchmark_runner.py --backend ollama --model llama3.2:3b
```
This writes `benchmark_report.md`, which includes both the agent's own
Final Answer and a ground-truth table recomputed straight from the tool
calls (not trusting the model's own arithmetic) — use the ground-truth
table to check the agent's numbers are actually correct.

**Custom query, for an extra trace or to stress self-correction:**
```bash
python react_agent.py --backend ollama --model llama3.2:3b \
  --task "Train a deep classifier on wine with batch_size=1, then fix it." \
  --log-dir logs
```
Deliberately asking for something like `batch_size=1` or a bad `lr` is a
good way to confirm the self-healing actually kicks in and recovers on the
next step — worth highlighting in the report.

## 5. Deliverables checklist

- [ ] `ml_tools.py`, `react_agent.py`, `benchmark_runner.py`, `requirements.txt`
- [ ] >= 3 saved multi-step execution traces (`RUN-*.txt`)
- [ ] Technical report (PDF/DOCX) covering local LLM architecture + prompt
      engineering, statistical comparison of the models (CV mean/std, not
      just one test split), and the ReAct loop/tool registry diagram
- [ ] `benchmark_report.md`

## 6. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `Connection refused` to `127.0.0.1:11434` | `ollama serve` not running | Run it in a separate terminal, or `nohup ollama serve &` |
| Model ignores the `Action:` / `Action Input:` format | 3B model is small | `temperature: 0.1` and the stop sequences are already set; try `mistral:7b` if you have >=8GB RAM |
| Agent answers with a Final Answer immediately, no tool calls | 3B model taking a shortcut on a dense prompt | Already guarded against — the controller refuses a Final Answer with zero real tool calls and pushes the model to try again |
| Agent loops repeating the same call | Small model stuck on one action | Already handled by the repetition guard — after 3 identical calls it stops itself and summarizes what it has |
| `NaN loss` from `train_deep_pytorch_classifier` / `train_pytorch_mlp` | Learning rate too high | Tool reports this as a structured error; agent should retry with a smaller `lr` — check this actually happens in your log |
| Request times out after a while | Slow CPU-only generation | Bump `--timeout` (default 300s) |
| WSL can't see your GPU | Needs NVIDIA's WSL-specific driver | Install "NVIDIA drivers for WSL" on the **Windows** side, not inside WSL — CPU-only torch is fine for these dataset sizes anyway |
