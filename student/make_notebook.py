"""Generate the student Kaggle notebook without editing instructor scripts."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build_notebook():
    cells = []
    def cell(kind, source):
        value = {"cell_type": kind, "id": f"lab-{len(cells)}", "metadata": {},
                 "source": source.strip("\n").splitlines(True)}
        if kind == "code":
            value.update(execution_count=None, outputs=[])
        cells.append(value)

    cell("markdown", """# Lab Day 10 — Kaggle E0–E4
Push this implementation to GitHub first, then import this notebook into Kaggle.
Enable Internet and GPU, attach the road-damage dataset, and grant access to GITHUB_TOKEN.
Optionally enable the HF_TOKEN secret for authenticated Hugging Face downloads.
Choose **Save Version → Save & Run All**. Each section below contains executable cells.
For a complete new run, start a fresh Kaggle session and keep `RESUME_FROM = None`.
Setup fixes the optional TorchAO conflict and checks E3 dense LoRA before any experiment.
Training functions live in `student/pipeline.py`; the notebook calls each step separately and displays its logs.
E5 is excluded. Complete REPORT.md locally after collecting real results. See KAGGLE.md.
""")
    cell("markdown", "## 0. Environment, clone and GPU diagnostics")
    cell("code", '''import os, sys, subprocess, json, urllib.request
from pathlib import Path
from datetime import datetime, timezone
from kaggle_secrets import UserSecretsClient

REPO_URL = "https://github.com/minhduong814/K4-Track-4-Day10-VisionLanguageModel.git"
SOURCE_REF = "main"
RESULT_BRANCH = "kaggle-results-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
REPO = Path("/kaggle/working/lab-repo")
BATCH_SIZE, GRAD_ACCUM = 1, 8
RESUME_FROM = None  # Full new E0–E4 run. Leave unchanged; no previous Output needed.
FAIL_VERSION_ON_ERROR = False  # Keep partial Output when a stage fails; status.json still says failed.
GITHUB_TOKEN = UserSecretsClient().get_secret("GITHUB_TOKEN")
request = urllib.request.Request("https://api.github.com/repos/minhduong814/K4-Track-4-Day10-VisionLanguageModel",
    headers={"Authorization": "Bearer " + GITHUB_TOKEN, "Accept": "application/vnd.github+json"})
with urllib.request.urlopen(request, timeout=30) as response:
    permissions = json.load(response).get("permissions", {})
assert permissions.get("push"), "GitHub token must have write access to this repository"
del request
DATASET_ROOTS = [p.parent for p in Path("/kaggle/input").rglob("labels-YOLO")
                 if p.is_dir() and (p.parent / "images").is_dir()]
assert len(DATASET_ROOTS) == 1, "Attach exactly one dataset with images/ and labels-YOLO/"
os.environ["HF_HOME"] = "/tmp/lab-hf-cache"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
if not REPO.exists():
    subprocess.run(["git", "clone", "--branch", SOURCE_REF, REPO_URL, str(REPO)], check=True)
else:
    # Permit rerunning setup after a preflight failure without deleting the clone.
    origin = subprocess.check_output(["git", "remote", "get-url", "origin"], cwd=REPO, text=True).strip()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=REPO, text=True).strip()
    changes = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO, text=True)
    assert origin.rstrip("/").removesuffix(".git") == REPO_URL.removesuffix(".git"), "Existing clone belongs to another repo"
    assert branch == SOURCE_REF and not changes, "Start a fresh session: existing clone has another branch or edited tracked files"
    subprocess.run(["git", "fetch", "origin", SOURCE_REF], cwd=REPO, check=True)
    subprocess.run(["git", "merge", "--ff-only", "FETCH_HEAD"], cwd=REPO, check=True)
os.chdir(REPO)
sys.path.insert(0, str(REPO))
assert (REPO / "student/kaggle_runtime.py").exists(), "Push the updated student code to GitHub first"
from student.kaggle_runtime import configure_hf_token, report_gpu, run_preflight_checks, prepare_optional_backends, restore_saved_output, verify_dense_lora_backend
from student.resume import reusable_stage, record_stage
if RESUME_FROM:
    restore_saved_output(REPO, RESUME_FROM)
HF_AUTH_ENABLED = configure_hf_token(UserSecretsClient(), os.environ)
subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements-colab.txt", "pytest"], check=True)
BACKEND_INFO = prepare_optional_backends()
BACKEND_INFO.update(verify_dense_lora_backend())
(REPO / "report/runs").mkdir(parents=True, exist_ok=True)
(REPO / "report/runs/optional_backends.json").write_text(json.dumps(BACKEND_INFO, indent=2), encoding="utf-8")
subprocess.run(["nvidia-smi", "-L"], check=True)
import torch
GPU_INFO = report_gpu(torch)  # No hard 15 GiB cutoff: show actual device capacity.
run_preflight_checks(REPO)
print("Preflight passed; result branch:", RESULT_BRANCH)
''')
    cell("code", '''# Each subprocess owns and releases its CUDA model; failures skip later experiments.
# The final cell preserves partial Output and prominently reports failure status.
pipeline_error = None
stage_history = []

def run_stage(stage, name=None):
    global pipeline_error
    if pipeline_error is not None:
        print(f"SKIP {stage} {name or ''}: an earlier stage failed")
        return False
    label = stage + ("_" + name if name else "")
    command = [sys.executable, "-u", "student/pipeline.py", "--stage", stage,
               "--batch-size", str(BATCH_SIZE), "--grad-accum", str(GRAD_ACCUM)]
    if name:
        command += ["--name", name]
    folder = REPO / "report/runs"
    folder.mkdir(parents=True, exist_ok=True)
    print(f"\\n===== START {label} =====", flush=True)
    try:
        if reusable_stage(REPO, stage, name, BATCH_SIZE, GRAD_ACCUM):
            stage_history.append({"stage": label, "status": "reused"})
            print(f"REUSE {label}: matching code/config/data and verified artifacts", flush=True)
            return True
        with (folder / f"{label}.log.txt").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=REPO, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            code = process.wait()
        if code:
            raise RuntimeError(f"{label} exited {code}; see report/runs/{label}.log.txt")
        record_stage(REPO, stage, name, BATCH_SIZE, GRAD_ACCUM)
        stage_history.append({"stage": label, "status": "completed"})
        print(f"===== DONE {label} =====", flush=True)
        return True
    except Exception as exc:
        pipeline_error = exc
        stage_history.append({"stage": label, "status": "failed", "error": str(exc)})
        (folder / "status.json").write_text(json.dumps({"pipeline": "failed", "stages": stage_history}, indent=2), encoding="utf-8")
        print("FAILED:", str(exc), flush=True)
        return False
    finally:
        (folder / "notebook_stages.json").write_text(json.dumps(stage_history, indent=2), encoding="utf-8")
''')
    cell("markdown", "## 1. Prepare data — fixed splits by recording session")
    cell("code", '''subprocess.run([sys.executable, "scripts/prepare_data.py", "--source", str(DATASET_ROOTS[0])], check=True)
split_meta = json.loads((REPO / "data/splits/meta.json").read_text(encoding="utf-8"))
print("Dataset version:", split_meta["dataset_version"])
print(json.dumps(split_meta["counts"], indent=2))
''')
    cell("markdown", "## 2. Convert YOLO labels to JSON on the 0–1000 scale\nPrint the actual prompt and ground truth; draw the converted boxes on a training image.")
    cell("code", '''if run_stage("labels"):
    from IPython.display import display, Image as NotebookImage
    display(NotebookImage(filename=str(REPO / "report/figures/train_ground_truth.png")))
''')
    cell("markdown", "## 3. E0 — zero-shot inference and original test scorer\nNF4 LLM; vision and lm_head remain unquantized. Predictions: work/e0_test.jsonl.")
    cell("code", 'run_stage("e0")')
    cell("markdown", "## 4. Encode chat examples and verify answer-only loss\nCheck the prompt prefix, 220 image tokens, and decode exactly the supervised JSON answer.")
    cell("code", 'run_stage("mask")')
    cell("markdown", "## 5. Configure E1/E2/E3 and inspect LoRA targets\nE1: LLM LoRA. E2: add full fp32 projector training. E3: add ViT LoRA. Training asserts 252/348 matched modules.")
    cell("code", 'run_stage("configs")')
    cell("markdown", "## 6. Train E1, E2 and E3\nEach cell runs 300 optimizer steps, batch 1 × accumulation 8, warmup/cosine and finite loss/gradient checks. Save resumable optimizer/model/RNG/batch checkpoints every 50 successful steps. Compatible completed setups are reused; interrupted training resumes at its last complete checkpoint. Final artifacts contain LoRA plus full projector weights for E2/E3.")
    for name in ("e1", "e2", "e3"):
        cell("code", f'# Train {name.upper()} and record real VRAM/time/loss.\nrun_stage("train", "{name}")')
    cell("markdown", "## 7. Reload adapters/projectors, infer and score E1–E3\nEach setup is evaluated on val and test. E4 base selection uses val only. Evidence uses the unmodified scripts/score.py.")
    for name in ("e1", "e2", "e3"):
        cell("code", f'run_stage("evaluate", "{name}")')
    cell("markdown", "## 8. E4 rank ablation, report artifacts and GitHub push\nChoose the base using validation F1, then change only rank (alpha=2×rank): 4 and 64. E5 is excluded.")
    cell("code", 'run_stage("select-e4")')
    for rank in (4, 64):
        cell("code", f'if run_stage("train", "e4_rank_{rank}"):\n    run_stage("evaluate", "e4_rank_{rank}")')
    cell("code", '''if run_stage("summary"):
    from IPython.display import display, Markdown
    display(Markdown((REPO / "report/RESULTS.md").read_text(encoding="utf-8")))
    print((REPO / "report/submission_check.txt").read_text(encoding="utf-8"))
''')
    cell("code", '''# Push only report artifacts. Weights, data and secrets stay out of GitHub.
from student.github_sync import push_results
if not GITHUB_TOKEN:
    GITHUB_TOKEN = UserSecretsClient().get_secret("GITHUB_TOKEN")
push_error = None
try:
    push_results(REPO, GITHUB_TOKEN, RESULT_BRANCH)
except Exception as exc:
    push_error = exc
    print("GitHub push failed; download report/ from Kaggle Output.")
finally:
    GITHUB_TOKEN = None
if pipeline_error is not None:
    print("PIPELINE FAILED:", pipeline_error)
    print("Partial results/checkpoints are kept in Kaggle Output. Attach this Output as Input in the next version and set RESUME_FROM.")
elif push_error is not None:
    print("Pipeline finished; GITHUB PUSH FAILED. Download report/ from Kaggle Output.")
else:
    print("Completed. Pull branch", RESULT_BRANCH, "at local to finish REPORT.md.")
if FAIL_VERSION_ON_ERROR:
    if pipeline_error is not None:
        raise pipeline_error
    if push_error is not None:
        raise push_error
''')
    return {"cells": cells, "metadata": {"accelerator": "GPU",
            "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
            "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


if __name__ == "__main__":
    (ROOT / "lab_day10.ipynb").write_text(json.dumps(build_notebook(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
