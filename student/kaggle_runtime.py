"""Small runtime helpers; GPU capacity is diagnostic, not an arbitrary cutoff."""
import subprocess
import sys
import warnings
import importlib
import importlib.metadata
import shutil
from pathlib import Path


def restore_saved_output(repo, source):
    """Restore a report snapshot with backups, while keeping checkpoint conflicts strict."""
    from student.resume import digest
    repo, source = Path(repo).resolve(), Path(source).resolve()
    if not (source / "work").is_dir():
        source = source / "lab-repo"
    if not (source / "work").is_dir():
        raise ValueError("RESUME_FROM must contain work/ or lab-repo/work/ from a saved Kaggle Output")
    session_files = {"report/runs/preflight.log.txt", "report/runs/optional_backends.json",
                     "report/runs/notebook_stages.json", "report/runs/status.json", "report/submission_check.txt"}
    candidates = []
    for folder in (source / "work/adapters", source / "work/resume", source / "report"):
        if folder.is_dir():
            candidates.extend(p for p in folder.rglob("*") if p.is_file() and not p.is_symlink())
    candidates.extend((source / "work").glob("*.jsonl"))
    copied = archived = 0
    for path in candidates:
        rel = path.relative_to(source)
        if path.name.endswith(".tmp") or not path.resolve().is_relative_to(source):
            continue
        if rel.parts[0] == "report" and path.suffix not in {".json", ".md", ".png", ".txt"}:
            continue
        target = repo / rel
        if not target.resolve().is_relative_to(repo):
            raise ValueError("Invalid resume artifact path")
        if target.exists():
            incoming_hash = digest(path)
            if digest(target) == incoming_hash:
                continue
            diagnostic = rel.as_posix() in session_files or (rel.parts[:2] == ("report", "runs") and path.name.endswith(".log.txt"))
            if not diagnostic and rel.parts[0] != "report":
                raise RuntimeError(f"Resume artifact conflicts with working output: {rel}; checkpoint/evidence was not overwritten")
            if not diagnostic:
                # environment.json, scores and run metadata belong to the saved snapshot.
                # Back up current report bytes, then restore the source bytes expected by its receipts.
                current_hash = digest(target)
                backup = repo / "work/resume/replaced-reports" / current_hash / rel
                if not backup.resolve().is_relative_to(repo):
                    raise ValueError("Invalid report backup path")
                if backup.exists():
                    if digest(backup) != current_hash:
                        raise RuntimeError(f"Corrupted report backup: {backup}")
                else:
                    backup.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(target, backup)
                shutil.copy2(path, target)
                archived += 1
                copied += 1
                continue
            # Keep the current session's log active and preserve the old one by content hash.
            target = repo / "work/resume/imported-logs" / incoming_hash / rel
            if not target.resolve().is_relative_to(repo):
                raise ValueError("Invalid diagnostic archive path")
            if target.exists():
                if digest(target) != incoming_hash:
                    raise RuntimeError(f"Corrupted diagnostic archive: {target}")
                continue
            archived += 1
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied += 1
    print(f"Restored {copied} artifact files; preserved {archived} conflicting reports/logs in work/resume/. Checkpoint compatibility is checked before reuse.")
    return copied


def configure_hf_token(secrets, environ):
    """Export the optional Kaggle secret before starting Hugging Face subprocesses."""
    try:
        token = secrets.get_secret("HF_TOKEN")
    except Exception:
        # Missing or ungranted secret: public model downloads can continue.
        # Never print exception details, which may contain authentication data.
        token = None
    if token and token.strip():
        environ["HF_TOKEN"] = token.strip()
    enabled = bool(environ.get("HF_TOKEN", "").strip())
    print("HF_TOKEN configured for model downloads." if enabled else
          "HF_TOKEN not configured; public model downloads will be unauthenticated. "
          "To authenticate, enable the HF_TOKEN secret in Kaggle Add-ons > Secrets.")
    return enabled


def prepare_optional_backends():
    """Remove old, unused TorchAO that PEFT probes while adapting dense ViT layers."""
    try:
        installed = importlib.metadata.version("torchao")
    except importlib.metadata.PackageNotFoundError:
        return {"torchao": "not_installed"}
    from packaging.version import Version
    if Version(installed) >= Version("0.16.0"):
        return {"torchao": installed, "action": "unchanged"}
    print(f"Removing unused torchao {installed}: PEFT 0.21.2 requires >=0.16.0 when probing this backend.", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=True)
    importlib.invalidate_caches()
    return {"torchao": installed, "action": "removed_incompatible_optional_backend", "quantization_backend": "bitsandbytes"}


def run_preflight_checks(repo):
    """Keep pytest output and surface the failure reason, not just its exit code."""
    repo = Path(repo)
    log_path = repo / "report/runs/preflight.log.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    command = [sys.executable, "-m", "pytest", "-q", "tests", "student/test_kaggle.py", "student/test_resume.py"]
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1)
        for line in process.stdout:
            lines.append(line)
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
        code = process.wait()
    if code:
        details = "".join(lines[-60:])
        hint = ""
        if "Regenerate with python student/make_notebook.py" in "".join(lines):
            hint = ("Notebook/generator mismatch: run python student/make_notebook.py LOCALLY, "
                    "commit and push lab_day10.ipynb together with student/make_notebook.py, "
                    "then re-import the updated notebook into Kaggle. ")
        raise RuntimeError(f"Preflight tests failed (exit {code}). {hint}"
                           f"Full log: {log_path}\n\n{details}")
    return log_path


def verify_dense_lora_backend():
    """Exercise E3's dense ViT LoRA dispatch in the actual installed environment."""
    code = '''import torch
from peft import LoraConfig, get_peft_model
class VisionLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = torch.nn.Linear(8, 8)
    def forward(self, x):
        return self.qkv(x)
model = get_peft_model(VisionLayer(), LoraConfig(r=2, lora_alpha=4, target_modules=["qkv"]))
model(torch.randn(2, 8)).square().mean().backward()
grads = [p.grad for n, p in model.named_parameters() if "lora_B" in n]
assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)
print("E3 dense LoRA backend: forward/backward passed")
'''
    # A fresh process sees packages installed/removed during setup, just like training.
    subprocess.run([sys.executable, "-u", "-c", code], check=True)
    return {"dense_lora_forward_backward": "passed"}


def report_gpu(torch):
    if not torch.cuda.is_available():
        raise RuntimeError("Enable a CUDA GPU in Kaggle Settings before running")
    devices = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        free, _ = torch.cuda.mem_get_info(index)
        info = {"index": index, "name": props.name,
                "total_gib": props.total_memory / 1024**3, "free_gib": free / 1024**3}
        devices.append(info)
        print(f"GPU {index}: {info['name']}; total={info['total_gib']:.2f} GiB; free={info['free_gib']:.2f} GiB")
    print("Experiments use GPU 0, batch_size=1 and grad_accum=8; GPU capacities are not combined.")
    if devices[0]["total_gib"] < 14:
        warnings.warn("GPU 0 has less than 14 GiB. E2/E3 may run out of memory; select a T4/P100 or larger GPU.")
    return devices
