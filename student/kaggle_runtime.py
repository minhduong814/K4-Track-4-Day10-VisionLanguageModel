"""Small runtime helpers; GPU capacity is diagnostic, not an arbitrary cutoff."""
import subprocess
import sys
import warnings
from pathlib import Path


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
