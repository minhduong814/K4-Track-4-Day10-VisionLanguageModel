"""Restore Kaggle artifacts and reuse only compatible, complete stages."""
import hashlib
import importlib.metadata
import json
import re
import shutil
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, sort_keys=True, indent=2), encoding="utf-8")
    tmp.replace(path)


def label(stage, name=None):
    value = stage + ("_" + name if name else "")
    if not re.fullmatch(r"[a-z0-9_-]+", value):
        raise ValueError("Invalid stage name")
    return value


def stage_signature(repo, stage, name=None, batch_size=1, grad_accum=8):
    repo = Path(repo)
    files = ["student/pipeline.py", "student/resume.py", "requirements-colab.txt", "data/splits/meta.json"]
    if stage == "train" and name and not name.startswith("e4_"):
        files.append(f"configs/{name}.yaml")
    if name and name.startswith("e4_"):
        files.append("report/runs/e4_selection.json")
        selection = repo / "report/runs/e4_selection.json"
        if selection.exists():
            base = json.loads(selection.read_text(encoding="utf-8"))["base"]
            files.append(f"report/runs/{base}.json")
    if stage == "evaluate" and name:
        files.append(f"work/resume/stages/train_{name}.json")
    if stage == "select-e4":
        files += [f"report/validation/e{i}.json" for i in (1, 2, 3)]
    if stage == "summary":
        files += [f"report/evidence/{n}.json" for n in ("e0", "e1", "e2", "e3", "e4_rank_4", "e4_rank_64")]
        files.append("report/runs/e4_selection.json")
    runtime = {}
    for package in ("torch", "transformers", "peft", "bitsandbytes", "accelerate"):
        try:
            runtime[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            runtime[package] = None
    value = {"format": 1, "stage": stage, "name": name, "batch_size": batch_size, "grad_accum": grad_accum, "runtime": runtime,
             "files": {f: digest(repo / f) if (repo / f).is_file() else None for f in files}}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def artifacts(repo, stage, name=None):
    repo = Path(repo)
    if stage == "e0":
        return ["work/e0_test.jsonl", "report/evidence/e0.json", "report/runs/splits.json", "report/runs/environment.json"]
    if stage == "train":
        base = f"work/adapters/{name}/final"
        files = [f"{base}/adapter_config.json", f"{base}/adapter_model.safetensors", f"report/runs/{name}.json",
                 f"report/runs/{name}_setup.json", f"report/runs/{name}_loss.json"]
        run = repo / f"report/runs/{name}.json"
        if run.exists() and json.loads(run.read_text(encoding="utf-8"))["config"]["train_projector"]:
            files.append(f"{base}/projector.pt")
        return files
    if stage == "evaluate":
        files = [f"work/{name}_test.jsonl", f"report/evidence/{name}.json"]
        if not name.startswith("e4_"):
            files += [f"work/{name}_val.jsonl", f"report/validation/{name}.json"]
        return files
    if stage == "select-e4":
        return ["report/runs/e4_selection.json"]
    # Cheap previews and summary are regenerated, preserving visible notebook output.
    return []


def reusable_stage(repo, stage, name=None, batch_size=1, grad_accum=8):
    repo = Path(repo)
    expected = artifacts(repo, stage, name)
    marker = repo / f"work/resume/stages/{label(stage, name)}.json"
    if not expected or not marker.exists():
        return False
    try:
        saved = json.loads(marker.read_text(encoding="utf-8"))
        if saved["signature"] != stage_signature(repo, stage, name, batch_size, grad_accum):
            return False
        if set(saved["artifacts"]) != set(expected):
            return False
        return all((repo / f).is_file() and digest(repo / f) == saved["artifacts"][f] for f in expected)
    except (ValueError, KeyError, OSError):
        return False


def record_stage(repo, stage, name=None, batch_size=1, grad_accum=8):
    repo = Path(repo)
    files = artifacts(repo, stage, name)
    if not files:
        return
    for f in files:
        if not (repo / f).is_file() or not (repo / f).stat().st_size:
            raise RuntimeError(f"Stage completed but required artifact is missing: {f}")
    atomic_json(repo / f"work/resume/stages/{label(stage, name)}.json", {
        "signature": stage_signature(repo, stage, name, batch_size, grad_accum),
        "artifacts": {f: digest(repo / f) for f in files}})


def restore_output(repo, source):
    """Copy artifacts from a read-only attached Notebook Output, never its code/.git."""
    repo, source = Path(repo).resolve(), Path(source).resolve()
    if not (source / "work").is_dir():
        source = source / "lab-repo"
    if not (source / "work").is_dir():
        raise ValueError("RESUME_FROM must contain work/ or lab-repo/work/ from a saved Kaggle Output")
    copied = 0
    candidates = []
    for folder in (source / "work/adapters", source / "work/resume", source / "report"):
        if folder.is_dir():
            candidates.extend(p for p in folder.rglob("*") if p.is_file() and not p.is_symlink())
    candidates.extend((source / "work").glob("*.jsonl"))
    for path in candidates:
        rel = path.relative_to(source)
        if rel.parts[0] == "report" and path.suffix not in {".json", ".md", ".png", ".txt"}:
            continue
        if path.name.endswith(".tmp"):
            continue
        if not path.resolve().is_relative_to(source):
            continue
        target = repo / rel
        if not target.resolve().is_relative_to(repo):
            raise ValueError("Invalid resume artifact path")
        if target.exists():
            if digest(target) != digest(path):
                raise RuntimeError(f"Resume artifact conflicts with working output: {rel}; start a fresh session")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied += 1
    print(f"Restored {copied} artifact files from {source}; compatibility is checked before reuse.")
    return copied


class BatchStream:
    """Shuffled batch cursor that can resume without re-encoding old images."""
    def __init__(self, dataset, batch_size, collate, new_order):
        self.dataset, self.batch_size, self.collate, self.new_order = dataset, batch_size, collate, new_order
        self.order, self.cursor = new_order(), 0

    def __next__(self):
        if self.cursor >= len(self.order):
            self.order, self.cursor = self.new_order(), 0
        indices = self.order[self.cursor:self.cursor+self.batch_size]
        batch = self.collate([self.dataset[i] for i in indices])
        self.cursor += len(indices)
        return batch

    def state_dict(self):
        return {"order": self.order, "cursor": self.cursor}

    def load_state_dict(self, state):
        if sorted(state["order"]) != list(range(len(self.dataset))) or not 0 <= state["cursor"] <= len(self.dataset):
            raise ValueError("Checkpoint batch order/cursor does not match the dataset")
        self.order, self.cursor = state["order"], state["cursor"]


def compatible_checkpoint(folder, signature):
    matches = []
    for marker in Path(folder).glob("resume-step-*/checkpoint.json"):
        try:
            data = json.loads(marker.read_text(encoding="utf-8"))
            state = marker.parent / "trainer.pt"
            if data["signature"] == signature and state.is_file() and digest(state) == data["sha256"]:
                matches.append((data["step"], marker.parent, data))
        except (OSError, ValueError, KeyError):
            pass
    return max(matches, key=lambda item: item[0]) if matches else None


def prune_checkpoints(folder, keep=2):
    root = Path(folder).resolve()
    paths = sorted((p for p in root.glob("resume-step-*") if (p / "checkpoint.json").is_file()),
                   key=lambda p: int(p.name.rsplit("-", 1)[1]), reverse=True)
    for path in paths[keep:]:
        absolute = path.resolve()
        if absolute.parent != root or not re.fullmatch(r"resume-step-\d+", absolute.name):
            raise ValueError("Unsafe checkpoint cleanup path")
        shutil.rmtree(absolute)
