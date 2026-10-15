import json
import random
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from student.resume import (BatchStream, atomic_json, compatible_checkpoint, digest, prune_checkpoints,
                            record_stage, restore_output, reusable_stage, stage_signature)
from student.kaggle_runtime import restore_saved_output


def fixture_repo(root):
    for name, text in {"student/pipeline.py": "# code", "student/resume.py": "# resume",
                       "requirements-colab.txt": "# deps", "data/splits/meta.json": '{"dataset_version":"test"}',
                       "configs/e1.yaml": "name: E1"}.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return root


def completed_e0(root):
    for name in ("work/e0_test.jsonl", "report/evidence/e0.json", "report/runs/splits.json", "report/runs/environment.json"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{}\n')
    record_stage(root, "e0")


def test_stage_cache_rejects_changed_code_data_or_corrupted_artifact(tmp_path):
    root = fixture_repo(tmp_path)
    completed_e0(root)
    assert reusable_stage(root, "e0")
    (root / "student/kaggle_runtime.py").write_text("# remove unused old torchao")
    assert reusable_stage(root, "e0")  # Runtime-only backend cleanup must not discard E0/E1/E2 receipts.
    assert not reusable_stage(root, "e0", batch_size=2)
    prediction = root / "work/e0_test.jsonl"
    prediction.write_text("corrupted")
    assert not reusable_stage(root, "e0")
    prediction.write_text('{}\n')
    source = root / "student/pipeline.py"
    source.write_text("# fixed quantization")
    assert not reusable_stage(root, "e0")
    source.write_text("# code")
    (root / "data/splits/meta.json").write_text('{"dataset_version":"different"}')
    assert not reusable_stage(root, "e0")


def test_restore_notebook_output_preserves_work_and_report_but_not_code_or_git(tmp_path):
    source = fixture_repo(tmp_path / "output/lab-repo")
    completed_e0(source)
    (source / ".git").mkdir()
    (source / ".git/token").write_text("must never be copied")
    dest = fixture_repo(tmp_path / "working")
    assert restore_output(dest, source.parent) > 0
    assert reusable_stage(dest, "e0")
    assert not (dest / ".git").exists()
    assert restore_output(dest, source) == 0
    (dest / "work/e0_test.jsonl").write_text("new output")
    with pytest.raises(RuntimeError, match="conflicts"):
        restore_output(dest, source)


def test_restore_archives_session_log_conflicts_and_reuses_completed_stage(tmp_path):
    source = fixture_repo(tmp_path / "source")
    completed_e0(source)
    dest = fixture_repo(tmp_path / "dest")
    for name in ("preflight.log.txt", "optional_backends.json", "status.json", "notebook_stages.json", "train_e2.log.txt"):
        old, current = source / "report/runs" / name, dest / "report/runs" / name
        current.parent.mkdir(parents=True, exist_ok=True)
        old.write_text("old diagnostic")
        current.write_text("current diagnostic")
    assert restore_saved_output(dest, source) > 0
    assert reusable_stage(dest, "e0")
    for name in ("preflight.log.txt", "optional_backends.json", "status.json", "notebook_stages.json", "train_e2.log.txt"):
        assert (dest / "report/runs" / name).read_text() == "current diagnostic"
        archived = list((dest / "work/resume/imported-logs").rglob(name))
        assert len(archived) == 1 and archived[0].read_text() == "old diagnostic"
    assert restore_saved_output(dest, source) == 0


def test_restore_still_rejects_conflicting_model_weights(tmp_path):
    source = fixture_repo(tmp_path / "source")
    dest = fixture_repo(tmp_path / "dest")
    rel = "work/adapters/e2/final/projector.pt"
    for root, value in [(source, b"old weights"), (dest, b"new weights")]:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
    with pytest.raises(RuntimeError, match="checkpoint/evidence was not overwritten"):
        restore_saved_output(dest, source)
    assert (dest / rel).read_bytes() == b"new weights"


def test_restore_recovers_entire_report_snapshot_and_keeps_prior_report_bytes(tmp_path):
    source = fixture_repo(tmp_path / "source")
    completed_e0(source)
    dest = fixture_repo(tmp_path / "dest")
    restore_saved_output(dest, source)
    for name in ("report/runs/environment.json", "report/runs/splits.json", "report/evidence/e0.json"):
        (dest / name).write_text('{"interrupted_current_run":true}')
    assert not reusable_stage(dest, "e0")
    assert restore_saved_output(dest, source) > 0
    assert reusable_stage(dest, "e0")
    for name in ("report/runs/environment.json", "report/runs/splits.json", "report/evidence/e0.json"):
        assert (dest / name).read_bytes() == (source / name).read_bytes()
        backups = list((dest / "work/resume/replaced-reports").glob("*/" + name))
        assert len(backups) == 1 and backups[0].read_text() == '{"interrupted_current_run":true}'
    assert restore_saved_output(dest, source) == 0


def test_checkpoint_selection_ignores_incomplete_or_incompatible_snapshots(tmp_path):
    for step, signature in [(50, "same"), (100, "same"), (150, "different")]:
        folder = tmp_path / f"resume-step-{step}"
        folder.mkdir()
        (folder / "trainer.pt").write_bytes(str(step).encode())
        atomic_json(folder / "checkpoint.json", {"signature": signature, "step": step, "sha256": digest(folder / "trainer.pt")})
    assert compatible_checkpoint(tmp_path, "same")[0] == 100
    (tmp_path / "resume-step-100/trainer.pt").write_bytes(b"truncated")
    assert compatible_checkpoint(tmp_path, "same")[0] == 50
    (tmp_path / "resume-step-200").mkdir()
    assert compatible_checkpoint(tmp_path, "same")[0] == 50
    prune_checkpoints(tmp_path, keep=2)
    assert not (tmp_path / "resume-step-50").exists()
    assert (tmp_path / "resume-step-200").exists()  # Incomplete directories are not treated as valid snapshots.


def test_batch_cursor_resumes_across_epoch_boundary_without_reading_old_items():
    rng = random.Random(42)
    reads = []
    class Dataset:
        def __len__(self):
            return 7
        def __getitem__(self, i):
            reads.append(i)
            return i
    def order():
        indices = list(range(7))
        rng.shuffle(indices)
        return indices
    stream = BatchStream(Dataset(), 2, lambda x: x, order)
    for _ in range(4):
        next(stream)
    saved, rng_state = stream.state_dict(), rng.getstate()
    expected = [next(stream) for _ in range(6)]
    restored = BatchStream(Dataset(), 2, lambda x: x, order)
    restored.load_state_dict(saved)
    rng.setstate(rng_state)
    reads.clear()
    assert [next(restored) for _ in range(6)] == expected
    assert len(reads) == sum(map(len, expected))


def test_training_resume_matches_uninterrupted_optimizer_and_rng(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file, load_file
    from student.test_kaggle import load_pipeline_functions
    cfg = {"seed": 42, "batch_size": 2, "grad_accum": 2, "max_steps": 60, "lr": .001, "train_projector": False}
    full = fixture_repo(tmp_path / "full")
    resumed = fixture_repo(tmp_path / "resumed")
    failure = [False]

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(2))
            self.config = SimpleNamespace(image_token_id=1)
            self._lab_setup_audit = {"test": True}
            self.calls = 0
        def forward(self, input_ids, labels, use_cache):
            assert use_cache is False
            self.calls += 1
            if failure[0] and self.calls == 101:
                raise RuntimeError("simulated interruption after step 50")
            noise = torch.rand(()) + random.random()
            return SimpleNamespace(loss=(self.weight.mean()*noise-labels.float().mean()).square())

    class Dataset:
        def __init__(self, *args):
            pass
        def __len__(self):
            return 7
        def __getitem__(self, i):
            return {"input_ids": torch.ones((1, 220), dtype=torch.long), "labels": torch.tensor([[i]])}

    def collate(batch, pad):
        return {key: torch.cat([b[key] for b in batch]) for key in batch[0]}
    def save(model, path, processor):
        path.mkdir(parents=True, exist_ok=True)
        (path / "adapter_config.json").write_text('{}')
        save_file({"weight": model.weight.detach().clone()}, str(path / "adapter_model.safetensors"))

    original_to = torch.Tensor.to
    def cpu_to(self, *args, **kwargs):
        if args and args[0] == "cuda:0":
            args = ("cpu", *args[1:])
        return original_to(self, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, "to", cpu_to)
    class CPUTraining:
        cuda = SimpleNamespace(manual_seed_all=lambda _: None, reset_peak_memory_stats=lambda: None,
            synchronize=lambda: None, max_memory_allocated=lambda: 1024**3,
            get_rng_state=lambda _: torch.get_rng_state(), set_rng_state=lambda state, _: torch.set_rng_state(state))
        amp = SimpleNamespace(GradScaler=lambda _: torch.amp.GradScaler("cpu"))
        autocast = staticmethod(lambda *args, **kw: nullcontext())
        def __getattr__(self, name):
            return getattr(torch, name)
    import time
    import math
    from student.resume import atomic_json
    def write_json(path, data):
        atomic_json(path, data)
    ns = load_pipeline_functions({"train", "optimizer_step"}, dict(torch=CPUTraining(), random=random, time=time, math=math,
        json=json, ROOT=full, BatchStream=BatchStream, Examples=Dataset, rows=lambda _: [], collate=collate,
        load_base=Model, setup=lambda m, _: m, save_model=save, write_json=write_json,
        projector=lambda _: False, digest=digest, atomic_json=atomic_json, stage_signature=stage_signature,
        compatible_checkpoint=compatible_checkpoint, prune_checkpoints=prune_checkpoints,
        record_stage=record_stage, reusable_stage=reusable_stage, gc=SimpleNamespace(collect=lambda: None)))
    ns["torch"].cuda.empty_cache = lambda: None
    processor = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0, decode=lambda _: "[]"))
    ns["train"](cfg, "e1", processor)
    ns["ROOT"] = resumed
    failure[0] = True
    with pytest.raises(RuntimeError, match="simulated interruption"):
        ns["train"](cfg, "e1", processor)
    assert compatible_checkpoint(resumed / "work/adapters/e1", stage_signature(resumed, "train", "e1", 2, 2))[0] == 50
    failure[0] = False
    ns["train"](cfg, "e1", processor)
    expected = load_file(str(full / "work/adapters/e1/final/adapter_model.safetensors"))["weight"]
    actual = load_file(str(resumed / "work/adapters/e1/final/adapter_model.safetensors"))["weight"]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert reusable_stage(resumed, "train", "e1", 2, 2)
    assert not list((resumed / "work/adapters/e1").glob("resume-step-*"))
