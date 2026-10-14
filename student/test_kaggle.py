"""CPU-only regression checks for Kaggle capacity and stage orchestration."""
import argparse
import ast
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from student.kaggle_runtime import report_gpu
from student.make_notebook import build_notebook


def fake_torch(capacities):
    return SimpleNamespace(cuda=SimpleNamespace(
        is_available=lambda: bool(capacities), device_count=lambda: len(capacities),
        get_device_properties=lambda i: SimpleNamespace(name="Tesla T4", total_memory=int(capacities[i]*1024**3)),
        mem_get_info=lambda i: (int((capacities[i]-1)*1024**3), int(capacities[i]*1024**3))))


def test_t4_below_15_gib_is_accepted(capsys):
    devices = report_gpu(fake_torch([14.75, 14.75]))
    assert len(devices) == 2
    assert devices[0]["total_gib"] == 14.75
    assert "GPU 0" in capsys.readouterr().out


def test_small_gpu_warns_but_does_not_abort():
    with pytest.warns(UserWarning, match="less than 14 GiB"):
        assert report_gpu(fake_torch([8]))[0]["total_gib"] == 8


def test_missing_cuda_has_actionable_error():
    with pytest.raises(RuntimeError, match="Enable a CUDA GPU"):
        report_gpu(fake_torch([]))


def test_notebook_code_compiles_and_every_section_executes():
    notebook = build_notebook()
    saved = json.loads((Path(__file__).resolve().parents[1] / "lab_day10.ipynb").read_text(encoding="utf-8"))
    assert saved == notebook, "Regenerate with python student/make_notebook.py"
    sections = {}
    current = None
    for cell in notebook["cells"]:
        source = "".join(cell["source"])
        if cell["cell_type"] == "markdown" and source.startswith("## "):
            current = source.split(".", 1)[0]
            sections[current] = 0
        if cell["cell_type"] == "code":
            compile(source, "notebook cell", "exec")
            sections[current] += 1
    assert len(sections) == 9 and all(sections.values())


def test_failed_stage_skips_later_training_and_pushes_partial_results(tmp_path, monkeypatch):
    cells = build_notebook()["cells"]
    runner = next("".join(c["source"]) for c in cells if "def run_stage" in "".join(c["source"]))
    commands = []
    def popen(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(stdout=iter(["CUDA out of memory\n"]), wait=lambda: 1)
    ns = dict(REPO=tmp_path, BATCH_SIZE=1, GRAD_ACCUM=8, sys=sys, json=json,
              subprocess=SimpleNamespace(Popen=popen, PIPE=-1, STDOUT=-2))
    exec(runner, ns)
    assert not ns["run_stage"]("train", "e2")
    assert not ns["run_stage"]("train", "e3")
    assert len(commands) == 1
    status = json.loads((tmp_path / "report/runs/status.json").read_text())
    assert status["pipeline"] == "failed" and status["stages"][0]["stage"] == "train_e2"
    from student import github_sync
    pushed = []
    monkeypatch.setattr(github_sync, "push_results", lambda *args: pushed.append(args))
    ns.update(GITHUB_TOKEN="test-only-placeholder", RESULT_BRANCH="kaggle-results-test")
    with pytest.raises(RuntimeError, match="train_e2 exited 1"):
        exec("".join(cells[-1]["source"]), ns)
    assert len(pushed) == 1 and ns["GITHUB_TOKEN"] is None


def load_pipeline_functions(names, namespace):
    path = Path(__file__).with_name("pipeline.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def test_e4_selects_val_only_and_changes_only_rank_alpha(tmp_path):
    for folder in ("report/validation", "report/runs"):
        (tmp_path / folder).mkdir(parents=True)
    for name, f1 in [("e1", .2), ("e2", .4), ("e3", .3)]:
        (tmp_path / f"report/validation/{name}.json").write_text(json.dumps({"micro": {"f1": f1}}))
    base = {"batch_size": 1, "grad_accum": 8, "max_steps": 300, "seed": 42,
            "train_projector": True, "lora": {"r": 16, "alpha": 32, "dropout": .05, "targets": "llm"}}
    (tmp_path / "report/runs/e2.json").write_text(json.dumps({"config": base}))
    def write_json(path, data):
        path.write_text(json.dumps(data))
    ns = load_pipeline_functions({"select_e4", "experiment_config"},
        dict(ROOT=tmp_path, json=json, copy=copy, yaml=yaml, write_json=write_json))
    assert ns["select_e4"]() == "e2"
    for rank in (4, 64):
        cfg = ns["experiment_config"](f"e4_rank_{rank}", 9, 99)
        assert cfg["lora"]["r"] == rank and cfg["lora"]["alpha"] == 2*rank
        cfg["lora"].update(r=16, alpha=32)
        assert cfg == base  # Even a changed CLI batch size must not contaminate ablation.


@pytest.mark.parametrize("stage,expected", [("train", "train"), ("evaluate", "evaluate_trained"),
    ("mask", "check_mask"), ("select-e4", "select_e4"), ("summary", "summarize")])
def test_cli_stages_do_not_accidentally_run_entire_pipeline(stage, expected, monkeypatch):
    calls = []
    ns = dict(argparse=argparse, torch=fake_torch([14.75]), MODEL="unused",
              AutoProcessor=SimpleNamespace(from_pretrained=lambda _: "processor"),
              experiment_config=lambda *args: {}, initialize=lambda: None)
    for name in ("train", "evaluate_trained", "check_mask", "select_e4", "summarize"):
        ns[name] = lambda *args, label=name: calls.append(label)
    load_pipeline_functions({"main"}, ns)
    monkeypatch.setattr(sys, "argv", ["pipeline.py", "--stage", stage, "--name", "e2"])
    ns["main"]()
    assert calls == [expected]
