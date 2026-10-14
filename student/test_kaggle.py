"""CPU-only regression checks for Kaggle capacity and stage orchestration."""
import argparse
import ast
import copy
import json
import importlib.util
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from student.kaggle_runtime import configure_hf_token, report_gpu, run_preflight_checks
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


def test_preflight_preserves_failure_details_and_notebook_sync_hint(tmp_path, monkeypatch):
    from student import kaggle_runtime
    output = ["FAILED test_notebook_code_compiles_and_every_section_executes\n",
              "AssertionError: Regenerate with python student/make_notebook.py\n", "1 failed, 94 passed\n"]
    monkeypatch.setattr(kaggle_runtime.subprocess, "Popen", lambda *a, **kw:
                        SimpleNamespace(stdout=iter(output), wait=lambda: 1))
    with pytest.raises(RuntimeError, match="commit and push lab_day10.ipynb") as error:
        run_preflight_checks(tmp_path)
    assert "1 failed, 94 passed" in str(error.value)
    assert (tmp_path / "report/runs/preflight.log.txt").read_text() == "".join(output)


def test_successful_preflight_returns_saved_log(tmp_path, monkeypatch):
    from student import kaggle_runtime
    monkeypatch.setattr(kaggle_runtime.subprocess, "Popen", lambda *a, **kw:
                        SimpleNamespace(stdout=iter(["100 passed\n"]), wait=lambda: 0))
    assert run_preflight_checks(tmp_path).read_text() == "100 passed\n"


def test_hf_secret_is_exported_without_printing_value(capsys):
    token = "test-only-secret-value"
    requested = []
    def get_secret(name):
        requested.append(name)
        return token
    env = {}
    assert configure_hf_token(SimpleNamespace(get_secret=get_secret), env)
    assert requested == ["HF_TOKEN"] and env["HF_TOKEN"] == token
    assert token not in capsys.readouterr().out


def test_missing_hf_secret_allows_public_downloads_without_leaking_error(capsys):
    def unavailable(name):
        raise RuntimeError("private diagnostic that should never be logged")
    env = {}
    assert not configure_hf_token(SimpleNamespace(get_secret=unavailable), env)
    assert "HF_TOKEN" not in env
    assert "private diagnostic" not in capsys.readouterr().out
    env["HF_TOKEN"] = "existing-environment-token"
    assert configure_hf_token(SimpleNamespace(get_secret=unavailable), env)
    assert env["HF_TOKEN"] == "existing-environment-token"


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
              subprocess=SimpleNamespace(Popen=popen, PIPE=-1, STDOUT=-2),
              reusable_stage=lambda *args: False, record_stage=lambda *args: None)
    exec(runner, ns)
    assert not ns["run_stage"]("train", "e2")
    assert not ns["run_stage"]("train", "e3")
    assert len(commands) == 1
    status = json.loads((tmp_path / "report/runs/status.json").read_text())
    assert status["pipeline"] == "failed" and status["stages"][0]["stage"] == "train_e2"
    from student import github_sync
    pushed = []
    monkeypatch.setattr(github_sync, "push_results", lambda *args: pushed.append(args))
    ns.update(GITHUB_TOKEN="test-only-placeholder", RESULT_BRANCH="kaggle-results-test", FAIL_VERSION_ON_ERROR=True)
    with pytest.raises(RuntimeError, match="train_e2 exited 1"):
        exec("".join(cells[-1]["source"]), ns)
    assert len(pushed) == 1 and ns["GITHUB_TOKEN"] is None
    ns.update(GITHUB_TOKEN="test-only-placeholder", FAIL_VERSION_ON_ERROR=False)
    exec("".join(cells[-1]["source"]), ns)
    assert len(pushed) == 2  # Preserve Output without pretending the pipeline completed.


def load_pipeline_functions(names, namespace):
    path = Path(__file__).with_name("pipeline.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    selected = [node for node in tree.body if
                (isinstance(node, ast.FunctionDef) and node.name in names) or
                (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in node.targets))]
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class FakeParameter:
    def __init__(self, size, dtype="torch.float16", requires_grad=True):
        self.size = size
        self.data = SimpleNamespace(dtype=dtype)
        self.data.float = lambda: SimpleNamespace(dtype="torch.float32")
        self.requires_grad = requires_grad

    @property
    def requires_grad(self):
        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self, enabled):
        if enabled and not self.is_floating_point():
            raise RuntimeError("only Tensors of floating point and complex dtype can require gradients")
        self._requires_grad = enabled

    @property
    def dtype(self):
        return self.data.dtype

    def is_floating_point(self):
        return self.dtype != "torch.uint8"

    def numel(self):
        return self.size


def parameter_fixture():
    prefix = "base_model.model.model."
    return {
        prefix + "language_model.layers.0.self_attn.q_proj.lora_A.default.weight": FakeParameter(20),
        prefix + "language_model.layers.0.self_attn.q_proj.lora_B.default.weight": FakeParameter(20),
        prefix + "visual.blocks.0.attn.qkv.lora_A.default.weight": FakeParameter(20),
        prefix + "visual.blocks.0.attn.qkv.lora_B.default.weight": FakeParameter(20),
        prefix + "visual.merger.linear_fc1.weight": FakeParameter(100),
        prefix + "visual.deepstack_merger_list.0.linear_fc2.weight": FakeParameter(200),
        prefix + "visual.blocks.0.attn.qkv.base_layer.weight": FakeParameter(100),
        prefix + "language_model.embed_tokens.weight": FakeParameter(50),
        prefix + "language_model.layers.0.mlp.gate_proj.base_layer.weight": FakeParameter(1000, "torch.uint8", False),
    }


@pytest.mark.parametrize("targets,projector_enabled,expected", [
    ("llm", False, {"llm_lora": 40, "vit_lora": 0, "projector": 0}),
    ("llm", True, {"llm_lora": 40, "vit_lora": 0, "projector": 300}),
    ("llm+vit", True, {"llm_lora": 40, "vit_lora": 40, "projector": 300}),
])
def test_setup_refreezes_unintended_params_and_promotes_all_trainables_after_checkpointing(targets, projector_enabled, expected):
    params = parameter_fixture()
    ns = load_pipeline_functions({"LLM_TARGETS", "VIT_TARGETS", "quantized_parameter", "trainable_group", "finalize_trainable_parameters", "setup"},
        dict(re=re, json=json, torch=SimpleNamespace(float32="torch.float32"), LoraConfig=lambda **kw: kw))
    module_names = [f"model.language_model.layers.{i}.{kind}.{proj}" for i in range(36)
                    for kind, projections in [("self_attn", ["q_proj", "k_proj", "v_proj", "o_proj"]),
                                              ("mlp", ["gate_proj", "up_proj", "down_proj"])] for proj in projections]
    if targets == "llm+vit":
        module_names += [f"model.visual.blocks.{i}.{kind}.{proj}" for i in range(24)
                         for kind, projections in [("attn", ["qkv", "proj"]),
                                                   ("mlp", ["linear_fc1", "linear_fc2"])] for proj in projections]
    def checkpointing(**kwargs):
        # Simulate a wrapper leaving unrelated half-precision parameters trainable.
        for p in params.values():
            if p.is_floating_point():
                p.requires_grad = True
    model = SimpleNamespace(parameters=lambda: params.values(), named_parameters=lambda: params.items(),
        named_modules=lambda: [(name, None) for name in module_names], enable_input_require_grads=lambda: None,
        gradient_checkpointing_enable=checkpointing, print_trainable_parameters=lambda: None,
        config=SimpleNamespace(use_cache=True, text_config=SimpleNamespace(use_cache=True)))
    ns["get_peft_model"] = lambda m, cfg: m
    cfg = {"train_projector": projector_enabled, "lora": {"targets": targets, "r": 16, "alpha": 32, "dropout": .05}}
    ns["setup"](model, cfg)
    audit = model._lab_setup_audit
    assert audit["groups"] == expected and audit["trainable_params"] == sum(expected.values())
    assert len(audit["matched_modules"]) == (348 if targets == "llm+vit" else 252)
    assert not model.config.use_cache and not model.config.text_config.use_cache
    for name, param in params.items():
        authorized = ns["trainable_group"](name, cfg) is not None
        assert param.requires_grad == authorized
        assert param.dtype == ("torch.float32" if authorized else "torch.uint8" if "gate_proj.base_layer" in name else "torch.float16")


def test_quantized_trainable_parameter_fails_before_optimizer():
    params = parameter_fixture()
    first = next(iter(params))
    params[first] = FakeParameter(20, "torch.uint8", False)
    ns = load_pipeline_functions({"LLM_TARGETS", "VIT_TARGETS", "quantized_parameter", "trainable_group", "finalize_trainable_parameters"},
                                dict(re=re, json=json, torch=SimpleNamespace(float32="torch.float32")))
    with pytest.raises(RuntimeError, match="quantized/nonfloating"):
        ns["finalize_trainable_parameters"](SimpleNamespace(named_parameters=lambda: params.items()),
                                            {"train_projector": True, "lora": {"targets": "llm"}})


def test_real_cpu_grad_scaler_accepts_finalized_parameters():
    # Runs on Kaggle's installed torch too; no GPU or model download needed.
    torch = pytest.importorskip("torch")
    cfg = {"train_projector": True, "lora": {"targets": "llm"}}
    ns = load_pipeline_functions({"LLM_TARGETS", "VIT_TARGETS", "quantized_parameter", "trainable_group", "finalize_trainable_parameters", "optimizer_step"},
                                dict(re=re, json=json, torch=torch))
    params = {
        "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.default.weight": torch.nn.Parameter(torch.ones(2, dtype=torch.float16)),
        "base_model.model.model.visual.merger.linear_fc1.weight": torch.nn.Parameter(torch.ones(2, dtype=torch.float16)),
        "base_model.model.model.visual.blocks.0.attn.qkv.weight": torch.nn.Parameter(torch.ones(2, dtype=torch.float16)),
    }
    trainables = list(params.values())[:2]
    opt = torch.optim.SGD(trainables, lr=.1)
    scaler = torch.amp.GradScaler("cpu", init_scale=128)
    scaler.scale(sum(p.square().sum() for p in trainables)).backward()
    with pytest.raises(ValueError, match="unscale FP16 gradients"):
        scaler.unscale_(opt)
    opt.zero_grad(set_to_none=True)
    ns["finalize_trainable_parameters"](SimpleNamespace(named_parameters=lambda: params.items()), cfg)
    trainables = [p for p in params.values() if p.requires_grad]
    opt = torch.optim.SGD(trainables, lr=.1)
    scaler = torch.amp.GradScaler("cpu", init_scale=128)
    scaler.scale(sum(p.square().sum() for p in trainables)).backward()
    scaler.unscale_(opt)
    scaler.step(opt)
    scaler.update()
    assert all(p.grad.dtype == torch.float32 and (p < 1).all() for p in trainables)
    frozen = list(params.values())[-1]
    assert not frozen.requires_grad and frozen.dtype == torch.float16 and (frozen == 1).all()
    opt.zero_grad(set_to_none=True)
    before = trainables[0].detach().clone()
    old_scale = scaler.get_scale()
    scaler.scale(trainables[0].sum() * float("inf")).backward()
    applied, norm = ns["optimizer_step"](trainables, opt, scaler)
    assert not applied and norm is None and scaler.get_scale() < old_scale
    assert torch.equal(trainables[0], before)


def load_upstream_skip_matcher():
    # Test the pinned library's real source without importing its heavy dependencies.
    spec = importlib.util.find_spec("transformers")
    if spec is None:
        pytest.skip("Transformers source is not installed")
    source = Path(spec.origin).parent / "quantizers/quantizers_utils.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "should_convert_module"]
    ns = {"re": re}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), "exec"), ns)
    return ns["should_convert_module"]


def test_nf4_exclusions_match_actual_transformers_full_paths():
    should_convert = load_upstream_skip_matcher()
    ns = load_pipeline_functions({"QUANTIZATION_SKIP_MODULES"}, {})
    skip = ns["QUANTIZATION_SKIP_MODULES"]
    protected = ["model.visual.blocks.0.attn.qkv", "model.visual.blocks.23.mlp.linear_fc2",
                 "model.visual.merger.linear_fc1", "model.visual.deepstack_merger_list.0.linear_fc2", "lm_head"]
    for name in protected:
        assert not should_convert(name, skip), name
    assert should_convert("model.language_model.layers.0.self_attn.q_proj", skip)
    assert should_convert("model.visual.merger.linear_fc1", ["visual", "lm_head"]), "Reproduce the old skip-list bug"


class Params4bit(FakeParameter):
    """Represent a packed bnb parameter, including its possible floating storage dtype."""


def test_model_load_rejects_quantized_projector_before_setup():
    params = {"model.visual.merger.linear_fc1.weight": Params4bit(50, "torch.uint8", False),
              "model.visual.blocks.0.attn.qkv.weight": FakeParameter(100),
              "model.language_model.layers.0.self_attn.q_proj.weight": Params4bit(50, "torch.uint8", False)}
    ns = load_pipeline_functions({"QUANTIZATION_SKIP_MODULES", "projector", "quantized_parameter", "validate_quantization"}, {})
    model = SimpleNamespace(named_parameters=lambda: params.items(), named_modules=lambda: [])
    with pytest.raises(RuntimeError, match="model.visual.merger.linear_fc1.weight"):
        ns["validate_quantization"](model)
    params["model.visual.merger.linear_fc1.weight"] = FakeParameter(200)
    audit = ns["validate_quantization"](model)
    assert audit["vision_params"] == 300 and audit["projector_params"] == 200 and audit["nf4_llm_weights"] == 1
    # Even floating-storage bnb parameters must not be trained as dense projectors.
    params["model.visual.merger.linear_fc1.weight"] = Params4bit(200, "torch.float16", False)
    with pytest.raises(RuntimeError, match="must not be quantized"):
        ns["validate_quantization"](model)


def test_quantized_projector_is_rejected_before_enabling_gradients():
    params = parameter_fixture()
    name = "base_model.model.model.visual.merger.linear_fc1.weight"
    packed = Params4bit(50, "torch.uint8", False)
    params[name] = packed
    ns = load_pipeline_functions({"LLM_TARGETS", "VIT_TARGETS", "quantized_parameter", "trainable_group", "finalize_trainable_parameters"},
                                dict(re=re, json=json, torch=SimpleNamespace(float32="torch.float32")))
    with pytest.raises(RuntimeError, match="quantized/nonfloating.*visual.merger"):
        ns["finalize_trainable_parameters"](SimpleNamespace(named_parameters=lambda: params.items()),
                                           {"train_projector": True, "lora": {"targets": "llm"}})
    assert not packed.requires_grad


def test_actual_bnb_conversion_keeps_vision_dense():
    torch = pytest.importorskip("torch")
    bnb = pytest.importorskip("bitsandbytes")
    from transformers import BitsAndBytesConfig
    from transformers.integrations.bitsandbytes import replace_with_bnb_linear

    def tiny_model():
        model = torch.nn.Module()
        model.model = torch.nn.Module()
        model.model.visual = torch.nn.Module()
        model.model.visual.merger = torch.nn.Sequential(torch.nn.Linear(4, 4))
        model.model.visual.deepstack_merger_list = torch.nn.ModuleList([torch.nn.Linear(4, 4)])
        model.model.visual.blocks = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(4, 4))])
        model.model.language_model = torch.nn.Module()
        model.model.language_model.layers = torch.nn.ModuleList([torch.nn.Linear(4, 4)])
        model.lm_head = torch.nn.Linear(4, 4)
        return model

    ns = load_pipeline_functions({"QUANTIZATION_SKIP_MODULES", "projector", "quantized_parameter", "validate_quantization"}, {})
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              llm_int8_skip_modules=ns["QUANTIZATION_SKIP_MODULES"])
    model = replace_with_bnb_linear(tiny_model(), modules_to_not_convert=quant.llm_int8_skip_modules, quantization_config=quant)
    assert type(model.model.visual.merger[0]) is torch.nn.Linear
    assert type(model.model.visual.deepstack_merger_list[0]) is torch.nn.Linear
    assert type(model.model.visual.blocks[0][0]) is torch.nn.Linear
    assert type(model.lm_head) is torch.nn.Linear
    assert isinstance(model.model.language_model.layers[0], bnb.nn.Linear4bit)
    assert ns["validate_quantization"](model)["nf4_llm_weights"] == 1
    old = replace_with_bnb_linear(tiny_model(), modules_to_not_convert=["visual", "lm_head"], quantization_config=quant)
    assert isinstance(old.model.visual.merger[0], bnb.nn.Linear4bit)
    with pytest.raises(RuntimeError, match="must not be quantized"):
        ns["validate_quantization"](old)


@pytest.mark.parametrize("overflow", [False, True])
def test_overflowed_updates_are_skipped_instead_of_aborting(overflow):
    events = []
    scale = [256.]
    scaler = SimpleNamespace(unscale_=lambda opt: events.append("unscale"),
        get_scale=lambda: scale[0], step=lambda opt: events.append("step"),
        update=lambda: scale.__setitem__(0, 128. if overflow else 256.))
    torch = SimpleNamespace(isfinite=lambda x: x != float("inf"),
        nn=SimpleNamespace(utils=SimpleNamespace(clip_grad_norm_=lambda *a: float("inf") if overflow else .5)))
    ns = load_pipeline_functions({"optimizer_step"}, dict(torch=torch))
    applied, norm = ns["optimizer_step"]([], object(), scaler)
    assert applied == (not overflow) and norm == (None if overflow else .5)
    assert events == ["unscale", "step"]


@pytest.mark.parametrize("padding,generation", [(True, True), (False, True), (False, False)])
def test_chat_processor_options_use_new_api(padding, generation):
    calls = []
    # Explicit signature rejects obsolete flat padding keyword arguments.
    def template(conversation, *, tokenize, add_generation_prompt, return_dict, return_tensors, processor_kwargs):
        calls.append((conversation, tokenize, add_generation_prompt, return_dict, return_tensors, processor_kwargs))
        return "encoded"
    ns = load_pipeline_functions({"encode_chat"}, {})
    processor = SimpleNamespace(apply_chat_template=template)
    conversation = [{"role": "user", "content": "test"}]
    assert ns["encode_chat"](processor, conversation, add_generation_prompt=generation, padding=padding) == "encoded"
    assert calls == [(conversation, True, generation, True, "pt", {"padding": padding})]


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
