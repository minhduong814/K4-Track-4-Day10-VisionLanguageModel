"""Student implementation; run from repository root on a CUDA Kaggle runtime."""
import argparse
import copy
import gc
import json
import math
import random
import re
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
from peft import LoraConfig, PeftModel, get_peft_model

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.score import evaluate, parse_output

MODEL = "Qwen/Qwen3-VL-4B-Instruct"
LLM_TARGETS = r"model\.language_model\.layers\.\d+\.(?:self_attn\.(?:q|k|v|o)_proj|mlp\.(?:gate|up|down)_proj)"
VIT_TARGETS = r"model\.visual\.blocks\.\d+\.(?:attn\.(?:qkv|proj)|mlp\.(?:linear_fc1|linear_fc2))"
PROMPT = ('Detect all potholes, cracks and manholes. Return only a JSON list of objects '
          'with keys "label" (pothole, crack, manhole) and "bbox_2d" '
          '([x1,y1,x2,y2], integer coordinates normalized to 0–1000). '
          'Return [] if none. No markdown or explanation.')


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def rows(split):
    return [json.loads(s) for s in (ROOT / f"data/splits/{split}.jsonl").read_text(encoding="utf-8").splitlines()]


def answer(entry):
    boxes = []
    for obj in entry["objects"]:
        x, y, w, h = obj["yolo"]
        box = [max(0, min(1000, round(v * 1000))) for v in (x-w/2, y-h/2, x+w/2, y+h/2)]
        boxes.append({"label": obj["label"], "bbox_2d": box})
    boxes.sort(key=lambda b: (b["bbox_2d"][1], b["bbox_2d"][0], b["label"]))
    return json.dumps(boxes, separators=(",", ":"))


def messages(image):
    return [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": PROMPT}]}]


def encode_chat(processor, conversation, *, add_generation_prompt, padding=False):
    # transformers 5.19 routes processor options through processor_kwargs.
    return processor.apply_chat_template(conversation, tokenize=True,
        add_generation_prompt=add_generation_prompt, return_dict=True, return_tensors="pt",
        processor_kwargs={"padding": padding})


def load_base():
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
                              llm_int8_skip_modules=["visual", "lm_head"])
    return Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL, quantization_config=quant, dtype=torch.float16,
        device_map={"": 0}, attn_implementation="sdpa")


def projector(name):
    return "visual.merger." in name or "visual.deepstack_merger_list." in name


def trainable_group(name, cfg):
    """Select exactly the parameters authorized by an E1/E2/E3 setup."""
    name = name.removeprefix("base_model.model.")
    adapter = re.fullmatch(r"(.+)\.lora_[AB]\.[^.]+\.weight", name)
    if adapter:
        module = adapter.group(1)
        if re.fullmatch(LLM_TARGETS, module):
            return "llm_lora"
        if cfg["lora"]["targets"] == "llm+vit" and re.fullmatch(VIT_TARGETS, module):
            return "vit_lora"
        return None
    if cfg["train_projector"] and re.match(r"model\.visual\.(?:merger\.|deepstack_merger_list\.\d+\.)", name):
        return "projector"
    return None


def finalize_trainable_parameters(model, cfg):
    # Apply this AFTER PEFT/checkpointing setup and BEFORE creating the optimizer.
    # Frozen NF4/vision weights retain their dtype; all optimizer parameters use fp32.
    groups = {"llm_lora": 0, "vit_lora": 0, "projector": 0}
    audit = {"groups": groups, "parameters": [], "refrozen": [], "promoted_to_fp32": []}
    for name, param in model.named_parameters():
        group = trainable_group(name, cfg)
        if param.requires_grad and group is None:
            audit["refrozen"].append(name)
        param.requires_grad = group is not None
        if group is None:
            continue
        if not param.is_floating_point():
            raise RuntimeError(f"Trainable parameter is quantized/nonfloating: {name}")
        if param.dtype != torch.float32:
            param.data = param.data.float()
            audit["promoted_to_fp32"].append(name)
        if param.dtype != torch.float32:
            raise RuntimeError(f"AMP optimizer parameter must be fp32: {name}: {param.dtype}")
        groups[group] += param.numel()
        audit["parameters"].append({"name": name, "group": group, "dtype": str(param.dtype), "numel": param.numel()})
    if not groups["llm_lora"]:
        raise RuntimeError("No LLM LoRA parameters selected")
    if cfg["train_projector"] and not groups["projector"]:
        raise RuntimeError("Projector training requested but no projector parameters found")
    if cfg["lora"]["targets"] == "llm+vit" and not groups["vit_lora"]:
        raise RuntimeError("ViT LoRA requested but no ViT adapter parameters found")
    audit["trainable_params"] = sum(groups.values())
    print("Trainable groups (all fp32):", json.dumps(groups), flush=True)
    if audit["refrozen"]:
        print("Refroze parameters outside the experiment setup:", audit["refrozen"][:10], flush=True)
    return audit


def setup(model, cfg):
    for p in model.parameters():
        p.requires_grad = False
    regex = LLM_TARGETS if cfg["lora"]["targets"] == "llm" else f"(?:{LLM_TARGETS}|{VIT_TARGETS})"
    matched = [n for n, _ in model.named_modules() if re.fullmatch(regex, n)]
    expected = 252 if cfg["lora"]["targets"] == "llm" else 348
    if len(matched) != expected:
        raise RuntimeError(f"Unexpected model module layout: {len(matched)} != {expected}")
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=cfg["lora"]["r"], lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"], target_modules=regex, task_type="CAUSAL_LM"))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.config.text_config.use_cache = False
    model._lab_setup_audit = finalize_trainable_parameters(model, cfg)
    model._lab_setup_audit["matched_modules"] = matched
    model.print_trainable_parameters()
    return model


class Examples(Dataset):
    def __init__(self, entries, processor):
        self.entries, self.processor = entries, processor

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        entry = self.entries[index]
        with Image.open(ROOT / "data" / entry["image_path"]) as im:
            user = messages(im.convert("RGB"))
            prompt = encode_chat(self.processor, user, add_generation_prompt=True)
            full = encode_chat(self.processor, user + [{"role": "assistant", "content": answer(entry)}],
                               add_generation_prompt=False)
        n = prompt["input_ids"].shape[1]
        assert torch.equal(full["input_ids"][:, :n], prompt["input_ids"]), "Chat prefix mismatch"
        full["labels"] = full["input_ids"].clone()
        full["labels"][:, :n] = -100
        assert (full["labels"] != -100).any()
        return full


def collate(batch, pad_id):
    result = {}
    for key in batch[0]:
        tensors = [x[key] for x in batch]
        if key in ("pixel_values", "image_grid_thw"):
            result[key] = torch.cat(tensors, dim=0)
        elif key in ("input_ids", "attention_mask", "mm_token_type_ids", "labels"):
            value = -100 if key == "labels" else pad_id if key == "input_ids" else 0
            result[key] = torch.nn.utils.rnn.pad_sequence([t[0] for t in tensors], batch_first=True, padding_value=value)
        else:
            raise RuntimeError(f"Unsupported processor field: {key}")
    return result


def save_model(model, path, processor):
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    processor.save_pretrained(path)
    extra = {n: p.detach().cpu() for n, p in model.named_parameters() if projector(n) and p.requires_grad}
    if extra:
        torch.save(extra, path / "projector.pt")


def optimizer_step(params, opt, scaler):
    """Let GradScaler skip overflowed updates and adapt its scale."""
    scaler.unscale_(opt)
    norm = torch.nn.utils.clip_grad_norm_(params, 1.)
    scale_before = scaler.get_scale()
    scaler.step(opt)
    scaler.update()
    applied = scaler.get_scale() >= scale_before
    finite = bool(torch.isfinite(norm))
    if applied and not finite:
        raise RuntimeError("Nonfinite gradients were not rejected by GradScaler")
    return applied, float(norm) if finite else None


def train(cfg, name, processor):
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    model = setup(load_base(), cfg)
    write_json(ROOT / f"report/runs/{name}_setup.json", model._lab_setup_audit)
    dataset = Examples(rows("train"), processor)
    sample = dataset[0]
    image_tokens = int((sample["input_ids"] == model.config.image_token_id).sum())
    assert image_tokens == 220, f"Expected 220 image tokens; got {image_tokens}"
    write_json(ROOT / f"report/runs/{name}_mask.json", {
        "image_tokens": image_tokens,
        "supervised_text": processor.tokenizer.decode(sample["labels"][sample["labels"] != -100]),
        "config": cfg})
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=0,
                        collate_fn=lambda b: collate(b, processor.tokenizer.pad_token_id))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(cfg["lr"]), weight_decay=0)
    steps = cfg["max_steps"]
    warmup = max(1, round(steps * .05))
    def schedule(step):
        if step < warmup:
            return (step + 1) / warmup
        return .5 * (1 + math.cos(math.pi * (step-warmup) / max(1, steps-warmup)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, schedule)
    scaler = torch.amp.GradScaler("cuda")
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    it = iter(loader)
    model.train()
    logs = []
    out = ROOT / f"work/adapters/{name}"
    step = attempts = consecutive_overflows = 0
    overflow_logs = []
    while step < steps:
        attempts += 1
        opt.zero_grad(set_to_none=True)
        loss_sum = 0.
        for _ in range(cfg["grad_accum"]):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader)
                batch = next(it)
            batch = {k: v.to("cuda:0") for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(**batch, use_cache=False).loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss in {name}, step {step+1}")
            loss_sum += loss.item() / cfg["grad_accum"]
            scaler.scale(loss / cfg["grad_accum"]).backward()
        applied, grad_norm = optimizer_step(params, opt, scaler)
        if not applied:
            consecutive_overflows += 1
            overflow_logs.append({"attempt": attempts, "optimizer_steps": step, "amp_scale": scaler.get_scale()})
            print(f"{name}: AMP overflow; retry step {step+1} at scale {scaler.get_scale()}", flush=True)
            write_json(ROOT / f"report/runs/{name}_amp_overflows.json", overflow_logs)
            if consecutive_overflows >= 10:
                raise RuntimeError(f"Repeated AMP overflows in {name}; inspect loss/dtype or reduce LR")
            continue
        consecutive_overflows = 0
        sched.step()
        step += 1
        logs.append({"step": step, "attempt": attempts, "loss": loss_sum, "grad_norm": grad_norm, "lr": sched.get_last_lr()[0]})
        if step % 10 == 0:
            print(name, logs[-1], flush=True)
            write_json(ROOT / f"report/runs/{name}_loss.json", logs)
        if step % 100 == 0:
            save_model(model, out / f"step-{step}", processor)
    torch.cuda.synchronize()
    meta = {"trainable_params": sum(p.numel() for p in params),
            "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3,
            "train_minutes": (time.perf_counter()-start) / 60}
    save_model(model, out / "final", processor)
    write_json(ROOT / f"report/runs/{name}.json", {**meta, "config": cfg, "amp_overflows": len(overflow_logs), "optimizer_steps": step})
    write_json(ROOT / f"report/runs/{name}_loss.json", logs)
    del model, opt, params, batch, loss, scaler, sched
    gc.collect()
    torch.cuda.empty_cache()
    return meta


def infer(name, split, processor, trained, batch_size=1):
    model = load_base()
    if trained:
        path = ROOT / f"work/adapters/{name}/final"
        model = PeftModel.from_pretrained(model, path)
        if (path / "projector.pt").exists():
            extra = torch.load(path / "projector.pt", map_location="cpu", weights_only=True)
            parameters = dict(model.named_parameters())
            for n, value in extra.items():
                assert n in parameters
                parameters[n].data = value.to(device=parameters[n].device, dtype=torch.float32)
            del extra
    model.eval()
    model.config.use_cache = True
    processor.tokenizer.padding_side = "left"
    entries = rows(split)
    path = ROOT / f"work/{name}_{split}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f, torch.inference_mode():
        for start in range(0, len(entries), batch_size):
            group = entries[start:start+batch_size]
            images = []
            for entry in group:
                with Image.open(ROOT / "data" / entry["image_path"]) as im:
                    images.append(im.convert("RGB"))
            enc = encode_chat(processor, [messages(im) for im in images],
                              add_generation_prompt=True, padding=True).to("cuda:0")
            with torch.autocast("cuda", dtype=torch.float16):
                outputs = model.generate(**enc, max_new_tokens=512, do_sample=False)
            decoded = processor.batch_decode(outputs[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for entry, raw in zip(group, decoded):
                f.write(json.dumps({"image_id": entry["image_id"], "raw": raw}, ensure_ascii=False)+"\n")
            f.flush()
            completed = start + len(group)
            if start == 0 or completed % 5 == 0 or completed == len(entries):
                print(f"{name} {split}: {completed}/{len(entries)} images completed", flush=True)
    del model, enc, outputs
    gc.collect()
    torch.cuda.empty_cache()
    return path


def score(name, split, path, meta):
    out = ROOT / (f"report/evidence/{name}.json" if split == "test" else f"report/validation/{name}.json")
    cmd = [sys.executable, "scripts/score.py", "--pred", str(path), "--name", name.upper(),
           "--split", split, "--out", str(out)]
    for k, v in meta.items():
        cmd += ["--meta", f"{k}={v}"]
    subprocess.run(cmd, cwd=ROOT, check=True)
    return json.loads(out.read_text(encoding="utf-8"))


def figures(name):
    predictions = {r["image_id"]: r["raw"] for r in
                   map(json.loads, (ROOT / f"work/{name}_test.jsonl").read_text(encoding="utf-8").splitlines())}
    ranked = []
    for entry in rows("test"):
        metric = evaluate([entry], predictions, n_boot=10)
        ranked.append((metric["micro"]["f1"], entry))
    ranked.sort(key=lambda x: (x[0], x[1]["image_id"]))
    selected = ranked[:3] + ranked[-3:]
    manifest = []
    folder = ROOT / "report/figures"
    folder.mkdir(parents=True, exist_ok=True)
    for i, (f1, entry) in enumerate(selected):
        with Image.open(ROOT / "data" / entry["image_path"]) as source:
            im = source.convert("RGB")
        draw = ImageDraw.Draw(im)
        predicted = [{"label": b["label"], "bbox_2d": b["bbox"]}
                     for b in (parse_output(predictions[entry["image_id"]]) or [])]
        for color, boxes in [("lime", json.loads(answer(entry))), ("red", predicted)]:
            for box in boxes:
                coords = [v * (im.width if j % 2 == 0 else im.height) / 1000 for j, v in enumerate(box["bbox_2d"])]
                if coords[2] < coords[0] or coords[3] < coords[1]:
                    continue
                draw.rectangle(coords, outline=color, width=2)
                draw.text((coords[0], max(0, coords[1]-12)), box["label"], fill=color)
        caption = f"{name}: F1={f1:.3f}; green=GT red=prediction"
        canvas = Image.new("RGB", (im.width, im.height+24), "white")
        canvas.paste(im)
        ImageDraw.Draw(canvas).text((4, im.height+4), caption, fill="black")
        filename = f"{name}_{i+1}.png"
        canvas.save(folder / filename)
        manifest.append({"file": filename, "image_id": entry["image_id"], "caption": caption,
                         "raw": predictions[entry["image_id"]]})
    write_json(folder / "captions.json", manifest)


def initialize():
    split_rows = {s: rows(s) for s in ("train", "val", "test")}
    sessions = {s: {r["session"] for r in rr} for s, rr in split_rows.items()}
    assert all(not sessions[a] & sessions[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test")))
    write_json(ROOT / "report/runs/splits.json", json.loads((ROOT / "data/splits/meta.json").read_text()))
    write_json(ROOT / "report/runs/environment.json", {"torch": torch.__version__,
               "gpu": torch.cuda.get_device_name(0), "source_commit": subprocess.check_output(
                   ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()})


def experiment_config(name, batch_size, grad_accum):
    if name.startswith("e4_rank_"):
        selection = json.loads((ROOT / "report/runs/e4_selection.json").read_text(encoding="utf-8"))
        # Copy the actual training configuration, not a fresh set of defaults.
        cfg = json.loads((ROOT / f"report/runs/{selection['base']}.json").read_text(encoding="utf-8"))["config"]
        cfg = copy.deepcopy(cfg)
        rank = int(name.rsplit("_", 1)[1])
        cfg["lora"].update(r=rank, alpha=2*rank)
    else:
        cfg = yaml.safe_load((ROOT / f"configs/{name}.yaml").read_text(encoding="utf-8"))
        cfg.update(batch_size=batch_size, grad_accum=grad_accum)
    return cfg


def evaluate_trained(name, processor):
    run = json.loads((ROOT / f"report/runs/{name}.json").read_text(encoding="utf-8"))
    meta = {k: run[k] for k in ("trainable_params", "peak_vram_gb", "train_minutes")}
    if not name.startswith("e4_"):
        score(name, "val", infer(name, "val", processor, True), meta)
    score(name, "test", infer(name, "test", processor, True), meta)


def select_e4():
    validation = {name: json.loads((ROOT / f"report/validation/{name}.json").read_text(encoding="utf-8"))
                  for name in ("e1", "e2", "e3")}
    best = max(validation, key=lambda n: validation[n]["micro"]["f1"])
    write_json(ROOT / "report/runs/e4_selection.json", {"base": best, "selection_split": "val",
                "variable": "rank", "values": [4, 64], "validation_f1": {n: v["micro"]["f1"] for n, v in validation.items()}})
    print(f"E4 base selected on validation: {best}; compare rank 4 vs 64", flush=True)
    return best


def summarize():
    best = json.loads((ROOT / "report/runs/e4_selection.json").read_text(encoding="utf-8"))["base"]
    results = {name: json.loads((ROOT / f"report/evidence/{name}.json").read_text(encoding="utf-8"))
               for name in ("e0", "e1", "e2", "e3", "e4_rank_4", "e4_rank_64")}
    figures(best)
    write_json(ROOT / "report/summary.json", results)
    fields = "| Setup | F1 | CI95 | FP/image | pothole | crack | manhole | VRAM GB | Train min |\n|---|---|---|---|---|---|---|---|---|\n"
    for name, r in results.items():
        values = [name, r["micro"]["f1"], str(r["f1_ci95"]), r["fp_per_image"],
                  *[r["per_class"][c]["f1"] for c in ("pothole", "crack", "manhole")],
                  r["meta"].get("peak_vram_gb", "N/A"), r["meta"].get("train_minutes", "N/A")]
        fields += "| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in values) + " |\n"
    (ROOT / "report/RESULTS.md").write_text(fields, encoding="utf-8")
    check = subprocess.run([sys.executable, "scripts/check_submission.py"], cwd=ROOT, capture_output=True, text=True)
    (ROOT / "report/submission_check.txt").write_text(check.stdout+check.stderr, encoding="utf-8")
    write_json(ROOT / "report/runs/status.json", {"pipeline": "completed", "submission_check_exit_code": check.returncode,
               "report_pending": True})


def check_mask(processor):
    entry = rows("train")[0]
    sample = Examples([entry], processor)[0]
    token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    image_tokens = int((sample["input_ids"] == token_id).sum())
    supervised = processor.tokenizer.decode(sample["labels"][sample["labels"] != -100], skip_special_tokens=True).strip()
    assert image_tokens == 220, f"Expected 220 image tokens; got {image_tokens}"
    assert supervised == answer(entry), "Supervised tokens must decode to the JSON answer"
    result = {"image_id": entry["image_id"], "image_tokens": image_tokens,
              "supervised_text": supervised, "input_shapes": {k: list(v.shape) for k, v in sample.items()}}
    write_json(ROOT / "report/runs/mask_check.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def preview_labels():
    entry = rows("train")[0]
    print("Prompt:", PROMPT, flush=True)
    print("Example image:", entry["image_id"], "Ground truth JSON:", answer(entry), flush=True)
    with Image.open(ROOT / "data" / entry["image_path"]) as source:
        im = source.convert("RGB")
    draw = ImageDraw.Draw(im)
    for box in json.loads(answer(entry)):
        coords = [v * (im.width if j % 2 == 0 else im.height) / 1000 for j, v in enumerate(box["bbox_2d"])]
        draw.rectangle(coords, outline="lime", width=2)
        draw.text((coords[0], max(0, coords[1]-12)), box["label"], fill="lime")
    folder = ROOT / "report/figures"
    folder.mkdir(parents=True, exist_ok=True)
    im.save(folder / "train_ground_truth.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--stage", default="all", choices=["all", "labels", "e0", "mask", "configs", "train", "evaluate", "select-e4", "summary"])
    ap.add_argument("--name", choices=["e1", "e2", "e3", "e4_rank_4", "e4_rank_64"])
    args = ap.parse_args()
    if args.batch_size < 1 or args.grad_accum < 1:
        ap.error("Batch size and accumulation must be positive")
    if args.stage in ("train", "evaluate") and not args.name:
        ap.error("--name is required for train/evaluate")
    if args.stage == "labels":
        preview_labels()
        return
    if args.stage == "configs":
        print("LLM LoRA targets:", LLM_TARGETS)
        print("ViT LoRA targets (E3):", VIT_TARGETS)
        for name in ("e1", "e2", "e3"):
            print(name, json.dumps(experiment_config(name, args.batch_size, args.grad_accum), ensure_ascii=False))
        return
    if args.stage == "select-e4":
        select_e4()
        return
    if args.stage == "summary":
        summarize()
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    processor = AutoProcessor.from_pretrained(MODEL)
    if args.stage in ("all", "e0"):
        initialize()
        score("e0", "test", infer("e0", "test", processor, False), {})
    if args.stage in ("all", "mask"):
        check_mask(processor)
    if args.stage == "train":
        train(experiment_config(args.name, args.batch_size, args.grad_accum), args.name, processor)
    if args.stage == "evaluate":
        evaluate_trained(args.name, processor)
    if args.stage == "all":
        for name in ("e1", "e2", "e3"):
            train(experiment_config(name, args.batch_size, args.grad_accum), name, processor)
            evaluate_trained(name, processor)
        select_e4()
        for rank in (4, 64):
            name = f"e4_rank_{rank}"
            train(experiment_config(name, args.batch_size, args.grad_accum), name, processor)
            evaluate_trained(name, processor)
        summarize()


if __name__ == "__main__":
    main()
