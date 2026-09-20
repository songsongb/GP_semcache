#!/usr/bin/env python3
"""Small deterministic, systems-only QKV LoRA training benchmark."""
import argparse
import sys
import tempfile
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from semcache.metrics.m8 import (environment_record, model_config,
                                 training_record, training_timing_summary)
from semcache.models.loader import load_model
from semcache.utils.io import write_json
from semcache.utils.seed import seed_everything

TEXTS = (
    "Please find a hotel near the station.",
    "Book an Italian restaurant tonight.",
    "What is the weather tomorrow?",
    "Find a morning train to Cambridge.",
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="facebook/opt-125m", choices=("facebook/opt-125m", "facebook/opt-2.7b"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype")
    p.add_argument("--samples", type=int, default=16)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--sequence-length", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--mixed-precision", choices=("none", "fp16"), default="none")
    p.add_argument("--allow-download", action="store_true")
    p.add_argument("--revision")
    p.add_argument("--output", type=Path, default=ROOT / "results/m8/lora_training.json")
    args = p.parse_args()
    if args.samples < 1 or args.epochs < 1 or args.batch_size < 1 or args.sequence_length < 2:
        p.error("samples, epochs, batch size and sequence length must be positive")
    cfg = model_config(args.model, dtype=args.dtype)
    print(f"M8 training work summary: model={args.model}, samples={args.samples}, epochs={args.epochs}, "
          f"batch={args.batch_size}, sequence_length={args.sequence_length}, dtype={cfg['dtype']}")
    seed_everything(42)
    import torch
    from peft import LoraConfig, get_peft_model
    model, tokenizer, metadata = load_model(dict(name=args.model, tokenizer=args.model,
        revision=args.revision, tokenizer_revision=args.revision, dtype=cfg["dtype"],
        device=args.device, local_files_only=not args.allow_download, attention_implementation="eager"))
    lora = LoraConfig(r=8, lora_alpha=8, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj"], bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    tokenizer.pad_token = tokenizer.eos_token
    encoded = tokenizer([TEXTS[i % len(TEXTS)] for i in range(args.samples)], padding="max_length",
        truncation=True, max_length=args.sequence_length, return_tensors="pt")
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    use_amp = args.mixed_precision == "fp16"
    if use_amp and not (torch.cuda.is_available() and args.device.startswith("cuda")):
        raise ValueError("fp16 mixed precision requires CUDA")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    if torch.cuda.is_available() and args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    step_ms, epoch_s, token_count = [], [], 0
    total_start = time.perf_counter_ns()
    for _ in range(args.epochs):
        epoch_start = time.perf_counter_ns()
        for start in range(0, args.samples, args.batch_size):
            batch = {k: v[start:start + args.batch_size].to(args.device) for k, v in encoded.items()}
            labels = batch["input_ids"].clone()
            labels[batch["attention_mask"] == 0] = -100
            if torch.cuda.is_available() and args.device.startswith("cuda"):
                torch.cuda.synchronize()
            step_start = time.perf_counter_ns()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                loss = model(**batch, labels=labels, use_cache=False).loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if torch.cuda.is_available() and args.device.startswith("cuda"):
                torch.cuda.synchronize()
            step_ms.append((time.perf_counter_ns() - step_start) / 1e6)
            token_count += int(batch["attention_mask"].sum())
        epoch_s.append((time.perf_counter_ns() - epoch_start) / 1e9)
    total_s = (time.perf_counter_ns() - total_start) / 1e9
    timing_summary = training_timing_summary(total_s, step_ms)
    allocated = torch.cuda.max_memory_allocated() if torch.cuda.is_available() and args.device.startswith("cuda") else None
    reserved = torch.cuda.max_memory_reserved() if torch.cuda.is_available() and args.device.startswith("cuda") else None
    with tempfile.TemporaryDirectory(prefix="semcache-m8-") as directory:
        model.save_pretrained(directory)
        checkpoint_bytes = sum(p.stat().st_size for p in Path(directory).rglob("*") if p.is_file())
    env = environment_record(torch)
    row = training_record(experiment_id=f"m8-train-{uuid.uuid4().hex[:12]}", gpu_name=env.get("gpu_name"),
        model_id=args.model, model_revision=metadata.get("resolved_model_revision"),
        tokenizer_revision=metadata.get("resolved_tokenizer_revision"), dtype=metadata["dtype"], rank=8,
        target_modules=["q_proj", "k_proj", "v_proj"], sample_count=args.samples, epochs=args.epochs,
        sequence_length=args.sequence_length, batch_size=args.batch_size,
        gradient_checkpointing=args.gradient_checkpointing, mixed_precision=args.mixed_precision,
        epoch_time_s=epoch_s, step_time_ms=step_ms, **timing_summary,
        samples_per_second=args.samples * args.epochs / total_s, tokens_per_second=token_count / total_s,
        peak_cuda_allocated_bytes=allocated, peak_cuda_reserved_bytes=reserved,
        trainable_parameter_count=trainable, adapter_checkpoint_bytes=checkpoint_bytes,
        measured_or_estimated="MEASURED", reproduction_choices=dict(dataset="deterministic synthetic fixture",
            optimizer="AdamW", learning_rate=args.learning_rate, convergence_goal=False,
            total_training_wall_scope="all optimization steps including first-step startup overhead",
            steady_state_scope="diagnostic only; all optimization steps except the first",
            gradient_checkpointing=args.gradient_checkpointing, mixed_precision=args.mixed_precision))
    write_json(args.output, {"environment": env, "result": row})
    print(f"Saved training timing to {args.output}")


if __name__ == "__main__":
    main()
