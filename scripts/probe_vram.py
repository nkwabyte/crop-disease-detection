#!/usr/bin/env python3
"""Peak-VRAM probe for the remaining detectors. Writes no checkpoints.

A few AMP training steps on synthetic 640px images, so it can run before the
dataset upload finishes. Set CUDA_VISIBLE_DEVICES yourself. This script refuses
to run without it, because device 0 on the shared box may belong to someone else.

  CUDA_VISIBLE_DEVICES=1 TRAIN_MEM_FRACTION=0.30 \
    .venv/bin/python scripts/probe_vram.py --model frcnn --batch 4
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/probe_vram.py --model all --batch 4
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch


def _refuse_unpinned() -> None:
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        sys.exit(
            "Set CUDA_VISIBLE_DEVICES first. An unset value uses physical GPU 0, "
            "which may already be running someone else's job."
        )
    if not torch.cuda.is_available():
        sys.exit("CUDA is not available in this process.")


def _build(name: str):
    if name == "frcnn":
        from src.fasterrcnn.train_alt_faster_rcnn import build_model, resolve_device
        return resolve_device(), build_model()
    if name == "final":
        from src.fasterrcnn.faster_rcnn_final import build_model, resolve_device
        return resolve_device(), build_model(anchor_sizes=None)
    if name == "vit":
        from src.vit.train_vit import build_model, resolve_device
        return resolve_device(), build_model(pretrained=True)
    if name == "swin":
        from src.swin.train_swin import build_model, resolve_device
        return resolve_device(), build_model(pretrained=True)
    raise SystemExit(f"unknown model {name}")


def _run(name: str, batch: int, steps: int) -> None:
    print(f"\n════ {name}  batch={batch}  steps={steps} ════")
    device, model = _build(name)
    model = model.to(device)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=1e-3, momentum=0.9)
    scaler = torch.amp.GradScaler("cuda")
    torch.cuda.reset_peak_memory_stats()
    for step in range(steps):
        images = [torch.rand(3, 640, 640, device=device) for _ in range(batch)]
        targets = [
            {
                "boxes": torch.tensor(
                    [[32, 32, 180, 180], [220, 200, 480, 500]],
                    device=device, dtype=torch.float32,
                ),
                "labels": torch.tensor([1, 16], device=device),
            }
            for _ in range(batch)
        ]
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss = sum(_loss_dict(model, images, targets).values())
        if not torch.isfinite(loss):
            print(f"  step {step + 1} non-finite loss")
            break
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 1024 ** 3
        print(f"  step {step + 1}  loss={float(loss):.4f}  peak={peak:.2f} GB")
    del model, opt, scaler
    torch.cuda.empty_cache()


def _loss_dict(model, images, targets):
    return model(images, targets)


def main() -> None:
    parser = argparse.ArgumentParser(description="VRAM probe, no checkpoints")
    parser.add_argument("--model", default="all",
                        choices=["all", "frcnn", "final", "vit", "swin"])
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--steps", type=int, default=2)
    args = parser.parse_args()
    _refuse_unpinned()
    names = ["frcnn", "final", "vit", "swin"] if args.model == "all" else [args.model]
    print(f"device {torch.cuda.get_device_name(0)}  "
          f"visible={os.environ.get('CUDA_VISIBLE_DEVICES')}  "
          f"fraction={os.environ.get('TRAIN_MEM_FRACTION', 'none')}")
    for name in names:
        _run(name, args.batch, args.steps)


if __name__ == "__main__":
    main()
