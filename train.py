"""Train one explicit TOML recipe, with small command-line overrides."""

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from config import DEFAULT_CONFIG, ROOT, load_config
from data import CLASS_NAMES
from model import build_model


def train(config_path=DEFAULT_CONFIG, *, batch=None, device=None, run_dir=None, resume=None):
    """Train a recipe; resume restores its checkpoint settings, without RNG replay guarantees."""
    config_path = Path(config_path).expanduser().resolve(strict=True)
    recipe = load_config(config_path)
    hard_normal = recipe["hard_normal"]
    loss_runtime = None
    paths = recipe["paths"]
    overrides = {k: v for k, v in {"batch": batch, "device": device}.items() if v is not None}
    if resume is not None and (overrides or run_dir is not None):
        raise ValueError("Resume uses its saved recipe; batch/device/output overrides are not allowed")
    data_yaml = paths["prepared"] / "data.yaml"
    rank, world_size = int(os.environ.get("RANK", -1)), int(os.environ.get("WORLD_SIZE", 1))
    primary = rank in (-1, 0)
    data_yaml = Path(data_yaml).expanduser().resolve(strict=True)
    model = build_model(resume if resume is not None else paths["weights"])
    if resume is not None:
        checkpoint = model.ckpt
        if checkpoint.get("epoch", -1) < 0 or checkpoint.get("optimizer") is None:
            raise ValueError("Checkpoint has no resumable optimizer/epoch state")
        expected = checkpoint["train_args"]
        if Path(expected["data"]).resolve() != data_yaml:
            raise ValueError("Resume requires the original data.yaml path")
        output = Path(model.ckpt_path).resolve().parent.parent
        original_receipt = json.loads((output / "run.json").read_text())
        saved_loss = original_receipt.get("hard_normal", {"weight": 0.0, "topk": 10})
        if hard_normal != saved_loss:
            raise ValueError("Resume requires the original hard_normal settings; use a new run for a changed loss")
        kwargs = {"resume": True, "save_dir": str(output)}
        receipt_name = datetime.now(timezone.utc).strftime("resume-%Y%m%dT%H%M%S%fZ.json")
    else:
        expected = {**recipe["train"], **overrides}
        output = (ROOT / Path(run_dir).expanduser()).resolve() if run_dir is not None else paths["output"]
        if primary and output.exists():
            raise FileExistsError(f"Run already exists: {output}; choose a new name or resume")
        kwargs = {**expected, "data": str(data_yaml), "project": str(output.parent),
                  "name": output.name, "exist_ok": True}
        receipt_name = "run.json"
    if type(expected["batch"]) is not int or expected["batch"] <= 0 or expected["batch"] % world_size:
        raise ValueError("batch must be a positive GLOBAL integer divisible by WORLD_SIZE")
    device = expected["device"]
    devices = str(device).split(",")
    if world_size > 1:
        if "LOCAL_RANK" not in os.environ or len(devices) != world_size or not all(d.isdigit() for d in devices):
            raise ValueError("Distributed runs require torchrun and one device index per worker")
    elif not (str(device).isdigit() or device in ("cpu", "mps")):
        raise ValueError("Use torchrun for multiple GPUs so every worker retains the runtime checks")
    if primary:
        output.mkdir(parents=True, exist_ok=resume is not None)
    elif receipt_name == "run.json":
        receipt_name = f"run-rank-{rank}.json"
    elif not primary:
        receipt_name = f"rank-{rank}-{receipt_name}"
    receipt_path = output / receipt_name
    expected_counts = {split: len((data_yaml.parent / f"{split}.csv").read_text().splitlines()) - 1
                       for split in ("train", "val") if (data_yaml.parent / f"{split}.csv").is_file()}
    receipt = {
        "status": "running", "requested": expected,
        "hard_normal": hard_normal,
        "config_file": str(config_path), "cli_overrides": {**overrides, "run_dir": str(run_dir) if run_dir else None},
        "config_role": "data path only; recipe from checkpoint" if resume else "training recipe", "resume": str(resume) if resume else None,
        "distributed": {"rank": rank, "world_size": world_size, "local_rank": os.environ.get("LOCAL_RANK")},
        "data_yaml": str(data_yaml), "checkpoint": str(model.ckpt_path),
        "sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                   (data_yaml, Path(model.ckpt_path), config_path, *(ROOT / n for n in ("config.py", "data.py", "model.py", "train.py", "loss.py")))},
        "versions": {p: version(p) for p in ("ultralytics", "torch", "torchvision", "numpy")},
        "selection": "native validation detection mAP50-95, not image false-alarm rate",
    }
    if primary:
        with receipt_path.open("x") as handle:
            json.dump(receipt, handle, indent=2, default=str)
        snapshot = "requested-config.toml" if resume is None else receipt_name.replace(".json", ".toml")
        (output / snapshot).write_bytes(config_path.read_bytes())

    def check_runtime(trainer):
        nonlocal loss_runtime
        # Upstream can halve the batch after OOM. Reject that protocol change.
        if trainer.batch_size != expected["batch"] or bool(trainer.amp) != expected["amp"]:
            raise RuntimeError("Effective batch/AMP differs from the requested recipe")
        if max(trainer.world_size, 1) != world_size or trainer.train_loader.batch_size != expected["batch"] // world_size:
            raise RuntimeError("Effective world size or per-worker batch differs from the recipe")
        if expected["optimizer"] != "auto" and type(trainer.optimizer).__name__ != expected["optimizer"]:
            raise RuntimeError("Effective optimizer differs from the requested recipe")
        if [trainer.data["names"][i] for i in range(len(trainer.data["names"]))] != CLASS_NAMES:
            raise ValueError("Dataset class order differs from the VSB ten-class contract")
        if "effective" not in receipt:
            if hard_normal["weight"]:
                from loss import install_hard_normal
                loss_runtime = install_hard_normal(trainer, hard_normal)
                ids = sorted(Path(p).name for p in loss_runtime.normal_paths)
                receipt["effective_hard_normal"] = {
                    **hard_normal, "criterion": type(loss_runtime).__name__,
                    "eligibility": "original empty annotation; no mixed-image augmentation; no augmented boxes",
                    "human_reviewed": False, "candidate_normal_source_images": len(ids),
                    "source_image_names_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
                    "scope": "training only; native EMA validation loss unchanged",
                }
            counts = {"train": len(trainer.train_loader.dataset), "val": len(trainer.test_loader.dataset)}
            if any(counts[split] != count for split, count in expected_counts.items()):
                raise RuntimeError(f"The native loader dropped samples: {counts}, expected {expected_counts}")
            initial_state = hashlib.sha256()
            for name, tensor in trainer.model.state_dict().items():
                initial_state.update(name.encode())
                initial_state.update(tensor.detach().contiguous().cpu().numpy().tobytes())
            receipt["effective"] = {
                "batch": trainer.batch_size, "amp": bool(trainer.amp),
                "per_worker_batch": trainer.train_loader.batch_size, "dataset_images": counts,
                "world_size": trainer.world_size, "train_sampler_images": len(trainer.train_loader.sampler),
                "device": str(trainer.device), "optimizer": type(trainer.optimizer).__name__,
                "initial_learning_rates": [g["lr"] for g in trainer.optimizer.param_groups],
                "optimizer_groups": [{k: v for k, v in g.items() if k != "params"}
                                     for g in trainer.optimizer.param_groups],
                "accumulate_after_warmup": trainer.accumulate,
                "parameter_dtype": str(next(trainer.model.parameters()).dtype),
                "parameters": sum(p.numel() for p in trainer.model.parameters()),
                "initial_model_state_sha256": initial_state.hexdigest(),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "class_names": trainer.data["names"], "args": vars(trainer.args),
            }
            receipt_path.write_text(json.dumps(receipt, indent=2, default=str) + "\n")

    def record_validation(validator):
        # Native training validation follows AMP, overriding the quantize request.
        phase = "during_training" if validator.training else "final_best"
        receipt.setdefault("validation", {})[phase] = {
            "quantize_arg": validator.args.quantize, "device": str(validator.device)}
        receipt_path.write_text(json.dumps(receipt, indent=2, default=str) + "\n")

    def record_loss_exposure(trainer):
        if loss_runtime is not None:
            receipt["hard_normal_observed"] = loss_runtime.summary()
            receipt_path.write_text(json.dumps(receipt, indent=2, default=str) + "\n")

    model.add_callback("on_val_start", record_validation)
    model.add_callback("on_train_start", check_runtime)
    model.add_callback("on_train_epoch_start", check_runtime)
    model.add_callback("on_train_epoch_end", record_loss_exposure)
    try:
        model.train(**kwargs)
        best = Path(model.trainer.best)
        if not best.is_file():
            raise FileNotFoundError(f"Training returned without best.pt: {best}")
        receipt.update(status="completed", best_weights=str(best))
        return best
    except BaseException as error:
        receipt.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                       error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if loss_runtime is not None:
            receipt["hard_normal_observed"] = loss_runtime.summary()
        if output.is_dir():
            receipt_path.write_text(json.dumps(receipt, indent=2, default=str) + "\n")
        if world_size > 1:
            import torch.distributed as dist
            if dist.is_initialized():
                dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--batch", type=int, help="Global batch across all GPUs")
    parser.add_argument("--device", help="Visible GPU indices, cpu or mps; multiple GPUs require torchrun")
    parser.add_argument("--run-dir", type=Path, help="New output directory, relative to project root")
    parser.add_argument("--resume", type=Path, help="Interrupted weights/last.pt; restore its saved settings")
    args = parser.parse_args()
    print(train(args.config, batch=args.batch, device=args.device, run_dir=args.run_dir, resume=args.resume))
