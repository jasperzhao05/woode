"""Read explicit experiment settings with Python 3.11+; no config framework."""

from pathlib import Path
import math
import tomllib

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "vsb-musgd.toml"


def load_config(path=DEFAULT_CONFIG):
    """Read flat recipe tables plus optional loss and quantization settings; resolve root-relative data paths."""
    path = Path(path).expanduser().resolve(strict=True)
    with path.open("rb") as handle:
        config = tomllib.load(handle)
    if not {"paths", "train", "predict"} <= set(config) or set(config) - {"paths", "train", "predict", "hard_normal", "quantize"}:
        raise ValueError("Config requires paths/train/predict and optionally hard_normal/quantize")
    if not all(isinstance(v, dict) for v in config.values()):
        raise ValueError("All top-level config entries must be tables")
    if set(config["paths"]) != {"source", "prepared", "weights", "output"}:
        raise ValueError("[paths] requires source, prepared, weights and output")
    for key, value in config["paths"].items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"paths.{key} must be a nonempty path")
        config["paths"][key] = (ROOT / Path(value).expanduser()).resolve()
    if not {"batch", "amp", "device", "optimizer"} <= config["train"].keys():
        raise ValueError("[train] requires batch, amp, device and optimizer")
    loss = config.setdefault("hard_normal", {"weight": 0.0, "topk": 10})
    if set(loss) != {"weight", "topk"}:
        raise ValueError("[hard_normal] requires exactly weight and topk")
    if type(loss["weight"]) not in (int, float) or not math.isfinite(loss["weight"]) or loss["weight"] < 0:
        raise ValueError("hard_normal.weight must be finite and nonnegative")
    if type(loss["topk"]) is not int or loss["topk"] < 1:
        raise ValueError("hard_normal.topk must be a positive integer")
    if loss["weight"]:
        if any(config["train"].get(key, 0) != 0 for key in ("mosaic", "mixup", "cutmix", "copy_paste")):
            raise ValueError("Hard-normal loss requires mosaic/mixup/cutmix/copy_paste = 0")
        if config["train"].get("compile", False):
            raise ValueError("Hard-normal loss has not been qualified with compile")
    if "quantize" in config:
        check_quantize(config["quantize"], config["predict"].get("imgsz"))
    return config


QUANTIZE_VARIANTS = ("fp16", "int8-dynamic", "int8-minmax", "int8-percentile")


def check_quantize(settings, imgsz):
    """Validate the explicit ONNX export, calibration and benchmark settings."""
    keys = {"input_shape", "opset", "variants", "calibration_images", "calibration_seed", "activation_type",
            "weight_type", "per_channel", "reduce_range", "warmup", "runs", "threads",
            "sensitivity_images", "sensitivity_seed", "mixed_keep_fp32", "mixed_method"}
    if set(settings) != keys:
        raise ValueError(f"[quantize] requires exactly {sorted(keys)}")
    shape = settings["input_shape"]
    if (not isinstance(shape, list) or len(shape) != 2 or any(type(side) is not int or side <= 0 or side % 32 for side in shape)
            or max(shape) != imgsz):
        raise ValueError("quantize.input_shape must be two positive multiples of 32 whose long side equals predict.imgsz")
    if type(settings["opset"]) is not int or not 13 <= settings["opset"] <= 20:
        raise ValueError("quantize.opset must be 13..20 (per-channel QDQ needs 13; ONNX Runtime static INT8 stops at 20)")
    variants = settings["variants"]
    if (not isinstance(variants, list) or not variants or len(set(variants)) != len(variants)
            or not set(variants) <= set(QUANTIZE_VARIANTS)):
        raise ValueError(f"quantize.variants must be distinct names from {QUANTIZE_VARIANTS}")
    for key in ("calibration_images", "calibration_seed", "warmup", "runs", "sensitivity_images", "sensitivity_seed", "mixed_keep_fp32"):
        if type(settings[key]) is not int or settings[key] < (0 if key.endswith("seed") else 1):
            raise ValueError(f"quantize.{key} must be a positive integer")
    if settings["mixed_method"] not in ("minmax", "percentile"):
        raise ValueError("quantize.mixed_method must be 'minmax' or 'percentile'")
    for key in ("activation_type", "weight_type"):
        if settings[key] not in ("uint8", "int8"):
            raise ValueError(f"quantize.{key} must be 'uint8' or 'int8'")
    for key in ("per_channel", "reduce_range"):
        if type(settings[key]) is not bool:
            raise ValueError(f"quantize.{key} must be a boolean")
    threads = settings["threads"]
    if (not isinstance(threads, list) or not threads or len(set(threads)) != len(threads)
            or any(type(count) is not int or count < 0 for count in threads)):
        raise ValueError("quantize.threads must be distinct nonnegative integers (0 = all logical cores)")
