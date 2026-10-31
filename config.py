"""Read explicit experiment settings with Python 3.11+; no config framework."""

from pathlib import Path
import math
import tomllib

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "vsb-musgd.toml"


def load_config(path=DEFAULT_CONFIG):
    """Read flat recipe tables plus optional loss settings; resolve root-relative data paths."""
    path = Path(path).expanduser().resolve(strict=True)
    with path.open("rb") as handle:
        config = tomllib.load(handle)
    if not {"paths", "train", "predict"} <= set(config) or set(config) - {"paths", "train", "predict", "hard_normal"}:
        raise ValueError("Config requires paths/train/predict and optionally hard_normal")
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
    return config
