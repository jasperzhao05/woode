"""Prepare the existing VSB splits for YOLO; keep image labels for later evaluation."""

import argparse
import csv
import json
import math
import shutil
from pathlib import Path

from config import DEFAULT_CONFIG, load_config


EXPECTED_IMAGES = {"train": 14196, "val": 3042, "test": 3038}
CLASS_NAMES = ["Live_Knot", "Dead_Knot", "Knot_missing", "knot_with_crack",
               "Crack", "Quartzity", "resin", "Marrow", "Blue_Stain", "overgrown"]


def read_boxes(path):
    """Read YOLO rows: class_id, centre_x, centre_y, width, height (normalized)."""
    boxes = []
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5 or fields[0] not in {str(i) for i in range(10)}:
            raise ValueError(f"Invalid label at {path}:{number}")
        coords = [float(value) for value in fields[1:]]
        if not all(math.isfinite(value) and 0 <= value <= 1 for value in coords):
            raise ValueError(f"Invalid coordinates at {path}:{number}")
        if coords[2] == 0 or coords[3] == 0:
            raise ValueError(f"Empty box at {path}:{number}")
        boxes.append((int(fields[0]), *coords))
    return boxes


def prepare(root, output, include_test=False):
    """Return data.yaml; preserve assignments, link BMPs and copy small label files."""
    root, output = Path(root).resolve(strict=True), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists; choose a new directory: {output}")
    splits = ("train", "val", "test") if include_test else ("train", "val")
    prepared, groups = {}, {}
    for split in splits:
        image_dir, label_dir = root / "images" / split, root / "labels" / split
        images = sorted(image_dir.glob("*.bmp"))
        labels = {path.stem for path in label_dir.glob("*.txt") if path.is_file()}
        if len(images) != EXPECTED_IMAGES[split] or {p.stem for p in images} != labels:
            raise ValueError(f"Missing or mismatched images/labels in {split}: {root}")
        rows = []
        for image in images:
            image_id = image.stem
            if not image.is_file() or not image_id.isdigit() or len(image_id) <= 5 or int(image_id[-5:]) > 84:
                raise ValueError(f"Unexpected VSB image: {image}")
            group = image_id[:-5]  # Filename sequence, not a confirmed physical board ID.
            if group in groups and groups[group] != split:
                raise ValueError(f"Filename sequence {group} occurs in two splits")
            groups[group] = split
            annotation = label_dir / f"{image_id}.txt"
            boxes = read_boxes(annotation)
            rows.append({"image": image.relative_to(root).as_posix(),
                         "label": int(bool(boxes)),
                         "annotation": annotation.relative_to(root).as_posix()})
        prepared[split] = rows

    # Validate all requested partitions before writing. Never overwrite old outputs.
    output.mkdir(parents=True)
    for split, rows in prepared.items():
        (output / "images" / split).mkdir(parents=True)
        (output / "labels" / split).mkdir(parents=True)
        for row in rows:
            (output / row["image"]).symlink_to(root / row["image"])
            shutil.copyfile(root / row["annotation"], output / row["annotation"])
        with (output / f"{split}.csv").open("x", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["image", "label", "annotation"])
            writer.writeheader()
            writer.writerows(rows)
        positive = sum(row["label"] for row in rows)
        print(f"{split}: {len(rows)} images, {positive} positive, {len(rows) - positive} candidate normal")
    config = {"path": str(output), **{s: f"images/{s}" for s in splits}, "names": CLASS_NAMES}
    data_yaml = output / "data.yaml"
    with data_yaml.open("x") as handle:
        json.dump(config, handle, indent=2)  # JSON is valid YAML; no YAML writer dependency.
    return data_yaml


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--include-test", action="store_true", help="Prepare held-out test too, only when ready for final evaluation")
    args = parser.parse_args()
    paths = load_config(args.config)["paths"]
    prepare(paths["source"], paths["prepared"], include_test=args.include_test)
