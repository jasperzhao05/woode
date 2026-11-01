"""YOLOv8n loading and image scoring; the network itself comes from Ultralytics."""

import os
from pathlib import Path


from config import DEFAULT_CONFIG, ROOT, load_config


# Local files only. Keep library settings inside the project, with cloud logging off.
settings_dir = ROOT / "runs" / ".ultralytics"
if "LOCAL_RANK" in os.environ:
    settings_dir /= f"{os.environ.get('TORCHELASTIC_RUN_ID', 'ddp')}-rank{os.environ['LOCAL_RANK']}"
os.environ.setdefault("YOLO_CONFIG_DIR", str(settings_dir))
os.environ["YOLO_OFFLINE"] = "true"
os.environ["YOLO_AUTOINSTALL"] = "false"


def build_model(weights):
    """Load a local .pt checkpoint or exported .onnx graph; missing weights are an error, not a download."""
    weights = Path(weights).expanduser().resolve(strict=True)
    if weights.suffix not in (".pt", ".onnx"):
        raise ValueError("Expected a local .pt detection checkpoint or an exported .onnx graph")
    from ultralytics import YOLO, settings

    settings.update({key: False for key in ("sync", "clearml", "comet", "dvc",
                     "mlflow", "neptune", "raytune", "tensorboard", "wandb")})
    model = YOLO(str(weights), task="detect")
    if model.task != "detect":
        raise ValueError(f"Expected a detection model, got {model.task}")
    return model


def predict_image(model, image, device="cpu", config_path=DEFAULT_CONFIG):
    """PIL image or local path -> score and Nx6 [x1,y1,x2,y2,confidence,class_id]."""
    from PIL import Image

    if isinstance(image, (str, Path)):
        with Image.open(Path(image).expanduser().resolve(strict=True)) as source:
            image = source.convert("RGB")
    else:
        image = image.convert("RGB")
    options = load_config(config_path)["predict"]
    backend = model.predictor.model if model.predictor is not None else None
    if backend is not None and backend.format != "pt" and not backend.dynamic:
        # An exported static graph fixes its input; the per-call imgsz must restate it instead of the recipe's long side.
        options["imgsz"] = list(backend.imgsz)
    result = model.predict(source=image, device=device, **options)[0]
    boxes = result.boxes.data.cpu().tolist()
    # Maximum post-NMS confidence over all classes, not a calibrated probability.
    score = max((box[4] for box in boxes), default=0.0)
    return {"score": score, "boxes": boxes}
