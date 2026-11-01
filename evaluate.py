"""One image-level metric contract: calibrate on val, freeze threshold for test."""

import argparse
import csv
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from config import DEFAULT_CONFIG, ROOT, load_config
from data import CLASS_NAMES, EXPECTED_IMAGES, read_boxes
from model import build_model, predict_image


def metrics(rows, threshold):
    """A zero score means no retained prediction; threshold must be positive."""
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("Use a finite positive threshold")
    tp = sum(row['label'] == 1 and row['score'] >= threshold for row in rows)
    fp = sum(row['label'] == 0 and row['score'] >= threshold for row in rows)
    positive = sum(row['label'] for row in rows)
    negative = len(rows) - positive
    return {'threshold': threshold, 'TP': tp, 'FP': fp, 'TN': negative - fp, 'FN': positive - tp,
            'precision': tp / (tp + fp) if tp + fp else None,
            'recall': tp / positive if positive else None,
            'false_positive_rate': fp / negative if negative else None}


def write_csv(path, rows):
    with path.open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def select_threshold(rows, target, score_floor):
    """Highest retained threshold attaining target recall; include ties together."""
    if not 0 < target <= 1:
        raise ValueError("Target recall must be in (0,1]")
    positives = sorted((r['score'] for r in rows if r['label'] == 1), reverse=True)
    if not positives:
        raise ValueError("Recall calibration requires positive validation images")
    threshold = positives[math.ceil(target * len(positives)) - 1]
    if threshold < score_floor or threshold <= 0:
        raise ValueError("Target is unattainable above retained prediction floor; do not claim zero-score detections")
    return metrics(rows, threshold)


def calibrate_from(source, output, target=0.999):
    """Freeze a threshold from a completed validation score file, without inference."""
    source, output = Path(source).resolve(strict=True), Path(output).resolve()
    record_path, scores_path = source / 'metrics.json', source / 'scores.csv'
    receipt = json.loads(record_path.read_text())
    if receipt.get('status') != 'completed' or receipt.get('split') != 'val':
        raise ValueError('Calibration requires a completed validation evaluation, never test')
    with scores_path.open() as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row['label'], row['score'] = int(row['label']), float(row['score'])
        if row['label'] not in (0, 1) or not math.isfinite(row['score']) or not 0 <= row['score'] <= 1:
            raise ValueError('Malformed validation score')
    counts = {'images': len(rows), 'positive': sum(r['label'] for r in rows),
              'candidate_normal': sum(r['label'] == 0 for r in rows)}
    if counts != receipt['counts'] or len({r['image'] for r in rows}) != len(rows):
        raise ValueError('Score count/identity mismatch with evaluation receipt')
    if not counts['candidate_normal']:
        raise ValueError('FPR calibration requires candidate-normal validation images')
    checkpoint_hashes = [digest for path, digest in receipt['sha256'].items() if path.endswith('.pt')]
    if len(checkpoint_hashes) != 1:
        raise ValueError('Expected exactly one checkpoint identity in evaluation receipt')
    selected = select_threshold(rows, target, receipt['requested_predict']['conf'])
    calibration = {'schema': 'image-rejection-calibration-v1', 'source_split': 'val',
                   'target_recall': target, 'threshold': selected['threshold'],
                   'validation_result': selected, 'counts': counts,
                   'weights_sha256': checkpoint_hashes[0],
                   'predict': receipt['requested_predict'], 'class_names': CLASS_NAMES,
                   'tf32': False, 'decision': 'score >= threshold',
                   'score_definition': receipt['score_definition'],
                   'positive_definition': receipt['positive_definition'],
                   'source': str(source),
                   'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (record_path, scores_path)},
                   'limitation': 'Selected and measured on val; test must use this unchanged threshold; no industrial certification'}
    output.mkdir(parents=True, exist_ok=False)
    with (output / 'calibration.json').open('x') as handle:
        json.dump(calibration, handle, indent=2)
        handle.write('\n')
    return calibration


def declared_onnx_input(path):
    """Shape of the single float32 input declared by an exported graph."""
    import onnx

    graph = onnx.load(str(path)).graph
    inputs = [tensor for tensor in graph.input if tensor.name not in {init.name for init in graph.initializer}]
    if len(inputs) != 1 or inputs[0].type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
        raise ValueError('Expected exactly one float32 graph input')
    shape = tuple(dim.dim_value for dim in inputs[0].type.tensor_type.shape.dim)
    if len(shape) != 4 or not all(side > 0 for side in shape):
        raise ValueError(f'Expected a static NCHW graph input, got {shape}')
    return shape


def read_calibration(path, weights, predict):
    calibration = json.loads(Path(path).read_text())
    expected = {'schema': 'image-rejection-calibration-v1', 'source_split': 'val',
                'weights_sha256': hashlib.sha256(Path(weights).read_bytes()).hexdigest(),
                'predict': predict, 'class_names': CLASS_NAMES, 'tf32': False,
                'decision': 'score >= threshold',
                'score_definition': 'maximum post-NMS confidence over all original classes; no boxes -> 0',
                'positive_definition': 'any of the original ten annotated defect classes'}
    if any(calibration.get(k) != v for k, v in expected.items()):
        raise ValueError('Calibration does not match checkpoint, prediction recipe or metric contract')
    threshold = calibration['threshold']
    if not math.isfinite(threshold) or not predict['conf'] <= threshold <= 1:
        raise ValueError('Calibration threshold is outside retained score support')
    return calibration


def evaluate(config_path, weights, prepared, output, device, split='val', calibration_path=None, target=0.999):
    config_path = Path(config_path).resolve(strict=True)
    config = load_config(config_path)
    prepared = Path(prepared or config['paths']['prepared']).resolve(strict=True)
    weights = Path(weights).resolve(strict=True)
    if weights.suffix not in ('.pt', '.onnx'):
        raise ValueError('Weights must be a .pt checkpoint or an exported .onnx graph')
    exported = weights.suffix == '.onnx'
    output = Path(output).resolve()
    options = config['predict']
    if not math.isfinite(target) or not 0 < target <= 1:
        raise ValueError('Target recall must be in (0,1]')
    if split not in ('val', 'test') or (split == 'test') != (calibration_path is not None):
        raise ValueError('Test requires --calibration from val; val must select its own threshold')
    calibration = read_calibration(calibration_path, weights, options) if calibration_path else None
    # Fixed decision recipe; only input size, letterbox mode and the retained-score floor are configurable.
    # Numeric precision comes from the weights file: a PyTorch FP32 checkpoint or an exported ONNX graph.
    required = {'iou': 0.7, 'max_det': 300, 'agnostic_nms': False, 'augment': False, 'quantize': 32, 'batch': 1}
    if any(options.get(key) != value for key, value in required.items()):
        raise ValueError('Evaluation requires the declared prediction recipe (iou/max_det/nms/augment/quantize/batch)')
    imgsz, rect, score_floor = options.get('imgsz'), options.get('rect'), options.get('conf')
    if type(imgsz) is not int or imgsz <= 0 or imgsz % 32:
        raise ValueError('predict.imgsz must be a positive multiple of 32')
    if type(rect) is not bool:
        raise ValueError('predict.rect must be a boolean')
    if type(score_floor) not in (int, float) or not 0 < score_floor <= 0.001:
        raise ValueError('predict.conf is the retained-score floor and must be in (0, 0.001]')
    recipe_keys = ('imgsz', 'rect', 'conf', *required)

    def shape_ok(shape):
        # rect=False letterboxes to a square; rect=True pads only to the stride, long side = imgsz.
        square = shape == (1, 3, imgsz, imgsz)
        padded = len(shape) == 4 and shape[:2] == (1, 3) and max(shape[2:]) == imgsz and not any(side % 32 for side in shape[2:])
        return padded if rect else square

    graph_input = declared_onnx_input(weights) if exported else None
    if exported and (not shape_ok(graph_input) or (not rect) != (graph_input[2] == graph_input[3])):
        raise ValueError(f'Static ONNX input {graph_input} does not match the letterbox recipe')
    backend = f'ONNX Runtime, static graph {weights.name}' if exported else 'PyTorch FP32, TF32 off'
    manifest = prepared / f'{split}.csv'
    with manifest.open() as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != EXPECTED_IMAGES[split] or len({row['image'] for row in rows}) != len(rows):
        raise ValueError(f'Expected all {EXPECTED_IMAGES[split]} unique {split} images')
    label_hash = hashlib.sha256()
    class_images = [0] * 10
    class_boxes = [0] * 10
    for row in rows:
        if not row['image'].startswith(f'images/{split}/') or not row['annotation'].startswith(f'labels/{split}/'):
            raise ValueError(f'Manifest contains a different split: {row}')
        row['label'] = int(row['label'])
        image = prepared / row['image']
        annotation = prepared / row['annotation']
        if not image.is_file() or not image.stem.isdigit():
            raise ValueError(f'Missing or unexpected image: {image}')
        boxes = read_boxes(annotation)
        row['gt_classes'] = sorted({box[0] for box in boxes})
        if row['label'] != int(bool(boxes)):
            raise ValueError(f'CSV/annotation disagreement: {image}')
        label_hash.update(row['annotation'].encode() + b'\0' + annotation.read_bytes() + b'\0')
        for class_id in row['gt_classes']:
            class_images[class_id] += 1
        for box in boxes:
            class_boxes[box[0]] += 1
    if split == 'val' and sum(row['label'] for row in rows) != 2718:
        raise ValueError('Expected 2,718 positive and 324 candidate-normal images')
    positive = sum(row['label'] for row in rows)
    output.mkdir(parents=True, exist_ok=False)
    receipt = {'status': 'running', 'started_at_utc': datetime.now(timezone.utc).isoformat(),
               'unit': 'one dataset image, not a verified physical product', 'split': split,
               'positive_definition': 'any of the original ten annotated defect classes',
               'score_definition': 'maximum post-NMS confidence over all original classes; no boxes -> 0',
               'decision': 'score >= threshold', 'requested_predict': options, 'device': device,
               'score_floor': score_floor,
               'input_recipe': f"long side {imgsz}, {'stride-padded letterbox (rect)' if rect else 'square letterbox'}, {backend}",
               'weights_format': 'onnx' if exported else 'pt',
               'fixed_thresholds': [0.05, 0.10, 0.25, 0.50, 0.75],
               'diagnostic_recall_targets': [0.90, 0.95, 0.99, target] if split == 'val' else [],
               'threshold_scope': 'validation selection; not independent performance' if split == 'val' else 'frozen validation threshold; no test retuning',
               'calibration': calibration,
               'checkpoint_selection': 'historical best selected by native validation mAP50-95',
               'counts': {'images': len(rows), 'positive': positive, 'candidate_normal': len(rows) - positive},
               'classes': {name: {'images': class_images[i], 'boxes': class_boxes[i]} for i, name in enumerate(CLASS_NAMES)},
               'sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in
                          (weights, config_path, manifest, *((Path(calibration_path),) if calibration_path else ()), *(ROOT / name for name in
                           ('evaluate.py', 'model.py', 'config.py', 'data.py')))},
               'annotation_sha256': label_hash.hexdigest(), 'image_content_rehashed': False,
               'versions': {name: version(name) for name in ('ultralytics', 'torch', 'torchvision', 'numpy', 'Pillow',
                                                             *(('onnx', 'onnxruntime') if exported else ()))}}
    record = output / 'metrics.json'
    record.write_text(json.dumps(receipt, indent=2) + '\n')
    (output / 'prediction-config.toml').write_bytes(config_path.read_bytes())
    started = time.monotonic()
    observed = set()

    def check_input(module, inputs):
        tensor = inputs[0]
        shape = tuple(tensor.shape)
        if not shape_ok(shape) or str(tensor.dtype) != 'torch.float32':
            raise RuntimeError(f'Unexpected model input: {shape}, {tensor.dtype}')
        observed.add((shape, str(tensor.dtype), str(tensor.device)))

    try:
        import torch
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model = build_model(weights)
        if [model.names[i] for i in range(10)] != CLASS_NAMES:
            raise ValueError('Checkpoint class mapping differs from labels')
        if exported:
            # A static graph rejects any other input shape at run time; record the declared one.
            hook = None
            observed.add((graph_input, 'float32 (static ONNX graph input)', 'onnxruntime'))
        else:
            hook = model.model.register_forward_pre_hook(check_input)
        scores = []
        with (output / 'scores.csv').open('x', newline='') as handle:
            writer = None
            for index, row in enumerate(rows, 1):
                result = predict_image(model, prepared / row['image'], device=device, config_path=config_path)
                score = result['score']
                if not math.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError(f'Invalid score: {score}')
                class_scores = [0.0] * 10
                for box in result['boxes']:
                    class_id = int(box[5])
                    class_scores[class_id] = max(class_scores[class_id], box[4])
                item = {'image': row['image'], 'sequence': Path(row['image']).stem[:-5],
                        'label': row['label'], 'gt_classes': json.dumps(row['gt_classes']),
                        'score': score, 'class_scores': json.dumps(class_scores)}
                if writer is None:
                    writer = csv.DictWriter(handle, fieldnames=list(item))
                    writer.writeheader()
                writer.writerow(item)
                scores.append(item)
                if index == 1:
                    predictor = model.predictor
                    effective = {'imgsz': predictor.imgsz, 'fp16': predictor.model.fp16,
                                 'device': str(predictor.device),
                                 'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
                                 'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
                                 'args': {key: getattr(predictor.args, key) for key in recipe_keys}}
                    expected_imgsz = list(graph_input[2:]) if exported else [imgsz, imgsz]
                    if effective['fp16'] or effective['imgsz'] != expected_imgsz or not observed:
                        raise RuntimeError(f'Unexpected effective prediction settings: {effective}')
                    receipt['effective_predict'] = effective
                    record.write_text(json.dumps(receipt, indent=2) + '\n')
                if index % 250 == 0 or index == len(rows):
                    handle.flush()
                    print(f'{weights.parent.parent.name}: {index}/{len(rows)} images', flush=True)
        if hook is not None:
            hook.remove()
        targets = []
        if split == 'val':
            thresholds = sorted({1.000001, options['conf'], *(row['score'] for row in scores if row['score'] > 0)}, reverse=True)
            curve = [metrics(scores, threshold) for threshold in thresholds]
            write_csv(output / 'threshold-curve.csv', curve)
            for diagnostic_target in dict.fromkeys(receipt['diagnostic_recall_targets']):
                candidates = [item for item in curve if item['recall'] >= diagnostic_target]
                targets.append({'target_recall': diagnostic_target, 'attainable_above_score_floor': bool(candidates),
                                'selected': candidates[0] if candidates else None})
        else:
            receipt['frozen_threshold_result'] = metrics(scores, calibration['threshold'])
        receipt.update(status='completed', finished_at_utc=datetime.now(timezone.utc).isoformat(),
                       elapsed_seconds=time.monotonic() - started,
                       observed_inputs=[{'shape': shape, 'dtype': dtype, 'device': dev} for shape, dtype, dev in sorted(observed)],
                       fixed_threshold_results=[metrics(scores, threshold) for threshold in receipt['fixed_thresholds']] if split == 'val' else [],
                       validation_selected_operating_points=targets)
    except BaseException as error:
        receipt.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        record.write_text(json.dumps(receipt, indent=2) + '\n')
    if split == 'val':
        try:
            calibrate_from(output, output / 'threshold-lock', target)
        except ValueError as error:
            receipt['calibration_error'] = str(error)
            record.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({key: receipt[key] for key in ('counts', 'fixed_threshold_results', 'validation_selected_operating_points', 'frozen_threshold_result') if key in receipt}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--weights', type=Path)
    parser.add_argument('--prepared', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--split', choices=('val', 'test'), default='val')
    parser.add_argument('--calibration', type=Path, help='Frozen validation calibration.json; required for test')
    parser.add_argument('--target-recall', type=float, default=0.999, help='Validation calibration target only')
    parser.add_argument('--calibrate-from', type=Path, help='Reuse a completed val evaluation directory; no model inference')
    args = parser.parse_args()
    if args.calibrate_from:
        if args.split != 'val' or args.calibration or args.weights:
            parser.error('--calibrate-from accepts val scores only, without weights/calibration')
        print(json.dumps(calibrate_from(args.calibrate_from, args.output, args.target_recall), indent=2))
    else:
        if args.weights is None:
            parser.error('--weights is required for model inference')
        evaluate(args.config, args.weights, args.prepared, args.output, args.device,
                 args.split, args.calibration, args.target_recall)
