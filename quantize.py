"""Export the FP32 checkpoint to ONNX, build reduced-precision variants, and measure size and CPU latency."""

import argparse
import collections
import csv
import hashlib
import json
import math
import os
import platform
import random
import shutil
import statistics
import subprocess
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np

from config import DEFAULT_CONFIG, ROOT, load_config
from data import CLASS_NAMES
from evaluate import metrics
from model import build_model

INPUT_NAME = 'images'
# Entropy is omitted: quantize_static fixes it at 128 histogram bins = 128 quantized bins, which reduces to MinMax.
STATIC_METHODS = {'int8-minmax': 'MinMax', 'int8-percentile': 'Percentile'}
QUANTIZED_OPS = {'Conv', 'Gemm', 'MatMul'}  # Weighted ops only; the box/score decode stays in floating point.
AGREEMENT_IMAGES = 8


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cpu_name():
    cpuinfo = Path('/proc/cpuinfo')
    if cpuinfo.is_file():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith('model name'):
                return line.split(':', 1)[1].strip()
    if platform.system() == 'Darwin':
        return subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], capture_output=True, text=True, check=True).stdout.strip()
    return platform.processor() or platform.machine()


def load_input(path, shape):
    """Reproduce the prediction preprocessing: RGB letterbox to the static input, scaled to [0, 1]."""
    from PIL import Image
    from ultralytics.data.augment import LetterBox

    with Image.open(path) as source:
        pixels = np.asarray(source.convert('RGB'))
    boxed = LetterBox(tuple(shape), auto=False, stride=32)(image=pixels)
    if boxed.shape != (*shape, 3):
        raise ValueError(f'Letterbox produced {boxed.shape}, expected {(*shape, 3)}')
    return np.ascontiguousarray(boxed.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0


def calibration_rows(prepared, count, seed):
    """Fixed-seed sample of training images; validation and test never calibrate the quantizer."""
    manifest = prepared / 'train.csv'
    with manifest.open() as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) < count or any(not row['image'].startswith('images/train/') for row in rows):
        raise ValueError(f'Need at least {count} training images in {manifest}')
    sample = random.Random(seed).sample(rows, count)
    images = [{'image': row['image'], 'label': int(row['label']), 'sha256': sha256(prepared / row['image'])} for row in sample]
    return {'split': 'train', 'manifest_rows': len(rows), 'manifest_sha256': sha256(manifest), 'seed': seed,
            'count': count, 'positive': sum(item['label'] for item in images), 'images': images}


def calibration_reader(paths, shape):
    from onnxruntime.quantization import CalibrationDataReader

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.position = 0

        def get_next(self):
            if self.position == len(paths):
                return None
            self.position += 1
            return {INPUT_NAME: load_input(paths[self.position - 1], shape)}

        def rewind(self):
            self.position = 0

    return Reader()


def export_fp32(checkpoint, output, shape, opset):
    """Copy the checkpoint next to its export so the historical run directory is never written."""
    import onnx

    source = output / 'source.pt'
    shutil.copyfile(checkpoint, source)
    model = build_model(source)
    if [model.names[i] for i in range(10)] != CLASS_NAMES:
        raise ValueError('Checkpoint class mapping differs from labels')
    exported = Path(model.export(format='onnx', imgsz=list(shape), batch=1, dynamic=False, simplify=False,
                                 opset=opset, device='cpu', verbose=False))
    target = output / 'fp32.onnx'
    exported.rename(target)
    graph = onnx.load(str(target))
    inputs = [(tensor.name, [dim.dim_value for dim in tensor.type.tensor_type.shape.dim]) for tensor in graph.graph.input]
    outputs = [(tensor.name, [dim.dim_value for dim in tensor.type.tensor_type.shape.dim]) for tensor in graph.graph.output]
    metadata = {prop.key: prop.value for prop in graph.metadata_props}
    if inputs != [(INPUT_NAME, [1, 3, *shape])] or len(outputs) != 1 or outputs[0][1][:2] != [1, 4 + len(CLASS_NAMES)]:
        raise ValueError(f'Unexpected exported graph: inputs {inputs}, outputs {outputs}')
    if metadata.get('imgsz') != str(list(shape)) or metadata.get('stride') != '32' or metadata.get('task') != 'detect':
        raise ValueError(f'Unexpected export metadata: {metadata}')
    return target, {'inputs': inputs, 'outputs': outputs, 'opset': [(entry.domain, entry.version) for entry in graph.opset_import],
                    'ir_version': graph.ir_version, 'metadata': metadata}


def sort_nodes(graph):
    """Restore topological order; the FP16 converter appends its input casts after the nodes that consume them."""
    available = {tensor.name for tensor in (*graph.input, *graph.initializer)} | {''}
    pending, ordered = list(graph.node), []
    while pending:
        ready = [node for node in pending if all(name in available for name in node.input)]
        if not ready:
            raise ValueError('Graph has inputs that no node or initializer produces')
        ordered.extend(ready)
        available.update(name for node in ready for name in node.output)
        taken = {id(node) for node in ready}
        pending = [node for node in pending if id(node) not in taken]
    del graph.node[:]
    graph.node.extend(ordered)


def finalize(path, reference):
    """Sort the graph and restore export metadata that the prediction backend reads (imgsz, names, stride)."""
    import onnx

    model = onnx.load(str(path))
    sort_nodes(model.graph)
    present = {prop.key: prop.value for prop in model.metadata_props}
    if any(present.get(key, value) != value for key, value in reference.items()):
        raise ValueError(f'Export metadata changed in {path.name}')
    for key, value in reference.items():
        if key not in present:
            prop = model.metadata_props.add()
            prop.key, prop.value = key, value
    onnx.save(model, str(path))
    return collections.Counter(node.op_type for node in model.graph.node)


def build_variant(name, work, output, calibration_paths, settings):
    import onnx
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_dynamic, quantize_static

    quant_types = {'uint8': QuantType.QUInt8, 'int8': QuantType.QInt8}
    target = output / f'{name}.onnx'
    if target.exists():
        raise FileExistsError(target)
    if name == 'fp16':
        from onnxruntime.transformers import float16

        onnx.save(float16.convert_float_to_float16(onnx.load(str(work)), keep_io_types=True), str(target))
        recipe = {'method': 'onnxruntime.transformers.float16.convert_float_to_float16', 'keep_io_types': True}
    elif name == 'int8-dynamic':
        # Weights quantized offline, activations per run; the CPU ConvInteger kernel only accepts uint8 weights.
        quantize_dynamic(str(work), str(target), weight_type=QuantType.QUInt8, per_channel=False)
        recipe = {'method': 'quantize_dynamic', 'weight_type': 'uint8', 'per_channel': False}
    else:
        graph = onnx.load(str(work)).graph
        exclude = [node.name for node in graph.node if node.op_type not in QUANTIZED_OPS]
        extra = {'CalibMaxIntermediateOutputs': 16, 'CalibPercentile': 99.999}
        quantize_static(str(work), str(target), calibration_reader(calibration_paths, settings['input_shape']),
                        quant_format=QuantFormat.QDQ, per_channel=settings['per_channel'], reduce_range=settings['reduce_range'],
                        activation_type=quant_types[settings['activation_type']], weight_type=quant_types[settings['weight_type']],
                        nodes_to_exclude=exclude, calibrate_method=getattr(CalibrationMethod, STATIC_METHODS[name]),
                        extra_options=extra)
        recipe = {'method': 'quantize_static', 'format': 'QDQ', 'calibrate_method': STATIC_METHODS[name],
                  'quantized_op_types': sorted(QUANTIZED_OPS), 'excluded_nodes': len(exclude),
                  **{key: settings[key] for key in ('activation_type', 'weight_type', 'per_channel', 'reduce_range')},
                  'extra_options': extra, 'calibration_images': len(calibration_paths)}
    return target, recipe


def session_for(path, threads):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])


def benchmark(path, sample, threads, warmup, runs):
    """Model forward only on the CPU execution provider: no decoding, letterbox or NMS."""
    session = session_for(path, threads or os.cpu_count())
    feed = {INPUT_NAME: sample}
    for _ in range(warmup):
        session.run(None, feed)
    timings = []
    for _ in range(runs):
        start = time.perf_counter()
        session.run(None, feed)
        timings.append((time.perf_counter() - start) * 1000)
    timings.sort()
    median = statistics.median(timings)
    return {'threads': threads or os.cpu_count(), 'warmup': warmup, 'runs': runs, 'median_ms': median,
            'mean_ms': statistics.fmean(timings), 'p90_ms': timings[math.ceil(0.9 * runs) - 1], 'min_ms': timings[0],
            'images_per_second': 1000 / median}


def host_record():
    import onnxruntime as ort

    return {'hostname': platform.node(), 'cpu': cpu_name(), 'machine': platform.machine(), 'system': platform.platform(),
            'logical_cores': os.cpu_count(), 'onnxruntime': ort.__version__, 'providers_available': ort.get_available_providers(),
            'measured_at_utc': datetime.now(timezone.utc).isoformat()}


def benchmark_all(directory, variants, settings):
    sample = np.load(directory / 'benchmark-input.npy')
    results = {}
    for name in variants:
        results[name] = [benchmark(directory / f'{name}.onnx', sample, threads, settings['warmup'], settings['runs'])
                         for threads in settings['threads']]
        print(f"{name}: " + ', '.join(f"{row['threads']} thr {row['median_ms']:.2f} ms" for row in results[name]), flush=True)
    return {**host_record(), 'input': 'benchmark-input.npy', 'scope': 'model forward only; excludes image decoding, letterbox and NMS',
            'results': results}


def agreement(reference, variant, samples):
    """Raw head output deltas against the FP32 graph; class rows are sigmoid probabilities, box rows are pixels."""
    sessions = session_for(reference, 0), session_for(variant, 0)
    class_deltas, box_deltas = [], []
    for sample in samples:
        base, other = (session.run(None, {INPUT_NAME: sample})[0] for session in sessions)
        box_deltas.append(float(np.abs(base[:, :4] - other[:, :4]).max()))
        class_deltas.append(np.abs(base[:, 4:] - other[:, 4:]))
    stacked = np.stack(class_deltas)
    return {'images': len(samples), 'max_abs_class_delta': float(stacked.max()), 'mean_abs_class_delta': float(stacked.mean()),
            'max_abs_box_delta_px': max(box_deltas)}


def quantize(config_path, weights, prepared, output):
    import onnx
    from onnxruntime.quantization.shape_inference import quant_pre_process

    config_path = Path(config_path).resolve(strict=True)
    config = load_config(config_path)
    settings = config.get('quantize')
    if settings is None:
        raise ValueError('Config needs a [quantize] table')
    weights = Path(weights).resolve(strict=True)
    if weights.suffix != '.pt':
        raise ValueError('Quantization starts from the FP32 .pt checkpoint')
    prepared = Path(prepared or config['paths']['prepared']).resolve(strict=True)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    shape = settings['input_shape']
    receipt = {'status': 'running', 'started_at_utc': datetime.now(timezone.utc).isoformat(),
               'checkpoint': {'path': str(weights), 'sha256': sha256(weights), 'bytes': weights.stat().st_size,
                              'note': 'Ultralytics checkpoints store FP16 weights plus optimizer-free training state'},
               'settings': settings, 'predict': config['predict'],
               'sha256': {str(path): sha256(path) for path in (config_path, *(ROOT / name for name in
                          ('quantize.py', 'evaluate.py', 'model.py', 'config.py', 'data.py')))},
               'versions': {name: version(name) for name in ('ultralytics', 'torch', 'onnx', 'onnxruntime', 'numpy', 'Pillow')},
               'limitation': 'Size and latency are measured here; detection quality comes from evaluate.py on the same files'}
    record = output / 'quantize.json'
    record.write_text(json.dumps(receipt, indent=2) + '\n')
    (output / 'quantize-config.toml').write_bytes(config_path.read_bytes())
    try:
        fp32, graph_info = export_fp32(weights, output, shape, settings['opset'])
        receipt['export'] = graph_info
        work = output / 'fp32-inferred.onnx'
        quant_pre_process(str(fp32), str(work))  # symbolic shape inference + ORT graph optimization for the quantizers
        metadata = graph_info['metadata']
        finalize(work, metadata)
        calibration = calibration_rows(prepared, settings['calibration_images'], settings['calibration_seed'])
        receipt['calibration'] = calibration
        calibration_paths = [prepared / item['image'] for item in calibration['images']]
        samples = [load_input(path, shape) for path in calibration_paths[:AGREEMENT_IMAGES]]
        np.save(output / 'benchmark-input.npy', samples[0])
        receipt['benchmark_input'] = {'image': calibration['images'][0]['image'], 'shape': list(samples[0].shape), 'dtype': 'float32'}
        fp32_bytes = fp32.stat().st_size
        variants = {'fp32': {'file': fp32.name, 'bytes': fp32_bytes, 'sha256': sha256(fp32), 'recipe': {'method': 'ultralytics export, fused, FP32'},
                             'ops': dict(collections.Counter(node.op_type for node in onnx.load(str(fp32)).graph.node))}}
        for name in settings['variants']:
            started = time.monotonic()
            target, recipe = build_variant(name, work, output, calibration_paths, settings)
            ops = finalize(target, metadata)
            onnx.checker.check_model(str(target))
            variants[name] = {'file': target.name, 'bytes': target.stat().st_size, 'sha256': sha256(target), 'build_seconds': time.monotonic() - started,
                              'recipe': recipe, 'ops': dict(ops), 'agreement_with_fp32': agreement(fp32, target, samples)}
            print(f'{name}: {target.stat().st_size / 1e6:.2f} MB, built in {variants[name]["build_seconds"]:.1f} s', flush=True)
        for entry in variants.values():
            entry['size_ratio_vs_fp32_onnx'] = fp32_bytes / entry['bytes']
        receipt['variants'] = variants
        receipt['benchmark'] = benchmark_all(output, list(variants), settings)
        receipt.update(status='completed', finished_at_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        receipt.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        record.write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


def benchmark_existing(directory):
    """Re-measure latency of a completed quantize.py or sensitivity.py directory on this host."""
    directory = Path(directory).resolve(strict=True)
    record = next((directory / name for name in ('quantize.json', 'sensitivity.json') if (directory / name).is_file()), None)
    if record is None:
        raise FileNotFoundError(f'No quantize.json or sensitivity.json in {directory}')
    receipt = json.loads(record.read_text())
    if receipt.get('status') != 'completed':
        raise ValueError('Benchmark a completed directory')
    for name, entry in receipt['variants'].items():
        if sha256(directory / entry['file']) != entry['sha256']:
            raise ValueError(f'{entry["file"]} changed since quantization')
    result = benchmark_all(directory, list(receipt['variants']), receipt['settings'])
    target = directory / f"benchmark-{platform.node()}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    with target.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    return result


def read_scores(directory):
    directory = Path(directory).resolve(strict=True)
    receipt = json.loads((directory / 'metrics.json').read_text())
    if receipt.get('status') != 'completed' or receipt.get('split') != 'val':
        raise ValueError(f'{directory} is not a completed validation evaluation')
    with (directory / 'scores.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row['label'], row['score'] = int(row['label']), float(row['score'])
    return receipt, rows


def compare(reference_dir, variant_dirs, output):
    """Apply the FP32 validation-selected thresholds unchanged to each variant's validation scores."""
    output = Path(output).resolve()
    reference, base_rows = read_scores(reference_dir)
    base = {row['image']: row for row in base_rows}
    thresholds = {f"recall_{point['target_recall']}": point['selected']['threshold']
                  for point in reference['validation_selected_operating_points'] if point['selected']}
    thresholds.update({f'fixed_{value}': value for value in reference['fixed_thresholds']})
    report = {'schema': 'quantization-comparison-v1', 'reference': str(Path(reference_dir).resolve()),
              'reference_weights': reference['input_recipe'], 'thresholds': thresholds, 'counts': reference['counts'],
              'decision': 'score >= threshold with the FP32 threshold frozen; no per-variant retuning', 'variants': {}}
    for directory in variant_dirs:
        receipt, rows = read_scores(directory)
        if receipt['counts'] != reference['counts'] or {row['image'] for row in rows} != set(base):
            raise ValueError(f'{directory} scores a different image set')
        deltas = [abs(row['score'] - base[row['image']]['score']) for row in rows]
        entry = {'input_recipe': receipt['input_recipe'], 'weights_sha256': [digest for path, digest in receipt['sha256'].items()
                                                                                if path.endswith(('.pt', '.onnx'))],
                 'score_agreement': {'max_abs_delta': max(deltas), 'mean_abs_delta': statistics.fmean(deltas),
                                     'images_delta_over_0.05': sum(delta > 0.05 for delta in deltas)},
                 'at_frozen_thresholds': {}, 'own_selected': receipt['validation_selected_operating_points']}
        for key, threshold in thresholds.items():
            result = metrics(rows, threshold)
            result['decision_flips'] = sum((row['score'] >= threshold) != (base[row['image']]['score'] >= threshold) for row in rows)
            entry['at_frozen_thresholds'][key] = result
        report['variants'][Path(directory).resolve().name] = entry
    with output.open('x') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
    print(f"{'variant':24s} {'thr':>8s} {'recall':>8s} {'FP':>4s} {'FPR':>7s} {'flips':>5s}")
    for name, entry in report['variants'].items():
        for key, result in entry['at_frozen_thresholds'].items():
            if key.startswith('recall_'):
                print(f"{name:24s} {result['threshold']:8.4f} {result['recall']:8.4%} {result['FP']:4d} "
                      f"{result['false_positive_rate']:7.2%} {result['decision_flips']:5d}")
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--weights', type=Path, help='FP32 .pt checkpoint to export and quantize')
    parser.add_argument('--prepared', type=Path)
    parser.add_argument('--output', type=Path, help='New directory for quantize, or the comparison JSON for --compare')
    parser.add_argument('--benchmark', type=Path, help='Re-measure latency of an existing quantization directory on this host')
    parser.add_argument('--compare', type=Path, nargs='+', metavar='EVAL_DIR',
                        help='FP32 validation evaluation directory followed by variant evaluation directories')
    args = parser.parse_args()
    if args.benchmark:
        print(json.dumps(benchmark_existing(args.benchmark), indent=2))
    elif args.compare:
        if len(args.compare) < 2 or not args.output:
            parser.error('--compare needs the FP32 reference, at least one variant, and --output JSON path')
        compare(args.compare[0], args.compare[1:], args.output)
    else:
        if not args.weights or not args.output:
            parser.error('--weights and --output are required to quantize')
        receipt = quantize(args.config, args.weights, args.prepared, args.output)
        print(json.dumps({name: {'bytes': entry['bytes'], 'ratio': round(entry['size_ratio_vs_fp32_onnx'], 2)}
                          for name, entry in receipt['variants'].items()}, indent=2))
