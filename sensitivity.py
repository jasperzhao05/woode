"""Per-layer INT8 sensitivity of the exported detector, and one mixed-precision variant built from the ranking."""

import argparse
import csv
import json
import random
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from config import DEFAULT_CONFIG, ROOT, load_config
from quantize import (INPUT_NAME, QUANTIZED_OPS, STATIC_METHODS, benchmark, finalize, host_record, load_input, session_for,
                      sha256)

MIXED = 'int8-mixed'


def image_score(output, floor):
    """Maximum class probability over anchors equals the post-NMS maximum confidence; below the floor nothing is retained."""
    best = float(output[0, 4:, :].max())
    return best if best >= floor else 0.0


def scores_for(path, arrays, floor):
    session = session_for(path, 0)
    return np.array([image_score(session.run(None, {INPUT_NAME: array})[0], floor) for array in arrays])


def deviation(reference, scores):
    delta = scores - reference
    return {'mean_abs_delta': float(np.abs(delta).mean()), 'max_abs_delta': float(np.abs(delta).max()),
            'mean_signed_delta': float(delta.mean()), 'images_delta_over_0.05': int((np.abs(delta) > 0.05).sum())}


def array_reader(arrays):
    from onnxruntime.quantization import CalibrationDataReader

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.position = 0

        def get_next(self):
            if self.position == len(arrays):
                return None
            self.position += 1
            return {INPUT_NAME: arrays[self.position - 1]}

        def rewind(self):
            self.position = 0

    return Reader()


def static_quantize(work, target, arrays, settings, method, nodes_to_quantize=None, nodes_to_exclude=None):
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static

    quant_types = {'uint8': QuantType.QUInt8, 'int8': QuantType.QInt8}
    quantize_static(str(work), str(target), array_reader(arrays), quant_format=QuantFormat.QDQ,
                    per_channel=settings['per_channel'], reduce_range=settings['reduce_range'],
                    activation_type=quant_types[settings['activation_type']], weight_type=quant_types[settings['weight_type']],
                    nodes_to_quantize=nodes_to_quantize, nodes_to_exclude=nodes_to_exclude,
                    calibrate_method=getattr(CalibrationMethod, method),
                    extra_options={'CalibMaxIntermediateOutputs': 16, 'CalibPercentile': 99.999})


def layer_table(graph):
    """Weighted nodes in graph order with their stage, read from Ultralytics layer indices: 0-9 backbone, 10-21 neck, 22 head."""
    initializers = {init.name: init for init in graph.initializer}
    layers = []
    for node in graph.node:
        if node.op_type not in QUANTIZED_OPS:
            continue
        weight = initializers.get(node.input[1])
        dims = list(weight.dims) if weight is not None else []
        index = int(node.name.split('/')[1].split('.')[1]) if node.name.startswith('/model.') else -1
        stage = 'backbone' if 0 <= index <= 9 else 'neck' if 10 <= index <= 21 else 'head' if index == 22 else 'other'
        layers.append({'node': node.name, 'op': node.op_type, 'layer_index': index, 'stage': stage,
                       'weight_shape': dims, 'parameters': int(np.prod(dims)) if dims else 0})
    return layers


def sensitivity(config_path, quantized, prepared, output):
    import onnx

    config_path = Path(config_path).resolve(strict=True)
    config = load_config(config_path)
    settings = config.get('quantize')
    if settings is None:
        raise ValueError('Config needs a [quantize] table')
    quantized = Path(quantized).resolve(strict=True)
    source = json.loads((quantized / 'quantize.json').read_text())
    if source.get('status') != 'completed':
        raise ValueError('Sensitivity needs a completed quantization directory')
    prepared = Path(prepared or config['paths']['prepared']).resolve(strict=True)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    shape, floor = settings['input_shape'], config['predict']['conf']
    work = quantized / 'fp32-inferred.onnx'
    fp32 = quantized / 'fp32.onnx'
    receipt = {'status': 'running', 'started_at_utc': datetime.now(timezone.utc).isoformat(), 'quantized': str(quantized),
               'quantized_sha256': {name: sha256(quantized / name) for name in ('fp32.onnx', 'fp32-inferred.onnx', 'quantize.json')},
               'settings': settings, 'score_floor': floor,
               'score_definition': 'maximum class probability over anchors of the raw head output; below predict.conf -> 0',
               'probe': 'one weighted node INT8 (MinMax calibration, same calibration images), all other nodes FP32',
               'sha256': {str(path): sha256(path) for path in (config_path, *(ROOT / name for name in
                          ('sensitivity.py', 'quantize.py', 'config.py', 'data.py')))},
               'limitation': 'Ranking and the mixed variant use training images only; detection quality comes from evaluate.py on val'}
    record = output / 'sensitivity.json'
    record.write_text(json.dumps(receipt, indent=2) + '\n')
    try:
        calibration_images = {item['image'] for item in source['calibration']['images']}
        with (prepared / 'train.csv').open() as handle:
            rows = [row for row in csv.DictReader(handle) if row['image'] not in calibration_images]
        if len(rows) < settings['sensitivity_images']:
            raise ValueError('Not enough training images outside the calibration set')
        sample = random.Random(settings['sensitivity_seed']).sample(rows, settings['sensitivity_images'])
        receipt['sensitivity_set'] = {'split': 'train', 'disjoint_from_calibration': True, 'seed': settings['sensitivity_seed'],
                                      'count': len(sample), 'positive': sum(int(row['label']) for row in sample),
                                      'images': [{'image': row['image'], 'label': int(row['label'])} for row in sample]}
        arrays = [load_input(prepared / row['image'], shape) for row in sample]
        calibration = [load_input(prepared / item['image'], shape) for item in source['calibration']['images']]
        reference = scores_for(fp32, arrays, floor)
        receipt['reference'] = {'file': fp32.name, 'mean_score': float(reference.mean()),
                                'positive_below_0.25': int(((reference < 0.25) & np.array([row['label'] == '1' for row in sample])).sum())}
        receipt['full_variants'] = {name: deviation(reference, scores_for(quantized / entry['file'], arrays, floor))
                                    for name, entry in source['variants'].items() if name != 'fp32'}
        layers = layer_table(onnx.load(str(work)).graph)
        probe = output / 'probe.onnx'
        for position, layer in enumerate(layers, 1):
            started = time.monotonic()
            static_quantize(work, probe, calibration, settings, 'MinMax', nodes_to_quantize=[layer['node']])
            layer.update(deviation(reference, scores_for(probe, arrays, floor)), probe_seconds=time.monotonic() - started)
            probe.unlink()
            print(f"{position}/{len(layers)} {layer['node']}: mean |d| {layer['mean_abs_delta']:.5f} max {layer['max_abs_delta']:.4f}", flush=True)
        ranked = sorted(layers, key=lambda layer: layer['mean_abs_delta'], reverse=True)
        for rank, layer in enumerate(ranked, 1):
            layer['rank'] = rank
        with (output / 'sensitivity.csv').open('x', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(layers[0]))
            writer.writeheader()
            writer.writerows(layers)
        receipt['layers'] = layers
        keep = [layer['node'] for layer in ranked[:settings['mixed_keep_fp32']]]
        method = STATIC_METHODS[f"int8-{settings['mixed_method']}"]
        graph = onnx.load(str(work)).graph
        exclude = [node.name for node in graph.node if node.op_type not in QUANTIZED_OPS] + keep
        target = output / f'{MIXED}.onnx'
        started = time.monotonic()
        static_quantize(work, target, calibration, settings, method, nodes_to_exclude=exclude)
        ops = finalize(target, source['export']['metadata'])
        onnx.checker.check_model(str(target))
        shutil.copyfile(quantized / 'benchmark-input.npy', output / 'benchmark-input.npy')
        fp32_bytes = source['variants']['fp32']['bytes']
        receipt['variants'] = {MIXED: {
            'file': target.name, 'bytes': target.stat().st_size, 'sha256': sha256(target), 'build_seconds': time.monotonic() - started,
            'size_ratio_vs_fp32_onnx': fp32_bytes / target.stat().st_size, 'ops': dict(ops),
            'recipe': {'method': 'quantize_static', 'format': 'QDQ', 'calibrate_method': method, 'kept_fp32_nodes': keep,
                       'rule': f"{settings['mixed_keep_fp32']} most sensitive weighted nodes by mean |score delta| on the sensitivity set",
                       **{key: settings[key] for key in ('activation_type', 'weight_type', 'per_channel', 'reduce_range')}},
            'sensitivity_set_deviation': deviation(reference, scores_for(target, arrays, floor))}}
        sample_input = np.load(output / 'benchmark-input.npy')
        results = {}
        for name, path in (('fp32', fp32), (f"int8-{settings['mixed_method']}", quantized / f"int8-{settings['mixed_method']}.onnx"), (MIXED, target)):
            results[name] = [benchmark(path, sample_input, threads, settings['warmup'], settings['runs']) for threads in settings['threads']]
            print(f"{name}: " + ', '.join(f"{row['threads']} thr {row['median_ms']:.2f} ms" for row in results[name]), flush=True)
        receipt['benchmark'] = {**host_record(), 'input': 'benchmark-input.npy',
                                'scope': 'model forward only; excludes image decoding, letterbox and NMS', 'results': results}
        receipt.update(status='completed', finished_at_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        receipt.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        record.write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--quantized', type=Path, required=True, help='Completed quantize.py output directory')
    parser.add_argument('--prepared', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    receipt = sensitivity(args.config, args.quantized, args.prepared, args.output)
    print(json.dumps({'top': [(layer['node'], round(layer['mean_abs_delta'], 5)) for layer in
                              sorted(receipt['layers'], key=lambda layer: layer['rank'])[:receipt['settings']['mixed_keep_fp32']]],
                      'full_variants': receipt['full_variants'],
                      MIXED: receipt['variants'][MIXED]['sensitivity_set_deviation']}, indent=2))
