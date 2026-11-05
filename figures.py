"""Render the README figures from the frozen receipts in results/; the inspection figure also needs images and weights."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from data import CLASS_NAMES, read_boxes  # noqa: E402

ROOT = Path(__file__).resolve().parent
QUARTZITY = CLASS_NAMES.index('Quartzity')
# Validation images chosen after the Quartzity audit: the two misses at the 99.9% operating point,
# one image with an unlabelled twin band, and one spot labelled Quartzity that the model calls Marrow.
INSPECTION = ('142100069', '164000051', '171700043', '170400076')
WINDOW = 1800  # pixels of the 2800-wide board shown per panel
STAGE_COLORS = {'backbone': '#4C72B0', 'neck': '#55A868', 'head': '#C44E52'}
GT_COLOR, Q_COLOR, OTHER_COLOR = '#2CA02C', '#D62728', '#FF7F0E'

plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'legend.fontsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
                     'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 300, 'savefig.bbox': 'tight'})


def read_rows(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def operating_curve(scores, labels):
    """Recall and false positive rate with every retained score as the threshold, highest first."""
    thresholds = np.unique(scores[scores > 0])[::-1]
    positive, negative = scores[labels == 1], scores[labels == 0]
    recall = np.array([(positive >= t).mean() for t in thresholds])
    fpr = np.array([(negative >= t).mean() for t in thresholds])
    return thresholds, recall, fpr


def figure_operating_points(results, output):
    receipt = json.loads((results / 'validation' / 'metrics.json').read_text())
    rows = read_rows(results / 'validation' / 'scores.csv')
    scores = np.array([float(row['score']) for row in rows])
    labels = np.array([int(row['label']) for row in rows])
    quartzity_only = np.array([json.loads(row['gt_classes']) == [QUARTZITY] for row in rows])
    points = [point['selected'] for point in receipt['validation_selected_operating_points'] if point['selected']]
    targets = [point['target_recall'] for point in receipt['validation_selected_operating_points'] if point['selected']]

    fig, (left, right) = plt.subplots(1, 2, figsize=(7.0, 2.7), gridspec_kw={'width_ratios': [1, 1.2], 'wspace': 0.3})
    _, recall, fpr = operating_curve(scores, labels)
    left.plot(fpr * 100, recall * 100, color=STAGE_COLORS['backbone'], linewidth=1.3)
    placements = ((6, -9, 'left'), (6, -9, 'left'), (6, -14, 'left'), (-8, -3, 'right'))  # the 99.9% point sits at the right edge
    for target, point, marker, (dx, dy, align) in zip(targets, points, 'osD^', placements):
        x, y = point['false_positive_rate'] * 100, point['recall'] * 100
        left.plot(x, y, marker, color=Q_COLOR, markersize=4.5)
        left.annotate(f"{target:.1%} recall: thr {point['threshold']:.3g}, FP {point['FP']}", (x, y), textcoords='offset points',
                      xytext=(dx, dy), ha=align, fontsize=6.5)
    left.set_xscale('log')
    left.set_xlim(0.3, 110)
    left.set_ylim(84, 100.6)
    left.set_xlabel('False positive rate on candidate normals (%)')
    left.set_ylabel('Recall on defective images (%)')
    left.grid(True, which='both', alpha=0.25, linewidth=0.5)
    left.text(-0.18, 1.02, '(a)', transform=left.transAxes, fontweight='bold')

    edges = np.logspace(-3, 0, 25)
    groups = [('defective images (2,718)', labels == 1, dict(color=STAGE_COLORS['backbone'], alpha=0.45)),
              ('Quartzity-only images (20)', quartzity_only, dict(color=Q_COLOR, alpha=0.8)),
              ('candidate normals (324)', labels == 0, dict(facecolor='none', edgecolor='black', linewidth=0.9))]
    slot = np.logspace(-3.75, -3.25, len(groups) + 1)  # a separate slot for images with no retained detection
    for index, (name, mask, style) in enumerate(groups):
        counts, _ = np.histogram(scores[mask & (scores > 0)], bins=edges)
        right.bar(edges[:-1], counts, width=np.diff(edges), align='edge', label=name, **style)
        zeros = int((mask & (scores == 0)).sum())
        if zeros:
            right.bar(slot[index], zeros, width=slot[index + 1] - slot[index], align='edge', **style)
    for point, text in zip(points[-2:], ('99%', '99.9%')):
        right.axvline(point['threshold'], color='gray', linestyle='--', linewidth=0.8)
        right.text(point['threshold'] * 1.12, 3000, f'thr for {text}', fontsize=6.5, color='dimgray')
    right.set_xscale('log')
    right.set_yscale('log')
    right.set_xlim(slot[0] / 1.2, 1.0)
    right.set_ylim(0.7, 100000)
    right.set_xticks([np.sqrt(slot[0] * slot[-1]), 1e-3, 1e-2, 1e-1, 1])
    right.set_xticklabels(['none', '0.001', '0.01', '0.1', '1'])
    right.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    right.set_xlabel('Image score (maximum detection confidence)')
    right.set_ylabel('Images')
    right.legend(loc='upper center', frameon=False, bbox_to_anchor=(0.5, 1.03))
    right.text(-0.15, 1.02, '(b)', transform=right.transAxes, fontweight='bold')
    fig.savefig(output / 'operating-points.png')
    plt.close(fig)


def short_name(node):
    parts = [part for part in node.strip('/').split('/') if part not in ('conv', 'Conv')]
    return parts[0].split('.')[1] + '.' + (parts[-1] if len(parts) > 1 else 'conv')


def figure_sensitivity(results, output):
    order = {stage: index for index, stage in enumerate(STAGE_COLORS)}
    # Graph order interleaves head and neck convs; group by stage and keep graph order within a stage.
    rows = sorted(read_rows(results / 'quantization' / 'sensitivity.csv'), key=lambda row: order[row['stage']])
    receipt = json.loads((results / 'quantization' / 'sensitivity.json').read_text())
    kept = set(receipt['variants']['int8-mixed']['recipe']['kept_fp32_nodes'])
    values = np.array([float(row['mean_abs_delta']) for row in rows])
    floor = 1e-5
    x = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    for index, (row, value) in enumerate(zip(rows, values)):
        style = dict(edgecolor='black', linewidth=0.7, hatch='////') if row['node'] in kept else dict(linewidth=0)
        if value > 0:
            ax.bar(index, value - floor, bottom=floor, width=0.8, color=STAGE_COLORS[row['stage']], **style)
        else:
            ax.plot(index, floor * 1.5, marker='x', color='gray', markersize=3, linewidth=0)
    for row, value, index in sorted(zip(rows, values, x), key=lambda item: -item[1])[:3]:
        ax.annotate(short_name(row['node']), (index, value), textcoords='offset points', xytext=(0, 3),
                    ha='left' if index < 2 else 'center', fontsize=6.5)
    ticks = [index for index, row in enumerate(rows) if index == 0 or row['layer_index'] != rows[index - 1]['layer_index']]
    labels, previous = [], -10
    for index in ticks:  # a layer with a single conv next to another one gets a tick but no label
        labels.append(rows[index]['layer_index'] if index - previous > 1 else '')
        previous = index
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    for index in (index for index, row in enumerate(rows) if index and row['stage'] != rows[index - 1]['stage']):
        ax.axvline(index - 0.5, color='lightgray', linewidth=0.8)
    ax.set_xlim(-1, len(rows))
    ax.set_yscale('log')
    ax.set_ylim(floor, 0.15)
    ax.set_xlabel('YOLOv8n layer index, grouped by stage (backbone 0–9, neck 10–21, detection head 22); one bar per Conv')
    ax.set_ylabel('Mean |Δ image score| vs FP32')
    ax.grid(True, axis='y', which='major', alpha=0.25, linewidth=0.5)
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in STAGE_COLORS.values()]
    handles += [plt.Rectangle((0, 0), 1, 1, facecolor='white', edgecolor='black', hatch='////'),
                plt.Line2D([], [], marker='x', color='gray', linewidth=0, markersize=4)]
    ax.legend(handles, [*STAGE_COLORS, f'kept in FP32 by int8-mixed ({len(kept)} layers)', 'exactly zero (box branches, P5 class branch)'],
              loc='upper left', ncol=2, frameon=False, bbox_to_anchor=(0.03, 1.02))
    fig.savefig(output / 'layer-sensitivity.png')
    plt.close(fig)


def figure_inspection(results, output, prepared, weights):
    from PIL import Image

    from model import build_model, predict_image

    scores = {Path(row['image']).stem: float(row['score']) for row in read_rows(results / 'validation' / 'scores.csv')}
    model = build_model(weights)
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.4), gridspec_kw={'wspace': 0.04, 'hspace': 0.2})
    for ax, stem in zip(axes.flat, INSPECTION):
        image_path = prepared / 'images' / 'val' / f'{stem}.bmp'
        with Image.open(image_path) as source:
            pixels = np.asarray(source.convert('RGB'))
        height, width = pixels.shape[:2]
        truth = [((cx - w / 2) * width, (cy - h / 2) * height, w * width, h * height)
                 for _, cx, cy, w, h in read_boxes(prepared / 'labels' / 'val' / f'{stem}.txt')]
        predictions = sorted(predict_image(model, image_path, device='cpu', config_path=ROOT / 'configs' / 'vsb-quantize.toml')['boxes'],
                             key=lambda box: -box[4])[:3]
        # Window around everything drawn: the annotations and the three strongest predictions.
        extents = [(x, x + w) for x, _, w, _ in truth] + [(box[0], box[2]) for box in predictions]
        centre = (min(x for x, _ in extents) + max(x for _, x in extents)) / 2
        left = int(np.clip(centre - WINDOW / 2, 0, width - WINDOW))
        ax.imshow(pixels[:, left:left + WINDOW])
        for x, y, w, h in truth:
            ax.add_patch(plt.Rectangle((x - left, y), w, h, fill=False, edgecolor=GT_COLOR, linewidth=1.4))
        for rank, (x1, y1, x2, y2, confidence, class_id) in enumerate(predictions):
            color = Q_COLOR if int(class_id) == QUARTZITY else OTHER_COLOR
            ax.add_patch(plt.Rectangle((x1 - left, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=1.0, linestyle='--'))
            near_right = x1 - left > WINDOW - 460
            ax.text(min(x2 - left - 8, WINDOW - 8) if near_right else x1 - left + 8, 40 + 60 * rank,
                    f"{CLASS_NAMES[int(class_id)].replace('_', ' ')} {confidence:.3f}", color=color, fontsize=6, clip_on=True,
                    ha='right' if near_right else 'left', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))
        ax.set_xlim(0, WINDOW)
        ax.set_ylim(height, 0)
        ax.set_title(f'{stem}   image score {scores[stem]:.4f}', fontsize=8)
        ax.set_axis_off()
    handles = [plt.Line2D([], [], color=GT_COLOR, linewidth=1.4), plt.Line2D([], [], color=Q_COLOR, linewidth=1.0, linestyle='--'),
               plt.Line2D([], [], color=OTHER_COLOR, linewidth=1.0, linestyle='--')]
    fig.legend(handles, ['annotation', 'Quartzity prediction', 'other-class prediction'], loc='lower center', ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.01))
    fig.savefig(output / 'quartzity-inspection.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=ROOT / 'results')
    parser.add_argument('--output', type=Path, default=ROOT / 'figures')
    parser.add_argument('--prepared', type=Path, help='Prepared split root with images/val and labels/val, for the inspection figure')
    parser.add_argument('--weights', type=Path, help='FP32 checkpoint, for the inspection figure')
    args = parser.parse_args()
    if (args.prepared is None) != (args.weights is None):
        parser.error('--prepared and --weights go together')
    args.output.mkdir(parents=True, exist_ok=True)
    figure_operating_points(args.results, args.output)
    figure_sensitivity(args.results, args.output)
    if args.weights:
        figure_inspection(args.results, args.output, args.prepared, args.weights)
    print('\n'.join(str(path) for path in sorted(args.output.glob('*.png'))))
