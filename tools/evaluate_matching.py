"""Compare production gallery distances on human-labeled tracklet pairs.

Input labels are never edited. Contradictory negatives are reported/excluded.
Thresholds are fixed by the caller, not optimized on this evaluation set.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from offline_association import RepresentativeGallery


def extract_samples(review, output, weights, repo):
    import cv2
    from transreid_adapter import TransReIDFeatureExtractor
    crop_root = Path(json.loads((review / 'summary.json').read_text(encoding='utf-8'))['input'])
    rows = [json.loads(line) for line in (review / 'tracklets.jsonl').read_text(encoding='utf-8').splitlines()]
    images, qualities, ids, offsets = [], [], [], [0]
    for row in rows:
        ids.append(row['tracklet_id'])
        for sample in row['representatives']:
            image = cv2.imread(str(crop_root / sample['path']))
            if image is None:
                raise ValueError('Unreadable representative: {}'.format(sample['path']))
            images.append(image)
            qualities.append(sample.get('quality', 1.))
        offsets.append(len(images))
    extractor = TransReIDFeatureExtractor(weight_path=str(weights), repo_dir=str(repo),
                                          batch_size=8, download=False)
    features = extractor.extract_numpy(images)
    if len(features) != len(images):
        raise ValueError('Feature count differs from input image count.')
    np.savez_compressed(output, tracklet_ids=np.asarray(ids), offsets=np.asarray(offsets),
        features=features, qualities=np.asarray(qualities), feature_mode='current_model_output',
        weights_sha256=hashlib.sha256(weights.read_bytes()).hexdigest())


def load_labels(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    parent = {}
    def root(key):
        parent.setdefault(key, key)
        if parent[key] != key:
            parent[key] = root(parent[key])
        return parent[key]
    for row in rows:
        if row['label'].strip() == '1':
            parent[root(row['tracklet_a'])] = root(row['tracklet_b'])
    valid, contradictions = [], []
    for row in rows:
        label = row['label'].strip()
        if label not in ('0', '1'):
            continue
        if label == '0' and root(row['tracklet_a']) == root(row['tracklet_b']):
            contradictions.append(row['pair_id'])
        else:
            valid.append(row)
    return valid, contradictions


def metrics(labels, distances, threshold):
    positive, accepted = labels == 1, distances < threshold
    tp = int(np.sum(positive & accepted))
    fp = int(np.sum(~positive & accepted))
    fn = int(np.sum(positive & ~accepted))
    tn = int(np.sum(~positive & ~accepted))
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=tp/max(1, tp+fp),
                recall=tp/max(1, tp+fn), f1=2*tp/max(1, 2*tp+fp+fn))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review-dir', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threshold', type=float, default=.3)
    parser.add_argument('--extract', action='store_true')
    parser.add_argument('--weights', type=Path)
    parser.add_argument('--repo', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.extract:
        if args.weights is None or args.repo is None:
            parser.error('--extract requires --weights and --repo')
        extract_samples(args.review_dir, args.cache, args.weights, args.repo)
    rows, contradictions = load_labels(args.review_dir / 'pairs.csv')
    with np.load(args.cache, allow_pickle=False) as cache:
        ids, offsets = cache['tracklet_ids'], cache['offsets']
        features, weights = cache['features'], cache['qualities']
        galleries = {str(key): RepresentativeGallery(features[offsets[i]:offsets[i+1]],
                        weights[offsets[i]:offsets[i+1]]) for i, key in enumerate(ids)}
        feature_mode = str(cache['feature_mode'].item())
    labels = np.asarray([int(row['label']) for row in rows])
    scores = {strategy: np.asarray([galleries[row['tracklet_a']].distance(
        galleries[row['tracklet_b']], strategy) for row in rows])
        for strategy in ('pairwise', 'bidirectional')}
    report = dict(pairs=len(rows), positives=int(labels.sum()), negatives=int((labels == 0).sum()),
        excluded_contradictions=contradictions, feature_mode=feature_mode,
        feature_dim=int(features.shape[1]), threshold=args.threshold,
        limitation='Historical, selected labeled pairs; not an independent video-level accuracy test.',
        results={})
    for strategy, distances in scores.items():
        if not np.isfinite(distances).all():
            raise ValueError('Evaluation contains invalid or empty feature sets.')
        result = metrics(labels, distances, args.threshold)
        comparisons = distances[labels == 1][:, None] - distances[labels == 0][None, :]
        result['auc'] = float(np.mean((comparisons < 0) + .5*(comparisons == 0)))
        report['results'][strategy] = result
    (args.output / 'matching_evaluation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    with (args.output / 'pair_decisions.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['pair_id', 'label', *scores])
        writer.writerows([row['pair_id'], int(row['label']), *[float(score[i]) for score in scores.values()]]
                        for i, row in enumerate(rows))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
