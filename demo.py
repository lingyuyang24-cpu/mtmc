# ! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import argparse
import json
import math
import os
import time

import cv2
import numpy as np
from PIL import Image

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker

from offline_association import Tracklet, as_numpy, feature_distance, fuse_tracklets


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ('yes', 'true', 't', '1', 'y'):
        return True
    if value in ('no', 'false', 'f', '0', 'n'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()
parser.add_argument(
    '--detector',
    default='yolo11l.pt',
    help='YOLO model path/name by default, or a torchvision model name.'
)
parser.add_argument(
    '--detector-backend',
    choices=('auto', 'ultralytics', 'torchvision'),
    default='auto',
    help='Detector backend. auto uses Ultralytics for YOLO-looking model names.'
)
parser.add_argument(
    '--detector-weights',
    default='default',
    help='For YOLO this may be a local .pt path; for torchvision use "default", "none", or a checkpoint path.'
)
parser.add_argument(
    '--detector-score',
    type=float,
    default=0.35,
    help='Minimum detector confidence for person boxes.'
)
parser.add_argument(
    '--reid-backend',
    choices=('torchreid', 'transreid', 'custom'),
    default='transreid',
    help='Feature extractor backend used by Deep SORT and final global-ID fusion.'
)
parser.add_argument(
    '--reid-model',
    default='resnet50',
    help='Torchreid model name for Deep SORT features and final ReID.'
)
parser.add_argument(
    '--reid-weights',
    default='model_data/models/model.pth',
    help='Local ReID checkpoint path. Used by torchreid and custom backends.'
)
parser.add_argument(
    '--reid-config',
    default=None,
    help='Optional config path for a custom ReID backend.'
)
parser.add_argument('--encoder-batch-size', type=int, default=32)
parser.add_argument('--reid-batch-size', type=int, default=32)
parser.add_argument('--transreid-variant', default='msmt17', choices=('msmt17', 'market1501', 'dukemtmc'))
parser.add_argument('--transreid-weights', default=None, help='Optional local TransReID checkpoint path.')
parser.add_argument('--transreid-repo', default=None, help='Optional local TransReID source repo path.')
parser.add_argument('--transreid-assets-root', default=None, help='Where TransReID source and weights are stored.')
parser.add_argument(
    '--transreid-download',
    nargs='?',
    const=True,
    type=str2bool,
    default=True,
    help='Download TransReID source/weights if they are missing.'
)
parser.add_argument(
    '--custom-reid-root',
    default='custom_reid_models',
    help='Root folder containing custom ReID model folders.'
)
parser.add_argument(
    '--custom-reid-name',
    default=None,
    help='Custom ReID model folder name under --custom-reid-root.'
)
parser.add_argument(
    '--custom-reid-dir',
    default=None,
    help='Direct path to one custom ReID model folder. Overrides --custom-reid-name.'
)
parser.add_argument(
    '--custom-reid-adapter',
    default=None,
    help='Optional direct path to adapter.py for the custom ReID backend.'
)
parser.add_argument(
    '--reid-distance',
    choices=('cosine', 'euclidean'),
    default='cosine',
    help='Distance metric used when fusing finished track IDs.'
)
parser.add_argument(
    '--reid-threshold',
    type=float,
    default=0.2,
    help='Fuse two track IDs when their mean ReID distance is below this value.'
)
parser.add_argument('--deep-sort-max-cosine-distance', type=float, default=0.2)
parser.add_argument('--nms-max-overlap', type=float, default=0.4)
parser.add_argument('--post-nms', choices=('detector', 'nms'), default='detector',
                    help='Keep detector output, or apply an additional IoU NMS before encoding.')
parser.add_argument('--reid-margin', type=float, default=0.05,
                    help='Minimum distance advantage over the second-best compatible identity.')
parser.add_argument('--reid-gallery-size', type=int, default=32,
                    help='Maximum representative features per fused identity.')
parser.add_argument('--tracker-max-age', type=int, default=100)
parser.add_argument('--tracker-n-init', type=int, default=3)
parser.add_argument('--min-reid-frames', type=int, default=10)
parser.add_argument(
    '--version',
    default=None,
    help='Deprecated legacy YOLO selector kept for old commands; ignored by the PyTorch path.'
)
parser.add_argument('--videos', nargs='+', help='List of videos', required=True)
parser.add_argument(
    '-all',
    '--all',
    nargs='?',
    const=True,
    type=str2bool,
    default=True,
    help='Write one combined tracking/ReID video.'
)
parser.add_argument(
    '--side-by-side',
    nargs='?',
    const=True,
    type=str2bool,
    default=True,
    help='Write side-by-side demo videos when multiple input videos are provided.'
)
parser.add_argument(
    '--side-by-side-scale',
    type=float,
    default=1.5,
    help='Scale side-by-side demo videos. Use 1.25 or 1.5 to make IDs easier to read.'
)


class LoadVideo:  # for inference
    def __init__(self, path, img_size=(1088, 608)):
        if not os.path.isfile(path):
            raise FileExistsError

        self.cap = cv2.VideoCapture(path)
        self.frame_rate = int(round(self.cap.get(cv2.CAP_PROP_FPS)))
        self.vw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.vh = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.vn = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = img_size[0]
        self.height = img_size[1]
        self.count = 0

        print('Length of {}: {:d} frames'.format(path, self.vn))

    def get_VideoLabels(self):
        return self.cap, self.frame_rate, self.vw, self.vh


def build_reid_encoder(args):
    from reid_backends import create_reid_encoder

    encoder = create_reid_encoder(
        backend=args.reid_backend,
        reid_model=args.reid_model,
        reid_weights=args.reid_weights,
        reid_config=args.reid_config,
        batch_size=args.encoder_batch_size,
        transreid_variant=args.transreid_variant,
        transreid_weights=args.transreid_weights,
        transreid_repo=args.transreid_repo,
        transreid_assets_root=args.transreid_assets_root,
        transreid_download=args.transreid_download,
        custom_reid_root=args.custom_reid_root,
        custom_reid_name=args.custom_reid_name,
        custom_reid_dir=args.custom_reid_dir,
        custom_reid_adapter=args.custom_reid_adapter
    )
    return encoder, encoder.extractor


def encode_frame_boxes(encoder, frame, boxes, camera_id):
    return encoder(frame, boxes, camera_id=camera_id)


def make_track_sample(frame, bbox, camera_id, confidence=1.0):
    if not np.isfinite(bbox).all():
        return None
    x1, y1, x2, y2 = map(int, bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2].copy()
    confidence = float(confidence) if confidence is not None else 0.0
    quality = float(np.clip(confidence, 0.0, 1.0)) if np.isfinite(confidence) else 0.0
    if quality <= 0:
        return None
    return {'image': crop, 'camera_id': int(camera_id),
            'confidence': confidence, 'quality': quality}


def extract_track_features(extractor, samples):
    if not samples:
        return np.empty((0, getattr(extractor, 'feature_dim', 0)), dtype=np.float32)
    if not getattr(extractor, 'uses_camera_id', False):
        return as_numpy(extractor.extract([sample['image'] for sample in samples]))

    features = None
    camera_ids = sorted(set(sample['camera_id'] for sample in samples))
    for camera_id in camera_ids:
        indices = [i for i, sample in enumerate(samples) if sample['camera_id'] == camera_id]
        batch = as_numpy(extractor.extract(
            [samples[i]['image'] for i in indices], camera_id=camera_id))
        if features is None:
            features = np.empty((len(samples), batch.shape[1]), dtype=batch.dtype)
        features[indices] = batch
    return features


def compute_reid_distance(qf, gf, metric='cosine'):
    return feature_distance(qf, gf, metric)


def resize_output_frame(frame, output_size):
    """Resize only rendered output; tracking and crops stay in source coordinates."""
    if (frame.shape[1], frame.shape[0]) == output_size:
        return frame
    return cv2.resize(frame, output_size, interpolation=cv2.INTER_LINEAR)


def prepare_detections(detector, encoder, frame, camera_id, post_nms='detector', max_overlap=0.4):
    image = Image.fromarray(frame[..., ::-1])
    if hasattr(detector, 'detect_image_with_scores'):
        boxes, scores = detector.detect_image_with_scores(image)
    else:
        boxes = detector.detect_image(image)
        scores = [1.0] * len(boxes)  # Legacy detectors have no confidence API.
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (len(boxes),):
        raise ValueError('Detector boxes and confidence scores must have equal lengths.')
    indices = preprocessing.select_detection_indices(
        boxes, scores, mode=post_nms, max_overlap=max_overlap)
    if not len(indices):
        return []
    boxes, scores = boxes[indices], scores[indices]
    features = as_numpy(encode_frame_boxes(encoder, frame, boxes, camera_id))
    if features.ndim != 2 or len(features) != len(boxes):
        raise ValueError('Encoder must return one feature vector per selected detection.')
    return [Detection(box, score, feature) for box, score, feature in zip(boxes, scores, features)]


def main(detector=None, args=None, encoder=None, out_dir='videos/output/'):
    # Parsing and model construction happen only when explicitly running main.
    if args is None:
        args = parser.parse_args()
    if args.side_by_side_scale <= 0:
        raise ValueError('--side-by-side-scale must be greater than 0.')
    if args.min_reid_frames < 1 or args.reid_gallery_size < 1:
        raise ValueError('--min-reid-frames and --reid-gallery-size must be positive.')
    if (not np.isfinite(args.reid_threshold) or args.reid_threshold < 0
            or not np.isfinite(args.reid_margin) or args.reid_margin < 0):
        raise ValueError('--reid-threshold and --reid-margin must be finite and nonnegative.')
    if args.version is not None:
        print('Warning: --version is deprecated. Use --detector yolov8n.pt for PyTorch YOLO.')
    if detector is None:
        from torch_detector import build_person_detector
        detector = build_person_detector(
            model_name=args.detector, backend=args.detector_backend,
            weights=args.detector_weights, score_threshold=args.detector_score)
    print(f'Using {detector} model')
    # Definition of the parameters
    max_cosine_distance = args.deep_sort_max_cosine_distance
    nn_budget = None
    nms_max_overlap = args.nms_max_overlap

    # deep_sort
    if encoder is None:
        encoder, reid_extractor = build_reid_encoder(args)
    else:
        reid_extractor = encoder.extractor

    trackers = {
        camera_id: Tracker(
            nn_matching.NearestNeighborDistanceMetric('cosine', max_cosine_distance, nn_budget),
            max_age=args.tracker_max_age, n_init=args.tracker_n_init)
        for camera_id in range(len(args.videos))
    }

    is_vis = True
    out_dir = os.path.join(os.fspath(out_dir), '')
    print('The output folder is', out_dir)
    os.makedirs(out_dir, exist_ok=True)

    all_frames = []
    all_frame_camera_ids = []
    all_local_frame_ids = []
    source_frame_counts = []
    source_names = []
    source_frame_rates = []
    for camera_id, video in enumerate(args.videos):
        loadvideo = LoadVideo(video)
        video_capture, frame_rate, w, h = loadvideo.get_VideoLabels()
        source_names.append(os.path.basename(video))
        source_frame_rates.append(frame_rate)
        video_frame_count = 0
        while True:
            ret, frame = video_capture.read()
            if ret is not True:
                video_capture.release()
                break
            all_frames.append(frame)
            all_frame_camera_ids.append(camera_id)
            all_local_frame_ids.append(video_frame_count)
            video_frame_count += 1
        source_frame_counts.append(video_frame_count)

    if not all_frames:
        raise ValueError('No frames were loaded from the input videos.')

    frame_rate = (source_frame_rates[0] if source_frame_rates else 25) or 25
    h, w = all_frames[0].shape[:2]
    output_size = (w, h)

    frame_nums = len(all_frames)
    tracking_path = out_dir + 'tracking' + '.avi'
    combined_path = out_dir + 'allVideos' + '.avi'
    if is_vis:
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(tracking_path, fourcc, frame_rate, (w, h))
        out2 = cv2.VideoWriter(combined_path, fourcc, frame_rate, (w, h))
        if not out.isOpened() or not out2.isOpened():
            out.release()
            out2.release()
            raise RuntimeError('Could not open tracking video writers in {}'.format(out_dir))
        # Combine all videos
        for frame in all_frames:
            out2.write(resize_output_frame(frame, output_size))
        out2.release()

    # Initialize tracking file
    filename = out_dir + '/tracking.txt'
    with open(filename, 'w'):
        pass

    frame_cnt = 0
    t1 = time.time()

    track_cnt = dict()
    images_by_id = dict()
    numeric_ids = {}
    track_spans = {}
    for source_frame, camera_id, local_frame in zip(
            all_frames, all_frame_camera_ids, all_local_frame_ids):
        original_frame = source_frame.copy()
        frame = original_frame.copy()
        frame_h, frame_w = frame.shape[:2]
        detections = prepare_detections(
            detector, encoder, original_frame, camera_id, args.post_nms, nms_max_overlap)
        text_scale, text_thickness, line_thickness = get_FrameLabels(frame)

        # Call the tracker
        tracker = trackers[camera_id]
        tracker.predict()
        tracker.update(detections)
        for track in tracker.tracks:
            key = (int(camera_id), int(track.track_id))
            if track.time_since_update <= 1:
                track_spans.setdefault(key, [local_frame, local_frame])[1] = local_frame
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            if key not in numeric_ids:
                numeric_ids[key] = len(numeric_ids) + 1
            output_id = numeric_ids[key]
            bbox = track.to_tlbr()
            area = (int(bbox[2]) - int(bbox[0])) * (int(bbox[3]) - int(bbox[1]))
            track_cnt.setdefault(key, []).append([
                frame_cnt, int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), area])
            images_by_id.setdefault(key, [])
            if track.time_since_update == 0 and getattr(track, 'last_feature', None) is not None:
                sample = make_track_sample(
                    original_frame, getattr(track, 'last_detection_bbox', bbox),
                    camera_id, getattr(track, 'last_confidence', None))
                if sample is not None:
                    images_by_id[key].append(sample)
            cv2_addBox(
                output_id,
                frame,
                int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]),
                line_thickness,
                text_thickness,
                text_scale
            )
            write_results(
                filename,
                'mot',
                frame_cnt + 1,
                str(output_id),
                int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]),
                frame_w, frame_h
            )

        # save a frame
        if is_vis:
            out.write(resize_output_frame(frame, output_size))
        t2 = time.time()

        frame_cnt += 1
        print(frame_cnt, '/', frame_nums)

    if is_vis:
        out.release()
    print('Tracking finished in {} seconds'.format(int(time.time() - t1)))
    print('Tracked video : {}'.format(tracking_path))
    print('Combined video : {}'.format(combined_path))

    print(f'Total IDs = {len(images_by_id)}')
    tracklets = []
    original_batch_size = getattr(reid_extractor, 'batch_size', None)
    if original_batch_size is not None:
        reid_extractor.batch_size = args.reid_batch_size
    try:
        for key, samples in images_by_id.items():
            print(f'ID number {numeric_ids[key]} -> Number of frames {len(samples)}')
            tracklets.append(Tracklet(
                key, numeric_ids[key], *track_spans[key],
                extract_track_features(reid_extractor, samples),
                [sample['quality'] for sample in samples]))
    finally:
        if original_batch_size is not None:
            reid_extractor.batch_size = original_batch_size

    global_ids, groups = fuse_tracklets(
        tracklets, threshold=args.reid_threshold, margin=args.reid_margin,
        min_frames=args.min_reid_frames, gallery_size=args.reid_gallery_size,
        metric=args.reid_distance)
    final_fuse_id = {gid: [member.key for member in group.members] for gid, group in groups.items()}
    mapping_path = os.path.join(out_dir, 'id_mapping.json')
    with open(mapping_path, 'w') as mapping_file:
        json.dump({
            'cameras': [{'camera_id': i, 'source': os.fspath(video)} for i, video in enumerate(args.videos)],
            'tracks': [
                {'camera_id': key[0], 'local_id': key[1], 'tracking_id': numeric_ids[key],
                 'global_id': global_ids[key], 'start_frame': track_spans[key][0],
                 'end_frame': track_spans[key][1], 'reid_samples': len(images_by_id[key])}
                for key in sorted(numeric_ids, key=numeric_ids.get)
            ],
            'frame_index_base': 0,
            'distance_metric': args.reid_distance,
        }, mapping_file, indent=2)
    print('Final ids and their sub-ids:', final_fuse_id)
    print('MOT took {} seconds'.format(int(time.time() - t1)))
    t2 = time.time()

    # To generate MOT for each person, declare 'is_vis' to True
    is_vis = False
    if is_vis:
        print('Writing videos for each ID...')
        output_dir = 'videos/output/tracklets/'
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        loadvideo = LoadVideo(combined_path)
        video_capture, frame_rate, w, h = loadvideo.get_VideoLabels()
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        for idx in final_fuse_id:
            tracking_path = os.path.join(output_dir, str(idx)+'.avi')
            out = cv2.VideoWriter(tracking_path, fourcc, frame_rate, (w, h))
            for i in final_fuse_id[idx]:
                for f in track_cnt[i]:
                    frame = all_frames[f[0]].copy()
                    text_scale, text_thickness, line_thickness = get_FrameLabels(frame)
                    cv2_addBox(idx, frame, f[1], f[2], f[3], f[4], line_thickness, text_thickness, text_scale)
                    out.write(resize_output_frame(frame, output_size))
            out.release()
        video_capture.release()

    # Generate a single video with complete MOT/ReID
    complete_path = None
    if args.all:
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        complete_path = out_dir+'/Complete'+'.avi'
        out = cv2.VideoWriter(complete_path, fourcc, frame_rate, output_size)
        if not out.isOpened():
            raise RuntimeError('Could not open global ID video writer: {}'.format(complete_path))

        for frame in range(len(all_frames)):
            frame2 = all_frames[frame].copy()
            for idx in final_fuse_id:
                for i in final_fuse_id[idx]:
                    for f in track_cnt[i]:
                        # print('frame {} f0 {}'.format(frame,f[0]))
                        if frame == f[0]:
                            text_scale, text_thickness, line_thickness = get_FrameLabels(frame2)
                            cv2_addBox(idx, frame2, f[1], f[2], f[3], f[4], line_thickness, text_thickness, text_scale)
            out.write(resize_output_frame(frame2, output_size))
        out.release()

    tracking_side_by_side_path = None
    complete_side_by_side_path = None
    if args.side_by_side and len(source_frame_counts) > 1:
        tracking_side_by_side_path = os.path.join(out_dir, 'tracking_side_by_side.avi')
        write_side_by_side_video(
            tracking_path,
            tracking_side_by_side_path,
            source_frame_counts,
            frame_rate,
            source_names,
            args.side_by_side_scale
        )
        if complete_path is not None:
            complete_side_by_side_path = os.path.join(out_dir, 'Complete_side_by_side.avi')
            write_side_by_side_video(
                complete_path,
                complete_side_by_side_path,
                source_frame_counts,
                frame_rate,
                source_names,
                args.side_by_side_scale
            )

    os.remove(combined_path)
    print('\nWriting videos took {} seconds'.format(int(time.time() - t2)))
    if complete_path is not None:
        print('Final video at {}'.format(complete_path))
    if tracking_side_by_side_path is not None:
        print('Side-by-side tracking video at {}'.format(tracking_side_by_side_path))
    if complete_side_by_side_path is not None:
        print('Side-by-side global ID video at {}'.format(complete_side_by_side_path))
    print('Total: {} seconds'.format(int(time.time() - t1)))
    print('Camera/local/global ID mapping at {}'.format(mapping_path))
    return global_ids


def get_FrameLabels(frame):
    text_scale = max(1, frame.shape[1] / 1600.)
    text_thickness = 1 if text_scale > 1.1 else 1
    line_thickness = max(1, int(frame.shape[1] / 500.))
    return text_scale, text_thickness, line_thickness


def write_side_by_side_video(input_path, output_path, source_frame_counts, frame_rate, labels=None, scale=1.0):
    if len(source_frame_counts) < 2:
        return None

    offsets = []
    offset = 0
    for count in source_frame_counts:
        offsets.append(offset)
        offset += count

    caps = []
    for start in offsets:
        cap = cv2.VideoCapture(input_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start))
        caps.append(cap)

    tile_w = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    tile_h = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    if tile_w <= 0 or tile_h <= 0:
        ret, sample = caps[0].read()
        if not ret:
            for cap in caps:
                cap.release()
            return None
        tile_h, tile_w = sample.shape[:2]
        caps[0].set(cv2.CAP_PROP_POS_FRAMES, int(offsets[0]))

    num_sources = len(source_frame_counts)
    cols = num_sources if num_sources <= 2 else int(math.ceil(math.sqrt(num_sources)))
    rows = int(math.ceil(float(num_sources) / cols))
    fps = frame_rate if frame_rate and frame_rate > 0 else 25
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    output_w = max(1, int(round(tile_w * cols * scale)))
    output_h = max(1, int(round(tile_h * rows * scale)))
    out = cv2.VideoWriter(output_path, fourcc, fps, (output_w, output_h))
    max_frames = max(source_frame_counts)

    for frame_idx in range(max_frames):
        canvas = np.zeros((tile_h * rows, tile_w * cols, 3), dtype=np.uint8)
        for source_idx, cap in enumerate(caps):
            if frame_idx < source_frame_counts[source_idx]:
                ret, frame = cap.read()
                if not ret:
                    frame = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
            else:
                frame = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)

            if frame.shape[1] != tile_w or frame.shape[0] != tile_h:
                frame = cv2.resize(frame, (tile_w, tile_h))
            else:
                frame = frame.copy()

            if labels and source_idx < len(labels):
                draw_source_label(frame, labels[source_idx])

            row = source_idx // cols
            col = source_idx % cols
            y1, y2 = row * tile_h, (row + 1) * tile_h
            x1, x2 = col * tile_w, (col + 1) * tile_w
            canvas[y1:y2, x1:x2] = frame

        if scale != 1.0:
            canvas = cv2.resize(canvas, (output_w, output_h), interpolation=cv2.INTER_LINEAR)
        out.write(canvas)

    out.release()
    for cap in caps:
        cap.release()
    return output_path


def draw_source_label(frame, label):
    label = str(label)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, frame.shape[1] / 1800.)
    thickness = max(1, int(frame.shape[1] / 900.))
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    x, y = 12, 12 + text_h
    cv2.rectangle(
        frame,
        (8, 8),
        (16 + text_w, 18 + text_h + baseline),
        (0, 0, 0),
        thickness=-1
    )
    cv2.putText(frame, label, (x, y), font, font_scale, (255, 255, 255), thickness)


def cv2_addBox(track_id, frame, x1, y1, x2, y2, line_thickness, text_thickness, text_scale):
    color = get_color(abs(track_id))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color=color, thickness=line_thickness)
    cv2.putText(
        frame, str(track_id), (x1, y1 + 30), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255), thickness=text_thickness)


def write_results(filename, data_type, w_frame_id, w_track_id, w_x1, w_y1, w_x2, w_y2, w_wid, w_hgt):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{x2},{y2},{w},{h}\n'
    else:
        raise ValueError(data_type)
    with open(filename, 'a') as f:
        line = save_format.format(frame=w_frame_id, id=w_track_id, x1=w_x1, y1=w_y1, x2=w_x2, y2=w_y2, w=w_wid, h=w_hgt)
        f.write(line)
    # print('save results to {}'.format(filename))


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color


if __name__ == '__main__':
    main()
