# ! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import argparse
import math
import os
import threading
import time
from collections import deque

import cv2
import numpy as np
from PIL import Image

from deep_sort import nn_matching
from deep_sort import preprocessing
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from reid_backends import create_reid_encoder
from global_identity import GlobalIDManager


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ('yes', 'true', 't', '1', 'y'):
        return True
    if value in ('no', 'false', 'f', '0', 'n'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def is_stream_url(source):
    if not isinstance(source, str):
        return False
    source = source.lower()
    return source.startswith(('rtsp://', 'http://', 'https://'))


def capture_api_value(name, source):
    name = (name or 'auto').lower()
    if name == 'auto':
        if is_stream_url(source) and hasattr(cv2, 'CAP_FFMPEG'):
            return cv2.CAP_FFMPEG
        return cv2.CAP_ANY

    mapping = {
        'ffmpeg': getattr(cv2, 'CAP_FFMPEG', cv2.CAP_ANY),
        'any': cv2.CAP_ANY,
    }
    if name not in mapping:
        raise ValueError('Unknown capture backend: {}'.format(name))
    return mapping[name]


def configure_ffmpeg_low_latency(args):
    if not args.ffmpeg_low_delay:
        return
    # FFmpeg's nobuffer option can discard file packets, including the first
    # frame. Capture options are process-wide: use them only for all-URL inputs,
    # never for local-file replay or mixed file/network sessions.
    if not all(is_stream_url(source) for source in args.streams):
        return

    options = [
        'rtsp_transport;{}'.format(args.rtsp_transport),
        'fflags;nobuffer',
        'flags;low_delay',
        'max_delay;{}'.format(int(args.ffmpeg_max_delay_us)),
        'stimeout;{}'.format(int(args.stream_timeout_ms * 1000)),
    ]
    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = '|'.join(options)


def reid_feature_quality(frame, bbox, confidence, args, track_hits, other_boxes=()):
    if confidence is None or float(confidence) < args.global_feature_min_confidence:
        return 0.0
    if int(track_hits) < args.global_feature_min_track_hits:
        return 0.0

    frame_h, frame_w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    width = x2 - x1
    height = y2 - y1
    if width <= 0 or height < args.global_feature_min_box_height:
        return 0.0
    area_ratio = float(width * height) / max(1.0, float(frame_w * frame_h))
    if area_ratio < args.global_feature_min_area_ratio:
        return 0.0

    # Keep tracking occluded people, but do not use mixed-person crops for ReID.
    max_occlusion = args.global_feature_max_occlusion
    for other in other_boxes:
        ox1, oy1, ox2, oy2 = other
        intersection = max(0, min(x2, ox2) - max(x1, ox1)) * max(
            0, min(y2, oy2) - max(y1, oy1))
        if intersection / float(width * height) > max_occlusion:
            return 0.0

    margin = args.global_feature_border_margin
    if margin > 0 and (
        x1 <= margin or y1 <= margin or
        x2 >= frame_w - 1 - margin or y2 >= frame_h - 1 - margin
    ):
        return 0.0

    if args.global_feature_min_blur > 0:
        crop = frame[max(0, y1):min(frame_h, y2), max(0, x1):min(frame_w, x2)]
        if crop.size == 0:
            return 0.0
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if blur_score < args.global_feature_min_blur:
            return 0.0

    return float(confidence)


class LatestFrameReader(object):
    def __init__(
        self,
        source,
        camera_id,
        loop_file=False,
        retry_interval=0.2,
        stream_mode='latest',
        queue_size=0,
        capture_backend='auto',
        buffer_size=1,
        stream_timeout_ms=3000,
        queue_max_bytes=0,
        queue_overflow='block',
        frame_observer=None
    ):
        self.source = source
        self.open_source = source
        self.camera_id = camera_id
        self.loop_file = loop_file
        self.retry_interval = retry_interval
        self.stream_mode = str(stream_mode or 'latest').lower()
        if self.stream_mode not in ('latest', 'queue'):
            raise ValueError('stream_mode must be "latest" or "queue".')
        self.queue_size = int(queue_size)
        if queue_overflow not in ('block', 'drop_oldest'):
            raise ValueError('queue_overflow must be block or drop_oldest')
        self.queue_overflow = queue_overflow
        self.queue_max_bytes = int(queue_max_bytes)
        self.frame_observer = frame_observer
        self.queue_bytes = 0
        self.skipped_frames = 0
        self.last_consumed_seq = 0
        self.last_capture_time = 0.0
        self.capture_backend = capture_backend
        self.buffer_size = int(buffer_size)
        self.stream_timeout_ms = int(stream_timeout_ms)
        self.is_file = isinstance(self.open_source, str) and os.path.isfile(self.open_source)
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.frame = None
        self.timestamp = 0.0
        self.seq = 0
        self.queue = deque()
        self.fps = 0.0
        self.stopped = False
        self.ready = False
        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True

    def start(self):
        self.thread.start()
        return self

    def _open_capture(self):
        api = capture_api_value(self.capture_backend, self.open_source)
        # FFmpeg 的超时是 open-only 参数，打开之后 set() 不会生效。
        params = []
        if is_stream_url(self.open_source) and api == getattr(cv2, 'CAP_FFMPEG', -1):
            if self.stream_timeout_ms > 0:
                params = [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self.stream_timeout_ms,
                          cv2.CAP_PROP_READ_TIMEOUT_MSEC, self.stream_timeout_ms]
        cap = (cv2.VideoCapture(self.open_source, api, params) if params else
               cv2.VideoCapture(self.open_source, api) if api != cv2.CAP_ANY else cv2.VideoCapture(self.open_source))

        if self.buffer_size > 0:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
        if self.stream_timeout_ms > 0:
            if hasattr(cv2, 'CAP_PROP_OPEN_TIMEOUT_MSEC'):
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self.stream_timeout_ms)
            if hasattr(cv2, 'CAP_PROP_READ_TIMEOUT_MSEC'):
                cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, self.stream_timeout_ms)
        return cap

    def _store_frame(self, frame):
        with self.condition:
            self.seq += 1
            item = (frame, time.time(), self.seq)
            self.last_capture_time = item[1]
            if self.frame_observer is not None:
                self.frame_observer(*item)
            if self.stream_mode == 'queue':
                if self.queue_max_bytes > 0 and frame.nbytes > self.queue_max_bytes:
                    self.skipped_frames += 1
                    return
                def full():
                    return ((self.queue_size > 0 and len(self.queue) >= self.queue_size) or
                            (self.queue_max_bytes > 0 and self.queue_bytes + frame.nbytes > self.queue_max_bytes))
                while full() and not self.stopped:
                    if self.queue_overflow == 'drop_oldest':
                        self.queue_bytes -= self.queue.popleft()[0].nbytes
                        self.skipped_frames += 1
                        continue
                    self.condition.wait(timeout=0.02)
                if self.stopped:
                    return
                self.queue.append(item)
                self.queue_bytes += frame.nbytes
                self.timestamp = item[1]
            else:
                if self.frame is not None and self.seq - 1 > self.last_consumed_seq:
                    self.skipped_frames += 1
                self.frame, self.timestamp, self.seq = item
            self.ready = True
            self.condition.notify_all()

    def _run(self):
        cap = None
        next_read_time = time.monotonic()
        try:
            while not self.stopped:
                if cap is None:
                    cap = self._open_capture()
                    if not cap.isOpened():
                        cap.release()
                        cap = None
                        if self.is_file:
                            break
                        time.sleep(self.retry_interval)
                        continue
                    self.fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
                    if not np.isfinite(self.fps) or self.fps < 0:
                        self.fps = 0.0
                if self.is_file and self.stream_mode == 'latest' and self.fps > 0:
                    # 本地文件模拟实时输入时按源帧率播放；queue 模式全速逐帧处理。
                    with self.condition:
                        self.condition.wait(timeout=max(0, next_read_time - time.monotonic()))
                    if self.stopped:
                        break
                    next_read_time = time.monotonic() + 1.0 / self.fps
                ret, frame = cap.read()
                if not ret:
                    if self.is_file and self.loop_file:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    if self.is_file:
                        break
                    # 断流后重新打开连接，而非在失效的 capture 上无限 read。
                    cap.release()
                    cap = None
                    time.sleep(self.retry_interval)
                    continue
                self._store_frame(frame)
        finally:
            if cap is not None:
                cap.release()
            with self.condition:
                self.stopped = True
                self.condition.notify_all()

    def read(self):
        with self.condition:
            if self.stream_mode == 'queue' and self.queue:
                frame, timestamp, seq = self.queue.popleft()
                self.queue_bytes -= frame.nbytes
                self.last_consumed_seq = seq
                self.condition.notify_all()
                return True, frame.copy(), timestamp, seq
            if self.frame is None:
                return False, None, self.timestamp, self.seq
            self.last_consumed_seq = self.seq
            return True, self.frame.copy(), self.timestamp, self.seq

    def stats(self):
        with self.condition:
            return {'decoded_frames': self.seq, 'inference_skipped': self.skipped_frames,
                    'inference_queue': len(self.queue), 'capture_fps': round(self.fps, 2),
                    'last_capture_time': self.last_capture_time}

    def stop(self):
        with self.condition:
            self.stopped = True
            self.condition.notify_all()
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)


class CameraTracker(object):
    def __init__(self, camera_id, detector, encoder, args):
        self.camera_id = int(camera_id)
        self.detector = detector
        self.encoder = encoder
        self.args = args
        metric = nn_matching.NearestNeighborDistanceMetric(
            'cosine',
            args.deep_sort_max_cosine_distance,
            args.deep_sort_nn_budget if args.deep_sort_nn_budget > 0 else None
        )
        self.tracker = Tracker(
            metric,
            max_age=args.tracker_max_age,
            n_init=args.tracker_n_init
        )

    def process(self, frame):
        image = Image.fromarray(frame[..., ::-1])
        if hasattr(self.detector, 'detect_image_with_scores'):
            boxes, detection_scores = self.detector.detect_image_with_scores(image)
        else:
            boxes = self.detector.detect_image(image)
            detection_scores = [1.0 for _ in boxes]
        keep = preprocessing.select_detection_indices(
            boxes, detection_scores, mode=self.args.post_nms,
            max_overlap=self.args.nms_max_overlap)
        boxes = [boxes[i] for i in keep]
        detection_scores = [detection_scores[i] for i in keep]
        features = self.encoder(frame, boxes, camera_id=self.camera_id)
        detections = [
            Detection(bbox, score, feature)
            for bbox, score, feature in zip(boxes, detection_scores, features)
        ]

        self.tracker.predict()
        self.tracker.update(detections)

        tracks = []
        frame_h, frame_w = frame.shape[:2]
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            x1 = max(0, int(bbox[0]))
            y1 = max(0, int(bbox[1]))
            x2 = min(frame_w - 1, int(bbox[2]))
            y2 = min(frame_h - 1, int(bbox[3]))
            if x2 <= x1 or y2 <= y1:
                continue
            updated_this_frame = track.time_since_update == 0
            feature = getattr(track, 'last_feature', None) if updated_this_frame else None
            confidence = getattr(track, 'last_confidence', None) if updated_this_frame else None
            feature_bbox = track.last_detection_bbox
            # Judge the detection crop that produced the feature, not the display box.
            other_boxes = [other.last_detection_bbox for other in self.tracker.tracks
                           if other is not track and other.time_since_update == 0]
            quality = reid_feature_quality(
                frame,
                feature_bbox,
                confidence,
                self.args,
                track.hits,
                other_boxes
            ) if feature is not None else 0.0
            tracks.append({
                'local_id': track.track_id,
                'bbox': (x1, y1, x2, y2),
                'feature': feature,
                'feature_quality': quality,
            })
        return tracks


def build_reid(args):
    return create_reid_encoder(
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


def get_color(idx):
    idx = int(idx) * 3
    return ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)


def draw_global_track(frame, bbox, global_id, show_pending=False, text_scale_factor=1.4):
    x1, y1, x2, y2 = bbox
    color = (120, 120, 120) if global_id is None else get_color(global_id)
    line_thickness = max(2, int(frame.shape[1] / 400.0))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color=color, thickness=line_thickness)

    if global_id is None and not show_pending:
        return

    label = 'GID:{}'.format(global_id if global_id is not None else '?')
    font = cv2.FONT_HERSHEY_PLAIN
    text_scale = float(text_scale_factor) * max(1.0, frame.shape[1] / 1600.0)
    text_thickness = max(2, int(round(text_scale)))
    (text_w, text_h), baseline = cv2.getTextSize(
        label,
        font,
        text_scale,
        text_thickness
    )
    label_y = max(text_h + 8, y1)
    cv2.rectangle(
        frame,
        (x1, label_y - text_h - baseline - 8),
        (x1 + text_w + 10, label_y + baseline + 4),
        (0, 0, 0),
        thickness=-1
    )
    cv2.putText(
        frame,
        label,
        (x1 + 5, label_y),
        font,
        text_scale,
        (0, 0, 255),
        thickness=text_thickness
    )


def compose_grid(frames, tile_width=0, tile_height=0, display_scale=1.0):
    valid_frames = [frame for frame in frames if frame is not None]
    if not valid_frames:
        return None

    if tile_width > 0 and tile_height > 0:
        tile_w, tile_h = int(tile_width), int(tile_height)
    else:
        tile_h, tile_w = valid_frames[0].shape[:2]

    count = len(frames)
    cols = count if count <= 2 else int(math.ceil(math.sqrt(count)))
    rows = int(math.ceil(float(count) / cols))
    canvas = np.zeros((tile_h * rows, tile_w * cols, 3), dtype=np.uint8)

    for idx, frame in enumerate(frames):
        if frame is None:
            tile = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
        elif frame.shape[1] != tile_w or frame.shape[0] != tile_h:
            tile = cv2.resize(frame, (tile_w, tile_h), interpolation=cv2.INTER_LINEAR)
        else:
            tile = frame

        row = idx // cols
        col = idx % cols
        y1, y2 = row * tile_h, (row + 1) * tile_h
        x1, x2 = col * tile_w, (col + 1) * tile_w
        canvas[y1:y2, x1:x2] = tile

    if display_scale != 1.0:
        out_w = max(1, int(round(canvas.shape[1] * display_scale)))
        out_h = max(1, int(round(canvas.shape[0] * display_scale)))
        canvas = cv2.resize(canvas, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    return canvas


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Realtime multi-camera MTMC demo')
    parser.add_argument(
        '--streams',
        nargs='+',
        required=True,
        help='RTSP/HTTP stream URLs. Local video files are also allowed for testing.'
    )
    parser.add_argument(
        '--stream-mode',
        choices=('latest', 'queue'),
        default='latest',
        help='latest keeps latency low by using newest frames; queue processes frames in order but can add delay.'
    )
    parser.add_argument(
        '--stream-queue-size',
        type=int,
        default=0,
        help='Queue size for --stream-mode queue. 0 means unlimited application queue.'
    )
    parser.add_argument(
        '--capture-backend',
        choices=('auto', 'ffmpeg', 'any'),
        default='auto',
        help='OpenCV capture backend. auto uses FFmpeg for RTSP/HTTP streams.'
    )
    parser.add_argument(
        '--capture-buffer-size',
        type=int,
        default=1,
        help='OpenCV capture buffer size. 1 reduces live stream delay.'
    )
    parser.add_argument(
        '--rtsp-transport',
        choices=('tcp', 'udp'),
        default='tcp',
        help='RTSP transport used by OpenCV FFmpeg. tcp is clearer/stabler; udp can be lower latency.'
    )
    parser.add_argument('--ffmpeg-low-delay', nargs='?', const=True, type=str2bool, default=True)
    parser.add_argument('--ffmpeg-max-delay-us', type=int, default=0)
    parser.add_argument('--stream-timeout-ms', type=int, default=3000)
    parser.add_argument('--detector', default='yolo11l.pt')
    parser.add_argument('--detector-backend', choices=('auto', 'ultralytics', 'torchvision'), default='auto')
    parser.add_argument('--detector-weights', default='default')
    parser.add_argument('--detector-score', type=float, default=0.3)

    parser.add_argument('--reid-backend', choices=('torchreid', 'transreid', 'custom'), default='transreid')
    parser.add_argument('--reid-model', default='resnet50')
    parser.add_argument('--reid-weights', default='model_data/models/model.pth')
    parser.add_argument('--reid-config', default=None)
    parser.add_argument('--encoder-batch-size', type=int, default=8)
    parser.add_argument('--transreid-variant', default='msmt17', choices=('msmt17', 'market1501', 'dukemtmc'))
    parser.add_argument('--transreid-weights', default=None)
    parser.add_argument('--transreid-repo', default=None)
    parser.add_argument('--transreid-assets-root', default=None)
    parser.add_argument('--transreid-download', nargs='?', const=True, type=str2bool, default=True)
    parser.add_argument('--custom-reid-root', default='custom_reid_models')
    parser.add_argument('--custom-reid-name', default=None)
    parser.add_argument('--custom-reid-dir', default=None)
    parser.add_argument('--custom-reid-adapter', default=None)

    parser.add_argument('--deep-sort-max-cosine-distance', type=float, default=0.2)
    parser.add_argument(
        '--deep-sort-nn-budget',
        type=int,
        default=100,
        help='Maximum ReID samples retained per active local track. <= 0 means unlimited.'
    )
    parser.add_argument('--post-nms', choices=('detector', 'nms'), default='detector',
                        help='Use detector suppression only, or add score-ordered IoU NMS.')
    parser.add_argument('--nms-max-overlap', type=float, default=0.4)
    parser.add_argument('--tracker-max-age', type=int, default=300)
    parser.add_argument('--tracker-n-init', type=int, default=3)

    parser.add_argument('--global-reid-threshold', type=float, default=0.3)
    parser.add_argument('--global-reid-margin', type=float, default=0.02)
    parser.add_argument('--global-delay', type=float, default=1.0)
    parser.add_argument('--global-min-features', type=int, default=5)
    parser.add_argument('--global-confirm-windows', type=int, default=3,
                        help='Consecutive non-overlapping feature windows required to reuse a GID.')
    parser.add_argument('--global-revalidate-threshold', type=float, default=0.4)
    parser.add_argument('--global-revalidate-windows', type=int, default=3)
    parser.add_argument('--global-pending-timeout', type=float, default=10.0)
    parser.add_argument('--global-gallery-size', type=int, default=80)
    parser.add_argument('--global-track-feature-size', type=int, default=40)
    parser.add_argument('--global-anchor-size', type=int, default=20)
    parser.add_argument('--global-anchor-min-confidence', type=float, default=0.65)
    parser.add_argument('--global-feature-aggregate-frames', type=int, default=5)
    parser.add_argument('--global-feature-min-novelty', type=float, default=0.03)
    parser.add_argument('--global-feature-update-max-distance', type=float, default=0.35)
    parser.add_argument('--global-feature-min-confidence', type=float, default=0.4)
    parser.add_argument('--global-feature-min-track-hits', type=int, default=5)
    parser.add_argument('--global-feature-min-box-height', type=int, default=48)
    parser.add_argument('--global-feature-min-area-ratio', type=float, default=0.0005)
    parser.add_argument('--global-feature-border-margin', type=int, default=2)
    parser.add_argument('--global-feature-min-blur', type=float, default=15.0)
    parser.add_argument('--global-feature-max-occlusion', type=float, default=0.6,
                        help='Reject ReID crops covered beyond this fraction by another person.')
    parser.add_argument('--global-distance', choices=('cosine', 'euclidean'), default='cosine')
    parser.add_argument('--global-topk', type=int, default=3)
    parser.add_argument('--global-feature-update-interval', type=float, default=0.2)
    parser.add_argument('--global-active-timeout', type=float, default=0.3)
    parser.add_argument('--global-stale-timeout', type=float, default=1800)
    parser.add_argument('--same-camera-reconnect', nargs='?', const=True, type=str2bool, default=True)
    parser.add_argument('--same-camera-reconnect-timeout', type=float, default=5.0)
    parser.add_argument('--same-camera-reconnect-distance', type=float, default=0.4)
    parser.add_argument('--same-camera-reconnect-min-iou', type=float, default=0.02)
    parser.add_argument('--same-camera-reconnect-confirm-frames', type=int, default=10)
    parser.add_argument('--same-camera-reconnect-confirm-ratio', type=float, default=0.6)
    parser.add_argument('--same-camera-reconnect-confirm-threshold', type=float, default=0.35)

    parser.add_argument('--display', nargs='?', const=True, type=str2bool, default=True)
    parser.add_argument('--window-name', default='MTMC Stream')
    parser.add_argument('--show-pending', nargs='?', const=True, type=str2bool, default=False)
    parser.add_argument('--display-wait-ms', type=int, default=1)
    parser.add_argument(
        '--gid-text-scale',
        type=float,
        default=1.4,
        help='Global ID text scale factor. Increase/decrease this to tune GID size.'
    )
    parser.add_argument('--display-scale', type=float, default=1.0)
    parser.add_argument('--tile-width', type=int, default=0)
    parser.add_argument('--tile-height', type=int, default=0)
    parser.add_argument('--output', default=None, help='Optional path to save the realtime output video.')
    parser.add_argument('--track-log', default=None,
                        help='Optional JSONL log of camera, local/global IDs, timestamps and boxes.')
    parser.add_argument('--loop-videos', nargs='?', const=True, type=str2bool, default=False)
    parser.add_argument('--max-frames', type=int, default=0)
    return parser.parse_args(argv)


def main(argv=None, detector=None, encoder=None):
    import json

    args = parse_args(argv)
    if args.display_scale <= 0:
        raise ValueError('--display-scale must be greater than 0.')
    if args.stream_queue_size < 0:
        raise ValueError('--stream-queue-size must be >= 0.')
    if args.capture_buffer_size < 0:
        raise ValueError('--capture-buffer-size must be >= 0.')
    if args.display_wait_ms < 1:
        args.display_wait_ms = 1
    if args.gid_text_scale <= 0:
        raise ValueError('--gid-text-scale must be greater than 0.')
    if args.same_camera_reconnect_timeout < 0:
        raise ValueError('--same-camera-reconnect-timeout must be >= 0.')
    if args.same_camera_reconnect_distance < 0:
        raise ValueError('--same-camera-reconnect-distance must be >= 0.')
    if args.same_camera_reconnect_min_iou < 0:
        raise ValueError('--same-camera-reconnect-min-iou must be >= 0.')
    if args.global_gallery_size < 1:
        raise ValueError('--global-gallery-size must be >= 1.')
    if args.global_track_feature_size < 1:
        raise ValueError('--global-track-feature-size must be >= 1.')
    if not 1 <= args.global_anchor_size <= args.global_gallery_size:
        raise ValueError('--global-anchor-size must be between 1 and --global-gallery-size.')
    if args.global_feature_aggregate_frames < 1:
        raise ValueError('--global-feature-aggregate-frames must be >= 1.')
    if not 0 <= args.global_feature_min_confidence <= 1:
        raise ValueError('--global-feature-min-confidence must be between 0 and 1.')
    if not 0 <= args.global_anchor_min_confidence <= 1:
        raise ValueError('--global-anchor-min-confidence must be between 0 and 1.')
    if args.global_feature_min_novelty < 0:
        raise ValueError('--global-feature-min-novelty must be >= 0.')
    if args.global_feature_update_max_distance <= 0:
        raise ValueError('--global-feature-update-max-distance must be > 0.')
    if args.global_feature_min_track_hits < 1:
        raise ValueError('--global-feature-min-track-hits must be >= 1.')
    if args.global_feature_min_box_height < 1:
        raise ValueError('--global-feature-min-box-height must be >= 1.')
    if args.global_feature_min_area_ratio < 0:
        raise ValueError('--global-feature-min-area-ratio must be >= 0.')
    if args.global_feature_border_margin < 0:
        raise ValueError('--global-feature-border-margin must be >= 0.')
    if args.global_feature_min_blur < 0:
        raise ValueError('--global-feature-min-blur must be >= 0.')
    if not 0 <= args.global_feature_max_occlusion <= 1:
        raise ValueError('--global-feature-max-occlusion must be between 0 and 1.')
    if not 0 <= args.nms_max_overlap <= 1:
        raise ValueError('--nms-max-overlap must be between 0 and 1.')
    if args.same_camera_reconnect_confirm_frames < 1:
        raise ValueError('--same-camera-reconnect-confirm-frames must be >= 1.')
    if not 0 <= args.same_camera_reconnect_confirm_ratio <= 1:
        raise ValueError('--same-camera-reconnect-confirm-ratio must be between 0 and 1.')
    if args.same_camera_reconnect_confirm_threshold <= 0:
        raise ValueError('--same-camera-reconnect-confirm-threshold must be > 0.')

    global_ids = GlobalIDManager(
        threshold=args.global_reid_threshold,
        margin=args.global_reid_margin,
        delay_seconds=args.global_delay,
        min_features=args.global_min_features,
        gallery_size=args.global_gallery_size,
        track_feature_size=args.global_track_feature_size,
        anchor_size=args.global_anchor_size,
        anchor_min_confidence=args.global_anchor_min_confidence,
        aggregate_frames=args.global_feature_aggregate_frames,
        feature_min_novelty=args.global_feature_min_novelty,
        feature_update_max_distance=args.global_feature_update_max_distance,
        metric=args.global_distance,
        topk=args.global_topk,
        feature_update_interval=args.global_feature_update_interval,
        active_timeout=args.global_active_timeout,
        stale_timeout=args.global_stale_timeout,
        same_camera_reconnect=args.same_camera_reconnect,
        same_camera_reconnect_timeout=args.same_camera_reconnect_timeout,
        same_camera_reconnect_distance=args.same_camera_reconnect_distance,
        same_camera_reconnect_min_iou=args.same_camera_reconnect_min_iou,
        reconnect_confirm_frames=args.same_camera_reconnect_confirm_frames,
        reconnect_confirm_ratio=args.same_camera_reconnect_confirm_ratio,
        reconnect_confirm_threshold=args.same_camera_reconnect_confirm_threshold,
        confirm_windows=args.global_confirm_windows,
        revalidate_threshold=args.global_revalidate_threshold,
        revalidate_windows=args.global_revalidate_windows,
        pending_timeout=args.global_pending_timeout
    )

    configure_ffmpeg_low_latency(args)
    if detector is None:
        from torch_detector import build_person_detector
        detector = build_person_detector(
            model_name=args.detector, backend=args.detector_backend,
            weights=args.detector_weights, score_threshold=args.detector_score)
    if encoder is None:
        encoder = build_reid(args)
    print('Detector:', detector)
    print('ReID backend:', args.reid_backend)
    print('Streams:', args.streams)
    trackers = [CameraTracker(idx, detector, encoder, args)
                for idx in range(len(args.streams))]
    readers = []

    writer = None
    processed_frames = 0
    display_enabled = args.display
    last_seq = [-1 for _ in args.streams]
    last_rendered = [None for _ in args.streams]
    track_log = None

    try:
        for idx, source in enumerate(args.streams):
            reader = LatestFrameReader(
                source, idx, loop_file=args.loop_videos, stream_mode=args.stream_mode,
                queue_size=args.stream_queue_size, capture_backend=args.capture_backend,
                buffer_size=args.capture_buffer_size, stream_timeout_ms=args.stream_timeout_ms)
            readers.append(reader)
            reader.start()
        if args.track_log:
            log_dir = os.path.dirname(args.track_log)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            track_log = open(args.track_log, 'w', encoding='utf-8')
        while True:
            now = time.time()
            frames = []
            any_alive = False
            processed_any_new = False
            for idx, (reader, camera_tracker) in enumerate(zip(readers, trackers)):
                ret, frame, frame_ts, seq = reader.read()
                if not reader.stopped:
                    any_alive = True
                if not ret:
                    frames.append(last_rendered[idx])
                    continue
                if seq == last_seq[idx]:
                    frames.append(last_rendered[idx])
                    continue

                last_seq[idx] = seq
                processed_any_new = True
                tracks = camera_tracker.process(frame)
                assignments = global_ids.update_camera(
                    camera_tracker.camera_id, tracks, frame_ts or now)
                for track in tracks:
                    global_id = assignments[track['local_id']]
                    if track_log:
                        track_log.write(json.dumps({
                            'camera_id': camera_tracker.camera_id,
                            'frame': seq,
                            'timestamp': frame_ts,
                            'local_id': track['local_id'],
                            'global_id': global_id,
                            'bbox': track['bbox'],
                            'feature_quality': track['feature_quality'],
                        }) + '\n')
                    draw_global_track(
                        frame,
                        track['bbox'],
                        global_id,
                        show_pending=args.show_pending,
                        text_scale_factor=args.gid_text_scale
                    )
                last_rendered[idx] = frame
                frames.append(frame)

            global_ids.cleanup(now)
            if not processed_any_new:
                if not any_alive:
                    break
                time.sleep(0.005)
                continue

            canvas = compose_grid(
                frames,
                tile_width=args.tile_width,
                tile_height=args.tile_height,
                display_scale=args.display_scale
            )

            if canvas is not None:
                if args.output and writer is None:
                    out_dir = os.path.dirname(args.output)
                    if out_dir and not os.path.exists(out_dir):
                        os.makedirs(out_dir)
                    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                    writer = cv2.VideoWriter(
                        args.output,
                        fourcc,
                        25.0,
                        (canvas.shape[1], canvas.shape[0])
                    )
                if writer is not None:
                    writer.write(canvas)

                if display_enabled:
                    try:
                        cv2.imshow(args.window_name, canvas)
                        key = cv2.waitKey(args.display_wait_ms) & 0xFF
                    except cv2.error as exc:
                        print('Display disabled because cv2.imshow failed:', exc)
                        display_enabled = False
                        key = 255
                    if key == ord('q') or key == 27:
                        break

            processed_frames += 1
            if args.max_frames > 0 and processed_frames >= args.max_frames:
                break
            if not any_alive and canvas is None:
                break

    finally:
        if track_log is not None:
            track_log.close()
        for reader in readers:
            reader.stop()
        if writer is not None:
            writer.release()
        if display_enabled:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
