"""复用项目检测、Deep SORT 与身份融合逻辑的 Web 推理流程。"""
import json
from pathlib import Path
import random
import time

import cv2
import numpy as np

from demo import prepare_detections
from demo_stream import (
    CameraTracker,
    LatestFrameReader,
    configure_ffmpeg_low_latency,
    draw_global_track,
    parse_args,
)
from deep_sort import nn_matching
from deep_sort.tracker import Tracker
from global_identity import GlobalIDManager
from offline_association import Tracklet, fuse_tracklets
from web.media import VideoReader, MP4Writer, LivePublisher


def build_stream_args(sources, options):
    """Translate validated Web options to the same CLI arguments as demo_stream."""
    values = [
        ('--detector-score', options['confidence']),
        ('--detector-imgsz', options['detector_imgsz']),
        ('--encoder-batch-size', options['batch_size']),
        ('--deep-sort-max-cosine-distance', options['deep_sort_max_cosine_distance']),
        ('--tracker-max-age', options['tracker_max_age']),
        ('--post-nms', 'nms'),
        ('--nms-max-overlap', options['nms_max_overlap']),
        ('--tracker-new-track-min-confidence', options['tracker_new_track_min_confidence']),
        ('--track-duplicate-containment', options['track_duplicate_containment']),
        ('--track-duplicate-max-cosine-distance', options['track_duplicate_max_cosine_distance']),
        ('--global-reid-threshold', options['reid_threshold']),
        ('--global-reid-strong-threshold', options['global_reid_strong_threshold']),
        ('--global-reid-margin', options['global_reid_margin']),
        ('--global-gallery-match', options['global_gallery_match']),
        ('--global-candidate-threshold', options['global_candidate_threshold']),
        ('--global-borderline-confirm-frames', options['global_borderline_confirm_frames']),
        ('--global-borderline-confirm-ratio', options['global_borderline_confirm_ratio']),
        ('--global-borderline-confirm-threshold', options['global_borderline_confirm_threshold']),
        ('--global-prototype-count', options['global_prototype_count']),
        ('--global-prototype-merge-threshold', options['global_prototype_merge_threshold']),
        ('--global-feature-min-confidence', options['global_feature_min_confidence']),
        ('--global-feature-min-box-height', options['global_feature_min_box_height']),
        ('--global-feature-max-occlusion', options['global_feature_max_occlusion']),
        ('--global-feature-update-max-distance', options['global_feature_update_max_distance']),
        ('--same-camera-reconnect', str(options['same_camera_reconnect']).lower()),
        ('--same-camera-reconnect-timeout', options['same_camera_reconnect_timeout']),
        ('--same-camera-reconnect-distance', options['same_camera_reconnect_distance']),
        ('--same-camera-reconnect-reid-threshold', options['same_camera_reconnect_reid_threshold']),
        ('--same-camera-reconnect-reid-margin', options['same_camera_reconnect_reid_margin']),
        ('--same-camera-reconnect-confirm-frames', options['same_camera_reconnect_confirm_frames']),
        ('--same-camera-reconnect-confirm-ratio', options['same_camera_reconnect_confirm_ratio']),
        ('--same-camera-reconnect-confirm-threshold', options['same_camera_reconnect_confirm_threshold']),
        ('--same-camera-conflict-continuity', options['same_camera_conflict_continuity']),
        ('--show-local-id', 'false'),
    ]
    # Keep the canonical Web profile identical to start_accuracy.ps1.  The
    # offline runner consumes these values directly; the online runner has its
    # own bounded live queue because it must be able to drop stale frames.
    argv = [
        '--streams', *sources,
        '--stream-mode', 'queue',
        '--stream-queue-size', '2',
        '--display', 'false',
    ]
    for name, value in values:
        argv.extend((name, str(value)))
    return parse_args(argv)


def build_global_id_manager(args):
    """Keep the Web identity manager aligned with demo_stream's full configuration."""
    return GlobalIDManager(
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
        pending_timeout=args.global_pending_timeout,
        match_strategy=args.global_gallery_match,
        candidate_threshold=args.global_candidate_threshold,
        strong_threshold=args.global_reid_strong_threshold,
        borderline_confirm_frames=args.global_borderline_confirm_frames,
        borderline_confirm_ratio=args.global_borderline_confirm_ratio,
        borderline_confirm_threshold=args.global_borderline_confirm_threshold,
        prototype_count=args.global_prototype_count,
        prototype_merge_threshold=args.global_prototype_merge_threshold,
        same_camera_reconnect_reid_threshold=args.same_camera_reconnect_reid_threshold,
        same_camera_reconnect_reid_margin=args.same_camera_reconnect_reid_margin,
        same_camera_conflict_continuity=args.same_camera_conflict_continuity,
    )


def video_writer(path, fps, shape):
    height, width = shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), fps, (width, height))
    if not writer.isOpened():
        writer.release()
        raise RuntimeError('无法创建结果视频，请检查磁盘空间与视频编码器。')
    return writer


def probe_video(path):
    cap = None
    try:
        cap = VideoReader(path)
        ok, frame = cap.read()
        if not ok:
            raise ValueError('视频无法解码或不含有效画面。')
        fps = float(cap.rate)
        return {'fps': fps if np.isfinite(fps) and 0 < fps <= 240 else 25.,
                'frames': max(0, int(cap.stream.frames)),
                'width': frame.shape[1], 'height': frame.shape[0]}
    except Exception as exc:
        raise ValueError('视频无法解码或不含有效画面。') from exc
    finally:
        if cap is not None:
            cap.release()


def run_offline_streaming(spec, detector, encoder, reporter, *, model_factory=None):
    """Run the terminal accuracy pipeline and publish per-camera Web artifacts.

    This deliberately uses ``LatestFrameReader(queue=2)`` and its capture-time
    clock, just like ``start_accuracy.ps1``/``demo_stream.py``.  Global identity
    thresholds were tuned on that clock.  Feeding media PTS here made one second
    equal thirty inference updates instead of roughly the terminal's observed
    three-to-five updates, delaying identity creation and starving galleries.
    """
    sources, options = spec['sources'], spec['options']
    directory = Path(spec['directory'])
    if not sources:
        raise ValueError('请至少提供一个离线视频。')
    reporter.check()
    if detector is None or encoder is None:
        if model_factory is None:
            raise ValueError('离线直接 GID 模式需要可用的检测与 ReID 模型。')
        detector, encoder = model_factory()
        reporter.check()

    metadata = []
    for camera_id, source in enumerate(sources):
        try:
            metadata.append(probe_video(source))
        except ValueError as exc:
            raise ValueError(f'摄像头 {camera_id + 1}：{exc}') from None
    total = sum(item['frames'] for item in metadata)
    args = build_stream_args(sources, options)
    trackers = [CameraTracker(i, detector, encoder, args) for i in range(len(sources))]
    global_ids = build_global_id_manager(args)
    readers, avi_writers = [], [None] * len(sources)
    counts, last_seq = [0] * len(sources), [-1] * len(sources)
    processed, complete = 0, False
    registry = {}
    artifacts = directory / 'artifacts'
    temp_avis = [directory / f'camera-{i}-partial.avi' for i in range(len(sources))]
    temp_mp4s = [directory / f'camera-{i}-partial.mp4' for i in range(len(sources))]
    temp_log = directory / 'tracks-final.tmp'

    reporter.emit(
        force=True, total_frames=total or None, offline_workers=1,
        message='使用终端同款 queue=2、采集时间轴与 GlobalIDManager 处理，结果直接显示 GID。')
    try:
        readers = [LatestFrameReader(
            source, camera_id, loop_file=False,
            stream_mode=args.stream_mode, queue_size=args.stream_queue_size,
            capture_backend=args.capture_backend,
            buffer_size=args.capture_buffer_size,
            stream_timeout_ms=args.stream_timeout_ms)
            for camera_id, source in enumerate(sources)]
        for reader in readers:
            reader.start()
        for camera in reporter.state['cameras']:
            camera['status'] = 'running'
        with temp_log.open('w', encoding='utf-8') as log:
            while True:
                reporter.check()
                now = time.time()
                progressed, any_alive = False, False
                rendered = []
                for camera_id, (reader, tracker) in enumerate(zip(readers, trackers)):
                    ok, frame, timestamp, seq = reader.read()
                    stats = reader.stats()
                    if not reader.stopped or stats['inference_queue']:
                        any_alive = True
                    if not ok or seq == last_seq[camera_id]:
                        continue
                    last_seq[camera_id] = seq
                    progressed = True
                    tracks = tracker.process(frame)
                    assignments = global_ids.update_camera(camera_id, tracks, timestamp or now)
                    for track in tracks:
                        local_id = int(track['local_id'])
                        global_id = assignments[local_id]
                        key = (camera_id, local_id)
                        state = registry.setdefault(key, {
                            'start': seq, 'end': seq,
                            'samples': 0, 'global_ids': set()})
                        state['end'] = seq
                        if track['feature_quality'] > 0:
                            state['samples'] += 1
                        if global_id is not None:
                            state['global_ids'].add(int(global_id))
                        log.write(json.dumps({
                            'camera_id': camera_id, 'frame': seq,
                            'timestamp': timestamp, 'local_id': local_id,
                            'global_id': global_id, 'bbox': track['bbox'],
                            'feature_quality': track['feature_quality']}) + '\n')
                        draw_global_track(
                            frame, track['bbox'], global_id, local_id=local_id,
                            show_local_id=False, show_pending=args.show_pending,
                            text_scale_factor=args.gid_text_scale)
                    rendered.append((camera_id, frame, len(tracks)))
                    counts[camera_id] += 1
                    processed += 1

                # demo_stream updates every camera before cleanup and rendering.
                # Keep slow JPEG/video encoding out of the inter-camera matching
                # order so Web presentation cannot change a GID decision.
                global_ids.cleanup(now)
                for camera_id, frame, track_count in rendered:
                    if avi_writers[camera_id] is None:
                        avi_writers[camera_id] = video_writer(
                            temp_avis[camera_id], metadata[camera_id]['fps'], frame.shape)
                    avi_writers[camera_id].write(frame)
                    reporter.preview(camera_id, frame, counts[camera_id], track_count)
                reporter.emit(
                    processed_frames=processed,
                    identity_count=len(global_ids.global_tracks),
                    progress=min(94., 94 * processed / total) if total else None)
                if not progressed:
                    if not any_alive:
                        break
                    time.sleep(.005)
            for camera_id, (declared, actual) in enumerate(
                    zip((item['frames'] for item in metadata), counts)):
                if actual == 0:
                    raise ValueError(f'摄像头 {camera_id + 1} 没有可读取的帧。')
                if declared and actual != declared:
                    raise ValueError(
                        f'摄像头 {camera_id + 1} 声明 {declared} 帧，实际只解码 {actual} 帧，'
                        '拒绝发布可能截断的视频。')
            complete = True
    finally:
        for reader in readers:
            reader.stop()
        for writer in avi_writers:
            if writer is not None:
                writer.release()

    # Encode MP4 only after every GID decision is final.  Encoding it inside
    # the inference loop would slow the consumer and therefore alter the same
    # capture-time clock that the terminal-tuned identity gates use.
    for camera_id, source in enumerate(temp_avis):
        reporter.check()
        reader, writer, rendered_count = None, None, 0
        try:
            reader = VideoReader(source)
            while True:
                reporter.check()
                ok, frame = reader.read()
                if not ok:
                    break
                if writer is None:
                    writer = MP4Writer(
                        temp_mp4s[camera_id], metadata[camera_id]['fps'], frame.shape)
                writer.write(frame, rendered_count / metadata[camera_id]['fps'])
                rendered_count += 1
            if rendered_count != counts[camera_id]:
                raise ValueError(
                    f'摄像头 {camera_id + 1} 结果转码帧数不完整：'
                    f'{rendered_count}/{counts[camera_id]}。')
        finally:
            if reader is not None:
                reader.release()
            if writer is not None:
                writer.close(verify=rendered_count == counts[camera_id])
        reporter.emit(
            processed_frames=processed,
            identity_count=len(global_ids.global_tracks),
            progress=94 + 5 * (camera_id + 1) / len(sources),
            message='GID 匹配完成，正在生成下载视频。')

    for camera_id in range(len(sources)):
        reporter.check()
        temp_avis[camera_id].replace(artifacts / f'camera-{camera_id + 1}.avi')
        temp_mp4s[camera_id].replace(artifacts / f'camera-{camera_id + 1}.mp4')
        reporter.state['cameras'][camera_id]['status'] = 'completed'
    temp_log.replace(artifacts / 'tracks.jsonl')

    mapping_data = {
        'frame_index_base': 1,
        'distance_metric': args.global_distance,
        'identity_mode': 'terminal_compatible',
        'timing_mode': 'capture_wall_clock',
        'stream_mode': args.stream_mode,
        'stream_queue_size': args.stream_queue_size,
        'cameras': [{'camera_id': i, 'source': name, **metadata[i], 'frames': counts[i]}
                    for i, name in enumerate(spec['names'])],
        'tracks': []}
    for key, state in sorted(registry.items()):
        local_state = global_ids.local_tracks.get(key)
        final_gid = local_state.global_id if local_state is not None else None
        mapping_data['tracks'].append({
            'camera_id': key[0], 'local_id': key[1], 'global_id': final_gid,
            'start_frame': state['start'], 'end_frame': state['end'],
            'reid_samples': state['samples'],
            'observed_global_ids': sorted(state['global_ids'])})
    (artifacts / 'id_mapping.json').write_text(
        json.dumps(mapping_data, ensure_ascii=False, indent=2), encoding='utf-8')
    reporter.emit(
        force=True, processed_frames=processed,
        identity_count=len(global_ids.global_tracks), progress=100,
        message='终端兼容 GID 追踪完成，结果已输出。')


def run_offline_posthoc(spec, detector, encoder, reporter, *, model_factory=None):
    """有界视频并行：每路顺序读帧，独立模型/跟踪器；统一融合后并行渲染。

    model_factory 在各工作线程中创建独立模型，线程处理后续文件时复用。
    直接传 detector/encoder 仅供单路兼容调用，绝不跨线程共享实例。
    """
    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
    import threading
    from web.worker import Cancelled

    sources, options = spec['sources'], spec['options']
    directory = Path(spec['directory'])
    requested = options.get('offline_workers', 2)
    if isinstance(requested, bool) or not isinstance(requested, int) or not 1 <= requested <= 4:
        raise ValueError('离线并行路数必须为 1–4 的整数。')
    workers = min(requested, len(sources))
    if not workers:
        raise ValueError('请至少提供一个离线视频。')
    reporter.check()
    if workers > 1 and model_factory is None:
        raise ValueError('并行离线处理需要 model_factory 为每个工作线程创建独立模型。')

    abort = threading.Event()
    progress_lock, model_init_lock = threading.Lock(), threading.Lock()
    local = threading.local()

    def check():
        reporter.check()
        if abort.is_set():
            raise Cancelled()

    def parallel(function):
        # 只保持 workers 个在途任务，不按输入数量创建线程或堆积待处理帧。
        results = [None] * len(sources)
        next_id = 0
        pending = {}
        pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='mtmc-offline')
        try:
            while next_id < len(sources) or pending:
                check()
                while next_id < len(sources) and len(pending) < workers:
                    pending[pool.submit(function, next_id)] = next_id
                    next_id += 1
                done, _ = wait(pending, timeout=.1, return_when=FIRST_COMPLETED)
                # 先检查这一批全部完成项，再启动后续文件；错误不被兄弟线程的取消掩盖。
                for future in done:
                    results[pending[future]] = future.result()
                for future in done:
                    del pending[future]
            return results
        except BaseException:
            abort.set()
            for future in pending:
                future.cancel()
            raise
        finally:
            pool.shutdown(wait=True, cancel_futures=True)

    def update_camera(camera_id, **values):
        with progress_lock:
            reporter.state['cameras'][camera_id].update(values)
            reporter.emit()

    def models():
        if not hasattr(local, 'models'):
            # TransReID 初始化会修改模块导入路径；仅初始化串行，实际推理不加共享锁。
            with model_init_lock:
                check()
                local.models = model_factory() if model_factory else (detector, encoder)
                check()
        return local.models

    metadata = []
    for i, source in enumerate(sources):
        check()
        try:
            metadata.append(probe_video(source))
        except ValueError as exc:
            raise ValueError(f'摄像头 {i + 1}：{exc}') from None
    total = sum(item['frames'] for item in metadata)
    reporter.emit(force=True, total_frames=total or None, offline_workers=workers,
                  message=f'离线最多 {workers} 路并行追踪，超出路数排队；全局 ID 将在统一融合后生成。')
    processed = 0

    def track_camera(camera_id):
        nonlocal processed
        check()
        update_camera(camera_id, status='loading')
        lane_detector, lane_encoder = models()
        update_camera(camera_id, status='running')
        tracker = Tracker(
            nn_matching.NearestNeighborDistanceMetric(
                'cosine', options['deep_sort_max_cosine_distance'], 100),
            max_age=options['tracker_max_age'], n_init=3,
            new_track_min_confidence=options['tracker_new_track_min_confidence'])
        registry, rng = {}, random.Random(camera_id)
        frame_index, next_track_id = 0, 0
        cap = VideoReader(sources[camera_id])
        try:
            with (directory / f'camera-{camera_id}.jsonl').open('w', encoding='utf-8') as records:
                while True:
                    check()
                    ok, frame = cap.read()
                    if not ok:
                        break
                    detections = prepare_detections(
                        lane_detector, lane_encoder, frame, camera_id,
                        post_nms='nms', max_overlap=options['nms_max_overlap'])
                    check()
                    tracker.predict()
                    tracker.update(detections)
                    observations = []
                    for track in tracker.tracks:
                        if track.time_since_update > 1:
                            continue
                        key = (camera_id, int(track.track_id))
                        state = registry.setdefault(key, {'start': frame_index, 'end': frame_index,
                                                          'id': None, 'features': [], 'qualities': [], 'seen': 0})
                        state['end'] = frame_index
                        if not track.is_confirmed():
                            continue
                        if state['id'] is None:
                            next_track_id += 1
                            state['id'] = next_track_id
                        if track.time_since_update == 0 and track.last_feature is not None:
                            state['seen'] += 1
                            feature = np.asarray(track.last_feature).copy()
                            quality = float(track.last_confidence or 0)
                            if len(state['features']) < 64:
                                state['features'].append(feature)
                                state['qualities'].append(quality)
                            else:
                                index = rng.randrange(state['seen'])
                                if index < 64:
                                    state['features'][index] = feature
                                    state['qualities'][index] = quality
                        bbox = [int(v) for v in track.to_tlbr()]
                        observations.append({'local_id': key[1], 'bbox': bbox})
                        cv2.rectangle(frame, tuple(bbox[:2]), tuple(bbox[2:]), (115, 210, 220), 2)
                        cv2.putText(frame, f'LID:{key[1]}', (max(0, bbox[0]), max(20, bbox[1])),
                                    cv2.FONT_HERSHEY_SIMPLEX, .7, (115, 210, 220), 2)
                    records.write(json.dumps({'frame': frame_index, 'timestamp': cap.timestamp,
                                              'tracks': observations}) + '\n')
                    frame_index += 1
                    with progress_lock:
                        processed += 1
                        reporter.preview(camera_id, frame, frame_index, len(observations))
                        reporter.emit(processed_frames=processed,
                                      progress=min(70., 70 * processed / total) if total else None)
        finally:
            cap.release()
        if frame_index == 0:
            raise ValueError(f'摄像头 {camera_id + 1} 没有可读取的帧。')
        declared = metadata[camera_id]['frames']
        if declared and frame_index != declared:
            raise ValueError(f'摄像头 {camera_id + 1} 声明 {declared} 帧，'
                             f'实际只解码 {frame_index} 帧，拒绝发布可能截断的视频。')
        update_camera(camera_id, status='fusing')
        return frame_index, registry

    tracked = parallel(track_camera)
    check()
    counts, registry, next_track_id = [], {}, 0
    # 按输入视频顺序和路内确认顺序编号，不能按线程完成顺序分配全局 ID。
    for count, camera_registry in tracked:
        counts.append(count)
        for key, state in sorted(camera_registry.items(), key=lambda item: item[1]['id'] or float('inf')):
            if state['id'] is not None:
                next_track_id += 1
                state['id'] = next_track_id
            registry[key] = state
    total = sum(counts)
    reporter.emit(force=True, message='所有视频追踪完成，正在统一融合跨摄像头身份…',
                  progress=72, total_frames=total)
    tracklets = [Tracklet(key, state['id'], state['start'], state['end'],
                         np.asarray(state['features']) if state['features'] else np.empty((0, 0)), state['qualities'])
                 for key, state in registry.items() if state['id'] is not None]
    mapping, groups = fuse_tracklets(
        tracklets, threshold=options['reid_threshold'],
        margin=options['global_reid_margin'], min_frames=options['min_reid_frames'],
        gallery_size=options['offline_gallery_size'],
        match_strategy=options['offline_gallery_match'])
    check()
    mapping_data = {'frame_index_base': 0, 'distance_metric': 'cosine',
                    'cameras': [{'camera_id': i, 'source': name, **metadata[i], 'frames': counts[i]}
                                for i, name in enumerate(spec['names'])],
                    'tracks': [{'camera_id': key[0], 'local_id': key[1], 'global_id': mapping[key],
                                'tracking_id': state['id'], 'start_frame': state['start'],
                                'end_frame': state['end'], 'reid_samples': state['seen'],
                                'retained_samples': len(state['features'])}
                               for key, state in registry.items() if key in mapping]}
    artifacts = directory / 'artifacts'
    (artifacts / 'id_mapping.json').write_text(json.dumps(mapping_data, ensure_ascii=False, indent=2), encoding='utf-8')
    reporter.emit(force=True, identity_count=len(groups),
                  message=f'身份融合完成，正在以最多 {workers} 路并行输出标注视频…', progress=75)
    rendered = 0

    def render_camera(camera_id):
        nonlocal rendered
        check()
        update_camera(camera_id, status='rendering')
        cap, writer, mp4 = VideoReader(sources[camera_id]), None, None
        temp_video = directory / f'camera-{camera_id}-partial.avi'
        temp_mp4 = directory / f'camera-{camera_id}-partial.mp4'
        temp_log = directory / f'camera-{camera_id}-tracks.tmp'
        complete, written = False, 0
        try:
            with (directory / f'camera-{camera_id}.jsonl').open(encoding='utf-8') as records, temp_log.open('w', encoding='utf-8') as log:
                for line in records:
                    check()
                    record = json.loads(line)
                    if record['frame'] != written:
                        raise RuntimeError('帧记录不连续，拒绝发布不完整结果。')
                    ok, frame = cap.read()
                    if not ok:
                        raise RuntimeError(f'摄像头 {camera_id + 1} 二次读取提前结束，结果未完整生成。')
                    if writer is None:
                        writer = video_writer(temp_video, metadata[camera_id]['fps'], frame.shape)
                        mp4 = MP4Writer(temp_mp4, metadata[camera_id]['fps'], frame.shape)
                    for track in record['tracks']:
                        key = (camera_id, track['local_id'])
                        gid = mapping[key]
                        draw_global_track(frame, track['bbox'], gid)
                        log.write(json.dumps({'camera_id': camera_id, 'frame': record['frame'],
                                              'timestamp': record['timestamp'], **track,
                                              'tracking_id': registry[key]['id'], 'global_id': gid}) + '\n')
                    writer.write(frame)
                    mp4.write(frame, record['timestamp'])
                    written += 1
                    with progress_lock:
                        rendered += 1
                        reporter.preview(camera_id, frame, written, len(record['tracks']),
                                         status='rendering', force=written == counts[camera_id])
                        reporter.emit(progress=75 + 24 * rendered / max(1, total))
                if written != counts[camera_id] or cap.read()[0]:
                    raise RuntimeError('两遍解码帧数不一致，拒绝发布不完整结果。')
                complete = True
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            if mp4 is not None:
                mp4.close(verify=complete)
        check()
        temp_video.replace(artifacts / f'camera-{camera_id + 1}.avi')
        temp_mp4.replace(artifacts / f'camera-{camera_id + 1}.mp4')
        update_camera(camera_id, status='completed')
        return temp_log

    camera_logs = parallel(render_camera)
    # 各路独立写日志，再按摄像头/帧顺序流式拼接；不并发写同一个文件。
    with (directory / 'tracks-final.tmp').open('w', encoding='utf-8') as log:
        for path in camera_logs:
            with path.open(encoding='utf-8') as camera_log:
                for line in camera_log:
                    check()
                    log.write(line)
    check()
    (directory / 'tracks-final.tmp').replace(artifacts / 'tracks.jsonl')


def run_offline(spec, detector, encoder, reporter, *, model_factory=None):
    if spec['options'].get('offline_identity_mode', 'streaming') == 'posthoc':
        return run_offline_posthoc(
            spec, detector, encoder, reporter, model_factory=model_factory)
    return run_offline_streaming(
        spec, detector, encoder, reporter, model_factory=model_factory)


def run_online(spec, detector, encoder, reporter):
    from tracking_contracts.events import EventWriter, frame_event
    sources, options = spec['sources'], spec['options']
    args = build_stream_args(sources, options)
    # Match demo_stream.py: Web jobs must configure OpenCV/FFmpeg before any
    # capture is opened.  In particular, RTSP defaults to TCP so cameras that
    # do not deliver usable UDP packets do not sit at decoded_frames == 0.
    configure_ffmpeg_low_latency(args)
    trackers = [CameraTracker(i, detector, encoder, args) for i in range(len(sources))]
    global_ids = build_global_id_manager(args)
    publishers = [LivePublisher(i, spec.get('live_queue')) for i in range(len(sources))]
    readers = [LatestFrameReader(source, i, stream_timeout_ms=3000, buffer_size=0, stream_mode='queue',
                                queue_size=30, queue_max_bytes=64*1024*1024,
                                queue_overflow='drop_oldest', frame_observer=publishers[i].submit)
               for i, source in enumerate(sources)]
    last_seq, last_seen = [-1] * len(sources), [time.monotonic()] * len(sources)
    log_path = Path(spec['directory']) / 'artifacts' / 'tracks.jsonl'
    frames = 0
    event_writer = EventWriter(Path(spec['directory'])/'events', Path(spec['directory']).name)
    last_health = 0.
    try:
        for publisher in publishers:
            publisher.start()
        for reader in readers:
            reader.start()
        with log_path.open('w', encoding='utf-8') as log:
            while True:
                reporter.check()
                changed = False
                for camera_id, (reader, tracker) in enumerate(zip(readers, trackers)):
                    reporter.check()
                    ok, frame, timestamp, seq = reader.read()
                    camera = reporter.state['cameras'][camera_id]
                    capture_stats = reader.stats()
                    camera.update(capture_stats)
                    camera.update(preview_dropped=publishers[camera_id].dropped,
                                  published_frames=publishers[camera_id].published)
                    if not ok or seq == last_seq[camera_id]:
                        capture_idle = (time.time()-capture_stats['last_capture_time']
                                        if capture_stats.get('last_capture_time') else time.monotonic()-last_seen[camera_id])
                        if capture_idle > 5:
                            camera['status'] = 'reconnecting'
                        continue
                    last_seq[camera_id], last_seen[camera_id] = seq, time.monotonic()
                    # 原画发布队列也引用此帧；标注必须使用副本，避免跨线程读写像素。
                    frame = frame.copy()
                    camera['source_width'], camera['source_height'] = frame.shape[1], frame.shape[0]
                    tracks = tracker.process(frame)
                    assignments = global_ids.update_camera(camera_id, tracks, timestamp)
                    event_writer.submit(frame_event(camera_id, seq, timestamp, frame.shape[1],
                                                    frame.shape[0], tracks, assignments, capture_stats))
                    for track in tracks:
                        gid = assignments[track['local_id']]
                        log.write(json.dumps({'camera_id': camera_id, 'frame': seq, 'timestamp': timestamp,
                                              'local_id': track['local_id'], 'global_id': gid,
                                              'bbox': track['bbox'], 'feature_quality': track['feature_quality']}) + '\n')
                        draw_global_track(frame, track['bbox'], gid, show_pending=True)
                    frames += 1
                    changed = True
                    camera['inference_age_ms'] = max(0, round((time.time()-timestamp)*1000))
                    reporter.preview(camera_id, frame, camera['frames'] + 1, len(tracks))
                now = time.monotonic()
                if now-last_health >= 1:
                    event_writer.submit({'type': 'health', 'timestamp': time.time(),
                                         'cameras': [{'camera_id': i,
                                                      'decoded_frames': r.stats().get('decoded_frames', 0),
                                                      'last_capture_time': r.stats().get('last_capture_time'),
                                                      'inference_skipped': r.stats().get('inference_skipped', 0)}
                                                     for i, r in enumerate(readers)]})
                    last_health = now
                reporter.state['event_log'] = {'dropped': event_writer.dropped, 'error': event_writer.error}
                if all((time.time()-reader.stats()['last_capture_time']
                        if reader.stats().get('last_capture_time') else now-last_seen[i]) > 30
                       for i, reader in enumerate(readers)):
                    raise RuntimeError('所有视频流连续 30 秒没有新画面，请检查地址、账号、网络与流编码。')
                global_ids.cleanup(time.time())
                reporter.emit(processed_frames=frames, identity_count=len(global_ids.global_tracks),
                              message='连续原画与模型标注独立显示；原画流畅不代表每帧均完成推理。')
                if changed:
                    log.flush()
                else:
                    time.sleep(.015)
    finally:
        for reader in readers:
            reader.stop()
        for publisher in publishers:
            publisher.stop()
        event_writer.close()
