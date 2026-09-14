"""复用项目检测、Deep SORT 与身份融合逻辑的 Web 推理流程。"""
import json
from pathlib import Path
import random
import time

import cv2
import numpy as np

from demo import prepare_detections
from demo_stream import CameraTracker, LatestFrameReader, draw_global_track, parse_args
from deep_sort import nn_matching
from deep_sort.tracker import Tracker
from global_identity import GlobalIDManager
from offline_association import Tracklet, fuse_tracklets
from web.media import VideoReader, MP4Writer, LivePublisher


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


def run_offline(spec, detector, encoder, reporter, *, model_factory=None):
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
        tracker = Tracker(nn_matching.NearestNeighborDistanceMetric('cosine', .2, 100), max_age=100, n_init=3)
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
                    detections = prepare_detections(lane_detector, lane_encoder, frame, camera_id)
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
    mapping, groups = fuse_tracklets(tracklets, threshold=options['reid_threshold'],
                                     margin=.05, min_frames=options['min_reid_frames'], gallery_size=32)
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


def run_online(spec, detector, encoder, reporter):
    from tracking_contracts.events import EventWriter, frame_event
    sources, options = spec['sources'], spec['options']
    args = parse_args(['--streams', *sources, '--display', 'false', '--encoder-batch-size', str(options['batch_size'])])
    trackers = [CameraTracker(i, detector, encoder, args) for i in range(len(sources))]
    global_ids = GlobalIDManager(threshold=options['reid_threshold'])
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
