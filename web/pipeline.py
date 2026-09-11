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


def run_offline(spec, detector, encoder, reporter):
    """两遍读文件：先跟踪和记录，再将融合 ID 渲染到原分辨率视频。

    不缓存视频帧或人体裁剪。每条轨迹最多保留 64 个抽样特征，
    检测记录逐帧写 JSONL，融合后第二遍顺序读取。
    """
    sources, options = spec['sources'], spec['options']
    directory = Path(spec['directory'])
    metadata = []
    for i, source in enumerate(sources):
        reporter.check()
        try:
            metadata.append(probe_video(source))
        except ValueError as exc:
            raise ValueError(f'摄像头 {i + 1}：{exc}') from None
    total = sum(item['frames'] for item in metadata)
    reporter.emit(force=True, total_frames=total or None, message='逐帧追踪中，画面显示本地 ID；全局 ID 将在融合后生成。')
    registry, rng = {}, random.Random(0)
    processed = 0
    counts = []
    for camera_id, source in enumerate(sources):
        tracker = Tracker(nn_matching.NearestNeighborDistanceMetric('cosine', .2, 100), max_age=100, n_init=3)
        cap = VideoReader(source)
        frame_index = 0
        records_path = directory / f'camera-{camera_id}.jsonl'
        try:
            with records_path.open('w', encoding='utf-8') as records:
                while True:
                    reporter.check()
                    ok, frame = cap.read()
                    if not ok:
                        break
                    detections = prepare_detections(detector, encoder, frame, camera_id)
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
                            state['id'] = sum(s['id'] is not None for s in registry.values()) + 1
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
                        observations.append({'local_id': key[1], 'tracking_id': state['id'], 'bbox': bbox})
                        cv2.rectangle(frame, tuple(bbox[:2]), tuple(bbox[2:]), (115, 210, 220), 2)
                        cv2.putText(frame, f'LID:{key[1]}', (max(0, bbox[0]), max(20, bbox[1])),
                                    cv2.FONT_HERSHEY_SIMPLEX, .7, (115, 210, 220), 2)
                    records.write(json.dumps({'frame': frame_index, 'timestamp': cap.timestamp,
                                              'tracks': observations}) + '\n')
                    frame_index += 1
                    processed += 1
                    reporter.preview(camera_id, frame, frame_index, len(observations))
                    reporter.emit(processed_frames=processed, progress=min(70., 70 * processed / total) if total else None)
        finally:
            cap.release()
        if frame_index == 0:
            raise ValueError(f'摄像头 {camera_id + 1} 没有可读取的帧。')
        if metadata[camera_id]['frames'] and frame_index != metadata[camera_id]['frames']:
            raise ValueError(f'摄像头 {camera_id + 1} 声明 {metadata[camera_id]["frames"]} 帧，'
                             f'实际只解码 {frame_index} 帧，拒绝发布可能截断的视频。')
        counts.append(frame_index)
        reporter.state['cameras'][camera_id]['status'] = 'fusing'
    # 不依赖容器帧数元数据作为实际工作量。
    total = sum(counts)
    reporter.emit(force=True, message='正在进行跨摄像头身份融合…', progress=72, total_frames=total)
    tracklets = [Tracklet(key, state['id'], state['start'], state['end'],
                         np.asarray(state['features']) if state['features'] else np.empty((0, 0)), state['qualities'])
                 for key, state in registry.items() if state['id'] is not None]
    mapping, groups = fuse_tracklets(tracklets, threshold=options['reid_threshold'],
                                     margin=.05, min_frames=options['min_reid_frames'], gallery_size=32)
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
    reporter.emit(force=True, identity_count=len(groups), message='正在输出全局 ID 标注视频…', progress=75)
    rendered = 0
    with (directory / 'tracks-final.tmp').open('w', encoding='utf-8') as log:
        for camera_id, source in enumerate(sources):
            cap, writer, mp4 = VideoReader(source), None, None
            temp_video = directory / f'camera-{camera_id}-partial.avi'
            temp_mp4 = directory / f'camera-{camera_id}-partial.mp4'
            complete = False
            try:
                with (directory / f'camera-{camera_id}.jsonl').open(encoding='utf-8') as records:
                    for line in records:
                        reporter.check()
                        record = json.loads(line)
                        ok, frame = cap.read()
                        if not ok:
                            raise RuntimeError(f'摄像头 {camera_id + 1} 二次读取提前结束，结果未完整生成。')
                        if writer is None:
                            writer = video_writer(temp_video, metadata[camera_id]['fps'], frame.shape)
                            mp4 = MP4Writer(temp_mp4, metadata[camera_id]['fps'], frame.shape)
                        for track in record['tracks']:
                            gid = mapping[(camera_id, track['local_id'])]
                            draw_global_track(frame, track['bbox'], gid)
                            log.write(json.dumps({'camera_id': camera_id, 'frame': record['frame'],
                                                  'timestamp': record['timestamp'],
                                                  **track, 'global_id': gid}) + '\n')
                        writer.write(frame)
                        mp4.write(frame, record['timestamp'])
                        rendered += 1
                        reporter.preview(camera_id, frame, record['frame'] + 1, len(record['tracks']),
                                         status='rendering', force=record['frame'] + 1 == counts[camera_id])
                        reporter.emit(progress=75 + 24 * rendered / max(1, total))
                    if cap.read()[0]:
                        raise RuntimeError('两遍解码帧数不一致，拒绝发布不完整结果。')
                    complete = True
            finally:
                cap.release()
                if writer is not None:
                    writer.release()
                if mp4 is not None:
                    mp4.close(verify=complete)
            temp_video.replace(artifacts / f'camera-{camera_id + 1}.avi')
            temp_mp4.replace(artifacts / f'camera-{camera_id + 1}.mp4')
            reporter.state['cameras'][camera_id]['status'] = 'completed'
    (directory / 'tracks-final.tmp').replace(artifacts / 'tracks.jsonl')


def run_online(spec, detector, encoder, reporter):
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
                    tracks = tracker.process(frame)
                    assignments = global_ids.update_camera(camera_id, tracks, timestamp)
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
