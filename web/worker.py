"""进程入口与限频预览输出；模型加载仅发生在任务子进程。"""
import os
from pathlib import Path
import queue
import re
import time


class Cancelled(Exception):
    pass


class Reporter:
    def __init__(self, spec, events, cancel):
        self.directory = Path(spec['directory'])
        self.events, self.cancel = events, cancel
        self.started = time.monotonic()
        self.last_send = 0
        self.last_preview = {}
        self.state = {'status': 'running', 'message': '正在追踪…', 'processed_frames': 0,
                      'identity_count': 0, 'fps': 0, 'progress': None,
                      'cameras': [{'id': i, 'name': name, 'status': 'waiting', 'frames': 0,
                                   'preview_version': 0, 'tracks': 0}
                                  for i, name in enumerate(spec['names'])]}

    def check(self):
        if self.cancel.is_set():
            raise Cancelled()

    def emit(self, force=False, **values):
        self.check()
        self.state.update(values)
        now = time.monotonic()
        if not force and now - self.last_send < 0.3:
            return
        self.state['fps'] = round(self.state['processed_frames'] / max(now - self.started, .001), 2)
        # Queue 在后台序列化，必须复制，不能继续修改已入队的对象。
        import copy
        try:
            self.events.put(copy.deepcopy(self.state), timeout=1 if force else 0)
            self.last_send = now
        except queue.Full:
            pass

    def preview(self, camera_id, frame, frame_num, tracks, status='running', force=False):
        import cv2
        self.check()
        camera = self.state['cameras'][camera_id]
        camera.update(frames=frame_num, tracks=tracks, status=status)
        now = time.monotonic()
        if force or now - self.last_preview.get(camera_id, 0) >= .3:
            height, width = frame.shape[:2]
            scale = min(1., 960 / width, 720 / height)
            if scale < 1:
                frame = cv2.resize(frame, (max(1, round(width * scale)), max(1, round(height * scale))))
            ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                path = self.directory / 'previews' / f'{camera_id}.jpg'
                temporary = path.with_suffix('.tmp')
                temporary.write_bytes(encoded.tobytes())
                temporary.replace(path)
                camera['preview_version'] += 1
                self.last_preview[camera_id] = now

    def artifacts(self):
        return [{'name': p.name, 'size': p.stat().st_size}
                for p in sorted((self.directory / 'artifacts').iterdir())
                if p.is_file() and p.suffix in ('.avi', '.mp4', '.json', '.jsonl')]


def load_models(spec, check):
    """为一个工作线程创建独立模型，不与其他视频推理线程共享可变状态。"""
    from web.jobs import ROOT
    from torch_detector import build_person_detector
    from reid_backends import create_reid_encoder
    detector_path = Path(os.environ.get('MTMC_DETECTOR', str(ROOT / 'yolo11l.pt')))
    if not detector_path.is_file():
        raise ValueError('本地 YOLO 权重不存在，请配置 MTMC_DETECTOR。')
    check()
    detector = build_person_detector(str(detector_path), backend='ultralytics',
                                     score_threshold=spec['options']['confidence'])
    check()
    encoder = create_reid_encoder(
        backend='transreid', batch_size=spec['options']['batch_size'],
        transreid_variant='msmt17', transreid_download=False,
        transreid_repo=os.environ.get('MTMC_TRANSREID_REPO', str(ROOT / 'external/transreid/repo')),
        transreid_weights=os.environ.get('MTMC_TRANSREID_WEIGHTS', str(ROOT / 'external/transreid/weights/vit_transreid_msmt.pth')))
    check()
    return detector, encoder


def run_worker(spec, events, cancel):
    from web.jobs import ROOT
    os.chdir(ROOT)
    from web.media import LiveTransport
    transport = LiveTransport(spec['live_connection']) if spec.get('live_connection') else None
    spec['live_queue'] = transport
    # 与 Web 的“尽可能保帧”要求一致，TCP 传输，不启用 nobuffer/discard。
    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'
    # OpenCV/FFmpeg 可能在原生 stderr 打印含密码的 URL；子进程禁用原始日志，
    # 仅通过下面经过脱敏的状态事件返回错误。
    with open(os.devnull, 'w') as sink:
        os.dup2(sink.fileno(), 1)
        os.dup2(sink.fileno(), 2)
    reporter = Reporter(spec, events, cancel)
    try:
        import torch
        torch.set_num_threads(max(1, int(os.environ.get('MTMC_CPU_THREADS', '2'))))
        from web.pipeline import run_offline, run_online
        if spec['mode'] == 'offline':
            run_offline(spec, None, None, reporter, model_factory=lambda: load_models(spec, reporter.check))
        else:
            detector, encoder = load_models(spec, reporter.check)
            reporter.started = time.monotonic()
            reporter.emit(force=True)
            run_online(spec, detector, encoder, reporter)
        reporter.emit(force=True, status='completed', message='追踪完成，结果可下载。',
                      progress=100 if spec['mode'] == 'offline' else None,
                      artifacts=reporter.artifacts())
    except Cancelled:
        events.put({'status': 'cancelled', 'message': '任务已停止。', 'artifacts': reporter.artifacts()})
    except Exception as exc:
        message = str(exc)
        if spec['mode'] == 'offline' and (isinstance(exc, MemoryError) or 'out of memory' in message.lower()):
            message += '；请降低离线并行路数或特征提取批大小后重试。'
        for source in spec['sources']:
            message = message.replace(source, '[输入源]')
        message = re.sub(r'(?i)(?:rtsp|https?)://\S+', '[流地址已隐藏]', message)
        events.put({'status': 'failed', 'message': '追踪失败。',
                    'error': f'{type(exc).__name__}: {message[:500]}'})
    finally:
        if transport:
            transport.close()
