"""单机单推理进程调度；API 进程不导入 PyTorch。"""
import copy
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import threading
import time
import uuid
from web.media import FrameHub

TERMINAL = {'completed', 'failed', 'cancelled'}
ROOT = Path(__file__).resolve().parents[1]


def valid_id(value):
    try:
        return uuid.UUID(value).hex == value
    except (ValueError, TypeError, AttributeError):
        return False


def write_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


class JobManager:
    def __init__(self, root, worker=None, cancel_grace=5):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / 'jobs').mkdir(exist_ok=True)
        (self.root / 'uploads').mkdir(exist_ok=True)
        self.lock = threading.RLock()
        self.jobs = {}
        self.active = None
        self.live_hubs = {}
        self.closed = False
        self.cancel_grace = cancel_grace
        self.context = mp.get_context('spawn')
        if worker is None:
            from web.worker import run_worker
            worker = run_worker
        self.worker = worker
        for path in (self.root / 'jobs').glob('*/job.json'):
            try:
                job = json.loads(path.read_text(encoding='utf-8'))
                if not valid_id(job['id']) or path.parent.name != job['id']:
                    continue
                if job['status'] not in TERMINAL:
                    job.update(status='failed', message='服务已重启，原任务中断，请重新启动。',
                               error='服务重启导致任务中断。', finished_at=time.time())
                    write_json(path, job)
                self.jobs[job['id']] = job
            except (OSError, ValueError, KeyError):
                continue
        self.wake = threading.Event()
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()

    def directory(self, job_id):
        if not valid_id(job_id):
            raise KeyError(job_id)
        return self.root / 'jobs' / job_id

    def get(self, job_id):
        with self.lock:
            if job_id not in self.jobs:
                raise KeyError(job_id)
            return copy.deepcopy(self.jobs[job_id])

    def list(self):
        with self.lock:
            return copy.deepcopy(sorted(self.jobs.values(), key=lambda j: j['created_at'], reverse=True)[:50])

    def _save(self, job):
        write_json(self.directory(job['id']) / 'job.json', job)

    def create(self, mode, sources, names, options):
        with self.lock:
            if self.closed or self.active is not None:
                raise RuntimeError('已有任务正在运行，请先停止或等待完成。')
            job_id = uuid.uuid4().hex
            directory = self.directory(job_id)
            directory.mkdir()
            (directory / 'previews').mkdir()
            (directory / 'artifacts').mkdir()
            job = {'id': job_id, 'mode': mode, 'status': 'starting',
                   'created_at': time.time(), 'finished_at': None,
                   'message': '正在加载检测与 ReID 模型…', 'error': None,
                   'progress': None, 'processed_frames': 0, 'total_frames': None,
                   'identity_count': 0, 'fps': 0, 'options': options,
                   'cameras': [{'id': i, 'name': name, 'status': 'waiting', 'frames': 0,
                                'preview_version': 0, 'tracks': 0} for i, name in enumerate(names)],
                   'artifacts': []}
            self.jobs[job_id] = job
            self._save(job)
            events, cancel = self.context.Queue(maxsize=64), self.context.Event()
            live_receiver, live_sender = self.context.Pipe(duplex=False)
            hub = FrameHub()
            self.live_hubs.clear()
            self.live_hubs[job_id] = hub
            # 原始流地址只经内存传入子进程，绝不写入任务 JSON。
            spec = {'mode': mode, 'sources': sources, 'names': names, 'options': options,
                    'directory': str(directory), 'live_connection': live_sender}
            process = self.context.Process(target=self.worker, args=(spec, events, cancel), daemon=True)
            try:
                process.start()
            except Exception:
                events.close()
                live_receiver.close()
                live_sender.close()
                job.update(status='failed', error='无法启动推理进程。', finished_at=time.time())
                self._save(job)
                raise
            live_sender.close()
            self.active = {'id': job_id, 'process': process, 'events': events,
                           'cancel': cancel, 'cancel_at': None, 'last_save': 0,
                           'live_receiver': live_receiver, 'hub': hub, 'live_eof': False}
            self.wake.set()
            return copy.deepcopy(job)

    def stop(self, job_id):
        with self.lock:
            job = self.jobs[job_id]
            if self.active and self.active['id'] == job_id and job['status'] not in TERMINAL:
                self.active['cancel'].set()
                if self.active['cancel_at'] is None:
                    self.active['cancel_at'] = time.monotonic()
                job.update(status='stopping', message='正在停止并释放视频流…')
                self._save(job)
            return copy.deepcopy(job)

    def _drain(self, active, job):
        while True:
            try:
                event = active['events'].get_nowait()
            except queue.Empty:
                break
            if active['cancel_at'] is not None:
                event.pop('status', None)
                event.pop('message', None)
            job.update(event)

    def _monitor(self):
        while not self.closed:
            self.wake.wait(0.01)
            self.wake.clear()
            with self.lock:
                active = self.active
                if not active:
                    continue
                job = self.jobs[active['id']]
                # 视频和状态分开传输，视频拥堵不阻塞取消/状态事件。
                for _ in range(128):
                    try:
                        if active['live_eof'] or not active['live_receiver'].poll():
                            break
                        active['hub'].append(active['live_receiver'].recv())
                    except (EOFError, OSError):
                        active['live_eof'] = True
                        break
                self._drain(active, job)
                process = active['process']
                if (active['cancel_at'] is not None and process.is_alive()
                        and time.monotonic() - active['cancel_at'] > self.cancel_grace):
                    process.terminate()
                    process.join(timeout=0.5)
                    if process.is_alive():
                        process.kill()
                        process.join(timeout=0.5)
                if not process.is_alive():
                    process.join()
                    self._drain(active, job)
                    if active['cancel_at'] is not None:
                        job.update(status='cancelled', message='任务已停止。')
                    elif job['status'] not in TERMINAL:
                        job.update(status='failed', message='推理进程异常退出。',
                                   error='推理进程异常退出，可能内存不足；请减少输入数量或批大小。')
                    job['finished_at'] = time.time()
                    if job['status'] in ('failed', 'cancelled'):
                        for camera in job.get('cameras', []):
                            if camera['status'] != 'completed':
                                camera['status'] = 'stopped' if job['status'] == 'cancelled' else 'failed'
                    active['events'].close()
                    active['live_receiver'].close()
                    process.close()
                    self.active = None
                    self._save(job)
                elif time.monotonic() - active['last_save'] > 1:
                    self._save(job)
                    active['last_save'] = time.monotonic()

    def close(self):
        with self.lock:
            if self.active:
                self.stop(self.active['id'])
        deadline = time.monotonic() + self.cancel_grace + 2
        while self.active and time.monotonic() < deadline:
            time.sleep(0.05)
        self.closed = True
        self.wake.set()
        self.thread.join(timeout=2)
