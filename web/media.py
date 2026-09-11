"""保留帧和时间戳的离线媒体读写，以及与推理解耦的在线播放缓冲。"""
from collections import deque
from fractions import Fraction
import queue
import threading
import time


class VideoReader:
    def __init__(self, path):
        import av
        self.container = av.open(str(path), options={'err_detect': 'explode'})
        try:
            self.stream = self.container.streams.video[0]
            self.stream.thread_type = 'AUTO'
            self.decoder = self.container.decode(self.stream)
            self.rate = self.stream.average_rate or Fraction(25)
            self.first_time = None
            self.timestamp = 0.0
            self.count = 0
        except BaseException:
            self.container.close()
            raise

    def read(self):
        try:
            frame = next(self.decoder)
        except StopIteration:
            return False, None
        # 不把解码异常伪装成正常 EOF；调用方会报告失败而非发布截断结果。
        source_time = frame.pts * frame.time_base if frame.pts is not None else Fraction(self.count, 1) / self.rate
        if self.first_time is None:
            self.first_time = source_time
        self.timestamp = float(source_time - self.first_time)
        self.count += 1
        return True, frame.to_ndarray(format='bgr24')

    def release(self):
        self.container.close()


class MP4Writer:
    """不设 fps 滤镜、不抽帧；逐帧编码并在结束后核对写出帧数。"""
    def __init__(self, path, fps, shape):
        import av
        self.path = path
        self.container = av.open(str(path), 'w', format='mp4', options={'movflags': '+faststart'})
        self.count = 0
        self.closed = False
        self.last_pts = -1
        self.time_base = Fraction(1, 90000)
        try:
            self.stream = self.container.add_stream('libx264', rate=Fraction(str(fps)).limit_denominator(1001000))
            height, width = shape[:2]
            self.stream.width, self.stream.height = width + width % 2, height + height % 2
            self.stream.pix_fmt = 'yuv420p'
            self.stream.time_base = self.time_base
            self.stream.codec_context.time_base = self.time_base
            self.stream.codec_context.thread_count = 2
            self.stream.options = {'preset': 'veryfast', 'crf': '20', 'bf': '0'}
        except BaseException:
            self.container.close()
            raise

    def write(self, pixels, timestamp):
        import av
        import cv2
        height, width = pixels.shape[:2]
        if height != self.stream.height or width != self.stream.width:
            pixels = cv2.copyMakeBorder(pixels, 0, self.stream.height-height, 0, self.stream.width-width,
                                       cv2.BORDER_CONSTANT, value=(0, 0, 0))
        frame = av.VideoFrame.from_ndarray(pixels, format='bgr24')
        # 有相同 PTS 的输入帧也保留；最多前移一个 1/90000 秒时钟单位。
        frame.pts = max(self.last_pts + 1, round(timestamp / self.time_base))
        frame.time_base = self.time_base
        self.last_pts = frame.pts
        for packet in self.stream.encode(frame):
            self.container.mux(packet)
        self.count += 1

    def close(self, verify=True):
        if self.closed:
            return
        self.closed = True
        try:
            for packet in self.stream.encode():
                self.container.mux(packet)
        finally:
            self.container.close()
        if verify:
            import av
            with av.open(str(self.path)) as result:
                if result.streams.video[0].frames != self.count:
                    raise RuntimeError('MP4 写出帧数与处理帧数不符，拒绝发布不完整结果。')


class FrameHub:
    """API 进程中的共享 JPEG 环形缓冲，限制总字节数而非每个客户复制。"""
    def __init__(self, max_bytes=32*1024*1024, max_frames=512):
        self.frames = deque()
        self.bytes = 0
        self.max_bytes, self.max_frames = max_bytes, max_frames
        self.lock = threading.Lock()

    def append(self, packet):
        size = len(packet['jpeg'])
        if size > self.max_bytes:
            return
        with self.lock:
            self.frames.append(packet)
            self.bytes += size
            while self.bytes > self.max_bytes or len(self.frames) > self.max_frames:
                self.bytes -= len(self.frames.popleft()['jpeg'])

    def next(self, camera_id, after):
        with self.lock:
            matching = [p for p in self.frames if p['camera_id'] == camera_id and p['seq'] > after]
            if not matching:
                return None
            if after == 0:
                # 新观众从接近实时的位置加入，而不是播放所有历史缓存。
                edge = matching[-1]['timestamp'] - .15
                return next((p for p in matching if p['timestamp'] >= edge), matching[-1])
            return matching[0]


class LivePublisher:
    """只发布连续原画，不把旧跟踪框错误地画到当前帧上。"""
    def __init__(self, camera_id, output, max_bytes=32*1024*1024, max_frames=60):
        self.camera_id, self.output = camera_id, output
        self.buffer = deque()
        self.bytes = 0
        self.max_bytes, self.max_frames = max_bytes, max_frames
        self.condition = threading.Condition()
        self.stopped = False
        self.dropped = 0
        self.published = 0
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def submit(self, frame, timestamp, seq):
        if self.output is None:
            return
        with self.condition:
            if self.stopped:
                return
            self.buffer.append((frame, timestamp, seq))
            self.bytes += frame.nbytes
            while self.bytes > self.max_bytes or len(self.buffer) > self.max_frames:
                self.bytes -= self.buffer.popleft()[0].nbytes
                self.dropped += 1
            self.condition.notify()

    def _run(self):
        import cv2
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.buffer or self.stopped)
                if self.stopped:
                    return
                frame, timestamp, seq = self.buffer.popleft()
                self.bytes -= frame.nbytes
            height, width = frame.shape[:2]
            scale = min(1., 960/width, 720/height)
            if scale < 1:
                frame = cv2.resize(frame, (max(1, round(width*scale)), max(1, round(height*scale))))
            ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok:
                self.dropped += 1
                continue
            try:
                self.output.put_nowait({'camera_id': self.camera_id, 'timestamp': timestamp,
                                        'seq': seq, 'jpeg': encoded.tobytes()})
                self.published += 1
            except queue.Full:
                self.dropped += 1

    def stop(self):
        with self.condition:
            self.stopped = True
            self.buffer.clear()
            self.bytes = 0
            self.condition.notify_all()
        if self.thread.is_alive():
            self.thread.join(timeout=1)


class LiveTransport:
    """有界线程队列 → 单向管道；父进程没有写端，子进程崩溃也能收到 EOF。"""
    def __init__(self, connection):
        self.connection = connection
        self.queue = queue.Queue(maxsize=64)
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def put_nowait(self, packet):
        if self.stopped.is_set():
            raise queue.Full
        self.queue.put_nowait(packet)

    def _run(self):
        try:
            while not self.stopped.is_set():
                try:
                    packet = self.queue.get(timeout=.1)
                except queue.Empty:
                    continue
                self.connection.send(packet)
        except (OSError, EOFError):
            self.stopped.set()

    def close(self):
        self.stopped.set()
        self.connection.close()
