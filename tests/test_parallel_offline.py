"""离线并行的重叠执行、隔离、有界调度、取消与逐帧结果一致性。"""
import json
from pathlib import Path
import queue
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import av
import numpy as np
from pydantic import ValidationError

from web.media import MP4Writer, VideoReader
from web.pipeline import run_offline
from web.schemas import TrackingOptions
from web.worker import Cancelled, Reporter


class Detector:
    def detect_image_with_scores(self, image):
        return [[20, 20, 40, 80]], [.95]


class Encoder:
    def __call__(self, frame, boxes, camera_id=None):
        return np.array([[1., 0.] for _ in boxes], dtype=np.float32)


class ParallelOfflineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='mtmc-parallel-')
        self.root = Path(self.temp.name)
        self.sources = []
        for camera_id, count in enumerate((78, 81, 9)):
            path = self.root / f'input-{camera_id}.mp4'
            shape = (160 + camera_id*16, 160 + camera_id*32, 3)
            writer = MP4Writer(path, 30, shape)
            for frame_id in range(count):
                # 可变帧率且逐帧可辨认，测试不只比较容器声明的帧数。
                writer.write(np.full(shape, 30+frame_id, dtype=np.uint8), self.stamp(frame_id))
            writer.close()
            self.sources.append(str(path))

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def stamp(frame_id):
        return frame_id/30 + (frame_id//10)*.01

    def setup_job(self, name, workers=2):
        directory = self.root / name
        directory.mkdir()
        (directory/'previews').mkdir()
        (directory/'artifacts').mkdir()
        spec = {'directory':str(directory), 'sources':self.sources, 'names':['a','b','c'],
                'options':TrackingOptions(offline_workers=workers, min_reid_frames=2).model_dump()}
        reporter = Reporter(spec, queue.Queue(maxsize=1024), threading.Event())
        return spec, reporter

    def test_concurrency_validation(self):
        self.assertEqual(TrackingOptions().offline_workers, 2)
        for invalid in (0, 5, -1, True, 1.5, '2'):
            with self.subTest(value=invalid), self.assertRaises(ValidationError):
                TrackingOptions(offline_workers=invalid)

    def test_parallel_inference_is_real_bounded_and_models_are_thread_owned(self):
        spec, reporter = self.setup_job('overlap')
        barrier = threading.Barrier(2, timeout=5)
        lock = threading.Lock()
        instances, active, peak = [], 0, 0

        class OwnedDetector(Detector):
            def __init__(self):
                self.owner, self.first = threading.get_ident(), True

            def detect_image_with_scores(self, image):
                nonlocal active, peak
                if threading.get_ident() != self.owner:
                    raise AssertionError('模型跨线程共享')
                with lock:
                    active += 1
                    peak = max(peak, active)
                try:
                    if self.first:
                        self.first = False
                        barrier.wait()
                    time.sleep(.002)
                    return super().detect_image_with_scores(image)
                finally:
                    with lock:
                        active -= 1

        def factory():
            instance = OwnedDetector()
            instances.append(instance)
            return instance, Encoder()

        run_offline(spec, None, None, reporter, model_factory=factory)
        self.assertEqual(peak, 2)
        self.assertEqual(len(instances), 2)  # 第三段视频复用空闲线程的模型。
        self.assertEqual(len({instance.owner for instance in instances}), 2)
        self.assertEqual(reporter.state['processed_frames'], 168)
        self.assertEqual(reporter.state['offline_workers'], 2)
        self.assertTrue(all(camera['status']=='completed' for camera in reporter.state['cameras']))
        directory = Path(spec['directory'])
        rows = [json.loads(line) for line in (directory/'artifacts/tracks.jsonl').read_text().splitlines()]
        self.assertEqual([(row['camera_id'],row['frame']) for row in rows],
                         sorted((row['camera_id'],row['frame']) for row in rows))
        for camera_id, count in enumerate((78,81,9)):
            with av.open(str(directory/f'artifacts/camera-{camera_id+1}.mp4')) as result:
                frames = list(result.decode(video=0))
            self.assertEqual(len(frames), count)
            for frame_id, frame in enumerate(frames):
                self.assertAlmostEqual(float(frame.pts*frame.time_base),self.stamp(frame_id),places=4)
            self.assertTrue((directory/f'artifacts/camera-{camera_id+1}.avi').is_file())

    def test_serial_and_parallel_have_same_mapping_logs_and_pixels(self):
        outputs = []
        for name, workers in [('serial',1),('parallel',2)]:
            spec, reporter = self.setup_job(name, workers)
            run_offline(spec, None, None, reporter, model_factory=lambda:(Detector(),Encoder()))
            outputs.append(Path(spec['directory'])/'artifacts')
        for filename in ('id_mapping.json','tracks.jsonl'):
            self.assertEqual((outputs[0]/filename).read_bytes(), (outputs[1]/filename).read_bytes())
        for camera_id in range(3):
            decoded = []
            for output in outputs:
                with av.open(str(output/f'camera-{camera_id+1}.mp4')) as result:
                    decoded.append([frame.to_ndarray(format='bgr24') for frame in result.decode(video=0)])
            self.assertEqual(len(decoded[0]),len(decoded[1]))
            for serial_frame, parallel_frame in zip(*decoded):
                np.testing.assert_array_equal(serial_frame,parallel_frame)

    def test_rendering_also_runs_in_parallel(self):
        spec, reporter = self.setup_job('render-overlap')
        barrier = threading.Barrier(2, timeout=5)

        class BarrierWriter(MP4Writer):
            def __init__(self, path, fps, shape):
                super().__init__(path,fps,shape)
                self.first = Path(path).name in ('camera-0-partial.mp4','camera-1-partial.mp4')

            def write(self, pixels, timestamp):
                if self.first:
                    self.first = False
                    barrier.wait()
                super().write(pixels,timestamp)

        with patch('web.pipeline.MP4Writer', BarrierWriter):
            run_offline(spec,None,None,reporter,model_factory=lambda:(Detector(),Encoder()))
        self.assertEqual(len(reporter.artifacts()),8)

    def test_cancel_stops_running_and_queued_videos(self):
        spec, reporter = self.setup_job('cancel')
        barrier = threading.Barrier(2,timeout=5)

        class CancellingDetector(Detector):
            def detect_image_with_scores(self, image):
                barrier.wait()
                reporter.cancel.set()
                return super().detect_image_with_scores(image)

        with self.assertRaises(Cancelled):
            run_offline(spec,None,None,reporter,model_factory=lambda:(CancellingDetector(),Encoder()))
        self.assertEqual(reporter.state['cameras'][2]['status'],'waiting')
        self.assertEqual(reporter.artifacts(),[])
        self.assertFalse(any(t.name.startswith('mtmc-offline') for t in threading.enumerate()))

    def test_failure_stops_siblings_and_keeps_original_error(self):
        spec, reporter = self.setup_job('failure')
        barrier = threading.Barrier(2,timeout=5)
        closed = []

        class TrackedReader(VideoReader):
            def release(self):
                closed.append(self)
                super().release()

        class BrokenEncoder(Encoder):
            def __init__(self):
                self.first = True

            def __call__(self, frame, boxes, camera_id=None):
                if self.first:
                    self.first = False
                    barrier.wait()
                if camera_id == 0:
                    raise RuntimeError('模拟单路推理失败')
                time.sleep(.01)
                return super().__call__(frame,boxes,camera_id)

        with patch('web.pipeline.VideoReader',TrackedReader), self.assertRaisesRegex(RuntimeError,'模拟单路推理失败'):
            run_offline(spec,None,None,reporter,model_factory=lambda:(Detector(),BrokenEncoder()))
        self.assertEqual(len(closed),5)  # 三次探测和两路追踪均释放解码器。
        self.assertEqual(reporter.state['cameras'][2]['status'],'waiting')
        self.assertEqual(reporter.artifacts(),[])
        self.assertFalse(any(t.name.startswith('mtmc-offline') for t in threading.enumerate()))

    def test_single_video_uses_only_one_model_even_if_four_workers_requested(self):
        spec, reporter = self.setup_job('single',4)
        spec['sources']=spec['sources'][:1]
        spec['names']=spec['names'][:1]
        reporter=Reporter(spec,queue.Queue(),threading.Event())
        calls=[]
        def factory():
            calls.append(1)
            return Detector(), Encoder()
        run_offline(spec,None,None,reporter,model_factory=factory)
        self.assertEqual(len(calls),1)
        self.assertEqual(reporter.state['offline_workers'],1)

    def test_render_failure_closes_encoders_and_stops_queued_video(self):
        spec, reporter = self.setup_job('render-failure')
        barrier = threading.Barrier(2,timeout=5)
        writers=[]

        class FailingWriter(MP4Writer):
            def __init__(self,path,fps,shape):
                super().__init__(path,fps,shape)
                self.first=True
                writers.append(self)

            def write(self,pixels,timestamp):
                if self.first:
                    self.first=False
                    barrier.wait()
                if Path(self.path).name=='camera-0-partial.mp4':
                    raise RuntimeError('模拟编码失败')
                time.sleep(.01)
                super().write(pixels,timestamp)

        with patch('web.pipeline.MP4Writer',FailingWriter), self.assertRaisesRegex(RuntimeError,'模拟编码失败'):
            run_offline(spec,None,None,reporter,model_factory=lambda:(Detector(),Encoder()))
        self.assertEqual(len(writers),2)
        self.assertTrue(all(writer.closed for writer in writers))
        self.assertEqual(list((Path(spec['directory'])/'artifacts').glob('*.mp4')),[])
        self.assertEqual(reporter.state['cameras'][2]['status'],'fusing')
        self.assertFalse(any(t.name.startswith('mtmc-offline') for t in threading.enumerate()))

    def test_parallel_mode_rejects_shared_model_instances(self):
        spec, reporter = self.setup_job('shared')
        with self.assertRaisesRegex(ValueError,'独立模型'):
            run_offline(spec,Detector(),Encoder(),reporter)


if __name__ == '__main__':
    unittest.main()
