"""Web API / 进程生命周期 / 流式离线推理，不下载模型权重。"""
import json
from decimal import Decimal
from html.parser import HTMLParser
from pathlib import Path
import queue
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from fastapi.testclient import TestClient

from web.app import create_app
from web.jobs import JobManager, TERMINAL, write_json
from web.pipeline import build_global_id_manager, build_stream_args, run_offline, run_online
from web.schemas import TrackingOptions
from web.worker import Cancelled, Reporter
from demo_stream import LatestFrameReader


class Detector:
    def detect_image_with_scores(self, image):
        return [[20, 20, 40, 80]], [.95]


class Encoder:
    def __call__(self, frame, boxes, camera_id=None):
        return np.array([[1., 0.] for box in boxes], dtype=np.float32)


class NumberInputParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.inputs = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == 'input' and attributes.get('type') == 'number':
            self.inputs.append(attributes)


class ParameterTests(unittest.TestCase):
    def test_frontend_number_defaults_pass_native_form_validation(self):
        parser = NumberInputParser()
        parser.feed((Path(__file__).parents[1] / 'web/static/index.html').read_text(encoding='utf-8'))
        self.assertTrue(parser.inputs)
        for field in parser.inputs:
            with self.subTest(field=field.get('id')):
                value = Decimal(field['value'])
                minimum = Decimal(field.get('min', '0'))
                maximum = Decimal(field['max']) if 'max' in field else None
                step = Decimal(field.get('step', '1'))
                self.assertGreaterEqual(value, minimum)
                if maximum is not None:
                    self.assertLessEqual(value, maximum)
                self.assertEqual((value - minimum) % step, 0)

    def test_web_defaults_match_high_accuracy_profile(self):
        options = TrackingOptions()
        self.assertEqual(options.confidence, .20)
        self.assertEqual(options.detector_imgsz, 1280)
        self.assertEqual(options.batch_size, 2)
        self.assertEqual(options.reid_threshold, .35)
        self.assertEqual(options.global_gallery_match, 'adaptive')
        self.assertEqual(options.global_prototype_count, 6)
        self.assertEqual(options.same_camera_reconnect_distance, 1.)
        self.assertEqual(options.offline_identity_mode, 'streaming')

    def test_all_online_parameters_reach_runtime_objects(self):
        options = TrackingOptions(
            tracker_max_age=47, global_prototype_count=9,
            global_borderline_confirm_frames=21,
            same_camera_reconnect_distance=1.25).model_dump()
        args = build_stream_args(['video.mp4'], options)
        manager = build_global_id_manager(args)
        self.assertEqual(args.stream_mode, 'queue')
        self.assertEqual(args.stream_queue_size, 2)
        self.assertEqual(args.tracker_max_age, 47)
        self.assertEqual(manager.prototype_count, 9)
        self.assertEqual(manager.borderline_confirm_frames, 21)
        self.assertEqual(manager.same_camera_reconnect_distance, 1.25)


def pipeline_worker(spec, events, cancel):
    reporter = Reporter(spec, events, cancel)
    try:
        run_offline(spec, None, None, reporter, model_factory=lambda: (Detector(), Encoder()))
        reporter.emit(force=True, status='completed', artifacts=reporter.artifacts())
    except Exception as exc:
        events.put({'status': 'failed', 'error': str(exc)})


def sleeping_worker(spec, events, cancel):
    events.put({'status': 'running', 'message': 'test'})
    cancel.wait(30)
    events.put({'status': 'cancelled'})


def stubborn_worker(spec, events, cancel):
    time.sleep(30)


def video(path, count=8, shape=(160, 160)):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 25, (shape[1], shape[0]))
    assert writer.isOpened()
    for i in range(count):
        frame = np.full((*shape, 3), 30 + i, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def await_terminal(client, job_id):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        job = client.get('/api/jobs/' + job_id).json()
        if job['status'] in TERMINAL:
            return job
        time.sleep(.05)
    raise AssertionError('job timeout')


class WebTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='mtmc-web-test-')
        self.root = Path(self.temp.name)
        self.client = TestClient(create_app(self.root / 'data', worker=pipeline_worker))
        self.client.__enter__()
        self.headers = {'X-MTMC-Client': 'web'}

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.temp.cleanup()

    def upload(self, path, filename=None):
        result = self.client.post('/api/uploads', params={'filename': filename or path.name},
                                  content=path.read_bytes(), headers=self.headers)
        self.assertEqual(result.status_code, 201, result.text)
        return result.json()['id']

    def test_health_and_embedded_frontend(self):
        self.assertEqual(self.client.get('/').status_code, 200)
        self.assertIn('实时在线', self.client.get('/').text)
        self.assertEqual(self.client.get('/static/app.js').status_code, 200)
        self.assertEqual(self.client.get('/api/health').json()['max_concurrent_jobs'], 1)
        self.assertEqual(self.client.get('/openapi.json').status_code, 200)
        html = self.client.get('/?mode=offline').text
        self.assertIn('class="module-sidebar"', html)
        self.assertLess(html.index('id="online-tab"'), html.index('id="offline-tab"'))
        self.assertLess(html.index('id="offline-tab"'), html.index('id="showroom-link"'))
        self.assertEqual(self.client.get('/ui/module-nav.css').status_code, 200)

    def test_origin_and_host_boundaries(self):
        url = '/api/jobs/online'
        self.assertEqual(self.client.post(url, json={}).status_code, 403)
        self.assertEqual(self.client.post(url, json={}, headers={**self.headers, 'Origin':'https://evil.test'}).status_code, 403)
        self.assertEqual(self.client.get('/api/jobs', headers={'Host':'evil.test'}).status_code, 400)

    def test_validates_streams_and_redacts_passwords(self):
        for streams in ([], [{'url':'file:///etc/passwd'}], [{'url':'rtsp://user:secret@host:bad/live'}]):
            response = self.client.post('/api/jobs/online', json={'streams': streams}, headers=self.headers)
            self.assertEqual(response.status_code, 422)
            self.assertNotIn('secret', response.text)
        response = self.client.post('/api/jobs/online', json={'streams':[{'url':'rtsp://host/live'}],
                                    'options':{'confidence':5}}, headers=self.headers)
        self.assertEqual(response.status_code, 422)

    def test_upload_sanitizes_filename_and_rejects_invalid(self):
        path = self.root / 'a.avi'
        video(path)
        uid = self.upload(path, '../../x.avi')
        metadata = json.loads((self.root / 'data/uploads' / (uid+'.json')).read_text())
        self.assertEqual(metadata['name'], 'x.avi')
        for filename, content, code in [('x.txt', b'test', 415), ('x.mp4', b'', 400)]:
            self.assertEqual(self.client.post('/api/uploads', params={'filename':filename}, content=content,
                                              headers=self.headers).status_code, code)

    def test_missing_and_duplicate_uploads(self):
        for ids, code in [([],422), (['../a'],400), (['a'*32],404), (['a'*32,'a'*32],422)]:
            self.assertEqual(self.client.post('/api/jobs/offline', json={'upload_ids':ids}, headers=self.headers).status_code, code)
        self.assertEqual(self.client.get('/api/jobs/no-such-job').status_code, 404)
        self.assertEqual(self.client.post('/api/jobs/no-such-job/stop', headers=self.headers).status_code, 404)

    def test_offline_worker_count_is_validated_before_job_creation(self):
        for workers in (0, 5, True, 1.5, '2'):
            response=self.client.post('/api/jobs/offline',json={
                'upload_ids':['a'*32], 'options':{'offline_workers':workers}},headers=self.headers)
            self.assertEqual(response.status_code,422,response.text)
        self.assertEqual(self.client.get('/api/health').json()['active_job_id'],None)

    def test_three_videos_infer_fuse_preview_download_and_replay(self):
        ids=[]
        for i in range(3):
            path=self.root/f'{i}.avi'
            video(path, count=8+i, shape=(160+i*16,160+i*32))
            ids.append(self.upload(path))
        response=self.client.post('/api/jobs/offline', json={'upload_ids':ids,'options':{'min_reid_frames':2}},headers=self.headers)
        self.assertEqual(response.status_code,202,response.text)
        job_id=response.json()['id']
        job=await_terminal(self.client,job_id)
        self.assertEqual(job['status'],'completed',job)
        self.assertEqual(job['processed_frames'],27)
        self.assertEqual(job['offline_workers'],1)
        self.assertEqual(job['identity_count'],0)  # 10-frame clips end before the 1 s GID delay.
        self.assertEqual(len(job['artifacts']),8)
        mapping=self.client.get(f'/api/jobs/{job_id}/artifacts/id_mapping.json').json()
        self.assertEqual(mapping['identity_mode'],'terminal_compatible')
        self.assertEqual(mapping['timing_mode'],'capture_wall_clock')
        self.assertEqual(mapping['stream_mode'],'queue')
        self.assertEqual(mapping['stream_queue_size'],2)
        self.assertEqual(mapping['frame_index_base'],1)
        self.assertEqual({t['global_id'] for t in mapping['tracks']},{None})
        self.assertEqual({t['camera_id'] for t in mapping['tracks']},{0,1,2})
        for i in range(3):
            frame=self.client.get(f'/api/jobs/{job_id}/cameras/{i}/frame.jpg')
            self.assertEqual(frame.status_code,200)
            self.assertTrue(frame.content.startswith(b'\xff\xd8'))
            cap=cv2.VideoCapture(str(self.root/f'data/jobs/{job_id}/artifacts/camera-{i+1}.avi'))
            self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),8+i)
            self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),160+i*32)
            cap.release()
        replay=self.client.get(f'/api/jobs/{job_id}/cameras/0/replay.mjpeg')
        self.assertEqual(replay.status_code,200)
        self.assertEqual(replay.content.count(b'Content-Type: image/jpeg'),8)
        self.assertEqual(self.client.get(f'/api/jobs/{job_id}/artifacts/job.json').status_code,404)
        self.assertEqual(self.client.get(f'/api/jobs/{job_id}/cameras/-1/frame.jpg').status_code,404)

    def test_invalid_video_fails_without_killing_server(self):
        path=self.root/'invalid.mp4';path.write_bytes(b'not a video')
        uid=self.upload(path)
        response=self.client.post('/api/jobs/offline',json={'upload_ids':[uid]},headers=self.headers)
        job=await_terminal(self.client,response.json()['id'])
        self.assertEqual(job['status'],'failed')
        self.assertIn('摄像头 1',job['error'])
        self.assertEqual(self.client.get('/api/health').status_code,200)

    def test_declared_frame_mismatch_is_not_published_as_success(self):
        path=self.root/'short.avi';video(path,count=5)
        directory=self.root/'mismatch';directory.mkdir();(directory/'previews').mkdir();(directory/'artifacts').mkdir()
        spec={'directory':str(directory),'sources':[str(path)],'names':['camera'],
              'options':TrackingOptions().model_dump()}
        with patch('web.pipeline.probe_video',return_value={'fps':25.,'frames':8,'width':160,'height':160}):
            with self.assertRaisesRegex(ValueError,'声明 8 帧'):
                run_offline(spec,Detector(),Encoder(),Reporter(spec,queue.Queue(),threading.Event()))
        self.assertEqual(list((directory/'artifacts').iterdir()),[])


class LifecycleTests(unittest.TestCase):
    def test_preview_uses_versioned_files_while_previous_image_is_open(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            (directory/'previews').mkdir()
            (directory/'artifacts').mkdir()
            spec = {'directory':root, 'names':['cam'], 'sources':['video.mp4'],
                    'options':TrackingOptions().model_dump()}
            reporter = Reporter(spec, queue.Queue(), threading.Event())
            frame = np.zeros((32, 32, 3), dtype=np.uint8)
            reporter.preview(0, frame, 1, 0, force=True)
            first = directory/'previews/0-1.jpg'
            with first.open('rb') as open_preview:
                reporter.preview(0, frame, 2, 0, force=True)
                self.assertTrue(open_preview.read(2).startswith(b'\xff\xd8'))
            self.assertTrue((directory/'previews/0-2.jpg').is_file())
            self.assertEqual(reporter.state['cameras'][0]['preview_version'], 2)

    def test_ffmpeg_timeouts_are_passed_during_open(self):
        reader=LatestFrameReader('rtsp://camera/live',0,stream_timeout_ms=1234)
        with patch('demo_stream.cv2.VideoCapture') as capture:
            reader._open_capture()
        args=capture.call_args.args
        self.assertEqual(args[1],cv2.CAP_FFMPEG)
        self.assertEqual(args[2],[cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,1234,cv2.CAP_PROP_READ_TIMEOUT_MSEC,1234])

    def test_reader_reopens_failed_network_capture(self):
        from unittest.mock import Mock
        reader=LatestFrameReader('http://camera/live',0,retry_interval=0)
        failed, connected=Mock(),Mock()
        failed.isOpened.return_value=False
        connected.isOpened.return_value=True
        connected.get.return_value=25
        def read():
            reader.stopped=True
            return True,np.zeros((16,16,3),dtype=np.uint8)
        connected.read.side_effect=read
        with patch.object(reader,'_open_capture',side_effect=[failed,connected]) as opened:
            reader._run()
        self.assertEqual(opened.call_count,2)
        failed.release.assert_called_once()
        connected.release.assert_called_once()

    def test_limit_upload_without_persisting_partial_data(self):
        with tempfile.TemporaryDirectory() as root:
            with TestClient(create_app(root,worker=sleeping_worker,max_upload_bytes=16)) as client:
                result=client.post('/api/uploads?filename=x.mp4',content=b'x'*32,headers={'X-MTMC-Client':'web'})
                self.assertEqual(result.status_code,413)
                self.assertEqual(list((Path(root)/'uploads').iterdir()),[])

    def test_concurrency_stop_persistence_and_password_not_on_disk(self):
        with tempfile.TemporaryDirectory() as root:
            with TestClient(create_app(root,worker=sleeping_worker)) as client:
                headers={'X-MTMC-Client':'web'}
                body={'streams':[{'url':'rtsp://user:secret@camera/live?token=private'} for _ in range(5)]}
                response=client.post('/api/jobs/online',json=body,headers=headers)
                self.assertEqual(response.status_code,202)
                job_id=response.json()['id']
                self.assertEqual(len(response.json()['cameras']),5)
                self.assertNotIn('secret',response.text)
                self.assertEqual(client.post('/api/jobs/online',json=body,headers=headers).status_code,409)
                self.assertEqual(client.post(f'/api/jobs/{job_id}/stop',headers=headers).status_code,200)
                self.assertEqual(await_terminal(client,job_id)['status'],'cancelled')
                self.assertNotIn('secret',(Path(root)/f'jobs/{job_id}/job.json').read_text())
            with TestClient(create_app(root,worker=sleeping_worker)) as client:
                self.assertEqual(client.get(f'/api/jobs/{job_id}').json()['status'],'cancelled')

    def test_force_cancel_stuck_worker(self):
        with tempfile.TemporaryDirectory() as root:
            manager=JobManager(root,worker=stubborn_worker,cancel_grace=.2)
            try:
                job=manager.create('online',['rtsp://host/live'],['cam'],TrackingOptions().model_dump())
                manager.stop(job['id'])
                deadline=time.monotonic()+5
                while manager.active and time.monotonic()<deadline:time.sleep(.05)
                self.assertIsNone(manager.active)
                self.assertEqual(manager.get(job['id'])['status'],'cancelled')
            finally:manager.close()

    def test_recovers_interrupted_manifest(self):
        with tempfile.TemporaryDirectory() as root:
            job_id='a'*32;path=Path(root)/'jobs'/job_id;path.mkdir(parents=True)
            write_json(path/'job.json',{'id':job_id,'status':'running'})
            manager=JobManager(root,worker=sleeping_worker)
            try:self.assertEqual(manager.get(job_id)['status'],'failed')
            finally:manager.close()

    def test_offline_checks_cancellation_before_decoding(self):
        with tempfile.TemporaryDirectory() as root:
            cancel=threading.Event();cancel.set()
            spec={'directory':root,'names':['cam'],'sources':['bad.mp4'],'options':TrackingOptions().model_dump()}
            with self.assertRaises(Cancelled):
                run_offline(spec,Detector(),Encoder(),Reporter(spec,queue.Queue(),cancel))

    def test_online_pipeline_emits_frames_and_cancels(self):
        captured=[]
        class Reader:
            def __init__(self,*args,**kwargs):self.index=0
            def start(self):return self
            def stop(self):pass
            def stats(self):return {'decoded_frames':self.index,'inference_skipped':0,'inference_queue':0}
            def read(self):
                self.index+=1
                frame=np.full((160,160,3),90,dtype=np.uint8)
                captured.append(frame)
                return True,frame,time.time(),self.index
        with tempfile.TemporaryDirectory() as root:
            (Path(root)/'previews').mkdir();(Path(root)/'artifacts').mkdir()
            cancel=threading.Event()
            spec={'directory':root,'names':['a','b'],'sources':['http://a/live','http://b/live'],'options':TrackingOptions().model_dump()}
            reporter=Reporter(spec,queue.Queue(),cancel)
            original=reporter.emit
            def emit(**values):
                original(**values)
                if reporter.state['processed_frames']>=24:cancel.set()
            reporter.emit=emit
            with (patch('web.pipeline.LatestFrameReader', Reader),
                  patch('web.pipeline.configure_ffmpeg_low_latency') as configure,
                  self.assertRaises(Cancelled)):
                run_online(spec,Detector(),Encoder(),reporter)
            configure.assert_called_once()
            configured_args = configure.call_args.args[0]
            self.assertEqual(configured_args.rtsp_transport, 'tcp')
            self.assertTrue(configured_args.ffmpeg_low_delay)
            self.assertEqual(reporter.state['processed_frames'],24)
            self.assertTrue(list((Path(root)/'previews').glob('1-*.jpg')))
            rows=[json.loads(line) for line in (Path(root)/'artifacts/tracks.jsonl').read_text().splitlines()]
            self.assertEqual({row['camera_id'] for row in rows},{0,1})
            # 发布线程持有原图时，推理/绘制不能修改同一数组。
            self.assertTrue(all(np.all(frame==90) for frame in captured))


if __name__=='__main__':unittest.main()
