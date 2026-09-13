"""合并回归：线程独立模型、检测尺寸、跨平台输出与脱敏错误报告。"""
from contextlib import ExitStack
import os
from pathlib import Path
import queue
import tempfile
import threading
import unittest
from unittest.mock import ANY, Mock, patch

from web.worker import Cancelled, load_models, run_worker


class WorkerMergeTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        (self.root / 'artifacts').mkdir()
        self.spec = {
            'directory': str(self.root), 'mode': 'offline',
            'sources': ['rtsp://user:private-password@camera/live'], 'names': ['camera'],
            'options': {'confidence': .35, 'batch_size': 4},
            'live_connection': object(),
        }
        self.events = queue.Queue()
        self.cancel = threading.Event()
        self.stack.enter_context(patch.dict(os.environ, {'MTMC_CPU_THREADS': '1'}))
        self.stack.enter_context(patch('web.worker.os.chdir'))
        self.stack.enter_context(patch('torch.set_num_threads'))
        self.silence = self.stack.enter_context(patch('web.worker.silence_worker_output'))
        self.loader = self.stack.enter_context(patch('web.worker.load_models'))
        self.loader.side_effect = lambda *args: (object(), object())
        self.offline = self.stack.enter_context(patch('web.pipeline.run_offline'))
        self.online = self.stack.enter_context(patch('web.pipeline.run_online'))
        self.transport = self.stack.enter_context(patch('web.media.LiveTransport')).return_value

    def run_job(self):
        run_worker(self.spec, self.events, self.cancel)
        events = []
        while not self.events.empty():
            events.append(self.events.get_nowait())
        return events[-1]

    def test_model_factory_keeps_default_and_custom_detector_size(self):
        # 调用保留的真实函数引用，只替换昂贵的检测器和特征模型构造。
        with patch('web.worker.Path.is_file', return_value=True), \
                patch('torch_detector.build_person_detector') as detector, \
                patch('reid_backends.create_reid_encoder') as encoder:
            detector.side_effect = lambda *args, **kwargs: object()
            encoder.side_effect = lambda *args, **kwargs: object()
            for options, expected in (({}, 640), ({'detector_imgsz': 1280}, 1280)):
                with self.subTest(imgsz=expected):
                    self.spec['options'] = {'confidence': .35, 'batch_size': 4, **options}
                    check = Mock()
                    first = load_models(self.spec, check)
                    second = load_models(self.spec, check)
                    self.assertIsNot(first[0], second[0])
                    self.assertIsNot(first[1], second[1])
                    self.assertEqual(detector.call_args.kwargs['imgsz'], expected)
                    self.assertEqual(detector.call_args.kwargs['score_threshold'], .35)
                    self.assertEqual(encoder.call_args.kwargs['batch_size'], 4)
                    self.assertEqual(check.call_count, 6)

    def test_offline_does_not_eagerly_load_or_share_models(self):
        def process(spec, detector, encoder, reporter, *, model_factory):
            self.assertIsNone(detector)
            self.assertIsNone(encoder)
            self.loader.assert_not_called()
            first, second = model_factory(), model_factory()
            self.assertIsNot(first[0], second[0])
            self.assertIsNot(first[1], second[1])

        self.offline.side_effect = process
        self.assertEqual(self.run_job()['status'], 'completed')
        self.assertEqual(self.loader.call_count, 2)
        self.online.assert_not_called()
        self.silence.assert_called_once_with()
        self.transport.close.assert_called_once_with()

    def test_online_loads_models_once_before_inference(self):
        self.spec['mode'] = 'online'
        detector, encoder = object(), object()
        self.loader.side_effect = None
        self.loader.return_value = detector, encoder
        self.assertEqual(self.run_job()['status'], 'completed')
        self.loader.assert_called_once_with(self.spec, ANY)
        self.online.assert_called_once_with(self.spec, detector, encoder, ANY)
        self.offline.assert_not_called()
        self.transport.close.assert_called_once_with()

    def test_memory_error_keeps_hint_and_redacts_both_error_outputs(self):
        for error_type in (MemoryError, RuntimeError):
            with self.subTest(error_type=error_type):
                self.offline.side_effect = error_type(
                    'CUDA out of memory ' + self.spec['sources'][0]
                    + '\nredirect https://other:redirect-secret@host/stream')
                result = self.run_job()
                self.assertEqual(result['status'], 'failed')
                self.assertIn('请降低离线并行路数', result['error'])
                diagnostic = (self.root / 'error.log').read_text(encoding='utf-8')
                self.assertIn('Traceback', diagnostic)
                for output in (result['error'], diagnostic):
                    self.assertIn('out of memory', output)
                    self.assertNotIn('private-password', output)
                    self.assertNotIn('redirect-secret', output)
                    self.assertNotIn('rtsp://', output)
                    self.assertNotIn('https://', output)

    def test_log_write_failure_does_not_hide_original_memory_error(self):
        self.offline.side_effect = MemoryError('allocation failed')
        with patch('web.worker.Path.write_text', side_effect=OSError('disk full')):
            result = self.run_job()
        self.assertEqual(result['status'], 'failed')
        self.assertIn('MemoryError: allocation failed', result['error'])
        self.assertIn('请降低离线并行路数', result['error'])
        self.assertNotIn('disk full', result['error'])
        self.transport.close.assert_called_once_with()

    def test_cancel_still_closes_transport_without_error_log(self):
        self.offline.side_effect = Cancelled()
        self.assertEqual(self.run_job()['status'], 'cancelled')
        self.assertFalse((self.root / 'error.log').exists())
        self.transport.close.assert_called_once_with()

    def test_output_setup_failure_is_reported_and_logged(self):
        self.silence.side_effect = OSError(1, 'stale console')
        result = self.run_job()
        self.assertEqual(result['status'], 'failed')
        self.assertIn('stale console', result['error'])
        self.assertIn('stale console', (self.root / 'error.log').read_text(encoding='utf-8'))
        self.loader.assert_not_called()
        self.transport.close.assert_called_once_with()


if __name__ == '__main__':
    unittest.main()
