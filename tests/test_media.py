"""保帧、时间轴、浏览器传输和过载缓冲回归。"""
import json
from pathlib import Path
import queue
import tempfile
import threading
import time
import unittest

import av
import numpy as np
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from demo_stream import LatestFrameReader
from web.app import create_app
from web.jobs import write_json
from web.media import FrameHub, LivePublisher, MP4Writer, VideoReader


class MediaTests(unittest.TestCase):
    def test_mp4_all_90_frames_preserve_order_and_30fps(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'result.mp4'
            writer=MP4Writer(path,30,(64,96,3))
            for i in range(90):writer.write(np.full((64,96,3),i*2,dtype=np.uint8),i/30)
            writer.close()
            with av.open(str(path)) as cap:
                frames=list(cap.decode(video=0))
            self.assertEqual(len(frames),90)
            for i,frame in enumerate(frames):
                self.assertAlmostEqual(float(frame.pts*frame.time_base),i/30,places=4)
                self.assertAlmostEqual(float(frame.to_ndarray(format='bgr24').mean()),i*2,delta=3)
            # faststart：moov 在 mdat 前，下载未结束即可建立播放时间轴。
            data=path.read_bytes()
            self.assertLess(data.index(b'moov'),data.index(b'mdat'))

    def test_variable_timestamps_and_odd_dimensions(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'vfr.mp4'
            timestamps=[0,.03,.09,.12,.20,.23]
            writer=MP4Writer(path,30,(65,97,3))
            for i,stamp in enumerate(timestamps):writer.write(np.full((65,97,3),i*30,dtype=np.uint8),stamp)
            writer.close()
            reader=VideoReader(path)
            result=[]
            try:
                while True:
                    ok,frame=reader.read()
                    if not ok:break
                    self.assertEqual(frame.shape,(66,98,3))
                    result.append(reader.timestamp)
            finally:reader.release()
            self.assertEqual(len(result),len(timestamps))
            np.testing.assert_allclose(result,timestamps,atol=1/90000)

    def test_queue_preserves_frames_without_overflow(self):
        reader=LatestFrameReader('http://test/live',0,stream_mode='queue',queue_size=30,
                                 queue_max_bytes=10000,queue_overflow='drop_oldest')
        for i in range(20):reader._store_frame(np.full((8,8,3),i,dtype=np.uint8))
        got=[reader.read() for _ in range(20)]
        self.assertEqual([item[3] for item in got],list(range(1,21)))
        self.assertEqual([int(item[1][0,0,0]) for item in got],list(range(20)))
        self.assertEqual(reader.stats()['inference_skipped'],0)
        self.assertEqual(reader.queue_bytes,0)

    def test_queue_overflow_is_bounded_counted_and_capture_stays_live(self):
        observed=[]
        reader=LatestFrameReader('http://test/live',0,stream_mode='queue',queue_size=30,
                                 queue_max_bytes=384,queue_overflow='drop_oldest',
                                 frame_observer=lambda frame,ts,seq:observed.append(seq))
        for i in range(100):reader._store_frame(np.zeros((8,8,3),dtype=np.uint8))
        self.assertEqual(observed,list(range(1,101)))
        self.assertEqual(reader.stats()['decoded_frames'],100)
        self.assertEqual(reader.stats()['inference_skipped'],98)
        self.assertEqual(reader.queue_bytes,384)
        self.assertEqual([reader.read()[3],reader.read()[3]],[99,100])

    def test_publisher_not_limited_by_slow_model_consumer(self):
        packets=queue.Queue(maxsize=60)
        publisher=LivePublisher(0,packets).start()
        reader=LatestFrameReader('http://test/live',0,stream_mode='queue',queue_size=2,
                                 queue_overflow='drop_oldest',frame_observer=publisher.submit)
        try:
            for i in range(30):reader._store_frame(np.zeros((64,96,3),dtype=np.uint8))
            received=[packets.get(timeout=3) for _ in range(30)]
            self.assertEqual([p['seq'] for p in received],list(range(1,31)))
            self.assertEqual(reader.stats()['inference_skipped'],28)
            self.assertEqual(publisher.dropped,0)
        finally:publisher.stop()

    def test_hub_eviction_preserves_order_and_memory_limit(self):
        hub=FrameHub(max_bytes=12,max_frames=10)
        for seq in range(1,8):hub.append({'camera_id':0,'seq':seq,'timestamp':seq*.04,'jpeg':b'xxx'})
        self.assertEqual(hub.bytes,12)
        self.assertEqual(hub.next(0,2)['seq'],4)
        self.assertEqual(hub.next(0,4)['seq'],5)
        self.assertIsNone(hub.next(1,0))

    def test_mp4_range_and_websocket_order_with_slow_consumer(self):
        with tempfile.TemporaryDirectory() as root:
            job_id='b'*32;directory=Path(root)/'jobs'/job_id;artifacts=directory/'artifacts';artifacts.mkdir(parents=True)
            path=artifacts/'camera-1.mp4';writer=MP4Writer(path,30,(64,96,3))
            for i in range(6):writer.write(np.zeros((64,96,3),dtype=np.uint8),i/30)
            writer.close()
            job={'id':job_id,'mode':'offline','status':'completed','created_at':0,'cameras':[{'id':0}],
                 'artifacts':[{'name':'camera-1.mp4','size':path.stat().st_size}]}
            write_json(directory/'job.json',job)
            app=create_app(root)
            with TestClient(app) as client:
                response=client.get(f'/api/jobs/{job_id}/cameras/0/video.mp4',headers={'Range':'bytes=0-127'})
                self.assertEqual(response.status_code,206)
                self.assertEqual(len(response.content),128)
                self.assertEqual(response.headers['content-type'],'video/mp4')
                exact_route=f'/api/jobs/{job_id}/cameras/0/frames'
                with self.assertRaises(WebSocketDisconnect):
                    with client.websocket_connect(exact_route,headers={'Origin':'http://evil.test'}):pass
                with client.websocket_connect(exact_route,headers={'Origin':'http://testserver'}) as ws:
                    for seq in range(1,7):
                        info=ws.receive_json()
                        self.assertEqual(info['seq'],seq)
                        self.assertEqual(info['skipped'],0)
                        self.assertAlmostEqual(info['timestamp'],(seq-1)/30,places=4)
                        self.assertTrue(ws.receive_bytes().startswith(b'\xff\xd8'))
                        time.sleep(.05)  # 慢于源帧率也必须逐帧交付，不补跳、不重复。
                        ws.send_text(str(seq))
                    self.assertEqual(ws.receive_json(),{'type':'ended','frames':6})
                # 中途离开释放解码器，其他请求仍能完成。
                with client.websocket_connect(exact_route,headers={'Origin':'http://testserver'}) as ws:
                    self.assertEqual(ws.receive_json()['seq'],1)
                    ws.receive_bytes()
                with client.websocket_connect(exact_route,headers={'Origin':'http://testserver'}) as ws:
                    ws.receive_json();ws.receive_bytes();ws.send_text('wrong-sequence')
                    with self.assertRaises(WebSocketDisconnect):ws.receive_json()
                app.state.manager.jobs[job_id]['mode']='online'
                with self.assertRaises(WebSocketDisconnect):
                    with client.websocket_connect(exact_route,headers={'Origin':'http://testserver'}):pass
                hub=FrameHub();app.state.manager.live_hubs[job_id]=hub
                for seq in range(1,5):hub.append({'camera_id':0,'seq':seq,'timestamp':seq*.02,'jpeg':b'jpeg'})
                route=f'/api/jobs/{job_id}/cameras/0/live'
                with self.assertRaises(WebSocketDisconnect):
                    with client.websocket_connect(route,headers={'Origin':'http://evil.test'}):pass
                with client.websocket_connect(route,headers={'Origin':'http://testserver'}) as ws:
                    seen=[]
                    for i in range(4):
                        info=ws.receive_json();seen.append(info['seq'])
                        self.assertEqual(info['skipped'],0)
                        self.assertEqual(ws.receive_bytes(),b'jpeg')
                        time.sleep(.03)
                        ws.send_text(str(info['seq']))
                    self.assertEqual(seen,[1,2,3,4])
                self.assertEqual(client.get('/api/health').status_code,200)


if __name__=='__main__':unittest.main()
