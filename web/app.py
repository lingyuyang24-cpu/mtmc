"""启动：python serve.py。前端与 API 同源，仅供受信任的本地/内网使用。"""
import asyncio
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
import shutil
import time
from urllib.parse import urlsplit
import uuid

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from starlette.middleware.trustedhost import TrustedHostMiddleware

from web.jobs import JobManager, ROOT, TERMINAL, valid_id, write_json
from web.schemas import OfflineRequest, OnlineRequest

STATIC = Path(__file__).parent / 'static'
EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v'}


def create_app(data_root=None, worker=None, max_upload_bytes=None):
    root = Path(data_root or os.environ.get('MTMC_DATA_DIR', ROOT / 'web_data')).resolve()
    upload_limit = max_upload_bytes or int(os.environ.get('MTMC_MAX_UPLOAD_MB', '2048')) * 1024 * 1024

    @asynccontextmanager
    async def lifespan(app):
        root.mkdir(parents=True, exist_ok=True)
        lock_file = (root / '.server.lock').open('a+b')
        try:
            if os.name == 'nt':
                import msvcrt
                lock_file.seek(0)
                lock_file.write(b'0')
                lock_file.flush()
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            lock_file.close()
            raise RuntimeError('该数据目录已有服务运行。请使用单 worker 或不同 MTMC_DATA_DIR。') from None
        manager = JobManager(root, worker=worker)
        app.state.manager = manager
        try:
            yield
        finally:
            await asyncio.to_thread(manager.close)
            lock_file.close()

    app = FastAPI(title='MTMC 多摄像头追踪 API', version='1.0.0', lifespan=lifespan,
                  description='同源本地工作台。写入请求必须附带 X-MTMC-Client: web。上传接口接收原始视频二进制。')
    hosts = os.environ.get('MTMC_ALLOWED_HOSTS', '127.0.0.1,localhost,::1,testserver').split(',')
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=[host.strip() for host in hosts])

    @app.middleware('http')
    async def local_boundary(request, call_next):
        if request.method not in ('GET', 'HEAD', 'OPTIONS'):
            origin = request.headers.get('origin')
            if (request.headers.get('x-mtmc-client') != 'web' or
                    (origin and urlsplit(origin).netloc != request.headers.get('host'))):
                return JSONResponse({'detail': '仅接受同源请求；请使用 X-MTMC-Client: web 请求头。'}, status_code=403)
        if request.url.path.startswith('/api/') and request.url.path != '/api/uploads':
            length = request.headers.get('content-length', '0')
            if length.isdigit() and int(length) > 1024 * 1024:
                return JSONResponse({'detail': '请求参数过大。'}, status_code=413)
        response = await call_next(request)
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Referrer-Policy'] = 'no-referrer'
        response.headers['X-Frame-Options'] = 'DENY'
        if request.url.path.startswith('/api/'):
            response.headers['Cache-Control'] = 'no-store'
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        # Pydantic 默认错误会回显输入，不能回显流地址密码。
        errors = [{'loc': e['loc'], 'msg': e['msg'], 'type': e['type']} for e in exc.errors()]
        return JSONResponse({'detail': errors}, status_code=422)

    def get_job(job_id):
        try:
            return app.state.manager.get(job_id)
        except KeyError:
            raise HTTPException(404, '任务不存在。') from None

    @app.get('/api/health')
    def health():
        detector = Path(os.environ.get('MTMC_DETECTOR', ROOT / 'yolo11l.pt'))
        reid = Path(os.environ.get('MTMC_TRANSREID_WEIGHTS', ROOT / 'external/transreid/weights/vit_transreid_msmt.pth'))
        repo = Path(os.environ.get('MTMC_TRANSREID_REPO', ROOT / 'external/transreid/repo'))
        active = app.state.manager.active
        return {'status': 'ok', 'model_files_ready': detector.is_file() and reid.is_file() and (repo / 'model').is_dir(),
                'showroom_url': os.environ.get('MTMC_SHOWROOM_URL'),
                'max_upload_bytes': upload_limit, 'max_concurrent_jobs': 1,
                'active_job_id': active['id'] if active else None,
                'models': {'detector': detector.name, 'reid': 'TransReID MSMT17'},
                'note': '权重文件检查不代表模型已加载；模型会在任务启动时加载。'}

    @app.post('/api/uploads', status_code=201, openapi_extra={
        'requestBody': {'required': True, 'content': {'application/octet-stream': {'schema': {'type': 'string', 'format': 'binary'}}}}})
    async def upload(request: Request, filename: str):
        name = Path(filename.replace('\\', '/')).name[:200]
        suffix = Path(name).suffix.lower()
        if suffix not in EXTENSIONS:
            raise HTTPException(415, '不支持此文件扩展名，请上传 MP4 / AVI / MOV / MKV / WEBM / M4V。')
        length = request.headers.get('content-length', '')
        if length.isdigit() and int(length) > upload_limit:
            raise HTTPException(413, '单个文件超过上传大小限制。')
        upload_id = uuid.uuid4().hex
        path = root / 'uploads' / (upload_id + suffix)
        temporary = path.with_suffix('.part')
        size = 0
        try:
            with temporary.open('xb') as output:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > upload_limit:
                        raise HTTPException(413, '单个文件超过上传大小限制。')
                    if shutil.disk_usage(root).free < len(chunk) + 64 * 1024 * 1024:
                        raise HTTPException(507, '磁盘剩余空间不足。')
                    output.write(chunk)
            if size == 0:
                raise HTTPException(400, '上传文件为空。')
            temporary.replace(path)
            data = {'id': upload_id, 'name': name, 'suffix': suffix, 'size': size, 'created_at': time.time()}
            write_json(root / 'uploads' / f'{upload_id}.json', data)
            return data
        except BaseException:
            temporary.unlink(missing_ok=True)
            path.unlink(missing_ok=True)
            raise

    def create_job(mode, sources, names, options):
        try:
            return app.state.manager.create(mode, sources, names, options.model_dump())
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from None

    @app.post('/api/jobs/online', status_code=202)
    def online(body: OnlineRequest):
        return create_job('online', [s.url for s in body.streams],
                          [s.name.strip() or f'摄像头 {i + 1:02d}' for i, s in enumerate(body.streams)], body.options)

    @app.post('/api/jobs/offline', status_code=202)
    def offline(body: OfflineRequest):
        sources, names = [], []
        for upload_id in body.upload_ids:
            if not valid_id(upload_id):
                raise HTTPException(400, '上传文件 ID 无效。')
            try:
                metadata = json.loads((root / 'uploads' / f'{upload_id}.json').read_text(encoding='utf-8'))
                if metadata['suffix'] not in EXTENSIONS:
                    raise ValueError()
                path = root / 'uploads' / (upload_id + metadata['suffix'])
                if not path.is_file():
                    raise ValueError()
            except (OSError, KeyError, ValueError):
                raise HTTPException(404, '上传文件不存在，请重新上传。') from None
            sources.append(str(path))
            names.append(metadata['name'])
        return create_job('offline', sources, names, body.options)

    @app.get('/api/jobs')
    def list_jobs():
        return {'jobs': app.state.manager.list()}

    @app.get('/api/jobs/{job_id}')
    def job(job_id: str):
        return get_job(job_id)

    @app.get('/api/jobs/{job_id}/events')
    def tracking_events(job_id: str, cursor: str = '0:0', limit: int = 250):
        from tracking_contracts.events import read_page
        current = get_job(job_id)
        if current['mode'] != 'online':
            raise HTTPException(409, '此事件接口当前仅支持在线任务。')
        try:
            page = read_page(app.state.manager.directory(job_id)/'events', cursor, limit)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None
        except OSError:
            raise HTTPException(503, '追踪事件日志暂不可读，请检查存储状态。') from None
        page['job_status'] = current['status']
        page['publisher_error'] = current.get('event_log',{}).get('error') or (
            'UnclosedEventLog' if page.get('available') and current['status'] in TERMINAL and not page.get('closed') else None)
        return page

    @app.post('/api/jobs/{job_id}/stop')
    def stop(job_id: str):
        get_job(job_id)
        return app.state.manager.stop(job_id)

    @app.get('/api/jobs/{job_id}/cameras/{camera_id}/frame.jpg')
    def frame(job_id: str, camera_id: int):
        current = get_job(job_id)
        if not 0 <= camera_id < len(current['cameras']):
            raise HTTPException(404, '摄像头不存在。')
        path = app.state.manager.directory(job_id) / 'previews' / f'{camera_id}.jpg'
        if not path.is_file():
            raise HTTPException(404, '尚未收到画面。')
        return FileResponse(path, media_type='image/jpeg')

    def artifact_path(job_id, name):
        current = get_job(job_id)
        if current['status'] not in TERMINAL or name not in {a['name'] for a in current['artifacts']}:
            raise HTTPException(404, '结果文件不存在或尚未完成。')
        if Path(name).name != name:
            raise HTTPException(404, '结果文件不存在。')
        path = app.state.manager.directory(job_id) / 'artifacts' / name
        if not path.is_file():
            raise HTTPException(404, '结果文件不存在。')
        return path

    @app.get('/api/jobs/{job_id}/artifacts/{name}')
    def artifact(job_id: str, name: str):
        return FileResponse(artifact_path(job_id, name), filename=name)

    @app.get('/api/jobs/{job_id}/cameras/{camera_id}/video.mp4')
    def video(job_id: str, camera_id: int):
        # FileResponse 支持 Range / 206，浏览器自行缓存与拖动，不逐帧发 HTTP 请求。
        return FileResponse(artifact_path(job_id, f'camera-{camera_id + 1}.mp4'), media_type='video/mp4')

    @app.websocket('/api/jobs/{job_id}/cameras/{camera_id}/live')
    async def live(websocket: WebSocket, job_id: str, camera_id: int):
        origin = websocket.headers.get('origin')
        if not origin or urlsplit(origin).netloc != websocket.headers.get('host'):
            await websocket.close(code=1008)
            return

        try:
            current = get_job(job_id)
        except HTTPException:
            await websocket.close(code=1008)
            return
        if current['mode'] != 'online' or not 0 <= camera_id < len(current['cameras']):
            await websocket.close(code=1008)
            return
        await websocket.accept()
        seq, anchor_time, anchor_timestamp = 0, None, None
        try:
            while True:
                current = get_job(job_id)
                hub = app.state.manager.live_hubs.get(job_id)
                packet = hub.next(camera_id, seq) if hub else None
                if packet is None:
                    if current['status'] in TERMINAL:
                        await websocket.close(code=1000)
                        return
                    await websocket.send_json({'type': 'waiting'})
                    if await asyncio.wait_for(websocket.receive_text(), timeout=10) != 'waiting':
                        await websocket.close(code=1008)
                        return
                    await asyncio.sleep(.1)
                    continue
                now = time.monotonic()
                if anchor_time is None or now-anchor_time-(packet['timestamp']-anchor_timestamp) > 1:
                    anchor_time, anchor_timestamp = now+.10, packet['timestamp']
                due = anchor_time + packet['timestamp']-anchor_timestamp
                await asyncio.sleep(max(0, min(.5, due-time.monotonic())))
                await websocket.send_json({'seq': packet['seq'], 'timestamp': packet['timestamp'],
                                           'skipped': max(0, packet['seq']-seq-1) if seq else 0})
                await websocket.send_bytes(packet['jpeg'])
                # 每个浏览器最多一帧待解码；慢客户端不拖慢其他观众或拉流线程。
                acknowledgement = await asyncio.wait_for(websocket.receive_text(), timeout=10)
                if acknowledgement != str(packet['seq']):
                    await websocket.close(code=1008)
                    return
                seq = packet['seq']
        except (WebSocketDisconnect, asyncio.TimeoutError, RuntimeError):
            try:
                await websocket.close(code=1001)
            except RuntimeError:
                pass
            return

    @app.websocket('/api/jobs/{job_id}/cameras/{camera_id}/frames')
    async def exact_frames(websocket: WebSocket, job_id: str, camera_id: int):
        """离线严格顺序回放：只在当前帧完成绘制确认后才解码下一帧。"""
        origin = websocket.headers.get('origin')
        if not origin or urlsplit(origin).netloc != websocket.headers.get('host'):
            await websocket.close(code=1008)
            return
        try:
            current = get_job(job_id)
            if current['mode'] != 'offline' or not 0 <= camera_id < len(current['cameras']):
                raise HTTPException(404, '离线摄像头不存在。')
            path = artifact_path(job_id, f'camera-{camera_id + 1}.mp4')
        except HTTPException:
            await websocket.close(code=1008)
            return
        await websocket.accept()
        from web.media import VideoReader
        reader = None
        try:
            reader = await run_in_threadpool(VideoReader, path)
            previous_time, previous_stamp = time.monotonic() + .1, 0.

            def decode():
                import cv2
                ok, pixels = reader.read()
                if not ok:
                    return None
                height, width = pixels.shape[:2]
                scale = min(1., 960/width, 720/height)
                if scale < 1:
                    pixels = cv2.resize(pixels, (max(1, round(width*scale)), max(1, round(height*scale))))
                ok, encoded = cv2.imencode('.jpg', pixels, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not ok:
                    raise RuntimeError('回放帧编码失败')
                return encoded.tobytes()

            while True:
                encoded = await run_in_threadpool(decode)
                if encoded is None:
                    await websocket.send_json({'type':'ended', 'frames':reader.count})
                    await websocket.close(code=1000)
                    return
                # 慢浏览器的积压不会被丢弃。隐藏页面会暂停 ACK，自然形成背压。
                due = previous_time + max(0, reader.timestamp-previous_stamp)
                await asyncio.sleep(max(0, due-time.monotonic()))
                previous_time, previous_stamp = time.monotonic(), reader.timestamp
                await websocket.send_json({'seq':reader.count,'timestamp':reader.timestamp,'skipped':0})
                await websocket.send_bytes(encoded)
                ack = await websocket.receive_text()
                if ack != str(reader.count):
                    await websocket.close(code=1008)
                    return
        except (WebSocketDisconnect, RuntimeError, OSError):
            try:
                await websocket.close(code=1011)
            except RuntimeError:
                pass
        finally:
            if reader is not None:
                await run_in_threadpool(reader.release)

    @app.get('/api/jobs/{job_id}/cameras/{camera_id}/replay.mjpeg')
    async def replay(job_id: str, camera_id: int, request: Request):
        path = artifact_path(job_id, f'camera-{camera_id + 1}.avi')

        async def frames():
            import cv2
            cap = cv2.VideoCapture(str(path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            interval = 1 / (fps if 0 < fps <= 240 else 25)

            def next_jpeg():
                ok, pixels = cap.read()
                if not ok:
                    return None
                height, width = pixels.shape[:2]
                scale = min(1., 960 / width, 720 / height)
                if scale < 1:
                    pixels = cv2.resize(pixels, (max(1, round(width * scale)), max(1, round(height * scale))))
                ok, encoded = cv2.imencode('.jpg', pixels)
                return encoded.tobytes() if ok else b''

            try:
                while not await request.is_disconnected():
                    started = time.monotonic()
                    # 解码/压缩在工作线程执行；取消时等待线程返回再释放 capture。
                    encoded = await run_in_threadpool(next_jpeg)
                    if encoded is None:
                        break
                    if encoded:
                        yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + encoded + b'\r\n'
                    await asyncio.sleep(max(0, interval - (time.monotonic() - started)))
                yield b'--frame--\r\n'
            finally:
                cap.release()
        return StreamingResponse(frames(), media_type='multipart/x-mixed-replace; boundary=frame')

    app.mount('/static', StaticFiles(directory=STATIC), name='static')
    app.mount('/ui', StaticFiles(directory=ROOT / 'ui'), name='shared-ui')

    @app.get('/', include_in_schema=False)
    def index():
        return FileResponse(STATIC / 'index.html')

    return app


app = create_app()
