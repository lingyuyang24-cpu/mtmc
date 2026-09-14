"""Standalone showroom API/UI. Does not import or load tracking/model internals."""

import asyncio
from contextlib import asynccontextmanager
from io import BytesIO
import os
from pathlib import Path
import sqlite3
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
import uuid
import warnings

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from starlette.middleware.trustedhost import TrustedHostMiddleware

from modules.showroom.client import TrackingClient
from modules.showroom.repository import Repository
from modules.showroom.reports import bounds, generate
from modules.showroom.schemas import (
    StoreCreate,
    LayoutCreate,
    BindingCreate,
    BindingPause,
    RoleSet,
)
from modules.showroom.service import Service

PREFIX = "/api/showroom/v1"
STATIC = Path(__file__).parent / "static"
ROOT = Path(__file__).resolve().parents[2]


def create_app(data_root=None, client=None, background=True):
    root = Path(
        data_root
        or os.environ.get("MTMC_SHOWROOM_DATA_DIR", ROOT / "web_data" / "showroom")
    ).resolve()
    upstream = client or TrackingClient(
        os.environ.get("MTMC_TRACKING_URL", "http://127.0.0.1:8765")
    )

    @asynccontextmanager
    async def lifespan(app):
        root.mkdir(parents=True, exist_ok=True)
        handle = (root / ".server.lock").open("a+b")
        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                handle.write(b"0")
                handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            handle.close()
            raise RuntimeError("展车数据目录已有服务运行，请使用单 worker。") from None
        repo = None
        service = None
        try:
            repo = Repository(root)
            service = Service(repo, upstream)
            app.state.repo, app.state.service = repo, service
            if background:
                service.start()
            yield
        finally:
            if service:
                await asyncio.to_thread(service.close)
            if repo:
                repo.close()
            handle.close()

    app = FastAPI(title="MTMC 独立展车分析", version="1.0.0", lifespan=lifespan)
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=[
            v.strip()
            for v in os.environ.get(
                "MTMC_SHOWROOM_ALLOWED_HOSTS", "127.0.0.1,localhost,::1,testserver"
            ).split(",")
        ],
    )

    @app.middleware("http")
    async def boundary(request, call_next):
        if request.method not in ("GET", "HEAD", "OPTIONS"):
            origin = request.headers.get("origin")
            if request.headers.get("x-mtmc-client") != "web" or (
                origin and urlsplit(origin).netloc != request.headers.get("host")
            ):
                return JSONResponse(
                    {"detail": "仅允许同源请求，并需要 X-MTMC-Client: web。"},
                    status_code=403,
                )
            # Bound actual bytes, including chunked requests; floorplans have a separate limit.
            limit = (
                8 * 1024 * 1024
                if request.url.path.endswith("/floorplans")
                else 1024 * 1024
            )
            parts, size = [], 0
            async for chunk in request.stream():
                size += len(chunk)
                if size > limit:
                    return JSONResponse({"detail": "上传内容过大。"}, status_code=413)
                parts.append(chunk)
            request._body = b"".join(parts)
        response = await call_next(request)
        response.headers.update(
            {
                "X-Content-Type-Options": "nosniff",
                "X-Frame-Options": "DENY",
                "Referrer-Policy": "no-referrer",
                "Cache-Control": "no-store",
                "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'",
            }
        )
        return response

    @app.exception_handler(KeyError)
    async def missing(request, exc):
        return JSONResponse({"detail": "门店或所请求的数据不存在。"}, status_code=404)

    @app.exception_handler(ValueError)
    async def invalid(request, exc):
        return JSONResponse({"detail": str(exc)[:300]}, status_code=422)

    @app.exception_handler(sqlite3.Error)
    async def database_error(request, exc):
        return JSONResponse(
            {"detail": "分析存储暂不可用，请检查磁盘空间和服务日志。"}, status_code=503
        )

    def repo():
        return app.state.repo

    def remote(method, *args):
        try:
            return getattr(upstream, method)(*args)
        except (HTTPError, URLError, OSError, TimeoutError):
            raise HTTPException(
                503, "追踪服务或画面暂不可用，请确认服务地址、任务和摄像头状态。"
            ) from None

    @app.get(PREFIX + "/health")
    def health():
        service = app.state.service
        return {
            "status": "ok",
            "tracking_url": upstream.base_url,
            "last_poll": service.last_poll,
            "last_reports": service.last_reports,
            "worker_error": service.error,
            "features": "memory_only",
        }

    @app.get(PREFIX + "/stores")
    def stores():
        return {"stores": repo().stores()}

    @app.post(PREFIX + "/stores", status_code=201)
    def create_store(body: StoreCreate):
        return repo().create_store(body.model_dump())

    @app.post(PREFIX + "/stores/{sid}/floorplans", status_code=201)
    async def upload(sid: str, request: Request):
        repo().store(sid)
        content = await request.body()

        def normalize():
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", Image.DecompressionBombWarning)
                    with Image.open(BytesIO(content)) as source:
                        if (
                            source.format not in ("PNG", "JPEG")
                            or source.width * source.height > 12_000_000
                        ):
                            raise ValueError(
                                "仅支持不超过 1200 万像素的 PNG / JPEG 平面图。"
                            )
                        source.load()
                        normalized = ImageOps.exif_transpose(source).convert("RGB")
                        output = BytesIO()
                        normalized.save(output, format="PNG")
                        return output.getvalue(), normalized.size
            except (
                UnidentifiedImageError,
                OSError,
                Image.DecompressionBombWarning,
                Image.DecompressionBombError,
            ):
                raise ValueError("图片损坏、格式不支持或尺寸过大。") from None

        data, (width, height) = await asyncio.to_thread(normalize)
        aid = uuid.uuid4().hex
        path = root / "floorplans" / f"{aid}.png"
        try:
            with path.open("xb") as output:
                output.write(data)
                output.flush()
                os.fsync(output.fileno())
            with repo().transaction() as db:
                db.execute(
                    "INSERT INTO assets VALUES(?,?,?,?)", (aid, sid, width, height)
                )
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        return {"id": aid, "width": width, "height": height}

    @app.get(PREFIX + "/stores/{sid}/floorplans/{aid}.png")
    def floorplan(sid: str, aid: str):
        repo().asset(sid, aid)
        return FileResponse(root / "floorplans" / f"{aid}.png", media_type="image/png")

    @app.get(PREFIX + "/stores/{sid}/layouts")
    def layouts(sid: str):
        repo().store(sid)
        return {"layouts": repo().layouts(sid)}

    @app.post(PREFIX + "/stores/{sid}/layouts", status_code=201)
    def publish(sid: str, body: LayoutCreate):
        return repo().publish_layout(sid, body)

    @app.get(PREFIX + "/tracking/jobs")
    def jobs():
        # Only safe metadata crosses the module boundary, never stream addresses.
        return {
            "jobs": [
                {k: j[k] for k in ("id", "mode", "status", "created_at", "cameras")}
                for j in remote("jobs")
                if j["mode"] == "online"
            ]
        }

    @app.get(PREFIX + "/tracking/jobs/{run}/cameras/{camera}/frame.jpg")
    def snapshot(run: str, camera: int):
        return Response(remote("frame", run, camera), media_type="image/jpeg")

    @app.get(PREFIX + "/stores/{sid}/bindings")
    def bindings(sid: str):
        repo().store(sid)
        return {"bindings": repo().bindings(sid)}

    @app.post(PREFIX + "/stores/{sid}/bindings", status_code=201)
    def bind(sid: str, body: BindingCreate):
        repo().bind(sid, remote("job", body.job_id))
        return {"bindings": repo().bindings(sid)}

    @app.post(PREFIX + "/stores/{sid}/bindings/{run}/pause")
    def pause(sid: str, run: str, body: BindingPause):
        repo().pause(sid, run, body.paused)
        return {"bindings": repo().bindings(sid)}

    @app.get(PREFIX + "/stores/{sid}/live")
    def live(sid: str):
        return repo().live(sid)

    @app.post(PREFIX + "/stores/{sid}/roles")
    def role(sid: str, body: RoleSet):
        repo().role(sid, body.run_id, body.global_id, body.role)
        return {"status": "ok"}

    @app.get(PREFIX + "/stores/{sid}/reports")
    def reports(sid: str):
        repo().store(sid)
        with repo().lock:
            rows = repo().db.execute(
                "SELECT day,MAX(revision) AS revision,MAX(generated) AS generated FROM reports WHERE store=? GROUP BY day ORDER BY day DESC LIMIT 366",
                (sid,),
            )
            return {"reports": [dict(r) for r in rows]}

    @app.post(PREFIX + "/stores/{sid}/reports/{day}")
    def report(sid: str, day: str):
        return generate(repo(), sid, day)

    @app.get(PREFIX + "/stores/{sid}/reports/{day}.{format}")
    def download(sid: str, day: str, format: str, download: bool = False):
        bounds(day, repo().store(sid)["timezone"])
        if format not in ("html", "json"):
            raise HTTPException(404, "报告格式仅支持 HTML / JSON。")
        with repo().lock:
            row = (
                repo()
                .db.execute(
                    "SELECT * FROM reports WHERE store=? AND day=? ORDER BY revision DESC LIMIT 1",
                    (sid, day),
                )
                .fetchone()
            )
            if row is None:
                raise HTTPException(404, "该日报尚未生成。")
            content = row["html"] if format == "html" else row["payload"]
        headers = (
            {"Content-Disposition": f'attachment; filename="showroom-{day}.{format}"'}
            if download
            else {}
        )
        return Response(
            content,
            media_type="text/html" if format == "html" else "application/json",
            headers=headers,
        )

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    app.mount("/ui", StaticFiles(directory=ROOT / "ui"), name="shared-ui")

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    return app
