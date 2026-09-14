"""Read-only public HTTP client. No imports from web or tracking internals."""

import json
import re
from urllib.parse import urlencode, urlsplit
from urllib.request import build_opener, HTTPRedirectHandler, ProxyHandler


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


class TrackingClient:
    def __init__(self, base_url):
        parsed = urlsplit(base_url)
        if (
            parsed.scheme not in ("http", "https")
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("追踪服务地址必须是无凭据的 HTTP/HTTPS 根地址。")
        self.base_url = base_url.rstrip("/")
        self.opener = build_opener(ProxyHandler({}), NoRedirect())

    def get(self, path):
        with self.opener.open(self.base_url + path, timeout=3) as response:
            content = response.read(2 * 1024 * 1024 + 1)
            if len(content) > 2 * 1024 * 1024:
                raise ValueError("追踪响应超出限制。")
            return json.loads(content)

    def jobs(self):
        return self.get("/api/jobs")["jobs"]

    def job(self, job_id):
        if not re.fullmatch(r"[a-f0-9]{32}", job_id):
            raise ValueError("无效任务编号。")
        return self.get("/api/jobs/" + job_id)

    def events(self, job_id, cursor):
        if not re.fullmatch(r"[a-f0-9]{32}", job_id):
            raise ValueError("无效任务编号。")
        return self.get(
            "/api/jobs/"
            + job_id
            + "/events?"
            + urlencode({"cursor": cursor, "limit": 500})
        )

    def frame(self, job_id, camera_id):
        if not re.fullmatch(r"[a-f0-9]{32}", job_id) or not 0 <= camera_id <= 10000:
            raise ValueError("无效任务或摄像头编号。")
        with self.opener.open(
            f"{self.base_url}/api/jobs/{job_id}/cameras/{camera_id}/frame.jpg",
            timeout=3,
        ) as response:
            data = response.read(8 * 1024 * 1024 + 1)
            if len(data) > 8 * 1024 * 1024 or not data.startswith(b"\xff\xd8"):
                raise ValueError("无法读取 JPEG 画面。")
            return data
