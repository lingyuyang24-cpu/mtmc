"""Synthetic contract/API/statistics tests. No video/model weights are required."""

import ast
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from io import BytesIO
import json
from pathlib import Path
import tempfile
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

from modules.showroom.app import create_app
from modules.showroom.client import TrackingClient
from modules.showroom.geometry import calibrate, polygon_valid, project
from modules.showroom.repository import Repository
from modules.showroom.reports import bounds, build_report, generate, exclusive
from modules.showroom.schemas import LayoutCreate, StoreCreate
from modules.showroom.service import Service
from tracking_contracts.events import EventWriter, frame_event, read_page

RUN = "a" * 32
AID = "b" * 32
NOW = time.time()
START = (
    datetime.fromtimestamp(NOW, timezone.utc)
    .replace(hour=9, minute=0, second=0, microsecond=0)
    .timestamp()
    - 86400
)
DAY = datetime.fromtimestamp(START, timezone.utc).date().isoformat()


def layout():
    return {
        "floorplan_id": AID,
        "width_m": 10,
        "height_m": 10,
        "effective_at": datetime.fromtimestamp(START - 60, timezone.utc).isoformat(),
        "cameras": [
            {
                "camera_id": i,
                "width": 100,
                "height": 100,
                "points": [
                    [0, 0, 0, 0],
                    [100, 0, 10, 0],
                    [100, 100, 10, 10],
                    [0, 100, 0, 10],
                ],
            }
            for i in (0, 1)
        ],
        "vehicles": [
            {
                "id": "car_a",
                "name": "展车 A",
                "model": "轿车",
                "polygon": [[1, 1], [4, 1], [4, 8], [1, 8]],
            },
            {
                "id": "car_b",
                "name": "展车 B",
                "model": "SUV",
                "polygon": [[6, 1], [9, 1], [9, 8], [6, 8]],
            },
        ],
    }


def event(seq, ts=None, camera=0, x=20, gid=1, empty=False, run=RUN):
    tracks = (
        []
        if empty
        else [
            {
                "local_id": 1,
                "bbox": [x - 5, 10, x + 5, 50],
                "feature_quality": 0.9,
                "feature": [99, 99],
            }
        ]
    )
    return {
        **frame_event(
            camera,
            seq,
            START + seq if ts is None else ts,
            100,
            100,
            tracks,
            {1: gid},
            {},
        ),
        "schema_version": 1,
        "run_id": run,
        "record_id": f"{run}:{camera}:{seq}",
    }


def job():
    return {
        "id": RUN,
        "mode": "online",
        "status": "running",
        "created_at": START,
        "cameras": [
            {"id": i, "name": f"摄像头 {i}", "source_width": 100, "source_height": 100}
            for i in (0, 1)
        ],
    }


class FakeClient:
    base_url = "http://127.0.0.1:8765"

    def __init__(self):
        self.page = {
            "schema_version": 1,
            "available": True,
            "events": [],
            "cursor": "0:0",
            "caught_up": True,
            "job_status": "running",
            "dropped": 0,
        }

    def jobs(self):
        return [job()]

    def job(self, run):
        return job()

    def events(self, run, cursor):
        return self.page

    def frame(self, run, camera):
        out = BytesIO()
        Image.new("RGB", (100, 100), "white").save(out, "JPEG")
        return out.getvalue()


class ContractTests(unittest.TestCase):
    def test_rotation_cursor_and_no_features(self):
        with tempfile.TemporaryDirectory() as folder:
            writer = EventWriter(folder, RUN, segment_bytes=500)
            for i in range(10):
                self.assertTrue(writer.submit(event(i)))
            writer.close()
            self.assertIsNone(writer.error)
            cursor, records = "0:0", []
            for _ in range(15):
                page = read_page(folder, cursor, 2)
                records.extend(page["events"])
                cursor = page["cursor"]
                if page["caught_up"]:
                    break
            self.assertEqual(len(records), 10)
            self.assertTrue(page["closed"])
            self.assertGreater(int(cursor.split(":")[0]), 0)
            self.assertNotIn("feature", json.dumps(records))
            self.assertNotIn("embedding", json.dumps(records))
            self.assertEqual(read_page(folder, cursor)["events"], [])
            with self.assertRaises(ValueError):
                read_page(folder, "0:3")
            with self.assertRaises(ValueError):
                read_page(folder, "../:0")

    def test_bounded_writer_and_error_isolation(self):
        with tempfile.TemporaryDirectory() as folder:
            writer = EventWriter(folder, RUN, max_bytes=10)
            self.assertFalse(writer.submit(event(1)))
            writer.close()
            self.assertEqual(writer.dropped, 1)
            self.assertEqual(read_page(folder)["dropped"], 1)
            second = EventWriter(folder, RUN)
            second.close()
            self.assertEqual(second.error, "FileExistsError")

    def test_missing_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertFalse(read_page(folder)["available"])


class GeometryTests(unittest.TestCase):
    def test_projection_and_degeneracy(self):
        matrix = calibrate(layout()["cameras"][0]["points"])["matrix"]
        x, y = project(matrix, [20, 50])
        self.assertAlmostEqual(x, 2)
        self.assertAlmostEqual(y, 5)
        with self.assertRaises(ValueError):
            calibrate([[i, i, i, i] for i in range(4)])

    def test_polygons(self):
        self.assertFalse(polygon_valid([[0, 0], [2, 2], [0, 2], [2, 0]]))
        self.assertTrue(
            polygon_valid(
                [[0, 0], [1, 0], [1, 1], [2, 1], [2, 0], [3, 0], [3, 2], [0, 2]]
            )
        )

    def test_conflicting_cameras_are_excluded(self):
        data, lost = exclusive({(RUN, 1, "a"): [[0, 10]], (RUN, 1, "b"): [[5, 15]]})
        self.assertEqual(lost, 5)
        self.assertEqual(data[(RUN, 1, "a")], [[0, 5]])
        self.assertEqual(data[(RUN, 1, "b")], [[10, 15]])

    def test_calendar_dst(self):
        a, b = bounds("2026-03-08", "America/New_York")
        self.assertEqual(b - a, 23 * 3600)
        a, b = bounds("2026-11-01", "America/New_York")
        self.assertEqual(b - a, 25 * 3600)


class RepositoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.repo = Repository(self.temp.name)
        self.sid = self.repo.create_store(
            StoreCreate(name="测试门店", timezone="UTC", min_dwell=2).model_dump()
        )["id"]
        with self.repo.transaction() as db:
            db.execute("INSERT INTO assets VALUES(?,?,?,?)", (AID, self.sid, 100, 100))
        self.repo.publish_layout(self.sid, LayoutCreate(**layout()))
        self.repo.bind(self.sid, job())
        self.client = FakeClient()

    def tearDown(self):
        self.repo.close()
        self.temp.cleanup()

    def ingest(self, events):
        b = self.repo.bindings(self.sid)[0]
        page = {
            **self.client.page,
            "events": events,
            "cursor": f"0:{int(b['cursor'].split(':')[1])+len(events)}",
        }
        return self.repo.ingest(self.sid, RUN, b["cursor"], page)

    def report(self):
        return build_report(self.repo, self.sid, DAY, NOW)

    def test_dwell_multicam_union_and_zero_cars(self):
        self.ingest([event(i, camera=c) for i in range(5) for c in (0, 1)])
        report = self.report()
        self.assertEqual(report["summary"]["people"], 1)
        self.assertEqual(report["summary"]["valid_seconds"], 4)
        self.assertEqual(report["vehicles"][0]["effective_people"], 1)
        self.assertEqual(report["vehicles"][1]["valid_seconds"], 0)
        self.assertEqual(report["status"], "partial")

    def test_empty_frame_and_sequence_gap_never_bridge(self):
        self.ingest(
            [
                event(0),
                event(1),
                event(2, empty=True),
                event(3),
                event(4),
                event(7),
                event(8),
            ]
        )
        self.assertEqual(self.report()["summary"]["valid_seconds"], 3)

    def test_idempotence_and_atomic_rollback(self):
        events = [event(0), event(1)]
        self.ingest(events)
        self.ingest(events)
        self.assertEqual(self.report()["summary"]["frames"], 2)
        before = self.repo.bindings(self.sid)[0]["cursor"]
        malformed = event(3)
        malformed["observations"][0]["bbox"] = [1, 2]
        with self.assertRaises(ValueError):
            self.ingest([event(2), malformed])
        self.assertEqual(self.repo.bindings(self.sid)[0]["cursor"], before)
        self.assertEqual(self.report()["summary"]["frames"], 2)

    def test_unknown_and_overlap_not_assigned(self):
        e = event(0)
        e["width"] = 200
        self.ingest([e, event(1, gid=None), event(2, x=50)])
        report = self.report()
        self.assertEqual(report["summary"]["unmapped_observations"], 1)
        self.assertEqual(report["summary"]["unlinked_observations"], 1)
        self.assertEqual(report["summary"]["valid_seconds"], 0)

    def test_report_conflict_exclusion(self):
        self.ingest(
            [
                event(i, camera=c, x=20 if c == 0 else 70)
                for i in range(5)
                for c in (0, 1)
            ]
        )
        report = self.report()
        self.assertEqual(report["summary"]["valid_seconds"], 0)
        self.assertEqual(report["summary"]["conflict_seconds"], 4)

    def test_overlapping_zones_stay_unknown(self):
        value = layout()
        value["effective_at"] = datetime.fromtimestamp(
            START - 59, timezone.utc
        ).isoformat()
        value["vehicles"][1]["polygon"] = value["vehicles"][0]["polygon"]
        self.repo.publish_layout(self.sid, LayoutCreate(**value))
        self.ingest([event(i) for i in range(3)])
        report = self.report()
        self.assertEqual(report["summary"]["unmapped_observations"], 3)
        self.assertEqual(report["summary"]["valid_seconds"], 0)

    def test_live_returns_saved_role(self):
        self.ingest([event(1)])
        self.repo.role(self.sid, RUN, 1, "staff")
        with patch("modules.showroom.repository.time.time", return_value=START + 2):
            self.assertEqual(self.repo.live(self.sid)["positions"][0]["role"], "staff")

    def test_restarted_tracking_keeps_separate_identities(self):
        self.ingest([event(i) for i in range(3)])
        other_run = "d" * 32
        self.repo.bind(self.sid, {**job(), "id": other_run})
        page = {
            **self.client.page,
            "events": [event(i, run=other_run) for i in range(3)],
            "cursor": "0:3",
        }
        self.repo.ingest(self.sid, other_run, "0:0", page)
        report = self.report()
        self.assertEqual(report["summary"]["people"], 2)
        self.assertEqual(report["summary"]["valid_seconds"], 4)
        self.assertTrue(any("跨运行无法自动去重" in w for w in report["warnings"]))

    def test_staff_and_revisions(self):
        self.ingest([event(i) for i in range(5)])
        p = generate(self.repo, self.sid, DAY, NOW)
        self.assertEqual(
            generate(self.repo, self.sid, DAY, NOW)["revision"], p["revision"]
        )
        self.repo.role(self.sid, RUN, 1, "staff")
        updated = generate(self.repo, self.sid, DAY, NOW)
        self.assertEqual(updated["summary"]["people"], 0)
        self.assertEqual(updated["summary"]["staff"], 1)
        self.assertEqual(updated["summary"]["valid_seconds"], 0)
        self.assertEqual(updated["revision"], p["revision"] + 1)

    def test_revisit_requires_observed_exit(self):
        self.ingest(
            [event(i) for i in range(4)]
            + [event(4, x=50)]
            + [event(i) for i in range(24, 28)]
        )
        self.assertEqual(self.report()["vehicles"][0]["revisits"], 1)

    def test_pause_resume(self):
        self.client.page = {
            **self.client.page,
            "events": [event(0), event(1)],
            "cursor": "0:2",
        }
        service = Service(self.repo, self.client)
        self.repo.pause(self.sid, RUN, True)
        service.poll()
        self.assertEqual(self.repo.bindings()[0]["cursor"], "0:0")
        self.repo.pause(self.sid, RUN, False)
        service.poll()
        self.assertEqual(self.repo.bindings()[0]["cursor"], "0:2")
        self.assertEqual(self.report()["summary"]["frames"], 2)

    def test_upstream_failure_keeps_cursor(self):
        service = Service(self.repo, self.client)
        with patch.object(self.client, "events", side_effect=OSError("secret/path")):
            service.poll()
        b = self.repo.bindings()[0]
        self.assertEqual(b["cursor"], "0:0")
        self.assertIn("暂不可用", b["error"])
        self.assertNotIn("secret", b["error"])

    def test_layout_version_does_not_rewrite_processed_history(self):
        self.ingest([event(1)])
        with self.assertRaises(ValueError):
            self.repo.publish_layout(self.sid, LayoutCreate(**layout()))
        value = layout()
        value["effective_at"] = datetime.fromtimestamp(
            START + 2, timezone.utc
        ).isoformat()
        self.assertEqual(
            self.repo.publish_layout(self.sid, LayoutCreate(**value))["version"], 2
        )

    def test_midnight_split(self):
        end = bounds(DAY, "UTC")[1]
        self.ingest([event(i, ts=end - 2 + i) for i in range(5)])
        yesterday = build_report(self.repo, self.sid, DAY, end + 3)
        today = build_report(
            self.repo,
            self.sid,
            datetime.fromtimestamp(end, timezone.utc).date().isoformat(),
            end + 3,
        )
        self.assertEqual(yesterday["summary"]["valid_seconds"], 2)
        self.assertEqual(today["summary"]["valid_seconds"], 2)
        self.assertEqual(today["summary"]["people"], 1)

    def test_daily_schedule_and_catchup(self):
        with self.repo.transaction() as db:
            db.execute(
                "UPDATE stores SET created=? WHERE id=?", (START - 86400, self.sid)
            )
        end = bounds(DAY, "UTC")[1]
        service = Service(self.repo, self.client)
        service.daily(end + 5 * 60)
        self.assertIsNone(
            self.repo.db.execute("SELECT 1 FROM reports WHERE day=?", (DAY,)).fetchone()
        )
        service.daily(end + 10 * 60)
        self.assertIsNotNone(
            self.repo.db.execute("SELECT 1 FROM reports WHERE day=?", (DAY,)).fetchone()
        )
        self.assertGreaterEqual(
            self.repo.db.execute("SELECT COUNT(*) FROM reports").fetchone()[0], 2
        )

    def test_restart_retains_statistics_and_cursor_not_features(self):
        self.ingest([event(i) for i in range(3)])
        cursor = self.repo.bindings()[0]["cursor"]
        self.repo.close()
        self.repo = Repository(self.temp.name)
        self.assertEqual(self.repo.bindings()[0]["cursor"], cursor)
        self.assertEqual(self.report()["summary"]["valid_seconds"], 2)
        self.assertNotIn(
            "feature",
            self.repo.db.execute(
                "SELECT GROUP_CONCAT(sql) FROM sqlite_master"
            ).fetchone()[0],
        )


class APITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.app = create_app(self.temp.name, FakeClient(), background=False)
        self.client = TestClient(self.app)
        self.client.__enter__()
        self.headers = {"X-MTMC-Client": "web"}

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.temp.cleanup()

    def post(self, path, value):
        return self.client.post(
            "/api/showroom/v1" + path, json=value, headers=self.headers
        )

    def test_full_configuration_and_report_api(self):
        response = self.post(
            "/stores", {"name": "<script>恶意名称</script>", "timezone": "UTC"}
        )
        self.assertEqual(response.status_code, 201, response.text)
        sid = response.json()["id"]
        image = BytesIO()
        Image.new("RGB", (100, 100), "white").save(image, format="PNG")
        uploaded = self.client.post(
            f"/api/showroom/v1/stores/{sid}/floorplans",
            content=image.getvalue(),
            headers=self.headers,
        )
        self.assertEqual(uploaded.status_code, 201, uploaded.text)
        value = layout()
        value["floorplan_id"] = uploaded.json()["id"]
        self.assertEqual(self.post(f"/stores/{sid}/layouts", value).status_code, 201)
        self.assertEqual(
            self.post(f"/stores/{sid}/bindings", {"job_id": RUN}).status_code, 201
        )
        p = self.post(f"/stores/{sid}/reports/{DAY}", {})
        self.assertEqual(p.status_code, 200, p.text)
        html = self.client.get(f"/api/showroom/v1/stores/{sid}/reports/{DAY}.html")
        self.assertEqual(html.status_code, 200)
        self.assertNotIn("<script>恶意名称</script>", html.text)
        self.assertIn("&lt;script&gt;", html.text)
        self.assertIn("data:image/png;base64,", html.text)
        js = self.client.get(
            f"/api/showroom/v1/stores/{sid}/reports/{DAY}.json?download=true"
        )
        self.assertIn("attachment", js.headers["content-disposition"])
        self.assertEqual(js.json()["revision"], p.json()["revision"])
        self.assertIn("展车分析", self.client.get("/").text)

    def test_boundaries(self):
        prefix = "/api/showroom/v1"
        self.assertEqual(
            self.client.post(prefix + "/stores", json={"name": "x"}).status_code, 403
        )
        self.assertEqual(
            self.client.post(
                prefix + "/stores",
                json={"name": "x"},
                headers={**self.headers, "Origin": "https://evil.example"},
            ).status_code,
            403,
        )
        self.assertEqual(
            self.client.post(
                prefix + "/stores",
                content=b"x" * (1024 * 1024 + 1),
                headers=self.headers,
            ).status_code,
            413,
        )
        sid = self.post("/stores", {"name": "x"}).json()["id"]
        self.assertEqual(
            self.client.post(
                f"{prefix}/stores/{sid}/floorplans",
                content=b"not a picture",
                headers=self.headers,
            ).status_code,
            422,
        )
        self.assertEqual(
            self.post(
                "/stores", {"name": "x", "timezone": "not/a/timezone"}
            ).status_code,
            422,
        )
        self.assertEqual(
            self.post(
                "/stores", {"name": "x", "opens": "18:00", "closes": "09:00"}
            ).status_code,
            422,
        )
        self.assertEqual(
            self.client.get(f"{prefix}/stores/{sid}/floorplans/{AID}.png").status_code,
            404,
        )

    def test_module_independence(self):
        for path in Path("modules/showroom").glob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                modules = (
                    [a.name for a in node.names]
                    if isinstance(node, ast.Import)
                    else [node.module or ""] if isinstance(node, ast.ImportFrom) else []
                )
                self.assertFalse(
                    any(
                        m.split(".")[0]
                        in {"web", "torch", "cv2", "ultralytics", "online_mtmc"}
                        for m in modules
                    ),
                    path,
                )

    def test_shared_vertical_navigation(self):
        html = self.client.get('/').text
        self.assertIn('class="module-sidebar"', html)
        self.assertLess(html.index('id="nav-online"'), html.index('id="nav-offline"'))
        self.assertIn('aria-current="page"', html)
        css = self.client.get('/ui/module-nav.css')
        self.assertEqual(css.status_code, 200)
        self.assertIn('flex-direction: column', css.text)


class HTTPIntegrationTests(unittest.TestCase):
    def test_public_tracking_api_to_separate_consumer(self):
        from web.app import create_app as tracking_app

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with TestClient(tracking_app(root / "tracking")) as source:
                source.app.state.manager.jobs[RUN] = job()
                writer = EventWriter(root / "tracking" / "jobs" / RUN / "events", RUN)
                for i in range(5):
                    writer.submit(event(i))
                writer.close()

                class Handler(BaseHTTPRequestHandler):
                    def do_GET(self):
                        response = source.get(self.path)
                        self.send_response(response.status_code)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        self.wfile.write(response.content)

                    def log_message(self, *args):
                        pass

                server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                client = TrackingClient(f"http://127.0.0.1:{server.server_port}")
                repo = Repository(root / "analytics")
                try:
                    sid = repo.create_store(
                        StoreCreate(name="HTTP集成测试", timezone="UTC").model_dump()
                    )["id"]
                    with repo.transaction() as db:
                        db.execute(
                            "INSERT INTO assets VALUES(?,?,?,?)", (AID, sid, 100, 100)
                        )
                    repo.publish_layout(sid, LayoutCreate(**layout()))
                    repo.bind(sid, client.job(RUN))
                    service = Service(repo, client)
                    service.poll()
                    p = generate(repo, sid, DAY, NOW)
                    self.assertEqual(p["summary"]["frames"], 5)
                    self.assertEqual(p["summary"]["valid_seconds"], 4)
                    service.poll()
                    self.assertEqual(
                        generate(repo, sid, DAY, NOW)["revision"], p["revision"]
                    )
                    self.assertEqual(
                        source.get(f"/api/jobs/{RUN}/events?cursor=0:3").status_code,
                        422,
                    )
                finally:
                    repo.close()
                    server.shutdown()
                    server.server_close()
                    thread.join(timeout=3)


if __name__ == "__main__":
    unittest.main()
