"""Showroom-owned SQLite state. Stores positions/events, NEVER embeddings."""

from contextlib import contextmanager
from datetime import datetime
import json
import math
from pathlib import Path
import sqlite3
import threading
import time
import uuid
from zoneinfo import ZoneInfo

from modules.showroom.geometry import calibrate, inside, project


def dumps(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


class Repository:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "floorplans").mkdir(exist_ok=True)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(
            self.root / "showroom.sqlite3", check_same_thread=False, timeout=5
        )
        self.db.row_factory = sqlite3.Row
        if self.db.execute("PRAGMA user_version").fetchone()[0] not in (0, 1):
            self.db.close()
            raise RuntimeError("展车数据库版本不兼容，请使用对应版本的分析服务。")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS stores(id TEXT PRIMARY KEY, config TEXT NOT NULL, created REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS assets(id TEXT PRIMARY KEY, store TEXT NOT NULL, width INT, height INT);
        CREATE TABLE IF NOT EXISTS layouts(store TEXT, version INT, effective REAL, config TEXT,
          PRIMARY KEY(store,version), UNIQUE(store,effective));
        CREATE TABLE IF NOT EXISTS bindings(store TEXT, run TEXT, cursor TEXT DEFAULT '0:0',
          status TEXT, error TEXT, created REAL, caught_up INT DEFAULT 0, dropped INT DEFAULT 0,
          PRIMARY KEY(store,run));
        CREATE TABLE IF NOT EXISTS processed(store TEXT, record TEXT, ts REAL, PRIMARY KEY(store,record));
        CREATE INDEX IF NOT EXISTS processed_time ON processed(ts);
        CREATE TABLE IF NOT EXISTS intervals(id INTEGER PRIMARY KEY, store TEXT, run TEXT, gid INT,
          camera INT, vehicle TEXT, layout INT, t0 REAL, t1 REAL);
        CREATE INDEX IF NOT EXISTS interval_time ON intervals(store,t0,t1);
        CREATE TABLE IF NOT EXISTS positions(store TEXT, run TEXT, camera INT, local INT, gid INT,
          ts REAL, vehicle TEXT, layout INT, interval_id INT, x REAL, y REAL,
          PRIMARY KEY(store,run,camera,local));
        CREATE TABLE IF NOT EXISTS exits(store TEXT, run TEXT, gid INT, vehicle TEXT, ts REAL);
        CREATE INDEX IF NOT EXISTS exit_time ON exits(store,ts);
        CREATE TABLE IF NOT EXISTS coverage(id INTEGER PRIMARY KEY, store TEXT, run TEXT, camera INT,
          t0 REAL,t1 REAL);
        CREATE INDEX IF NOT EXISTS coverage_time ON coverage(store,t0,t1);
        CREATE TABLE IF NOT EXISTS camera_state(store TEXT,run TEXT,camera INT,ts REAL,seq INT,
          coverage_id INT,skipped INT, PRIMARY KEY(store,run,camera));
        CREATE TABLE IF NOT EXISTS counters(store TEXT,day TEXT,run TEXT,camera INT,frames INT DEFAULT 0,
          unknown INT DEFAULT 0, unlinked INT DEFAULT 0, skipped INT DEFAULT 0, last_record TEXT,
          PRIMARY KEY(store,day,run,camera));
        CREATE TABLE IF NOT EXISTS roles(store TEXT,run TEXT,gid INT,role TEXT,
          PRIMARY KEY(store,run,gid));
        CREATE TABLE IF NOT EXISTS presence(store TEXT,day TEXT,run TEXT,gid INT,t0 REAL,t1 REAL,
          PRIMARY KEY(store,day,run,gid));
        CREATE TABLE IF NOT EXISTS reports(store TEXT,day TEXT,revision INT,fingerprint TEXT,payload TEXT,
          html TEXT,generated REAL, PRIMARY KEY(store,day,revision));
        PRAGMA user_version=1;
        """)
        if "paused" not in {
            r["name"] for r in self.db.execute("PRAGMA table_info(bindings)")
        }:
            self.db.execute(
                "ALTER TABLE bindings ADD COLUMN paused INT NOT NULL DEFAULT 0"
            )
            self.db.commit()

    @contextmanager
    def transaction(self):
        with self.lock:
            try:
                self.db.execute("BEGIN IMMEDIATE")
                yield self.db
                self.db.commit()
            except BaseException:
                self.db.rollback()
                raise

    def close(self):
        with self.lock:
            self.db.close()

    def create_store(self, value):
        sid = uuid.uuid4().hex
        with self.transaction() as db:
            db.execute(
                "INSERT INTO stores VALUES(?,?,?)", (sid, dumps(value), time.time())
            )
        return self.store(sid)

    def store(self, sid):
        with self.lock:
            row = self.db.execute("SELECT * FROM stores WHERE id=?", (sid,)).fetchone()
            if row is None:
                raise KeyError("门店不存在。")
            return {
                "id": sid,
                **json.loads(row["config"]),
                "created_at": row["created"],
            }

    def stores(self):
        with self.lock:
            return [
                self.store(r[0])
                for r in self.db.execute("SELECT id FROM stores ORDER BY created")
            ]

    def layouts(self, sid):
        with self.lock:
            return [
                dict(
                    version=r["version"],
                    effective_at=r["effective"],
                    **json.loads(r["config"])
                )
                for r in self.db.execute(
                    "SELECT * FROM layouts WHERE store=? ORDER BY effective", (sid,)
                )
            ]

    def publish_layout(self, sid, model):
        self.store(sid)
        config = model.model_dump(mode="json")
        effective = model.effective_at.timestamp()
        config.pop("effective_at")
        for camera in config["cameras"]:
            camera.update(calibrate(camera["points"]))
        with self.transaction() as db:
            if not db.execute(
                "SELECT 1 FROM assets WHERE store=? AND id=?", (sid, model.floorplan_id)
            ).fetchone():
                raise ValueError("请先上传该门店的平面图。")
            last = db.execute(
                "SELECT MAX(ts) FROM camera_state WHERE store=?", (sid,)
            ).fetchone()[0]
            if last is not None and effective <= last:
                raise ValueError(
                    "新布局只能影响尚未处理的数据；请将生效时间设为当前之后。历史修订暂不支持。"
                )
            version = db.execute(
                "SELECT COALESCE(MAX(version),0)+1 FROM layouts WHERE store=?", (sid,)
            ).fetchone()[0]
            try:
                db.execute(
                    "INSERT INTO layouts VALUES(?,?,?,?)",
                    (sid, version, effective, dumps(config)),
                )
            except sqlite3.IntegrityError:
                raise ValueError("该生效时间已有布局，请使用不同时间。") from None
        return {"version": version, "effective_at": effective, **config}

    def asset(self, sid, aid):
        with self.lock:
            row = self.db.execute(
                "SELECT * FROM assets WHERE store=? AND id=?", (sid, aid)
            ).fetchone()
            if row is None:
                raise KeyError("平面图不存在。")
            return dict(row)

    def bind(self, sid, job):
        self.store(sid)
        if job["mode"] != "online":
            raise ValueError("一期只绑定在线任务；离线视频需提供绝对起始时间后再扩展。")
        if not self.layouts(sid):
            raise ValueError("请先发布平面图、车辆区域与摄像头标定。")
        valid = {c["id"] for c in job["cameras"]}
        if not {c["camera_id"] for c in self.layouts(sid)[-1]["cameras"]}.issubset(
            valid
        ):
            raise ValueError("布局中的摄像头编号不在该追踪任务内。")
        with self.transaction() as db:
            db.execute(
                "INSERT OR IGNORE INTO bindings(store,run,status,created) VALUES(?,?,?,?)",
                (sid, job["id"], job["status"], job.get("created_at", time.time())),
            )

    def bindings(self, sid=None):
        with self.lock:
            rows = self.db.execute(
                "SELECT * FROM bindings" + (" WHERE store=?" if sid else ""),
                (sid,) if sid else (),
            )
            return [dict(r) for r in rows]

    def pause(self, sid, run, paused):
        with self.transaction() as db:
            if not db.execute(
                "UPDATE bindings SET paused=?,caught_up=0 WHERE store=? AND run=?",
                (int(paused), sid, run),
            ).rowcount:
                raise KeyError("分析绑定不存在。")

    def ingest(self, sid, run, expected_cursor, page):
        if page.get("schema_version") != 1:
            raise ValueError("不支持的追踪事件协议版本。")
        if not page.get("available"):
            raise ValueError(
                "此任务没有新版事件日志，请在升级后的追踪服务中新建在线任务。"
            )
        store = self.store(sid)
        layouts = self.layouts(sid)
        with self.transaction() as db:
            binding = db.execute(
                "SELECT * FROM bindings WHERE store=? AND run=?", (sid, run)
            ).fetchone()
            if (
                binding is None
                or binding["paused"]
                or binding["cursor"] != expected_cursor
            ):
                return False
            for event in page.get("events", []):
                if event.get("schema_version") != 1 or event.get("run_id") != run:
                    raise ValueError("追踪事件运行或版本不匹配。")
                if event.get("type") not in ("frame", "health"):
                    raise ValueError("出现未知追踪事件类型，已停止消费。")
                ts = float(event["timestamp"])
                if not math.isfinite(ts) or ts < 946684800 or ts > time.time() + 300:
                    raise ValueError("事件缺少可信的绝对时间。")
                if not db.execute(
                    "INSERT OR IGNORE INTO processed VALUES(?,?,?)",
                    (sid, event["record_id"], ts),
                ).rowcount:
                    continue
                if event["type"] == "frame":
                    self._frame(db, store, run, event, layouts)
            publisher_error = (
                "上游事件日志写入异常或未正常封存，请检查追踪服务。"
                if page.get("publisher_error")
                else None
            )
            db.execute(
                "UPDATE bindings SET cursor=?,status=?,error=?,caught_up=?,dropped=? WHERE store=? AND run=?",
                (
                    page["cursor"],
                    page.get("job_status", "running"),
                    publisher_error,
                    int(page.get("caught_up", False) and not publisher_error),
                    page.get("dropped", 0),
                    sid,
                    run,
                ),
            )
        return True

    def _frame(self, db, store, run, e, layouts):
        sid, ts, cam, seq = (
            store["id"],
            float(e["timestamp"]),
            int(e["camera_id"]),
            int(e["seq"]),
        )
        if cam not in {c["camera_id"] for layout in layouts for c in layout["cameras"]}:
            return
        day = datetime.fromtimestamp(ts, ZoneInfo(store["timezone"])).date().isoformat()
        previous = db.execute(
            "SELECT * FROM camera_state WHERE store=? AND run=? AND camera=?",
            (sid, run, cam),
        ).fetchone()
        if previous and seq <= previous["seq"]:
            return
        if previous and ts <= previous["ts"]:
            raise ValueError("摄像头时间倒退；请检查时钟后新建任务。")
        cover = None
        if (
            previous
            and seq == previous["seq"] + 1
            and ts - previous["ts"] <= store["max_gap"]
        ):
            cover = previous["coverage_id"]
            db.execute("UPDATE coverage SET t1=? WHERE id=?", (ts, cover))
        if cover is None:
            cover = db.execute(
                "INSERT INTO coverage(store,run,camera,t0,t1) VALUES(?,?,?,?,?)",
                (sid, run, cam, ts, ts),
            ).lastrowid
        skipped = int(e.get("inference_skipped", 0))
        delta_skip = max(0, skipped - (previous["skipped"] if previous else 0))
        db.execute(
            "INSERT OR REPLACE INTO camera_state VALUES(?,?,?,?,?,?,?)",
            (sid, run, cam, ts, seq, cover, skipped),
        )
        layout = next((v for v in reversed(layouts) if v["effective_at"] <= ts), None)
        camera = (
            next((c for c in layout["cameras"] if c["camera_id"] == cam), None)
            if layout
            else None
        )
        if camera and (
            e["width"] != camera["width"] or e["height"] != camera["height"]
        ):
            camera = None  # no unannounced rescaling of calibrated coordinates
        unknown, unlinked = 0, 0
        observations = e.get("observations", [])
        if not isinstance(observations, list) or len(observations) > 1000:
            raise ValueError("无效观察列表。")
        active = set()
        for observation in observations:
            local, gid = int(observation["local_id"]), observation.get("global_id")
            if local in active:
                raise ValueError("同一帧局部身份重复。")
            active.add(local)
            if gid is None:
                unlinked += 1
                continue
            if type(gid) is not int or gid < 1:
                raise ValueError("全局身份必须为正整数或空。")
            db.execute(
                """INSERT INTO presence VALUES(?,?,?,?,?,?) ON CONFLICT(store,day,run,gid)
                          DO UPDATE SET t1=MAX(t1,excluded.t1)""",
                (sid, day, run, gid, ts, ts),
            )
            bbox = observation["bbox"]
            if len(bbox) != 4 or not all(math.isfinite(float(v)) for v in bbox):
                raise ValueError("无效人员框。")
            x1, y1, x2, y2 = map(float, bbox)
            foot = [(x1 + x2) / 2, y2]
            point, vehicle, certain = None, None, False
            if (
                camera
                and 0 <= x1 < x2 <= e["width"]
                and 0 <= y1 < y2 < e["height"] - 1
                and float(observation.get("quality", 0)) > 0
                and inside(foot, camera["valid_polygon"])
            ):
                point = project(camera["matrix"], foot)
                if point is not None:
                    candidates = [
                        v["id"]
                        for v in layout["vehicles"]
                        if inside(point, v["polygon"])
                    ]
                    certain = len(candidates) <= 1
                    vehicle = candidates[0] if len(candidates) == 1 else None
            if not certain:
                unknown += 1
            old = db.execute(
                "SELECT * FROM positions WHERE store=? AND run=? AND camera=? AND local=?",
                (sid, run, cam, local),
            ).fetchone()
            segment = None
            continuous = (
                old is not None
                and old["gid"] == gid
                and previous is not None
                and seq == previous["seq"] + 1
                and 0 < ts - old["ts"] <= store["max_gap"]
            )
            if (
                certain
                and old
                and old["gid"] == gid
                and old["vehicle"]
                and vehicle != old["vehicle"]
            ):
                db.execute(
                    "INSERT INTO exits VALUES(?,?,?,?,?)",
                    (sid, run, gid, old["vehicle"], ts),
                )
            if vehicle:
                if (
                    continuous
                    and old["vehicle"] == vehicle
                    and old["layout"] == layout["version"]
                ):
                    segment = old["interval_id"]
                    db.execute("UPDATE intervals SET t1=? WHERE id=?", (ts, segment))
                else:
                    segment = db.execute(
                        "INSERT INTO intervals(store,run,gid,camera,vehicle,layout,t0,t1) VALUES(?,?,?,?,?,?,?,?)",
                        (sid, run, gid, cam, vehicle, layout["version"], ts, ts),
                    ).lastrowid
            db.execute(
                "INSERT OR REPLACE INTO positions VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (
                    sid,
                    run,
                    cam,
                    local,
                    gid,
                    ts,
                    vehicle,
                    layout["version"] if layout else None,
                    segment,
                    point[0] if point else None,
                    point[1] if point else None,
                ),
            )
        # An explicitly empty/occluded frame breaks continuity, even for a short gap.
        db.execute(
            "DELETE FROM positions WHERE store=? AND run=? AND camera=? AND ts<?",
            (sid, run, cam, ts),
        )
        db.execute(
            """INSERT INTO counters VALUES(?,?,?,?,1,?,?,?,?) ON CONFLICT(store,day,run,camera)
                      DO UPDATE SET frames=frames+1,unknown=unknown+excluded.unknown,
                      unlinked=unlinked+excluded.unlinked,skipped=skipped+excluded.skipped,last_record=excluded.last_record""",
            (sid, day, run, cam, unknown, unlinked, delta_skip, e["record_id"]),
        )

    def mark_error(self, sid, run, message):
        with self.transaction() as db:
            db.execute(
                "UPDATE bindings SET error=?,caught_up=0 WHERE store=? AND run=?",
                (message, sid, run),
            )

    def role(self, sid, run, gid, value):
        if value not in ("visitor", "staff", "unknown"):
            raise ValueError("角色仅支持 visitor / staff / unknown。")
        with self.transaction() as db:
            if not db.execute(
                "SELECT 1 FROM bindings WHERE store=? AND run=?", (sid, run)
            ).fetchone():
                raise ValueError("该运行不属于此门店。")
            db.execute(
                "INSERT OR REPLACE INTO roles VALUES(?,?,?,?)", (sid, run, gid, value)
            )

    def live(self, sid):
        self.store(sid)
        with self.lock:
            return {
                "bindings": self.bindings(sid),
                "positions": [
                    dict(r)
                    for r in self.db.execute(
                        """SELECT p.*,COALESCE(r.role,'unknown') AS role FROM positions p
                           LEFT JOIN roles r ON p.store=r.store AND p.run=r.run AND p.gid=r.gid
                           WHERE p.store=? AND p.ts>? ORDER BY p.ts DESC LIMIT 500""",
                        (sid, time.time() - 10),
                    )
                ],
                "layouts": self.layouts(sid),
            }
