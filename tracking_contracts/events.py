"""Bounded asynchronous JSONL publication and independent durable cursors.

Only plain observations are published: never embeddings, model objects or URLs.
An acknowledged manifest points only at fsynced complete records.
"""

import json
import os
from pathlib import Path
import queue
import re
import threading
import time

from tracking_contracts import SCHEMA_VERSION


def encode(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def frame_event(camera_id, seq, timestamp, width, height, tracks, assignments, stats):
    return {
        "type": "frame",
        "camera_id": int(camera_id),
        "seq": int(seq),
        "timestamp": float(timestamp),
        "timestamp_source": "receiver",
        "width": int(width),
        "height": int(height),
        "inference_skipped": int(stats.get("inference_skipped", 0)),
        "observations": [
            {
                "local_id": int(t["local_id"]),
                "global_id": assignments.get(t["local_id"]),
                "bbox": [float(v) for v in t["bbox"]],
                "quality": float(t.get("feature_quality", 0)),
            }
            for t in tracks
        ],
    }


class EventWriter:
    def __init__(
        self,
        directory,
        run_id,
        max_records=2048,
        max_bytes=16 * 1024 * 1024,
        segment_bytes=16 * 1024 * 1024,
        sync_interval=1.0,
    ):
        self.directory = Path(directory)
        self.run_id = str(run_id)
        self.queue = queue.Queue(maxsize=max_records)
        self.max_bytes, self.pending_bytes = max_bytes, 0
        self.segment_bytes, self.sync_interval = segment_bytes, sync_interval
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.dropped, self.error, self.count = 0, None, 0
        self.thread = threading.Thread(
            target=self._run, name="tracking-event-writer", daemon=True
        )
        self.thread.start()

    def submit(self, event):
        # Serialization snapshots the record; callers may reuse their dictionaries.
        with self.lock:
            if self.stop.is_set() or self.error:
                self.dropped += 1
                return False
            self.count += 1
            data = (
                encode(
                    {
                        **event,
                        "schema_version": SCHEMA_VERSION,
                        "run_id": self.run_id,
                        "record_id": f"{self.run_id}:{self.count}",
                        "published_at": time.time(),
                        "publisher_dropped": self.dropped,
                    }
                )
                + "\n"
            ).encode()
            if (
                len(data) > 512 * 1024
                or self.pending_bytes + len(data) > self.max_bytes
            ):
                self.dropped += 1
                return False
            try:
                self.queue.put_nowait(data)
            except queue.Full:
                self.dropped += 1
                return False
            self.pending_bytes += len(data)
            return True

    def _run(self):
        output = None
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            # A run must never silently overwrite an existing stream.
            output = (self.directory / "000000.jsonl").open("xb")
            segments, index, size = {}, 0, 0
            hour, last_sync = int(time.time() // 3600), 0.0

            def checkpoint(closed=False):
                output.flush()
                os.fsync(output.fileno())
                segments[str(index)] = size
                manifest = {
                    "schema_version": SCHEMA_VERSION,
                    "run_id": self.run_id,
                    "segments": segments,
                    "closed": closed,
                    "updated_at": time.time(),
                    "dropped": self.dropped,
                }
                temporary = self.directory / "manifest.tmp"
                with temporary.open("w", encoding="utf-8") as f:
                    f.write(encode(manifest))
                    f.flush()
                    os.fsync(f.fileno())
                temporary.replace(self.directory / "manifest.json")

            while not self.stop.is_set() or not self.queue.empty():
                try:
                    data = self.queue.get(timeout=0.1)
                except queue.Empty:
                    data = None
                if data is not None:
                    with self.lock:
                        self.pending_bytes -= len(data)
                    if size and (
                        size + len(data) > self.segment_bytes
                        or int(time.time() // 3600) != hour
                    ):
                        checkpoint()
                        output.close()
                        index += 1
                        size, hour = 0, int(time.time() // 3600)
                        output = (self.directory / f"{index:06d}.jsonl").open("xb")
                    output.write(data)
                    size += len(data)
                if time.monotonic() - last_sync >= self.sync_interval:
                    checkpoint()
                    last_sync = time.monotonic()
            checkpoint(closed=True)
        except Exception as exc:
            self.error = type(
                exc
            ).__name__  # do not expose arbitrary filesystem details
        finally:
            if output:
                output.close()

    def close(self):
        self.stop.set()
        self.thread.join(timeout=3)
        if self.thread.is_alive():
            self.error = "FlushTimeout"


def read_page(directory, cursor="0:0", limit=250):
    if not re.fullmatch(r"\d{1,8}:\d{1,16}", cursor) or not 1 <= limit <= 1000:
        raise ValueError("无效的事件游标或页大小。")
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        return {
            "schema_version": SCHEMA_VERSION,
            "available": False,
            "events": [],
            "cursor": cursor,
            "closed": False,
        }
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    index, offset = map(int, cursor.split(":"))
    segments = manifest["segments"]
    if str(index) not in segments or offset > segments[str(index)]:
        raise ValueError("事件游标已失效或超出已保存范围。")
    events, total = [], 0
    while len(events) < limit and total < 1024 * 1024:
        end = segments[str(index)]
        if offset == end:
            if str(index + 1) not in segments:
                break
            index, offset = index + 1, 0
            continue
        with (directory / f"{index:06d}.jsonl").open("rb") as f:
            # Validate boundaries: clients cannot start parsing from the middle of a line.
            if offset:
                f.seek(offset - 1)
                if f.read(1) != b"\n":
                    raise ValueError("事件游标不在记录边界。")
            f.seek(offset)
            line = f.readline(min(end - offset, 512 * 1024 + 1))
        if not line.endswith(b"\n"):
            raise ValueError("已发布事件段不完整，无法安全读取。")
        events.append(json.loads(line))
        offset += len(line)
        total += len(line)
    return {
        "schema_version": SCHEMA_VERSION,
        "available": True,
        "events": events,
        "cursor": f"{index}:{offset}",
        "closed": manifest["closed"],
        "caught_up": index == max(map(int, segments))
        and offset == segments[str(index)],
        "dropped": manifest.get("dropped", 0),
        "updated_at": manifest["updated_at"],
    }
