"""Deterministic daily reports; no model inference and no feature storage."""

import base64
from collections import defaultdict
from datetime import date, datetime, timedelta
import hashlib
from html import escape
import json
import time
from zoneinfo import ZoneInfo

from modules.showroom.repository import dumps


def union(intervals):
    result = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if result and start <= result[-1][1] + 1e-6:
            result[-1][1] = max(end, result[-1][1])
        else:
            result.append([start, end])
    return result


def duration(intervals):
    return sum(b - a for a, b in union(intervals))


def subtract(intervals, removed):
    result = []
    cuts = union(removed)
    for start, end in union(intervals):
        for a, b in cuts:
            if b <= start:
                continue
            if a >= end:
                break
            if a > start:
                result.append([start, a])
            start = max(start, b)
        if start < end:
            result.append([start, end])
    return result


def exclusive(raw):
    """Remove simultaneous attribution to different cars, not guess the right camera."""
    by_person = defaultdict(list)
    for (run, gid, vehicle), intervals in raw.items():
        for a, b in union(intervals):
            by_person[(run, gid)].extend([(a, vehicle, 1), (b, vehicle, -1)])
    conflicts = {}
    for person, changes in by_person.items():
        active, last, cuts = defaultdict(int), None, []
        for ts, vehicle, delta in sorted(changes):
            if (
                last is not None
                and ts > last
                and sum(n > 0 for n in active.values()) > 1
            ):
                cuts.append([last, ts])
            active[vehicle] += delta
            last = ts
        conflicts[person] = union(cuts)
    return {
        key: subtract(values, conflicts[key[:2]]) for key, values in raw.items()
    }, sum(duration(v) for v in conflicts.values())


def bounds(day, timezone):
    d, tz = date.fromisoformat(day), ZoneInfo(timezone)
    if d.isoformat() != day:
        raise ValueError("日期格式必须为 YYYY-MM-DD。")
    return (
        datetime.combine(d, datetime.min.time(), tz).timestamp(),
        datetime.combine(d + timedelta(days=1), datetime.min.time(), tz).timestamp(),
    )


def build_report(repo, sid, day, now=None):
    now = time.time() if now is None else now
    store = repo.store(sid)
    start, end = bounds(day, store["timezone"])
    if start > now:
        raise ValueError("不能生成未来日期的报告。")
    tz = ZoneInfo(store["timezone"])
    with repo.lock:
        # All mutations use this lock; this is a consistent in-process read snapshot.
        db = repo.db
        layouts_all = repo.layouts(sid)
        layouts = []
        for i, layout in enumerate(layouts_all):
            stop = (
                layouts_all[i + 1]["effective_at"]
                if i + 1 < len(layouts_all)
                else float("inf")
            )
            if layout["effective_at"] < end and stop > start:
                layouts.append(
                    {
                        **layout,
                        "valid_from": max(start, layout["effective_at"]),
                        "valid_until": min(end, stop),
                    }
                )
        intervals = [
            dict(r)
            for r in db.execute(
                "SELECT * FROM intervals WHERE store=? AND t0<? AND t1>?",
                (sid, end, start),
            )
        ]
        presence = [
            dict(r)
            for r in db.execute(
                "SELECT * FROM presence WHERE store=? AND day=?", (sid, day)
            )
        ]
        counters = [
            dict(r)
            for r in db.execute(
                "SELECT * FROM counters WHERE store=? AND day=?", (sid, day)
            )
        ]
        roles = {
            (r["run"], r["gid"]): r["role"]
            for r in db.execute("SELECT * FROM roles WHERE store=?", (sid,))
        }
        exits = [
            dict(r)
            for r in db.execute(
                "SELECT * FROM exits WHERE store=? AND ts>=? AND ts<?",
                (sid, start, end),
            )
        ]
        cover = [
            dict(r)
            for r in db.execute(
                "SELECT * FROM coverage WHERE store=? AND t0<? AND t1>?",
                (sid, end, start),
            )
        ]
        bindings = repo.bindings(sid)
    entities = {(r["run"], r["gid"]) for r in presence} | {
        (r["run"], r["gid"]) for r in intervals
    }
    staff = {k for k in entities if roles.get(k) == "staff"}
    people = entities - staff
    vehicles, raw = {}, defaultdict(list)
    for layout in layouts:
        for vehicle in layout["vehicles"]:
            key = vehicle["id"]
            vehicles.setdefault(
                key,
                {
                    "id": key,
                    "name": vehicle["name"],
                    "model": vehicle["model"],
                    "people": set(),
                    "effective_people": set(),
                    "returning_people": set(),
                    "valid_seconds": 0.0,
                    "entries": 0,
                    "revisits": 0,
                    "durations": [],
                    "layouts": [],
                },
            )
            vehicles[key]["layouts"].append(layout["version"])
    for item in intervals:
        identity = (item["run"], item["gid"])
        if identity in staff or item["vehicle"] not in vehicles:
            continue
        raw[(item["run"], item["gid"], item["vehicle"])].append(
            (max(start, item["t0"]), min(end, item["t1"]))
        )
    raw, conflict_seconds = exclusive(raw)
    episodes, hourly = [], []
    t = start
    while t < end:
        hourly.append(
            {
                "start": t,
                "end": min(t + 3600, end),
                "label": datetime.fromtimestamp(t, tz).isoformat(),
                "valid_seconds": 0.0,
                "people": set(),
            }
        )
        t += 3600
    for (run, gid, vehicle), values in sorted(raw.items()):
        merged = union(values)
        groups = []
        for interval in merged:
            if groups and interval[0] - groups[-1][-1][1] <= 3:
                groups[-1].append(interval)
            else:
                groups.append([interval])
        previous_end = None
        for group in groups:
            seconds = duration(group)
            first, last = group[0][0], group[-1][1]
            effective = seconds >= store["min_dwell"]
            is_return = (
                effective
                and previous_end is not None
                and first - previous_end >= store["revisit_gap"]
                and any(
                    x["run"] == run
                    and x["gid"] == gid
                    and x["vehicle"] == vehicle
                    and previous_end <= x["ts"] <= first
                    for x in exits
                )
            )
            person = f"{run}:{gid}"
            episode = {
                "person_id": person,
                "run_id": run,
                "global_id": gid,
                "vehicle_id": vehicle,
                "start": first,
                "end": last,
                "valid_seconds": round(seconds, 3),
                "unknown_seconds": round(last - first - seconds, 3),
                "effective": effective,
                "revisit": bool(is_return),
                "role": roles.get((run, gid), "unknown"),
            }
            episodes.append(episode)
            metric = vehicles[vehicle]
            metric["people"].add(person)
            metric["valid_seconds"] += seconds
            metric["durations"].append(seconds)
            if effective:
                metric["effective_people"].add(person)
                metric["entries"] += 1
                previous_end = last
            if is_return:
                metric["returning_people"].add(person)
                metric["revisits"] += 1
            for hour in hourly:
                seconds_in_hour = duration(
                    [(max(a, hour["start"]), min(b, hour["end"])) for a, b in group]
                )
                hour["valid_seconds"] += seconds_in_hour
                if seconds_in_hour:
                    hour["people"].add(person)
    metrics = []
    for value in vehicles.values():
        durations = sorted(value.pop("durations"))
        value["people"] = len(value["people"])
        value["effective_people"] = len(value["effective_people"])
        returning = len(value.pop("returning_people"))
        value["revisit_rate"] = (
            round(returning / value["effective_people"], 4)
            if value["effective_people"]
            else None
        )
        value["valid_seconds"] = round(value["valid_seconds"], 3)
        n = len(durations)
        value["median_seconds"] = (
            round((durations[(n - 1) // 2] + durations[n // 2]) / 2, 3) if n else None
        )
        metrics.append(value)
    for hour in hourly:
        hour["people"] = len(hour["people"])
        hour["valid_seconds"] = round(hour["valid_seconds"], 3)
        hour["observed_seconds"] = round(
            duration(
                [
                    (max(hour["start"], r["t0"]), min(hour["end"], r["t1"]))
                    for r in cover
                ]
            ),
            3,
        )
        hour["has_data"] = hour["observed_seconds"] > 0

    def clock(text):
        return datetime.fromisoformat(day + "T" + text).replace(tzinfo=tz).timestamp()

    expected_start, expected_end = clock(store["opens"]), min(
        clock(store["closes"]), now
    )
    camera_ids = {c["camera_id"] for layout in layouts for c in layout["cameras"]}
    coverage = []
    for camera in sorted(camera_ids):
        available = union(
            [
                (max(expected_start, r["t0"]), min(expected_end, r["t1"]))
                for r in cover
                if r["camera"] == camera
            ]
        )
        expected = max(0, expected_end - expected_start)
        coverage.append(
            {
                "camera_id": camera,
                "seconds": round(duration(available), 3),
                "expected_seconds": round(expected, 3),
                "ratio": round(duration(available) / expected, 4) if expected else None,
            }
        )
    source_runs = sorted({r["run"] for r in counters})
    warnings = [
        "人员框底边中点为地面近似；接近车辆不等于购买意愿。",
        "时间采用服务器接收时间，尚未提供硬件级跨摄像头同步。",
        "人数为模型关联会话数；未标为员工的未知角色仍包括在人员统计中。",
        "停留秒数包含可靠观察到的短暂停留；阈值仅用于有效访问和回访判定。",
    ]
    if len(source_runs) > 1:
        warnings.append(
            "当天包含多个追踪运行；特征仅内存，跨运行无法自动去重，人数可能重复。"
        )
    if sum(r["unknown"] for r in counters):
        warnings.append("部分观察未标定、被截断、质量不足或区域重叠，未计入展车停留。")
    if conflict_seconds:
        warnings.append(
            "同一身份被不同摄像头同时分配到不同车辆的冲突时段已剔除，未重复计时。"
        )
    if sum(r["unlinked"] for r in counters):
        warnings.append("尚未获得全局身份的观察不计入人数与停留，暂不做历史身份回填。")
    warnings.append("跨日停留按当天时间截断，访问段数不等同于当天新入店次数。")
    if any(b["error"] or not b["caught_up"] for b in bindings if b["created"] < end):
        warnings.append("存在上游不可用或消费积压；日报可能需要补算。")
    dropped = sum(b["dropped"] for b in bindings if b["run"] in source_runs)
    no_data = not counters
    partial = (
        not layouts
        or not coverage
        or any((c["ratio"] or 0) < 0.99 for c in coverage)
        or dropped > 0
        or any(b["error"] or not b["caught_up"] for b in bindings if b["created"] < end)
    )
    status = (
        "draft"
        if now < end
        else "no_data" if no_data else "partial" if partial else "complete"
    )
    return {
        "report_schema": 1,
        "store_id": sid,
        "store_name": store["name"],
        "date": day,
        "timezone": store["timezone"],
        "period": {
            "start": start,
            "end": end,
            "opens": store["opens"],
            "closes": store["closes"],
        },
        "status": status,
        "source_runs": source_runs,
        "rule_version": "showroom-1.0",
        "rules": {k: store[k] for k in ("min_dwell", "max_gap", "revisit_gap")},
        "summary": {
            "people": len(people),
            "staff": len(staff),
            "unknown_role": sum(roles.get(k, "unknown") == "unknown" for k in people),
            "vehicles": len(metrics),
            "valid_seconds": round(sum(v["valid_seconds"] for v in metrics), 3),
            "frames": sum(r["frames"] for r in counters),
            "unlinked_observations": sum(r["unlinked"] for r in counters),
            "unmapped_observations": sum(r["unknown"] for r in counters),
            "skipped_frames": sum(r["skipped"] for r in counters),
            "publisher_dropped": dropped,
            "conflict_seconds": round(conflict_seconds, 3),
        },
        "vehicles": sorted(metrics, key=lambda v: -v["valid_seconds"]),
        "visits": sorted(episodes, key=lambda x: x["start"]),
        "hourly": hourly,
        "coverage": coverage,
        "layouts": layouts,
        "warnings": warnings,
        "input_checkpoints": [
            {k: r[k] for k in ("run", "camera", "frames", "last_record")}
            for r in counters
        ],
    }


STATUS = {
    "draft": "当日草稿",
    "complete": "统计已完成",
    "partial": "数据不完整",
    "no_data": "无有效数据",
}


def render_html(payload, repo):
    p, h = payload, escape
    tz = ZoneInfo(p["timezone"])

    def dt(value):
        return datetime.fromtimestamp(value, tz).strftime("%H:%M:%S")

    def table(headers, rows):
        return (
            "<table><thead><tr>"
            + "".join("<th>" + h(str(x)) + "</th>" for x in headers)
            + "</tr></thead><tbody>"
            + "".join(
                "<tr>" + "".join("<td>" + h(str(x)) + "</td>" for x in row) + "</tr>"
                for row in rows
            )
            + "</tbody></table>"
        )

    summary = p["summary"]
    cards = "".join(
        f'<div class="card"><small>{h(label)}</small><strong>{h(str(value))}</strong></div>'
        for label, value in [
            ("人员会话（非实名客户）", summary["people"]),
            ("展车数", summary["vehicles"]),
            ("有效停留 / 分钟", round(summary["valid_seconds"] / 60, 1)),
            ("员工会话", summary["staff"]),
        ]
    )
    vehicle_table = table(
        ["展车", "车型", "到访会话", "有效会话", "停留/分钟", "中位/秒", "回访次数"],
        [
            [
                v["name"],
                v["model"],
                v["people"],
                v["effective_people"],
                round(v["valid_seconds"] / 60, 2),
                v["median_seconds"] if v["median_seconds"] is not None else "—",
                v["revisits"],
            ]
            for v in p["vehicles"]
        ],
    )
    visit_table = table(
        ["运行 / GID", "角色", "展车", "开始", "结束", "有效/秒", "未知/秒", "回访"],
        [
            [
                v["run_id"][:8] + " / " + str(v["global_id"]),
                v["role"],
                v["vehicle_id"],
                dt(v["start"]),
                dt(v["end"]),
                v["valid_seconds"],
                v["unknown_seconds"],
                "是" if v["revisit"] else "否",
            ]
            for v in p["visits"]
        ],
    )
    coverage_table = table(
        ["摄像头", "营业时段观测/秒", "应观测/秒", "覆盖率"],
        [
            [
                c["camera_id"],
                c["seconds"],
                c["expected_seconds"],
                f"{c['ratio']*100:.1f}%" if c["ratio"] is not None else "—",
            ]
            for c in p["coverage"]
        ],
    )
    hourly_table = table(
        ["时段", "区域停留人员会话", "有效停留/秒", "观测/秒"],
        [
            [
                v["label"],
                v["people"] if v["has_data"] else "未知",
                v["valid_seconds"] if v["has_data"] else "—",
                v["observed_seconds"],
            ]
            for v in p["hourly"]
        ],
    )
    plans = []
    metrics = {v["id"]: v for v in p["vehicles"]}
    for layout in p["layouts"]:
        aid = layout["floorplan_id"]
        try:
            repo.asset(p["store_id"], aid)
            data = base64.b64encode(
                (repo.root / "floorplans" / f"{aid}.png").read_bytes()
            ).decode()
        except (KeyError, OSError):
            plans.append("<p>此版本平面图不可用。</p>")
            continue
        shapes = []
        for vehicle in layout["vehicles"]:
            polygon = vehicle["polygon"]
            points = " ".join(f"{x},{y}" for x, y in polygon)
            x = sum(v[0] for v in polygon) / len(polygon)
            y = sum(v[1] for v in polygon) / len(polygon)
            label = f"{vehicle['name']} · {metrics[vehicle['id']]['valid_seconds']/60:.1f} 分钟（当日合计）"
            shapes.append(
                f'<polygon points="{points}" fill="#36b7a655" stroke="#087e75" stroke-width=".04"/><text x="{x}" y="{y}" text-anchor="middle" font-size="{layout["width_m"]*.018}">{h(label)}</text>'
            )
        plans.append(
            f'<h3>布局 {layout["version"]} · {dt(layout["valid_from"])} 起</h3><svg viewBox="0 0 {layout["width_m"]} {layout["height_m"]}" role="img" aria-label="展车区域停留分布"><image width="100%" height="100%" preserveAspectRatio="none" href="data:image/png;base64,{data}"/>'
            + "".join(shapes)
            + "</svg>"
        )
    return (
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'
        + h(p["date"] + " 展车日报")
        + """</title><style>
    body{font:15px/1.65 system-ui,sans-serif;background:#f2f5f6;color:#182d38;margin:0}main{max-width:1100px;margin:auto;padding:40px 24px}h1{font-size:32px}h2{margin-top:36px}small{color:#5e737c}.cards{display:flex;gap:14px;flex-wrap:wrap}.card{background:white;padding:20px;flex:1;min-width:150px;border-radius:12px}.card strong{display:block;font-size:30px}table{width:100%;border-collapse:collapse;background:white;font-size:13px}th,td{text-align:left;padding:10px;border-bottom:1px solid #dce5e8}th{background:#e4eeee}.scroll{overflow:auto}li{margin:7px 0}svg{max-height:520px;width:100%;background:white}footer{color:#5e737c;margin-top:30px}@media print{body{background:white}main{padding:0}tr{break-inside:avoid}h2{break-after:avoid}}
    </style><main><small>MTMC / SHOWROOM · 日报</small><h1>"""
        + h(p["store_name"])
        + " · "
        + h(p["date"])
        + "</h1><p>"
        + h(STATUS[p["status"]])
        + f' · 版本 {p["revision"]} · '
        + h(p["timezone"])
        + '</p><div class="cards">'
        + cards
        + '</div><h2>全部展车</h2><div class="scroll">'
        + vehicle_table
        + "</div><h2>平面图与区域停留</h2>"
        + "".join(plans)
        + '<h2>按小时统计</h2><div class="scroll">'
        + hourly_table
        + '</div><h2>当日访问明细</h2><div class="scroll">'
        + visit_table
        + "</div><h2>观测覆盖</h2>"
        + coverage_table
        + "<h2>数据质量与边界</h2><ul>"
        + "".join("<li>" + h(w) + "</li>" for w in p["warnings"])
        + "</ul><footer>特征仅存内存 · 本报告不含特征向量 · 断流、未知位置不计入有效停留。<br>生成时间："
        + h(datetime.fromtimestamp(p["generated_at"], tz).isoformat())
        + "</footer></main></html>"
    )


def generate(repo, sid, day, now=None):
    with repo.lock:
        payload = build_report(repo, sid, day, now)
        fingerprint = hashlib.sha256(dumps(payload).encode()).hexdigest()
        current = repo.db.execute(
            "SELECT * FROM reports WHERE store=? AND day=? ORDER BY revision DESC LIMIT 1",
            (sid, day),
        ).fetchone()
        if current and current["fingerprint"] == fingerprint:
            return json.loads(current["payload"])
        revision = current["revision"] + 1 if current else 1
        payload.update(
            revision=revision, generated_at=time.time() if now is None else now
        )
        html = render_html(payload, repo)
        # HTML + JSON in the same committed row avoids cross-file publication races.
        with repo.transaction() as db:
            db.execute(
                "INSERT INTO reports VALUES(?,?,?,?,?,?,?)",
                (
                    sid,
                    day,
                    revision,
                    fingerprint,
                    dumps(payload),
                    html,
                    payload["generated_at"],
                ),
            )
        return payload
