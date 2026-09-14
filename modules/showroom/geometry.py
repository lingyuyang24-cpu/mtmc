"""Ground-plane calibration, intentionally independent of detection models."""

import math
import numpy as np


def inside(point, polygon):
    x, y = point
    result = False
    for a, b in zip(polygon, polygon[1:] + polygon[:1]):
        cross = (b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0])
        if (
            abs(cross) < 1e-8
            and min(a[0], b[0]) - 1e-8 <= x <= max(a[0], b[0]) + 1e-8
            and min(a[1], b[1]) - 1e-8 <= y <= max(a[1], b[1]) + 1e-8
        ):
            return True
        if (a[1] > y) != (b[1] > y) and x < (b[0] - a[0]) * (y - a[1]) / (
            b[1] - a[1]
        ) + a[0]:
            result = not result
    return result


def hull(points):
    points = sorted(set(tuple(p) for p in points))

    def turn(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    lower, upper = [], []
    for p in points:
        while len(lower) >= 2 and turn(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(points):
        while len(upper) >= 2 and turn(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return [list(p) for p in lower[:-1] + upper[:-1]]


def polygon_valid(points):
    if len(set(map(tuple, points))) != len(points):
        return False
    area = sum(
        a[0] * b[1] - b[0] * a[1] for a, b in zip(points, points[1:] + points[:1])
    )
    if abs(area) < 1e-6:
        return False

    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    edges = list(zip(points, points[1:] + points[:1]))
    for i, (a, b) in enumerate(edges):
        for j, (c, d) in enumerate(edges):
            if j <= i or j == i + 1 or (i == 0 and j == len(edges) - 1):
                continue
            overlaps = max(min(a[0], b[0]), min(c[0], d[0])) <= min(
                max(a[0], b[0]), max(c[0], d[0])
            ) and max(min(a[1], b[1]), min(c[1], d[1])) <= min(
                max(a[1], b[1]), max(c[1], d[1])
            )
            if (
                overlaps
                and orient(a, b, c) * orient(a, b, d) <= 0
                and orient(c, d, a) * orient(c, d, b) <= 0
            ):
                return False
    return True


def calibrate(points):
    source = np.asarray([p[:2] for p in points], dtype=float)
    target = np.asarray([p[2:] for p in points], dtype=float)

    def normalize(values):
        mean = values.mean(axis=0)
        spread = np.sqrt(((values - mean) ** 2).sum(axis=1)).mean()
        if spread < 1e-8:
            raise ValueError("标定点重合，无法计算。")
        scale = math.sqrt(2) / spread
        matrix = np.array(
            [[scale, 0, -mean[0] * scale], [0, scale, -mean[1] * scale], [0, 0, 1.0]]
        )
        return (values - mean) * scale, matrix

    s, st = normalize(source)
    t, tt = normalize(target)
    rows = []
    for (x, y), (u, v) in zip(s, t):
        rows.extend(
            [
                [-x, -y, -1, 0, 0, 0, u * x, u * y, u],
                [0, 0, 0, -x, -y, -1, v * x, v * y, v],
            ]
        )
    a = np.array(rows)
    if np.linalg.matrix_rank(a) < 8 or len(hull(source.tolist())) < 3:
        raise ValueError("标定点共线或退化，请选择分散的地面对应点。")
    _, _, vt = np.linalg.svd(a, full_matrices=True)
    matrix = np.linalg.inv(tt) @ vt[-1].reshape(3, 3) @ st
    matrix /= np.linalg.norm(matrix)
    if abs(np.linalg.det(matrix)) < 1e-12:
        raise ValueError("标定矩阵不可逆。")
    denominators = np.c_[source, np.ones(len(source))] @ matrix[2]
    if not (np.all(denominators > 1e-10) or np.all(denominators < -1e-10)):
        raise ValueError("映射在控制点范围内穿过无穷远，请核对地面对应点顺序。")
    projected = [project(matrix.tolist(), p) for p in source]
    if any(p is None for p in projected):
        raise ValueError("标定无有效映射。")
    error = float(np.sqrt(((np.asarray(projected) - target) ** 2).sum(axis=1)).max())
    if error > 0.5:
        raise ValueError(
            f"控制点最大误差 {error:.2f} 米，超过 0.5 米，请核对对应关系。"
        )
    return {
        "matrix": matrix.tolist(),
        "valid_polygon": hull(source.tolist()),
        "fit_error_m": error,
    }


def project(matrix, point):
    v = np.asarray(matrix) @ [point[0], point[1], 1.0]
    if abs(v[2]) < 1e-10:
        return None
    result = (v[:2] / v[2]).tolist()
    return result if all(math.isfinite(x) for x in result) else None
