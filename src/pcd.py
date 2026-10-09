"""Read an unorganized PCD into an (N, 3) array of x, y, z.

The bytes are used as stored. A Viam camera's GetPointCloud PCD is in meters,
which is the same numbers as a .pcd file before viam-server scales it internally.
"""

from __future__ import annotations

import numpy as np


def parse_pcd(data: bytes) -> np.ndarray:
    newline = data.find(b"\n", data.find(b"DATA"))
    if newline < 0:
        raise ValueError("point cloud is not a PCD")
    header = data[:newline].decode("ascii", errors="replace")
    body = data[newline + 1 :]
    meta: dict[str, list[str]] = {}
    for line in header.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            meta[parts[0]] = parts[1:]

    fields = meta.get("FIELDS", [])
    try:
        ix, iy, iz = fields.index("x"), fields.index("y"), fields.index("z")
    except ValueError as err:
        raise ValueError("PCD must contain x y z fields") from err

    sizes = [int(value) for value in meta.get("SIZE", [])]
    types = meta.get("TYPE", [])
    counts = [int(value) for value in meta.get("COUNT", ["1"] * len(fields))]
    if not (len(sizes) == len(types) == len(counts) == len(fields)):
        raise ValueError("PCD field header is incomplete")

    npoints = int(meta["POINTS"][0]) if "POINTS" in meta else int(meta["WIDTH"][0])
    kind = meta.get("DATA", ["ascii"])[0]
    if kind == "ascii":
        return _parse_ascii(body, npoints, ix, iy, iz)
    if kind == "binary":
        return _parse_binary(body, npoints, sizes, types, counts, ix, iy, iz)
    raise ValueError(f"unsupported PCD data {kind!r}")


def _parse_ascii(body: bytes, npoints: int, ix: int, iy: int, iz: int) -> np.ndarray:
    points = np.empty((npoints, 3), dtype=np.float64)
    written = 0
    for line in body.splitlines():
        if not line.strip():
            continue
        cols = line.split()
        points[written] = (float(cols[ix]), float(cols[iy]), float(cols[iz]))
        written += 1
        if written == npoints:
            break
    return points[:written]


def _parse_binary(
    body: bytes,
    npoints: int,
    sizes: list[int],
    types: list[str],
    counts: list[int],
    ix: int,
    iy: int,
    iz: int,
) -> np.ndarray:
    step = sum(size * count for size, count in zip(sizes, counts))
    raw = np.frombuffer(body[: npoints * step], dtype=np.uint8)
    if raw.size < npoints * step:
        npoints = raw.size // step
        raw = raw[: npoints * step]
    rows = raw.reshape(npoints, step)
    offsets: list[int] = []
    cursor = 0
    for size, count in zip(sizes, counts):
        offsets.append(cursor)
        cursor += size * count
    columns = [_field(rows, offsets[i], sizes[i], types[i]) for i in (ix, iy, iz)]
    return np.stack(columns, axis=1)


def _field(rows: np.ndarray, offset: int, size: int, kind: str) -> np.ndarray:
    dtype = _dtype(size, kind)
    column = np.ascontiguousarray(rows[:, offset : offset + size])
    return column.view(dtype).reshape(-1).astype(np.float64)


def _dtype(size: int, kind: str) -> str:
    table = {
        ("F", 4): "<f4",
        ("F", 8): "<f8",
        ("I", 1): "<i1",
        ("I", 2): "<i2",
        ("I", 4): "<i4",
        ("I", 8): "<i8",
        ("U", 1): "<u1",
        ("U", 2): "<u2",
        ("U", 4): "<u4",
        ("U", 8): "<u8",
    }
    try:
        return table[(kind, size)]
    except KeyError as err:
        raise ValueError(f"unsupported PCD field type {kind} size {size}") from err
