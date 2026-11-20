#!/usr/bin/env python3
"""Package an array store as release assets, and unpack a pinned release.

Standard library only, so the Vercel build can run it with whatever python3 the
image has and the GitHub workflow can run it before anything is installed.

  package   STORE_DIR OUT_DIR --tag T --base-url U ...   -> OUT_DIR/*.tar.gz[.partNN],
            sha256 files and release.json (the manifest)
  fetch     MANIFEST DEST                                -> download, verify, unpack

The archive is one gzip level-1 tar stream of the store's contents (paths relative
to the store, `meta.json` last so a half-unpacked store never looks finished). Level
1 because most of the 930 MB is incompressible (detail files are already gzip) while
the vectors and columns shrink by about half, for a couple of seconds of CPU. If the
compressed stream would pass ``--max-part-bytes`` (GitHub caps a release asset at
2 GiB) it is cut into numbered parts that are simply concatenated back together.

`fetch` streams: bytes go from the network through the hash and gzip straight into
tarfile, so nothing is held on disk twice. It unpacks into ``DEST.partial`` and
renames to DEST only after every part's sha256 and size match the manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import IO, Any

MANIFEST_VERSION = 1
#: Exit status of `fetch` when the manifest is `null`: nothing is pinned yet.
EXIT_NOT_PINNED = 10
DEFAULT_MAX_PART = 1_900 * 1024 * 1024  # under GitHub's 2 GiB asset limit
CHUNK = 1024 * 1024


# --------------------------------------------------------------------- package


class _PartWriter:
    """File-like sink that rolls over to a new numbered file past ``max_bytes``."""

    def __init__(self, stem: Path, max_bytes: int) -> None:
        self._stem = stem
        self._max = max_bytes
        self.parts: list[dict[str, Any]] = []
        self._fh: IO[bytes] | None = None
        self._hash = hashlib.sha256()
        self._size = 0

    def _open(self) -> None:
        path = self._stem.parent / f"{self._stem.name}.part{len(self.parts):02d}"
        self._fh = path.open("wb")
        self._hash = hashlib.sha256()
        self._size = 0
        self.parts.append({"path": path})

    def _close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self.parts[-1].update(sha256=self._hash.hexdigest(), bytes=self._size)
            self._fh = None

    def write(self, data: bytes) -> int:
        view = memoryview(data)
        while len(view):
            if self._fh is None or self._size >= self._max:
                self._close()
                self._open()
            assert self._fh is not None
            room = self._max - self._size
            piece = view[:room]
            self._fh.write(piece)
            self._hash.update(piece)
            self._size += len(piece)
            view = view[len(piece):]
        return len(data)

    def flush(self) -> None:
        if self._fh is not None:
            self._fh.flush()

    def finish(self) -> list[dict[str, Any]]:
        self._close()
        return self.parts


def _store_files(store: Path) -> list[Path]:
    files = sorted(p for p in store.rglob("*") if p.is_file())
    # meta.json is the "complete" marker, so it goes last.
    return [p for p in files if p.name != "meta.json" or p.parent != store] + [store / "meta.json"]


def _normalise(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mtime = 0
    info.mode = 0o644 if info.isfile() else 0o755
    return info


def cmd_package(args: argparse.Namespace) -> int:
    store = Path(args.store)
    meta_path = store / "meta.json"
    if not meta_path.exists():
        print(f"error: {meta_path} is missing; the store was not finalized", file=sys.stderr)
        return 2
    meta = json.loads(meta_path.read_text())
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    base = args.name

    stem = out / f"{base}.tar.gz"
    writer = _PartWriter(stem, args.max_part_bytes)
    unpacked = 0
    with tarfile.open(fileobj=writer, mode="w|gz", compresslevel=1) as tar:  # type: ignore[call-overload]
        for path in _store_files(store):
            info = _normalise(tar.gettarinfo(str(path), arcname=path.relative_to(store).as_posix()))
            unpacked += info.size
            with path.open("rb") as fh:
                tar.addfile(info, fh)
    parts = writer.finish()

    # One part keeps the plain name; several keep their .partNN suffix.
    assets: list[dict[str, Any]] = []
    for i, part in enumerate(parts):
        src: Path = part["path"]
        name = stem.name if len(parts) == 1 else src.name
        if len(parts) == 1:
            src.rename(out / name)
        (out / f"{name}.sha256").write_text(f"{part['sha256']}  {name}\n")
        assets.append(
            {
                "name": name,
                "url": f"{args.base_url.rstrip('/')}/{name}",
                "sha256": part["sha256"],
                "bytes": part["bytes"],
            }
        )

    counts = meta.get("counts", {})
    manifest = {
        "schema": MANIFEST_VERSION,
        "tag": args.tag,
        "statsbomb_commit": args.statsbomb_commit,
        "code_sha": args.code_sha,
        "encodings": sorted(meta.get("encodings", {})),
        "matches": counts.get("matches"),
        "events": counts.get("events"),
        "frames": counts.get("frames"),
        "format": "tar+gzip",
        "unpacked_bytes": unpacked,
        "packed_bytes": sum(a["bytes"] for a in assets),
        "assets": assets,
    }
    (out / "release.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


# ----------------------------------------------------------------------- fetch


class _Concat:
    """Read several byte streams one after another as a single stream."""

    def __init__(self, streams: Iterator[IO[bytes]]) -> None:
        self._streams = streams
        self._cur: IO[bytes] | None = None

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = CHUNK
        while True:
            if self._cur is None:
                self._cur = next(self._streams, None)
                if self._cur is None:
                    return b""
            data = self._cur.read(n)
            if data:
                return data
            self._cur = None


class _Verified:
    """Wraps a stream, hashing and counting what passes through."""

    def __init__(self, raw: IO[bytes]) -> None:
        self._raw = raw
        self.hash = hashlib.sha256()
        self.size = 0

    def read(self, n: int = -1) -> bytes:
        data = self._raw.read(n)
        self.hash.update(data)
        self.size += len(data)
        return data


def _open_url(url: str) -> tuple[IO[bytes], Any]:
    """Stream a URL with `curl -fL --retry 5` when curl exists, else urllib."""

    if shutil.which("curl"):
        proc = subprocess.Popen(
            ["curl", "-fsSL", "--retry", "5", "--retry-delay", "2", "--retry-all-errors", url],
            stdout=subprocess.PIPE,
        )
        assert proc.stdout is not None
        return proc.stdout, proc
    resp = urllib.request.urlopen(url, timeout=60)
    return resp, resp


def _download_unpack(manifest: dict[str, Any], partial: Path) -> None:
    streams: list[_Verified] = []
    procs: list[Any] = []

    def gen() -> Iterator[IO[bytes]]:
        for asset in manifest["assets"]:
            print(f"fetching {asset['url']} ({asset['bytes'] / 1e6:.0f} MB)", flush=True)
            raw, handle = _open_url(asset["url"])
            procs.append(handle)
            v = _Verified(raw)
            streams.append(v)
            yield v  # type: ignore[misc]

    reader = _Concat(gen())
    with tarfile.open(fileobj=reader, mode="r|gz") as tar:  # type: ignore[call-overload]
        for member in tar:
            target = (partial / member.name).resolve()
            if not target.is_relative_to(partial.resolve()) or not (
                member.isfile() or member.isdir()
            ):
                raise RuntimeError(f"refusing archive member {member.name!r}")
            if hasattr(tarfile, "data_filter"):
                tar.extract(member, partial, filter="data")
            else:  # older interpreters; the member check above is the guard
                tar.extract(member, partial)
    # Drain whatever the gzip trailer left so the hashes cover every byte.
    while reader.read(CHUNK):
        pass
    for handle in procs:
        if isinstance(handle, subprocess.Popen) and handle.wait() != 0:
            raise RuntimeError(f"curl exited with status {handle.returncode}")
    for asset, v in zip(manifest["assets"], streams, strict=True):
        if v.size != asset["bytes"]:
            raise RuntimeError(f"{asset['name']}: {v.size} bytes, expected {asset['bytes']}")
        if v.hash.hexdigest() != asset["sha256"]:
            raise RuntimeError(f"{asset['name']}: sha256 mismatch")
    if not (partial / "meta.json").exists():
        raise RuntimeError("archive has no meta.json")


def cmd_fetch(args: argparse.Namespace) -> int:
    manifest = json.loads(Path(args.manifest).read_text())
    if manifest is None:
        print("no data release pinned (data/release.json is null)")
        return EXIT_NOT_PINNED
    if manifest.get("schema") != MANIFEST_VERSION or not manifest.get("assets"):
        print("error: unrecognised release manifest", file=sys.stderr)
        return 2
    dest = Path(args.dest)
    partial = dest.with_name(dest.name + ".partial")
    started = time.time()
    for attempt in range(1, args.attempts + 1):
        shutil.rmtree(partial, ignore_errors=True)
        partial.mkdir(parents=True)
        try:
            _download_unpack(manifest, partial)
            break
        except Exception as exc:  # noqa: BLE001 - retry any transport or integrity failure
            print(f"attempt {attempt}/{args.attempts} failed: {exc}", file=sys.stderr)
            shutil.rmtree(partial, ignore_errors=True)
            if attempt == args.attempts:
                return 1
            time.sleep(2 * attempt)
    shutil.rmtree(dest, ignore_errors=True)
    os.replace(partial, dest)
    print(
        f"unpacked {manifest.get('tag')} into {dest} "
        f"({manifest.get('frames')} frames, {manifest.get('matches')} matches) "
        f"in {time.time() - started:.1f}s"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("package", help="tar+gzip a finalized store and write the manifest")
    q.add_argument("store")
    q.add_argument("out_dir")
    q.add_argument("--name", required=True, help="asset base name, e.g. pitchscanner-data-4b73468-25cf143")
    q.add_argument("--tag", required=True)
    q.add_argument("--base-url", required=True, help="where the assets will be downloadable")
    q.add_argument("--statsbomb-commit", default=None)
    q.add_argument("--code-sha", default=None)
    q.add_argument("--max-part-bytes", type=int, default=DEFAULT_MAX_PART)
    q.set_defaults(fn=cmd_package)
    f = sub.add_parser("fetch", help="download, verify and unpack a pinned release")
    f.add_argument("manifest", help="data/release.json")
    f.add_argument("dest")
    f.add_argument("--attempts", type=int, default=3)
    f.set_defaults(fn=cmd_fetch)
    args = p.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    sys.exit(main())
