"""Download Mars 2020 Perseverance Navcam L/R raw images and JSON metadata.

Resumable, parallel, polite. Persists state under ``<data_root>/.state/``.

Programmatic entry point: :func:`fetch`. CLI exposure: ``rock-snitch fetch``.
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import math
import random
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from tqdm import tqdm


API = "https://mars.nasa.gov/rss/api/"
FEED_PARAMS = {
    "feed": "raw_images",
    "category": "mars2020",
    "feedtype": "json",
    "search": "NAVCAM_LEFT|NAVCAM_RIGHT",
    "order": "sol asc",
    "num": "100",
}
USER_AGENT = "rock-snitch-v2/0.1 (research; contact om.sanan007@gmail.com)"

DOWNLOAD_WORKERS = 8
INDEX_WORKERS = 1  # NASA's raw-images API 403s on concurrent index requests.
RETRIES = 6


@dataclass
class FetchPaths:
    """Filesystem layout for one fetch run."""

    root: Path
    img_dir: Path = field(init=False)
    meta_dir: Path = field(init=False)
    state_dir: Path = field(init=False)
    index_file: Path = field(init=False)
    pages_done_file: Path = field(init=False)
    total_file: Path = field(init=False)
    done_file: Path = field(init=False)
    fail_file: Path = field(init=False)
    log_file: Path = field(init=False)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.img_dir = self.root / "images"
        self.meta_dir = self.root / "metadata"
        self.state_dir = self.root / ".state"
        self.index_file = self.state_dir / "index.jsonl"
        self.pages_done_file = self.state_dir / "pages_done.txt"
        self.total_file = self.state_dir / "total_results.txt"
        self.done_file = self.state_dir / "done.txt"
        self.fail_file = self.state_dir / "failed.txt"
        self.log_file = self.state_dir / "log.txt"
        for d in (self.img_dir, self.meta_dir, self.state_dir):
            d.mkdir(parents=True, exist_ok=True)


_log_lock = threading.Lock()
_done_lock = threading.Lock()
_index_write_lock = threading.Lock()
_pages_lock = threading.Lock()


def _log(paths: FetchPaths, msg: str, *, to_console: bool = True) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    with _log_lock:
        if to_console:
            tqdm.write(line)
        with paths.log_file.open("a") as f:
            f.write(line + "\n")


def _http_get(url: str, *, timeout: int = 60, status_cb=None) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last_exc: Exception | None = None
    for attempt in range(RETRIES):
        if status_cb:
            status_cb(f"attempt {attempt+1}/{RETRIES}")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            last_exc = e
            if e.code in (403, 429, 503):
                sleep = min(240, 20 + (2 ** attempt) * 5 + random.random() * 15)
            else:
                sleep = min(60, (2 ** attempt) + random.random())
            if status_cb:
                status_cb(f"HTTP {e.code} sleep {sleep:.0f}s")
            time.sleep(sleep)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last_exc = e
            sleep = min(60, (2 ** attempt) + random.random())
            if status_cb:
                status_cb(f"{type(e).__name__} sleep {sleep:.0f}s")
            time.sleep(sleep)
    raise RuntimeError(f"GET failed after {RETRIES} retries: {url}: {last_exc}")


def _build_api_url(page: int) -> str:
    from urllib.parse import urlencode
    params = dict(FEED_PARAMS, page=str(page))
    return API + "?" + urlencode(params)


def _load_seen_ids(paths: FetchPaths) -> set[str]:
    seen: set[str] = set()
    if paths.index_file.exists():
        with paths.index_file.open() as f:
            for line in f:
                try:
                    seen.add(json.loads(line)["imageid"])
                except Exception:
                    pass
    return seen


def _load_pages_done(paths: FetchPaths) -> set[int]:
    if not paths.pages_done_file.exists():
        return set()
    with paths.pages_done_file.open() as f:
        return {int(line.strip()) for line in f if line.strip().isdigit()}


def _mark_page_done(paths: FetchPaths, page: int) -> None:
    with _pages_lock, paths.pages_done_file.open("a") as f:
        f.write(f"{page}\n")


def _load_done(paths: FetchPaths) -> set[str]:
    if not paths.done_file.exists():
        return set()
    with paths.done_file.open() as f:
        return {line.strip() for line in f if line.strip()}


def _mark_done(paths: FetchPaths, imageid: str) -> None:
    with _done_lock, paths.done_file.open("a") as f:
        f.write(imageid + "\n")


def _mark_failed(paths: FetchPaths, imageid: str, reason: str) -> None:
    with _done_lock, paths.fail_file.open("a") as f:
        f.write(f"{imageid}\t{reason}\n")


def _fetch_page(paths: FetchPaths, page: int, seen: set[str], bar: tqdm) -> tuple[int, int, Optional[int]]:
    url = _build_api_url(page)
    t0 = time.time()
    raw = _http_get(
        url,
        timeout=180,
        status_cb=lambda s, p=page: bar.set_postfix_str(f"p{p} {s}"),
    )
    data = json.loads(raw)
    total_results = int(data.get("total_results") or 0) or None
    images = data.get("images") or []
    new = 0
    if images:
        with _index_write_lock, paths.index_file.open("a") as f:
            for img in images:
                iid = img.get("imageid")
                if not iid or iid in seen:
                    continue
                f.write(json.dumps(img) + "\n")
                seen.add(iid)
                new += 1
    _mark_page_done(paths, page)
    bar.set_postfix_str(f"p{page} +{new} in {time.time()-t0:.0f}s")
    return page, new, total_results


def build_index(paths: FetchPaths, *, workers: int = INDEX_WORKERS) -> int:
    """Paginate the raw-images API and write all records to ``index.jsonl``."""
    per_page = int(FEED_PARAMS["num"])
    seen = _load_seen_ids(paths)
    pages_done = _load_pages_done(paths)
    _log(paths, f"Index: resuming - records={len(seen)} pages_done={len(pages_done)}")

    total_results: Optional[int] = None
    if paths.total_file.exists():
        try:
            total_results = int(paths.total_file.read_text().strip()) or None
        except Exception:
            total_results = None

    bar = tqdm(
        desc="index",
        unit="page",
        initial=len(pages_done),
        total=None,
        smoothing=0.1,
        dynamic_ncols=True,
    )
    try:
        if 0 not in pages_done:
            _, _, tr = _fetch_page(paths, 0, seen, bar)
            bar.update(1)
            if tr:
                total_results = tr
                paths.total_file.write_text(str(tr))
        elif total_results is None:
            _, _, tr = _fetch_page(paths, 0, seen, bar)
            if tr:
                total_results = tr
                paths.total_file.write_text(str(tr))

        if not total_results:
            raise RuntimeError("Could not determine total_results from API")
        total_pages = math.ceil(total_results / per_page)
        _log(paths, f"API total_results={total_results} total_pages={total_pages} workers={workers}")

        pending = [p for p in range(total_pages) if p not in pages_done]
        bar.total = total_pages
        bar.refresh()
        if not pending:
            _log(paths, "All pages already indexed.")
            return len(seen)

        def _worker(page: int) -> tuple[int, int]:
            time.sleep(random.random() * 0.5)
            for _ in range(3):
                try:
                    _, n, _ = _fetch_page(paths, page, seen, bar)
                    return page, n
                except Exception as e:
                    _log(paths, f"[index] page {page} retry: {e}", to_console=False)
                    time.sleep(10 + random.random() * 10)
            raise RuntimeError(f"page {page} giving up")

        with cf.ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_worker, p): p for p in pending}
            for fut in cf.as_completed(futures):
                page = futures[fut]
                try:
                    fut.result()
                except Exception as e:
                    _log(paths, f"[index] page {page} FAILED: {e}")
                bar.update(1)
    finally:
        bar.close()
    _log(paths, f"Index complete: {len(seen)} records across {len(_load_pages_done(paths))} pages")
    return len(seen)


def _sol_bucket(sol: int) -> str:
    return f"{sol:05d}"


def _download_one(paths: FetchPaths, record: dict, done: set[str]) -> tuple[str, bool, str]:
    iid = record["imageid"]
    if iid in done:
        return iid, True, "skip"
    sol = int(record.get("sol") or 0)
    bucket = _sol_bucket(sol)
    img_subdir = paths.img_dir / bucket
    meta_subdir = paths.meta_dir / bucket
    img_subdir.mkdir(parents=True, exist_ok=True)
    meta_subdir.mkdir(parents=True, exist_ok=True)

    files = record.get("image_files") or {}
    full = files.get("full_res") or files.get("large") or files.get("medium") or files.get("small")
    if not full:
        _mark_failed(paths, iid, "no image_files url")
        return iid, False, "no-url"

    ext = Path(full).suffix.lower() or ".png"
    img_path = img_subdir / f"{iid}{ext}"
    meta_path = meta_subdir / f"{iid}.json"

    if img_path.exists() and img_path.stat().st_size > 0:
        if not meta_path.exists():
            meta_path.write_text(json.dumps(record, indent=2))
        _mark_done(paths, iid)
        return iid, True, "already"

    try:
        data = _http_get(full, timeout=180)
    except Exception as e:
        _mark_failed(paths, iid, f"download: {e}")
        return iid, False, f"dl-fail: {e}"

    tmp = img_path.with_suffix(img_path.suffix + ".part")
    tmp.write_bytes(data)
    tmp.rename(img_path)
    meta_path.write_text(json.dumps(record, indent=2))
    _mark_done(paths, iid)
    return iid, True, f"ok {len(data)//1024}KB"


def _iter_index(paths: FetchPaths):
    if not paths.index_file.exists():
        return
    with paths.index_file.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def download_all(paths: FetchPaths, *, workers: int = DOWNLOAD_WORKERS) -> None:
    """Download every image listed in ``index.jsonl`` that isn't already present."""
    done = _load_done(paths)
    total_indexed = sum(1 for _ in _iter_index(paths))
    pending: list[dict] = [r for r in _iter_index(paths) if r["imageid"] not in done]
    _log(paths, f"Download: {len(done)} done, {len(pending)} pending, {total_indexed} indexed")

    ok = fail = bytes_total = 0
    bar = tqdm(
        total=len(done) + len(pending),
        initial=len(done),
        desc="download",
        unit="img",
        smoothing=0.05,
        dynamic_ncols=True,
    )
    try:
        with cf.ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_download_one, paths, rec, done) for rec in pending]
            for fut in cf.as_completed(futures):
                iid, success, status = fut.result()
                if success:
                    ok += 1
                    if status.startswith("ok "):
                        try:
                            bytes_total += int(status.split()[1].rstrip("KB")) * 1024
                        except Exception:
                            pass
                else:
                    fail += 1
                bar.update(1)
                bar.set_postfix(ok=ok, fail=fail, MB=f"{bytes_total/1e6:.0f}")
    finally:
        bar.close()
    _log(paths, f"DONE. ok={ok} fail={fail} bytes~{bytes_total/1e6:.1f}MB")


def fetch(
    data_root: Path = Path("data"),
    *,
    command: str = "all",
    index_workers: int = INDEX_WORKERS,
    download_workers: int = DOWNLOAD_WORKERS,
) -> None:
    """Run the index and/or download phases.

    Parameters
    ----------
    data_root : path to write images/, metadata/, .state/ into.
    command : one of ``"index"``, ``"download"``, ``"all"``.
    """
    paths = FetchPaths(root=Path(data_root))
    if command in ("index", "all"):
        build_index(paths, workers=index_workers)
    if command in ("download", "all"):
        download_all(paths, workers=download_workers)


__all__ = ["fetch", "FetchPaths", "build_index", "download_all"]
