#!/usr/bin/env python3
"""Download all Mars 2020 Perseverance Navcam L/R raw images + metadata.

Resumable, parallel, polite. Progress + state in data/.state/.
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import math
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

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
ROOT = Path(__file__).resolve().parent / "data"
IMG_DIR = ROOT / "images"
META_DIR = ROOT / "metadata"
STATE_DIR = ROOT / ".state"
INDEX_FILE = STATE_DIR / "index.jsonl"
PAGES_DONE_FILE = STATE_DIR / "pages_done.txt"
TOTAL_FILE = STATE_DIR / "total_results.txt"
DONE_FILE = STATE_DIR / "done.txt"
FAIL_FILE = STATE_DIR / "failed.txt"
LOG_FILE = STATE_DIR / "log.txt"

WORKERS = 8
INDEX_WORKERS = 1  # NASA's raw-images API 403s on concurrent requests; must be serial.
RETRIES = 6
USER_AGENT = "rock-snitch-v2/0.1 (research; contact om.sanan007@gmail.com)"

for d in (IMG_DIR, META_DIR, STATE_DIR):
    d.mkdir(parents=True, exist_ok=True)

_log_lock = threading.Lock()
_done_lock = threading.Lock()


def log(msg: str, *, to_console: bool = True) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    with _log_lock:
        if to_console:
            tqdm.write(line)
        with LOG_FILE.open("a") as f:
            f.write(line + "\n")


def http_get(url: str, timeout: int = 60, status_cb=None) -> bytes:
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
            # 403/429 → CDN / rate-limit: back off harder and longer
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


def build_api_url(page: int) -> str:
    from urllib.parse import urlencode
    params = dict(FEED_PARAMS, page=str(page))
    return API + "?" + urlencode(params)


def load_done() -> set[str]:
    if not DONE_FILE.exists():
        return set()
    with DONE_FILE.open() as f:
        return {line.strip() for line in f if line.strip()}


def mark_done(imageid: str) -> None:
    with _done_lock, DONE_FILE.open("a") as f:
        f.write(imageid + "\n")


def mark_failed(imageid: str, reason: str) -> None:
    with _done_lock, FAIL_FILE.open("a") as f:
        f.write(f"{imageid}\t{reason}\n")


_index_write_lock = threading.Lock()
_pages_lock = threading.Lock()


def _load_seen_ids() -> set[str]:
    seen: set[str] = set()
    if INDEX_FILE.exists():
        with INDEX_FILE.open() as f:
            for line in f:
                try:
                    seen.add(json.loads(line)["imageid"])
                except Exception:
                    pass
    return seen


def _load_pages_done() -> set[int]:
    if not PAGES_DONE_FILE.exists():
        return set()
    with PAGES_DONE_FILE.open() as f:
        return {int(line.strip()) for line in f if line.strip().isdigit()}


def _mark_page_done(page: int) -> None:
    with _pages_lock, PAGES_DONE_FILE.open("a") as f:
        f.write(f"{page}\n")


def _fetch_page(page: int, seen: set[str], bar: tqdm) -> tuple[int, int, int | None]:
    """Fetch one page, append new records to INDEX_FILE, return (page, n_new, total_results)."""
    url = build_api_url(page)
    t0 = time.time()
    raw = http_get(
        url,
        timeout=180,
        status_cb=lambda s, p=page: bar.set_postfix_str(f"p{p} {s}"),
    )
    data = json.loads(raw)
    total_results = int(data.get("total_results") or 0) or None
    images = data.get("images") or []
    new = 0
    if images:
        with _index_write_lock, INDEX_FILE.open("a") as f:
            for img in images:
                iid = img.get("imageid")
                if not iid or iid in seen:
                    continue
                f.write(json.dumps(img) + "\n")
                seen.add(iid)
                new += 1
    _mark_page_done(page)
    bar.set_postfix_str(f"p{page} +{new} in {time.time()-t0:.0f}s")
    return page, new, total_results


def build_index(workers: int = INDEX_WORKERS) -> int:
    """Parallel paginator. Resumable via pages_done.txt and index.jsonl.

    Strategy:
      1) Fetch page 0 synchronously (also gives us total_results).
      2) Enumerate pages 1..N-1 and submit them to a thread pool.
      3) Each worker appends records (lock-guarded), marks page done.
    """
    per_page = int(FEED_PARAMS["num"])
    seen = _load_seen_ids()
    pages_done = _load_pages_done()

    log(f"Index: resuming — records={len(seen)} pages_done={len(pages_done)}")

    # Discover total_results: cached or from page 0.
    total_results: int | None = None
    if TOTAL_FILE.exists():
        try:
            total_results = int(TOTAL_FILE.read_text().strip()) or None
        except Exception:
            total_results = None

    # Progress bar driven by PAGES (each ~100 records) so ETA is stable.
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
            _, _, tr = _fetch_page(0, seen, bar)
            bar.update(1)
            if tr:
                total_results = tr
                TOTAL_FILE.write_text(str(tr))
        elif total_results is None:
            # Need it; fetch page 0 metadata cheaply by re-fetching (pages_done idempotent).
            _, _, tr = _fetch_page(0, seen, bar)
            if tr:
                total_results = tr
                TOTAL_FILE.write_text(str(tr))

        if not total_results:
            raise RuntimeError("Could not determine total_results from API")
        total_pages = math.ceil(total_results / per_page)
        log(f"API total_results={total_results}  total_pages={total_pages}  workers={workers}")

        pending = [p for p in range(total_pages) if p not in pages_done]
        bar.total = total_pages
        bar.refresh()
        if not pending:
            log("All pages already indexed.")
            return len(seen)

        # Light jitter per-worker startup to avoid thundering herd.
        def _worker(page: int) -> tuple[int, int]:
            time.sleep(random.random() * 0.5)
            for _ in range(3):  # outer retry for transient index failures
                try:
                    _, n, _ = _fetch_page(page, seen, bar)
                    return page, n
                except Exception as e:
                    log(f"[index] page {page} outer retry: {e}", to_console=False)
                    time.sleep(10 + random.random() * 10)
            raise RuntimeError(f"page {page} giving up")

        with cf.ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_worker, p): p for p in pending}
            for fut in cf.as_completed(futures):
                page = futures[fut]
                try:
                    fut.result()
                except Exception as e:
                    log(f"[index] page {page} FAILED: {e}")
                bar.update(1)
    finally:
        bar.close()
    log(f"Index complete: {len(seen)} records across {len(_load_pages_done())} pages")
    return len(seen)


def sol_bucket(sol: int) -> str:
    return f"{sol:05d}"


def download_one(record: dict, done: set[str]) -> tuple[str, bool, str]:
    iid = record["imageid"]
    if iid in done:
        return iid, True, "skip"
    sol = int(record.get("sol") or 0)
    bucket = sol_bucket(sol)
    img_subdir = IMG_DIR / bucket
    meta_subdir = META_DIR / bucket
    img_subdir.mkdir(parents=True, exist_ok=True)
    meta_subdir.mkdir(parents=True, exist_ok=True)

    files = record.get("image_files") or {}
    full = files.get("full_res") or files.get("large") or files.get("medium") or files.get("small")
    if not full:
        mark_failed(iid, "no image_files url")
        return iid, False, "no-url"

    ext = Path(full).suffix.lower() or ".png"
    img_path = img_subdir / f"{iid}{ext}"
    meta_path = meta_subdir / f"{iid}.json"

    if img_path.exists() and img_path.stat().st_size > 0:
        if not meta_path.exists():
            meta_path.write_text(json.dumps(record, indent=2))
        mark_done(iid)
        return iid, True, "already"

    try:
        data = http_get(full, timeout=180)
    except Exception as e:
        mark_failed(iid, f"download: {e}")
        return iid, False, f"dl-fail: {e}"

    tmp = img_path.with_suffix(img_path.suffix + ".part")
    tmp.write_bytes(data)
    tmp.rename(img_path)
    meta_path.write_text(json.dumps(record, indent=2))
    mark_done(iid)
    return iid, True, f"ok {len(data)//1024}KB"


def iter_index():
    with INDEX_FILE.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def download_all() -> None:
    done = load_done()
    total_indexed = sum(1 for _ in iter_index())
    pending: list[dict] = [r for r in iter_index() if r["imageid"] not in done]
    log(f"Download: {len(done)} done, {len(pending)} pending, {total_indexed} indexed")

    ok = 0
    fail = 0
    bytes_total = 0
    bar = tqdm(
        total=len(done) + len(pending),
        initial=len(done),
        desc="download",
        unit="img",
        smoothing=0.05,
        dynamic_ncols=True,
    )
    try:
        with cf.ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futures = [ex.submit(download_one, rec, done) for rec in pending]
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
    log(f"DONE. ok={ok} fail={fail} bytes≈{bytes_total/1e6:.1f}MB")


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    workers = INDEX_WORKERS
    if len(sys.argv) > 2:
        try:
            workers = int(sys.argv[2])
        except ValueError:
            pass
    if cmd in ("index", "all"):
        build_index(workers=workers)
    if cmd in ("download", "all"):
        download_all()


if __name__ == "__main__":
    main()
