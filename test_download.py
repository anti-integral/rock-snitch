#!/usr/bin/env python3
"""Small end-to-end test: fetch 10 Navcam L/R records and download them.

Writes to ./data_test/ so it doesn't touch the real dataset.
Run: python3 test_download.py [N]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from urllib.parse import urlencode

from tqdm import tqdm

import download_navcam as dn

N = int(sys.argv[1]) if len(sys.argv) > 1 else 10

dn.ROOT = Path(__file__).resolve().parent / "data_test"
dn.IMG_DIR = dn.ROOT / "images"
dn.META_DIR = dn.ROOT / "metadata"
dn.STATE_DIR = dn.ROOT / ".state"
dn.INDEX_FILE = dn.STATE_DIR / "index.jsonl"
dn.DONE_FILE = dn.STATE_DIR / "done.txt"
dn.FAIL_FILE = dn.STATE_DIR / "failed.txt"
dn.LOG_FILE = dn.STATE_DIR / "log.txt"
for d in (dn.IMG_DIR, dn.META_DIR, dn.STATE_DIR):
    d.mkdir(parents=True, exist_ok=True)


def fetch_n_records(n: int) -> list[dict]:
    params = dict(dn.FEED_PARAMS, num=str(n), page="0")
    url = dn.API + "?" + urlencode(params)
    print(f"GET {url}")
    t0 = time.time()
    raw = dn.http_get(url, timeout=120)
    print(f"  got {len(raw)} bytes in {time.time()-t0:.1f}s")
    data = json.loads(raw)
    return data.get("images") or []


def main() -> None:
    print(f"=== test_download.py: fetching {N} images ===")
    records = fetch_n_records(N)
    print(f"received {len(records)} records")
    for r in records:
        print(f"  sol={r['sol']:>5}  {r['camera']['instrument']:<14}  {r['imageid']}")

    done: set[str] = set()
    bar = tqdm(total=len(records), desc="download", unit="img", dynamic_ncols=True)
    ok = fail = 0
    bytes_total = 0
    for rec in records:
        iid, success, status = dn.download_one(rec, done)
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
        bar.set_postfix(ok=ok, fail=fail, last=status[:30])
    bar.close()

    print(f"\nresult: ok={ok} fail={fail} bytes≈{bytes_total/1e6:.1f}MB")
    print(f"images:  {dn.IMG_DIR}")
    print(f"meta:    {dn.META_DIR}")


if __name__ == "__main__":
    main()
