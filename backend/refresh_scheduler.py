"""
Keeps "latest version/amendment" data current by periodically re-fetching
known standards from the BIS portal and rebuilding the index.

This directly targets the PS requirement: "Highlight the latest published
version and amendments of the recommended standards" — a static demo
dataset can't actually claim that in production; this is what makes it true
on an ongoing basis.

Run:  python refresh_scheduler.py
(runs once immediately, then on the configured interval — Ctrl+C to stop)

Needs internet access (unlike this dev sandbox) — run on your deployment
server, not inside a hackathon judging environment with no network.
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

from apscheduler.schedulers.blocking import BlockingScheduler

import scraper

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
STATUS_FILE = os.path.join(DATA_DIR, "refresh_status.json")
SEED_IDS_FILE = os.path.join(DATA_DIR, "crawl_seed_ids.json")

REFRESH_INTERVAL_HOURS = 24  # BIS publishes revisions/amendments infrequently;
                              # daily is more than enough and stays polite.
MAX_PAGES_PER_REFRESH = 200  # cap per run — stays polite, and re-crawling
                              # your existing known ids each run re-confirms
                              # their current version/amendment/certification


def seed_page_ids() -> list:
    """Page ids to re-crawl from. First run seeds from scraper.py's default;
    subsequent runs reuse whatever crawl_seed_ids.json already has (every id
    the crawl has discovered so far), so coverage grows over time instead of
    resetting. Delete crawl_seed_ids.json to force a fresh start."""
    if os.path.exists(SEED_IDS_FILE):
        with open(SEED_IDS_FILE, encoding="utf-8") as f:
            return json.load(f)
    return ["8140"]  # scraper.py's confirmed-working starting point


def refresh_job():
    print(f"[{datetime.now(timezone.utc).isoformat()}] Starting refresh...")
    ids = seed_page_ids()
    try:
        scraper.crawl(ids, max_pages=MAX_PAGES_PER_REFRESH,
                       out_path=os.path.join(DATA_DIR, "scraped_standards.json"))
    except Exception as e:
        print(f"Refresh fetch failed: {e}")
        _write_status(success=False, error=str(e))
        return

    try:
        subprocess.run([sys.executable, "build_index.py"], check=True,
                        cwd=os.path.dirname(__file__))
    except subprocess.CalledProcessError as e:
        print(f"Index rebuild failed: {e}")
        _write_status(success=False, error=str(e))
        return

    # Persist every page id the crawl has visited so the next refresh grows
    # coverage outward instead of restarting from the single seed each time.
    scraped_path = os.path.join(DATA_DIR, "scraped_standards.json")
    if os.path.exists(scraped_path):
        with open(scraped_path, encoding="utf-8") as f:
            scraped = json.load(f)
        all_page_ids = sorted({rec["source_page_id"] for rec in scraped if rec.get("source_page_id")})
        with open(SEED_IDS_FILE, "w", encoding="utf-8") as f:
            json.dump(all_page_ids, f, indent=2)

    _write_status(success=True)
    print("Refresh complete. Index rebuilt with latest version/amendment data.")


def _write_status(success: bool, error: str = None):
    status = {
        "last_run": datetime.now(timezone.utc).isoformat(),
        "success": success,
    }
    if error:
        status["error"] = error
    with open(STATUS_FILE, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2)


if __name__ == "__main__":
    refresh_job()  # run once immediately on startup

    scheduler = BlockingScheduler()
    scheduler.add_job(refresh_job, "interval", hours=REFRESH_INTERVAL_HOURS)
    print(f"Scheduler started — refreshing every {REFRESH_INTERVAL_HOURS}h. Ctrl+C to stop.")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        pass
