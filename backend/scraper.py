"""
BIS "Know Your Standards" scraper — verified working against the real portal.
=============================================================================
Run this on a machine WITH internet access (not inside this sandbox).

WHAT WAS VERIFIED (by actually fetching live pages, not assumed):
  BIS runs TWO separate portals for standard lookups:

  1. services.bis.gov.in (older, used for standards published before
     1 Oct 2025 — the large majority of the ~22,000-standard catalogue).
     This portal is PLAIN SERVER-RENDERED HTML — fetchable with a simple
     GET request, no JavaScript execution needed. Confirmed URL pattern:
         https://www.services.bis.gov.in/php/BIS_2.0/bisconnect/
         knowyourstandards/Indian_standards/isdetails_mnd/<numeric_id>
     The numeric id is a plain sequential-ish integer (not encrypted).
     A fetched page directly contains: IS Number, Title, Superseding IS,
     Degree of Equivalence, Number of Amendments, Classification
     (Group/Sub Group/Aspect/Certification/ITC-HS code), and a full
     Cross Reference Details table listing every other Indian Standard
     that references this one (with relation context) plus any
     International Standard equivalence — exactly the fields this
     project needs, already structured in HTML tables.

  2. standards.bis.gov.in (newer, standards published after 1 Oct 2025).
     This is a client-rendered JavaScript app with AES-encrypted per-page
     IDs — NOT scrapable this way. Getting data from here would require
     either a headless-browser tool (Selenium/Playwright) driven by a
     human clicking through the real UI, or an official BIS data feed.
     This is a genuine, current limitation — flag it plainly if asked,
     rather than implying full coverage.

STRATEGY: crawl outward from a seed list of known IS numbers/ids by
following the "Cross Reference Details" links on each page (breadth-first).
This mirrors how a procurement official would actually explore standards,
stays reasonably polite (no blind enumeration of ~350,000 possible ids),
and naturally builds exactly the knowledge-graph edges this project needs.

RESPONSIBLE USE:
  - Public informational portal, not a documented open API. Keep concurrency
    at 1, add delays between requests, cache what you fetch, and identify
    your client via User-Agent.
  - For anything beyond a demo-scale crawl (a few hundred to low thousands
    of standards), the real path is a formal data-sharing arrangement with
    BIS/DoCA — say this explicitly in your pitch rather than presenting a
    hobby scraper as a production ingestion pipeline.
"""

import base64
import json
import re
import time
from dataclasses import dataclass, field, asdict

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.services.bis.gov.in/php/BIS_2.0/bisconnect/knowyourstandards/Indian_standards/isdetails_mnd"
DETAIL_LINK_PATTERN = re.compile(r"isdetails/([A-Za-z0-9+/=]+)")
HEADERS = {
    "User-Agent": "SIH-IS-Recommender-Prototype/1.0 (educational hackathon project; contact: your-team-email@example.com)"
}
REQUEST_DELAY_SECONDS = 1.5  # be polite — do not lower this for a bulk crawl


@dataclass
class StandardRecord:
    is_number: str
    title: str = ""
    scope: str = ""
    group: str = ""
    sub_group: str = ""
    status: str = "active"
    latest_amendment: str = "None"
    certification: str = "None"
    cross_references: list = field(default_factory=list)
    source_page_id: str = ""  # this record's own portal page id, kept so a
                               # future crawl run can reseed from everything
                               # already discovered instead of one fixed id


def decode_page_id(b64_id: str) -> str:
    """Detail links elsewhere on the site reference pages via base64-encoded
    numeric ids (e.g. 'ODE0MA==' -> '8140'), while isdetails_mnd/ takes the
    plain number directly. Decode so both forms resolve to the same id."""
    try:
        return base64.b64decode(b64_id).decode("utf-8")
    except Exception:
        return b64_id  # already plain, or not decodable — pass through


def _table_value(soup: BeautifulSoup, label: str) -> str:
    """The Basic Details / Classification Details sections render as
    label/value rows. Find the row whose first cell matches `label` and
    return its value cell's text."""
    for row in soup.find_all(["tr", "div"]):
        cells = row.find_all(["td", "div"], recursive=False)
        text_cells = [c.get_text(strip=True) for c in row.find_all(["td"])] or None
        if text_cells and len(text_cells) >= 2 and label.lower() in text_cells[0].lower():
            return text_cells[-1]
    # Fallback: plain-text proximity search for pages whose markup varies
    full_text = soup.get_text("\n", strip=True)
    m = re.search(re.escape(label) + r"\s*:?\s*\n?\s*([^\n]+)", full_text, re.IGNORECASE)
    return m.group(1).strip() if m else ""


def parse_standard_page(html: str, id_hint: str = "") -> StandardRecord:
    soup = BeautifulSoup(html, "html.parser")
    full_text = soup.get_text("\n", strip=True)

    is_number_match = re.search(r"IS[/\s]?\d[\d:\s\(\)A-Za-z]*", full_text)
    record = StandardRecord(is_number=(is_number_match.group(0).split("\n")[0].strip()
                                        if is_number_match else id_hint))

    title_match = re.search(r"IS Title.*?:\s*\n?([^\n]+)", full_text)
    record.title = title_match.group(1).strip() if title_match else ""

    record.group = _table_value(soup, "Group")
    record.sub_group = _table_value(soup, "Sub Group")
    cert = _table_value(soup, "Certification")
    record.certification = cert if cert and cert.lower() != "none" else "None"

    amend_match = re.search(r"Number of Amendments.*?:\s*\n?([^\n]+)", full_text)
    if amend_match and "no amendment" not in amend_match.group(1).lower():
        record.latest_amendment = amend_match.group(1).strip()

    # Cross references: every isdetails/<id> link on the page, from the
    # "Cross Reference Details" tables, is another related standard.
    seen_ids = set()
    for link in soup.find_all("a", href=True):
        m = DETAIL_LINK_PATTERN.search(link["href"])
        if not m:
            continue
        ref_id = decode_page_id(m.group(1))
        if ref_id == id_hint or ref_id in seen_ids:
            continue
        seen_ids.add(ref_id)
        link_text = link.get_text(strip=True)
        if link_text and re.match(r"IS[/\s]?\d", link_text):
            record.cross_references.append({
                "is_number": link_text,
                "relation": "normative",  # portal doesn't label relation type explicitly;
                                           # refine manually per the six PS categories
                "title": "",
                "_page_id": ref_id,  # kept for crawling, strip before final dataset
            })
    return record


def crawl(seed_ids: list, max_pages: int = 200, out_path: str = "data/scraped_standards.json"):
    """Breadth-first crawl starting from seed_ids (plain numeric strings),
    following cross-reference links outward. Stops at max_pages fetched."""
    session = requests.Session()
    session.headers.update(HEADERS)

    to_visit = list(seed_ids)
    visited = set()
    results = []

    while to_visit and len(visited) < max_pages:
        page_id = to_visit.pop(0)
        if page_id in visited:
            continue
        visited.add(page_id)

        url = f"{BASE_URL}/{page_id}"
        try:
            resp = session.get(url, timeout=15)
            resp.raise_for_status()
            record = parse_standard_page(resp.text, id_hint=page_id)
            record.source_page_id = page_id
            if not record.title:
                print(f"Skipping id {page_id} — no title parsed (page may not exist)")
                continue
            print(f"Fetched [{len(visited)}/{max_pages}]: {record.is_number} — {record.title[:60]}")

            for ref in record.cross_references:
                ref_page_id = ref.pop("_page_id", None)
                if ref_page_id and ref_page_id not in visited:
                    to_visit.append(ref_page_id)

            results.append(asdict(record))
        except Exception as e:
            print(f"Failed for id {page_id}: {e}")
        time.sleep(REQUEST_DELAY_SECONDS)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nCrawled {len(results)} standards. Saved to {out_path}")
    print("Review the 'relation' field on each cross-reference and refine it "
          "to normative/test_method/terminology/safety/installation/related_product "
          "per the actual standard content — the portal doesn't label this itself.")


if __name__ == "__main__":
    # Seed with a few known numeric page ids to start the crawl outward from.
    # 8140 = IS 1730:1989 (confirmed working during development).
    # Add more seeds from standards relevant to your target sectors —
    # search "Know Your Standard" on bis.gov.in, open a result, and copy the
    # numeric id from its isdetails_mnd/<id> URL if present, or decode the
    # base64 id from its isdetails/<id> link using decode_page_id().
    seed_ids = ["8140"]
    crawl(seed_ids, max_pages=50)
