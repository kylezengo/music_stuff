#!/usr/bin/env python
"""Scrape Pitchfork album reviews.

Usage:
  python scraper.py             collect new links + scrape new reviews
  python scraper.py --links     collect new links only
  python scraper.py --test 5    scrape 5 links as a dry run
"""

import argparse
import datetime
import json
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup

from config import LINKS_DIR, REVIEWS_DIR

_HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
_DELAY = 1.5  # seconds between requests


def _get(url, retries=3):
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=15)
            if resp.status_code == 200:
                return resp
        except requests.RequestException:
            pass
        time.sleep(2 ** attempt)
    return None


def collect_links(known: set) -> list:
    """Paginate pitchfork.com/reviews/albums and return links not in known."""
    new_links = []
    page = 1

    while True:
        resp = _get(f"https://pitchfork.com/reviews/albums/?page={page}")
        if not resp:
            break

        soup = BeautifulSoup(resp.content, "lxml")
        page_links = {
            f"https://pitchfork.com{a['href']}"
            for a in soup.find_all("a", href=True)
            if a["href"].startswith("/reviews/albums/")
            and a["href"] != "/reviews/albums/"
            and not a["href"].startswith("/reviews/albums/?")
        }

        if not page_links:
            break

        fresh = [l for l in page_links if l not in known]
        new_links.extend(fresh)
        print(f"  Page {page}: {len(fresh)} new / {len(page_links)} total")

        if not fresh:
            break

        page += 1
        time.sleep(_DELAY)

    return list(set(new_links))


def parse_review(url: str) -> dict | None:
    resp = _get(url)
    if not resp:
        return None

    soup = BeautifulSoup(resp.content, "html.parser")

    # Score, artist, album, genre, date live in __PRELOADED_STATE__
    state_script = next(
        (s.string for s in soup.find_all("script") if "__PRELOADED_STATE__" in (s.string or "")),
        None,
    )
    if not state_script:
        return None

    state = json.loads(state_script.split("window.__PRELOADED_STATE__ = ", 1)[1].rstrip(";"))
    review = state.get("transformed", {}).get("review", {})
    header = review.get("headerProps", {})

    artists = header.get("artists", [])
    artist = artists[0]["name"] if artists else None

    hed = header.get("dangerousHed", "")
    album = BeautifulSoup(hed, "html.parser").get_text().strip()

    music_rating = header.get("musicRating", {})
    score = music_rating.get("score") if music_rating.get("score") is not None else review.get("rating")

    info = header.get("infoSliceFields", {})
    genre = info.get("genre")
    reviewed_date = info.get("reviewDate") or header.get("publishDate")
    best = 1 if (music_rating.get("isBestNewMusic") or music_rating.get("isBestNewReissue")) else 0

    # Review text lives in JSON-LD reviewBody
    review_text = None
    for s in soup.find_all("script", type="application/ld+json"):
        try:
            ld = json.loads(s.string)
            if ld.get("reviewBody"):
                review_text = ld["reviewBody"]
                break
        except Exception:
            pass

    if not artist or not album or score is None:
        return None

    return {
        "artist": artist,
        "album": album,
        "score": score,
        "genre": genre,
        "review": review_text,
        "best": best,
        "reviewed_date": reviewed_date,
        "link": url,
    }


def _load_known_links() -> set:
    links = set()
    if LINKS_DIR.exists():
        for f in LINKS_DIR.glob("*.txt"):
            links.update(l.strip() for l in f.read_text().splitlines() if l.strip().startswith("http"))
    return links


def _load_scraped_links() -> set:
    scraped = set()
    if REVIEWS_DIR.exists():
        for f in REVIEWS_DIR.glob("*.csv"):
            try:
                scraped.update(pd.read_csv(f, usecols=["link"])["link"].dropna().tolist())
            except Exception:
                pass
    return scraped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--links", action="store_true", help="Collect links only, skip scraping")
    parser.add_argument("--test", type=int, metavar="N", help="Scrape only N reviews (for testing)")
    args = parser.parse_args()

    today = datetime.date.today()

    print("Loading existing data...")
    known_links = _load_known_links()
    scraped_links = _load_scraped_links()
    all_known = known_links | scraped_links
    print(f"  {len(known_links):,} known links  |  {len(scraped_links):,} already scraped")

    print("\nCollecting new links from Pitchfork...")
    new_links = collect_links(all_known)
    print(f"  {len(new_links):,} new links found")

    if new_links:
        LINKS_DIR.mkdir(parents=True, exist_ok=True)
        link_file = LINKS_DIR / f"new_links_{today}.txt"
        link_file.write_text("\n".join(new_links))
        print(f"  Saved → {link_file.name}")

    if args.links:
        return

    # Scrape: anything known but not yet scraped, plus brand new links
    unscraped = list((known_links - scraped_links) | set(new_links))
    if not unscraped:
        print("Nothing new to scrape.")
        return

    print(f"\n{len(unscraped):,} links to scrape ({len(known_links - scraped_links):,} backlog + {len(new_links):,} new)")
    to_scrape = unscraped[:args.test] if args.test else unscraped
    print(f"\nScraping {len(to_scrape):,} reviews...")

    results, errors = [], []
    for i, url in enumerate(to_scrape, 1):
        print(f"  [{i}/{len(to_scrape)}] {url.split('/')[-2]}")
        try:
            row = parse_review(url)
            if row:
                results.append(row)
            else:
                errors.append(url)
        except Exception as e:
            print(f"    Error: {e}")
            errors.append(url)
        time.sleep(_DELAY)

    if results:
        out = REVIEWS_DIR / f"new_reviews_{today}.csv"
        pd.DataFrame(results).to_csv(out, index=False)
        print(f"\nSaved {len(results):,} reviews → {out.name}")

    if errors:
        print(f"\n{len(errors)} failed URLs (check manually):")
        for u in errors[:10]:
            print(f"  {u}")


if __name__ == "__main__":
    main()
