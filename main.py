#!/usr/bin/env python
import sys
import pandas as pd

from config import LIBRARY_XML, REVIEWS_DIR
from library import load_library
from reviews import load_reviews
from ratings import build_albums, slept_on, hidden_gems, worth_revisiting
from matching import fuzzy_match_reviews


def _check_paths():
    missing = [p for p in [LIBRARY_XML, REVIEWS_DIR] if not p.exists()]
    if missing:
        print("Missing data paths:")
        for p in missing:
            print(f"  {p}")
        sys.exit(1)


def _section(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)


def _fmt_time(ms):
    h = int(ms / 3_600_000)
    return f"{h}h" if h < 100 else f"{h:,}h"


def main():
    _check_paths()

    print("Loading library...")
    songs = load_library(LIBRARY_XML, playlist_names=["Party"])

    print("Loading reviews...")
    reviews = load_reviews(REVIEWS_DIR)

    print("Building ratings...")
    albums = build_albums(songs, reviews)

    print("Fuzzy matching unreviewed albums...")
    albums, n_fuzzy = fuzzy_match_reviews(albums, reviews)

    rated = albums[albums["your_rating"].notna()].copy()
    coverage = albums["pitchfork_score"].notna().mean()

    _section("LIBRARY SUMMARY")
    print(f"  Songs:           {len(songs):,}")
    print(f"  Albums:          {len(albums):,}")
    print(f"  Pitchfork match: {coverage:.0%} of albums  (+{n_fuzzy} via fuzzy match)")
    print(f"  Rated albums:    {len(rated):,}  (in library 1+ yr)")

    _section("YOUR TOP 20 ALBUMS")
    cols = ["album", "album_artist", "your_rating", "pitchfork_score",
            "play_count", "skip_count", "genre_clean", "decade"]
    top = rated.sort_values("your_rating", ascending=False).head(20)
    for _, row in top[cols].iterrows():
        p4k = f"P4K {row['pitchfork_score']:.1f}" if pd.notna(row["pitchfork_score"]) else "no review"
        print(f"  {row['your_rating']:4.1f}  {row['album'][:40]:<40}  {row['album_artist'][:25]:<25}  {p4k}")

    _section("CRITIC'S PICKS YOU'VE SLEPT ON  (P4K ≥ 8.5, < 5 plays per track)")
    for _, row in slept_on(albums).head(15)[["album", "album_artist", "pitchfork_score", "plays_per_track", "play_count"]].iterrows():
        print(f"  P4K {row['pitchfork_score']:.1f}  {row['plays_per_track']:.1f} plays/track  "
              f"({int(row['play_count'])} total)  {row['album'][:38]:<38}  {row['album_artist']}")

    _section("YOUR HIDDEN GEMS  (your rating top 25%, P4K < 7.5 or no review)")
    for _, row in hidden_gems(rated).head(15)[["album", "album_artist", "your_rating", "pitchfork_score", "play_count"]].iterrows():
        p4k = f"{row['pitchfork_score']:.1f}" if pd.notna(row["pitchfork_score"]) else "no review"
        print(f"  you {row['your_rating']:.1f}  P4K {p4k:>9}  plays {int(row['play_count']):>4}  "
              f"{row['album'][:38]:<38}  {row['album_artist']}")

    _section("WORTH REVISITING  (top-rated, not played in 2+ years)")
    for _, row in worth_revisiting(rated, songs).head(15)[["album", "album_artist", "your_rating", "last_played"]].iterrows():
        last = row["last_played"].strftime("%Y-%m") if pd.notna(row["last_played"]) else "never"
        print(f"  you {row['your_rating']:.1f}  last played {last}  "
              f"{row['album'][:38]:<38}  {row['album_artist']}")

    _section("YOUR LISTENING BY GENRE  (top 10 by total listen time)")
    genre_stats = (
        songs[songs["genre_clean"].notna()]
        .groupby("genre_clean")
        .agg(songs=("song_name", "count"), listen_time=("total_listen_time", "sum"))
        .sort_values("listen_time", ascending=False)
        .head(10)
        .reset_index()
    )
    for _, row in genre_stats.iterrows():
        print(f"  {_fmt_time(row['listen_time']):>6}  {row['songs']:>5} songs  {row['genre_clean']}")

    print()


if __name__ == "__main__":
    main()
