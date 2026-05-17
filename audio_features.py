"""Analyze local audio files with Essentia to extract mood/energy features.

Usage:
  python audio_features.py            analyze all unprocessed songs
  python audio_features.py --test 10  analyze 10 songs as a dry run
  python audio_features.py --top 2000 only analyze top 2000 most-played songs

Features extracted per track:
  bpm       - tempo in beats per minute
  loudness  - perceived loudness (0-1)
  mode      - 1=major (brighter), 0=minor (darker)
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config import LIBRARY_XML
from library import load_library

_CACHE = Path(__file__).parent / "data" / "audio_features.csv"
_FEATURE_COLS = ["bpm", "loudness", "mode"]


def _analyze_subprocess(file_path):
    """Run Essentia in a child process so segfaults don't kill the main script."""
    code = f"""
import warnings, json, sys, numpy as np
warnings.filterwarnings("ignore")
import essentia.standard as es
try:
    audio = es.MonoLoader(filename={json.dumps(file_path)}, sampleRate=22050)()
    bpm = float(es.PercivalBpmEstimator(sampleRate=22050)(audio))
    loudness_power = float(es.Loudness()(audio))
    loudness_db = 10 * np.log10(max(loudness_power, 1e-10))
    loudness_norm = float(np.clip((loudness_db + 60) / 60, 0, 1))
    key, scale, _ = es.KeyExtractor()(audio)
    mode = 1 if scale == "major" else 0
    print(json.dumps({{"bpm": bpm, "loudness": loudness_norm, "mode": mode}}))
except Exception as e:
    print(json.dumps({{"error": str(e)}}))
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=30
        )
        # Essentia prints many WARNING lines before the result; scan from the end
        for line in reversed(result.stdout.strip().splitlines()):
            try:
                data = json.loads(line)
                if "error" not in data:
                    return data
                return None
            except json.JSONDecodeError:
                continue
    except subprocess.TimeoutExpired:
        pass
    except Exception:
        pass
    return None


def _load_cache():
    if _CACHE.exists():
        return pd.read_csv(_CACHE)
    return pd.DataFrame(columns=["file_path"] + _FEATURE_COLS)


def _save_cache(cache):
    _CACHE.parent.mkdir(parents=True, exist_ok=True)
    cache.to_csv(_CACHE, index=False)


def enrich_songs(df_songs):
    """Merge cached audio features into df_songs."""
    cache = _load_cache()
    if cache.empty or "file_path" not in df_songs.columns:
        return df_songs
    return df_songs.merge(cache[["file_path"] + _FEATURE_COLS], on="file_path", how="left")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=int, metavar="N", help="Analyze N songs only")
    parser.add_argument("--top", type=int, metavar="N", help="Only analyze top N most-played songs")
    args = parser.parse_args()

    print("Loading library...")
    songs = load_library(LIBRARY_XML)
    songs = songs[songs["file_path"].notna() & (songs["play_count"] > 0)].copy()

    if args.top:
        songs = songs.nlargest(args.top, "play_count")

    cache = _load_cache()
    cached_paths = set(cache["file_path"].dropna())
    to_analyze = songs[~songs["file_path"].isin(cached_paths)].drop_duplicates("file_path")

    if args.test:
        to_analyze = to_analyze.head(args.test)

    total = len(to_analyze)
    print(f"{total:,} files to analyze ({len(cached_paths):,} already cached)")

    if to_analyze.empty:
        print("Nothing to analyze.")
        return

    new_rows = []
    errors = 0

    for i, (_, row) in enumerate(to_analyze.iterrows(), 1):
        path = row["file_path"]
        result = _analyze_subprocess(path)

        if result:
            new_rows.append({"file_path": path, **result})
        else:
            new_rows.append({"file_path": path, "bpm": None, "loudness": None, "mode": None})
            errors += 1

        if i % 50 == 0:
            cache = pd.concat([cache, pd.DataFrame(new_rows)], ignore_index=True)
            _save_cache(cache)
            new_rows = []
            pct = i / total * 100
            print(f"  {i:,} / {total:,} ({pct:.0f}%)  errors so far: {errors}")

    if new_rows:
        cache = pd.concat([cache, pd.DataFrame(new_rows)], ignore_index=True)
        _save_cache(cache)

    succeeded = len(cache[cache["bpm"].notna()])
    print(f"\nDone. {succeeded:,} tracks with features in cache ({errors} failed).")


if __name__ == "__main__":
    main()
