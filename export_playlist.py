#!/usr/bin/env python
"""Export a generated playlist as an M3U file for import into Music.app.

Usage:
  python export_playlist.py                    # list available playlists
  python export_playlist.py "Alternative"      # export by name
  python export_playlist.py "Folk & Country"   # export by name
  python export_playlist.py --all              # export all playlists
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from config import LIBRARY_XML
from library import load_library
from playlists import load_playlist_index, _PLAYLIST_DIR

_EXPORT_DIR = Path(__file__).parent / "data" / "m3u"


def _load_index():
    raw = load_playlist_index()
    if not raw:
        print("No playlists found. Run `python playlists.py` first.")
        sys.exit(1)
    return {v["name"]: k for k, v in raw.items()}


def _build_path_lookup():
    songs = load_library(LIBRARY_XML)
    return songs[["song_name", "artist", "file_path"]].dropna(subset=["file_path"])


def _export(playlist_name, filename, path_lookup):
    csv_path = _PLAYLIST_DIR / filename
    if not csv_path.exists():
        print(f"  Playlist file not found: {filename}")
        return

    pl = pd.read_csv(csv_path)
    merged = pl.merge(path_lookup, on=["song_name", "artist"], how="left")

    found = merged["file_path"].notna().sum()
    total = len(merged)

    _EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = _EXPORT_DIR / f"{playlist_name.replace(' ', '_').replace('/', '-').replace('—', '-')}.m3u"

    with open(out, "w", encoding="utf-8") as f:
        f.write("#EXTM3U\n")
        for _, row in merged.iterrows():
            if pd.notna(row["file_path"]):
                f.write(f"{row['file_path']}\n")

    print(f"  {playlist_name:<35} {found}/{total} tracks matched → {out.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("playlist", nargs="?", help="Playlist name to export")
    parser.add_argument("--all", action="store_true", help="Export all playlists")
    args = parser.parse_args()

    index = _load_index()

    if not args.playlist and not args.all:
        print("Available playlists:\n")
        for name in index:
            print(f"  {name}")
        print(f"\nUsage: python export_playlist.py \"<name>\"  or  --all")
        return

    print("Loading library file paths...")
    path_lookup = _build_path_lookup()

    if args.all:
        for name, filename in index.items():
            _export(name, filename, path_lookup)
    else:
        name = args.playlist
        if name not in index:
            close = [n for n in index if args.playlist.lower() in n.lower()]
            if close:
                print(f"No exact match for '{name}'. Did you mean:")
                for c in close:
                    print(f"  {c}")
            else:
                print(f"No playlist named '{name}'. Run without arguments to see options.")
            sys.exit(1)
        _export(name, index[name], path_lookup)

    print(f"\nTo import: open Music.app → File → Import → select the .m3u file from data/m3u/")


if __name__ == "__main__":
    main()
