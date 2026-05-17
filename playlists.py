#!/usr/bin/env python
"""Generate mood-based playlists via k-means clustering on song features.

Usage:
  python playlists.py           # cluster into 4 playlists
  python playlists.py --k 5     # use 5 clusters
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler

from config import LIBRARY_XML, REVIEWS_DIR, GENRE_BUCKETS, PITCHFORK_GENRE_BUCKETS, GENRE_ENERGY, GENRE_CHILL
from library import load_library
from ratings import build_albums
from reviews import load_reviews
from mood import extract_mood, MOOD_COLS

_PLAYLIST_DIR = Path(__file__).parent / "data" / "playlists"


def load_playlist_index():
    """Return {filename: {name, silhouette}} for all generated playlists."""
    index_file = _PLAYLIST_DIR / "index.json"
    if not index_file.exists():
        return {}
    raw = json.loads(index_file.read_text())
    if not raw:
        return {}
    if isinstance(next(iter(raw.values())), dict):
        return raw
    # Upgrade old format {filename: name} to new format
    return {k: {"name": v, "silhouette": None} for k, v in raw.items()}


def _build_features(songs, complete_cases=False):
    df = songs[
        songs["genre_clean"].notna()
        & (songs["play_count"] > 0)
        & songs["album"].notna()
    ].copy()

    # Consolidate into broad buckets; fall back to Pitchfork genre if iTunes genre doesn't map
    df["genre_bucket"] = df["genre_clean"].map(GENRE_BUCKETS)
    mask = df["genre_bucket"].isna() & df["pitchfork_genre"].notna()
    df.loc[mask, "genre_bucket"] = df.loc[mask, "pitchfork_genre"].map(PITCHFORK_GENRE_BUCKETS)
    df = df[df["genre_bucket"].notna()].copy()

    # Zero BPM means Essentia got silence (DRM file) — treat as missing
    if "bpm" in df.columns:
        df.loc[df["bpm"] == 0, ["bpm", "loudness", "mode"]] = np.nan

    if complete_cases:
        audio_cols = ["bpm", "loudness", "mode"]
        mood_cols = [c for c in MOOD_COLS if c in df.columns]
        has_audio = df[audio_cols].notna().all(axis=1) if all(c in df.columns for c in audio_cols) else pd.Series(False, index=df.index)
        has_mood = df[mood_cols].notna().all(axis=1) if mood_cols else pd.Series(False, index=df.index)
        df = df[has_audio & has_mood].copy()

    df["skip_ratio"] = df["skip_count"] / (df["play_count"] + 1)
    df["log_plays"] = np.log1p(df["play_count"])

    genre_dummies = pd.get_dummies(df["genre_bucket"], prefix="g")

    num = pd.DataFrame({
        "decade":     df["decade"].clip(1950, 2030) / 10,
        "log_plays":  df["log_plays"],
        "skip_ratio": df["skip_ratio"],
        "p4k_score":  df["pitchfork_score"].fillna(df["pitchfork_score"].median()),
    }, index=df.index)

    audio_cols = [c for c in ["bpm", "loudness", "mode"] if c in df.columns and df[c].notna().mean() > 0.1]
    if audio_cols:
        audio_num = df[audio_cols].copy()
        if "bpm" in audio_num.columns:
            audio_num["bpm"] = (audio_num["bpm"].clip(40, 220) - 40) / 180
        if "loudness" in audio_num.columns:
            audio_num["loudness"] = audio_num["loudness"].clip(0, 1)
        audio_num = audio_num.fillna(audio_num.median())
        num = pd.concat([num, audio_num], axis=1)

    available_mood = [c for c in MOOD_COLS if c in df.columns and df[c].notna().mean() > 0.05]
    if available_mood:
        mood_num = df[available_mood].fillna(0.5)
        num = pd.concat([num, mood_num], axis=1)

    features = pd.concat([num, genre_dummies * 0.2], axis=1).fillna(0)
    return df, features


def _audio_energy(avg_bpm, avg_loudness):
    """0-1 energy score from BPM and loudness."""
    bpm_norm = np.clip((avg_bpm - 40) / 180, 0, 1) if avg_bpm else 0.5
    loud_norm = np.clip(avg_loudness, 0, 1) if avg_loudness else 0.5
    return 0.6 * bpm_norm + 0.4 * loud_norm


def _mood_label(energy_rank, avg):
    """Evocative name from the combination of mood dimensions."""
    bpm = avg.get("bpm") or 100
    mode = avg.get("mode") or 0.5
    valence = avg.get("mood_valence", 0.5)
    acoustic = avg.get("mood_acousticness", 0.5)
    dance = avg.get("mood_danceability", 0.5)

    is_major = mode > 0.5
    fast = bpm > 120
    slow = bpm < 90

    if energy_rank == "high" and fast and dance > 0.52:
        return "Dance Floor"
    elif energy_rank == "high" and fast and not is_major:
        return "Raw & Loud"
    elif energy_rank == "high" and fast:
        return "Road Trip"
    elif energy_rank == "high" and not is_major:
        return "Dark Energy"
    elif energy_rank == "high":
        return "Pump Up"
    elif energy_rank == "low" and slow and acoustic > 0.5:
        return "Sunday Morning"
    elif energy_rank == "low" and not is_major and valence < 0.45:
        return "Late Night"
    elif energy_rank == "low" and acoustic > 0.5:
        return "Unplugged"
    elif energy_rank == "low" and not is_major:
        return "Rainy Day"
    elif energy_rank == "low":
        return "Wind Down"
    elif dance > 0.55 and is_major:
        return "Good Vibes"
    elif not is_major and valence < 0.45:
        return "Introspective"
    elif is_major and valence > 0.55:
        return "Feel Good"
    else:
        return "Cruising"


def _name_clusters(clusters):
    """Name all clusters relative to each other using audio + mood dimension rankings."""
    mood_present = [c for c in MOOD_COLS if any(
        c in cl.columns and cl[c].notna().mean() > 0.1 for cl in clusters
    )]
    has_audio = all("bpm" in cl.columns and cl["bpm"].notna().mean() > 0.1 for cl in clusters)

    avgs = []
    for cluster in clusters:
        a = {c: cluster[c].mean() for c in mood_present} if mood_present else {}
        if has_audio:
            a["bpm"] = cluster["bpm"].median()
            a["loudness"] = cluster["loudness"].median()
            a["mode"] = cluster["mode"].mean()
        avgs.append(a)

    # Determine energy rank per cluster — prefer audio signal, fall back to text
    if has_audio:
        energy_scores = [_audio_energy(a.get("bpm"), a.get("loudness")) for a in avgs]
    else:
        energy_scores = [a.get("mood_energy", 0.5) for a in avgs]

    k = len(clusters)
    # Rank clusters by energy score; assign high/mid/low by thirds
    ranked = sorted(range(k), key=lambda i: energy_scores[i])
    low_cutoff = k // 3
    high_cutoff = k - k // 3
    energy_ranks = {}
    for pos, idx in enumerate(ranked):
        if pos < low_cutoff:
            energy_ranks[idx] = "low"
        elif pos >= high_cutoff:
            energy_ranks[idx] = "high"
        else:
            energy_ranks[idx] = "mid"

    names = []
    seen = {}
    for i, (cluster, avg) in enumerate(zip(clusters, avgs)):
        top_genre = cluster["genre_bucket"].mode()[0]

        label = _mood_label(energy_ranks[i], avg)

        # Deduplicate: append top genre when label would repeat
        if label in seen:
            label = f"{label} — {top_genre}"
        seen[label] = True
        names.append(label)

    return names


def _find_optimal_k(X, k_range=range(2, 13)):
    print(f"Testing k={k_range.start}..{k_range.stop - 1}...")
    scores = {}
    for k in k_range:
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
        scores[k] = silhouette_score(X, labels)
        print(f"  k={k:2d}  silhouette={scores[k]:.4f}")
    best_k = max(scores, key=scores.get)
    print(f"\nBest k={best_k} (silhouette={scores[best_k]:.4f})")
    return best_k


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=9, help="Number of clusters (default: 9)")
    parser.add_argument("--size", type=int, default=30, help="Max songs per playlist (default: 30)")
    parser.add_argument("--no-audio", action="store_true", help="Skip Essentia audio features")
    parser.add_argument("--no-complete", action="store_true", help="Include songs missing audio or review data")
    parser.add_argument("--optimize", action="store_true", help="Find optimal k via silhouette score (k=2-12)")
    args = parser.parse_args()

    print("Loading library and reviews...")
    songs = load_library(LIBRARY_XML)
    reviews = extract_mood(load_reviews(REVIEWS_DIR))
    songs = songs.merge(
        reviews[["album", "album_artist", "pitchfork_score", "pitchfork_genre"] + MOOD_COLS],
        how="left", on=["album", "album_artist"]
    )

    albums = build_albums(songs, reviews)
    songs = songs.merge(
        albums[["album", "album_artist", "your_rating"]],
        how="left", on=["album", "album_artist"]
    )
    # quality: 60% personal rating, 40% Pitchfork (both 0-10 scale)
    songs["quality"] = (
        0.6 * songs["your_rating"].fillna(songs["your_rating"].median()) +
        0.4 * songs["pitchfork_score"].fillna(songs["pitchfork_score"].median())
    )

    if not args.no_audio:
        from audio_features import enrich_songs
        print("Enriching with Essentia audio features...")
        songs = enrich_songs(songs)

    print("Building features...")
    df, features = _build_features(songs, complete_cases=not args.no_complete)

    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    if args.optimize:
        args.k = _find_optimal_k(X)

    print(f"  {len(df):,} songs  |  {features.shape[1]} features  |  k={args.k}")

    print("Clustering...")
    km = KMeans(n_clusters=args.k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    df["cluster"] = labels

    sil_scores = silhouette_samples(X, labels)
    sil_by_cluster = pd.Series(sil_scores).groupby(labels).mean()

    _PLAYLIST_DIR.mkdir(parents=True, exist_ok=True)
    for f in _PLAYLIST_DIR.glob("playlist_*.csv"):
        f.unlink()
    (_PLAYLIST_DIR / "index.json").unlink(missing_ok=True)

    cluster_sizes = df["cluster"].value_counts()
    clusters = [(c, df[df["cluster"] == c].copy()) for c in range(args.k) if cluster_sizes.get(c, 0) >= 30]
    names = _name_clusters([cl for _, cl in clusters])

    index = {}
    saved = 0
    for (orig_c, cluster), name in zip(clusters, names):
        playlist = (
            cluster.sort_values("quality", ascending=False)
            .drop_duplicates(subset="artist")
            .drop_duplicates(subset="album")
            .head(args.size)
        )
        if len(playlist) < 30:
            continue

        saved += 1
        avg_decade = cluster[cluster["decade"] > 0]["decade"].mean()
        top_genre = cluster["genre_bucket"].mode()[0]
        print(f"  [{saved}] {name} — {len(playlist)} songs | {top_genre} | {int(avg_decade)}s")

        out = _PLAYLIST_DIR / f"playlist_{saved}.csv"
        playlist[["song_name", "artist", "album", "genre_clean", "decade",
                   "play_count", "skip_count", "pitchfork_score", "your_rating", "quality"]].to_csv(out, index=False)
        index[out.name] = {"name": name, "silhouette": float(sil_by_cluster[orig_c])}

    (_PLAYLIST_DIR / "index.json").write_text(json.dumps(index, indent=2))
    print()


if __name__ == "__main__":
    main()
