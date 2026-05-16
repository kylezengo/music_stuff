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
from sklearn.metrics import silhouette_samples
from sklearn.preprocessing import StandardScaler

from config import LIBRARY_XML, REVIEWS_DIR, GENRE_BUCKETS, PITCHFORK_GENRE_BUCKETS, GENRE_ENERGY, GENRE_CHILL
from library import load_library
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


def _build_features(songs):
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

    df["skip_ratio"] = df["skip_count"] / (df["play_count"] + 1)
    df["log_plays"] = np.log1p(df["play_count"])

    genre_dummies = pd.get_dummies(df["genre_bucket"], prefix="g")

    num = pd.DataFrame({
        "decade":     df["decade"].clip(1950, 2030) / 10,
        "log_plays":  df["log_plays"],
        "skip_ratio": df["skip_ratio"],
        "p4k_score":  df["pitchfork_score"].fillna(df["pitchfork_score"].median()),
    }, index=df.index)

    # Add Essentia audio features if available
    audio_cols = [c for c in ["bpm", "loudness", "mode"] if c in df.columns and df[c].notna().mean() > 0.1]
    if audio_cols:
        audio_num = df[audio_cols].copy()
        if "bpm" in audio_num.columns:
            audio_num["bpm"] = (audio_num["bpm"].clip(40, 220) - 40) / 180
        if "loudness" in audio_num.columns:
            audio_num["loudness"] = audio_num["loudness"].clip(0, 1)
        audio_num = audio_num.fillna(audio_num.median())
        num = pd.concat([num, audio_num], axis=1)

    # Add Pitchfork review mood features (energy, valence, danceability, acousticness)
    available_mood = [c for c in MOOD_COLS if c in df.columns and df[c].notna().mean() > 0.05]
    if available_mood:
        mood_num = df[available_mood].fillna(0.5)
        num = pd.concat([num, mood_num], axis=1)

    features = pd.concat([num, genre_dummies], axis=1).fillna(0)
    return df, features


def _mood_label(energy_rank, valence, acoustic, dance):
    """Return a mood label based on relative ranks and absolute mood averages."""
    if energy_rank == "high" and dance > valence:
        return "High Energy"
    elif energy_rank == "high" and valence < 0.45:
        return "Intense & Dark"
    elif energy_rank == "high":
        return "Energetic"
    elif energy_rank == "low" and acoustic > 0.55:
        return "Acoustic & Intimate"
    elif energy_rank == "low" and valence < 0.45:
        return "Dark & Brooding"
    elif energy_rank == "low":
        return "Chill"
    elif valence > 0.55:
        return "Warm & Mellow"
    else:
        return "Mid Energy"


def _name_clusters(clusters):
    """Name all clusters relative to each other using mood dimension rankings."""
    mood_present = [c for c in MOOD_COLS if any(
        c in cl.columns and cl[c].notna().mean() > 0.1 for cl in clusters
    )]

    avgs = []
    for cluster in clusters:
        if mood_present:
            avgs.append({c: cluster[c].mean() for c in mood_present})
        else:
            avgs.append({})

    energy_scores = [a.get("mood_energy", 0.5) for a in avgs]
    energy_max = max(energy_scores)
    energy_min = min(energy_scores)

    names = []
    for i, (cluster, avg) in enumerate(zip(clusters, avgs)):
        top_genre = cluster["genre_bucket"].value_counts().index[0]

        if mood_present and energy_max > energy_min:
            e = energy_scores[i]
            energy_range = energy_max - energy_min
            if e >= energy_max - energy_range * 0.25:
                energy_rank = "high"
            elif e <= energy_min + energy_range * 0.25:
                energy_rank = "low"
            else:
                energy_rank = "mid"

            label = _mood_label(
                energy_rank,
                avg.get("mood_valence", 0.5),
                avg.get("mood_acousticness", 0.5),
                avg.get("mood_danceability", 0.5),
            )
        else:
            if top_genre in GENRE_ENERGY:
                label = "High Energy"
            elif top_genre in GENRE_CHILL:
                label = "Chill"
            else:
                label = "Mid Energy"

        if label == "Mid Energy":
            names.append(top_genre)
        else:
            names.append(f"{label} — {top_genre}")

    return names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=8, help="Number of clusters (default: 8)")
    parser.add_argument("--size", type=int, default=50, help="Max songs per playlist (default: 50)")
    parser.add_argument("--audio", action="store_true", help="Enrich with Essentia audio features")
    args = parser.parse_args()

    print("Loading library and reviews...")
    songs = load_library(LIBRARY_XML)
    reviews = extract_mood(load_reviews(REVIEWS_DIR))
    songs = songs.merge(
        reviews[["album", "album_artist", "pitchfork_score", "pitchfork_genre"] + MOOD_COLS],
        how="left", on=["album", "album_artist"]
    )

    if args.audio:
        from audio_features import enrich_songs
        print("Enriching with Essentia audio features...")
        songs = enrich_songs(songs)

    print("Building features...")
    df, features = _build_features(songs)
    print(f"  {len(df):,} songs  |  {features.shape[1]} features  |  k={args.k}")

    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    print("Clustering...")
    km = KMeans(n_clusters=args.k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    df["cluster"] = labels

    sil_scores = silhouette_samples(X, labels)
    sil_by_cluster = pd.Series(sil_scores).groupby(labels).mean()

    _PLAYLIST_DIR.mkdir(parents=True, exist_ok=True)

    clusters = [df[df["cluster"] == c].copy() for c in range(args.k)]
    names = _name_clusters(clusters)
    cluster_sil = {c: float(sil_by_cluster[c]) for c in range(args.k)}

    for c, (cluster, name) in enumerate(zip(clusters, names)):
        genre_counts = cluster["genre_bucket"].value_counts()
        genre_pct = (genre_counts / len(cluster) * 100).head(5)
        avg_decade = cluster[cluster["decade"] > 0]["decade"].mean()
        skip_ratio = (cluster["skip_count"] / (cluster["play_count"] + 1)).mean()

        print(f"\n{'=' * 60}\n  PLAYLIST {c + 1}: {name.upper()}\n{'=' * 60}")
        top_genre = genre_counts.index[0]
        print(f"  Songs: {len(cluster):,}  |  Top genre: {top_genre}  |  Avg decade: {int(avg_decade):,}s")
        print(f"\n  Genre mix:")
        for genre, pct in genre_pct.items():
            bar = "█" * int(pct / 3)
            print(f"    {genre:<20} {bar} {pct:.0f}%")

        # One song per artist, capped at --size, ordered by play count
        playlist = (
            cluster.sort_values("play_count", ascending=False)
            .drop_duplicates(subset="artist")
            .drop_duplicates(subset="album")
            .head(args.size)
        )

        print(f"\n  Playlist ({len(playlist)} songs — 1 per artist, top {args.size}):")
        for _, row in playlist[["song_name", "artist", "play_count"]].iterrows():
            print(f"    {int(row['play_count']):>4}x  {row['song_name'][:40]:<40}  {row['artist']}")

        out = _PLAYLIST_DIR / f"playlist_{c + 1}.csv"
        playlist[["song_name", "artist", "album", "genre_clean", "decade",
                   "play_count", "skip_count", "pitchfork_score"]].to_csv(out, index=False)
        print(f"\n  Saved → {out.name}")

    index = {
        f"playlist_{c + 1}.csv": {"name": name, "silhouette": cluster_sil[c]}
        for c, name in enumerate(names)
    }
    (_PLAYLIST_DIR / "index.json").write_text(json.dumps(index, indent=2))
    print()


if __name__ == "__main__":
    main()
