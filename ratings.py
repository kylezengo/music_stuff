import numpy as np
import pandas as pd


def build_albums(df_songs, reviews):
    album_song_count = (
        df_songs.groupby(["album", "album_artist", df_songs["disc_number"].fillna(1)])["track_count"]
        .max()
        .reset_index()
        .groupby(["album", "album_artist"])["track_count"]
        .sum()
        .reset_index()
    )

    df = (
        df_songs.groupby(["album", "album_artist"])
        .agg(
            play_count=("play_count", "sum"),
            skip_count=("skip_count", "sum"),
            total_listen_time=("total_listen_time", "sum"),
            playMinusSkip=("playMinusSkip", "sum"),
            playMinusSkip_time=("playMinusSkip_time", "sum"),
            song_count_lib=("song_name", "count"),
            date_added=("date_added", "min"),
            year_of_release=("year_of_release", "min"),
            genre_clean=("genre_clean", lambda x: x.mode()[0] if x.notna().any() else None),
        )
        .reset_index()
    )

    df = df.merge(album_song_count, how="left", on=["album", "album_artist"])

    today = pd.Timestamp.now(tz="UTC").normalize()
    df["date_added"] = pd.to_datetime(df["date_added"], utc=True)
    df["lib_time"] = (today - df["date_added"]).dt.days

    df["PS_ratio"] = df["play_count"] / df["skip_count"].replace(0, np.nan)
    df["album_complete"] = df["song_count_lib"] / df["track_count"].replace(0, np.nan)
    df["PCT_ratio"] = df["total_listen_time"] / df["lib_time"].replace(0, np.nan)
    df["log_lib_time"] = np.log(df["lib_time"].replace(0, np.nan))
    df["PCT_L_ratio"] = df["total_listen_time"] / df["log_lib_time"]
    df["engagement"] = df["PS_ratio"] * df["album_complete"] * df["PCT_ratio"]
    df["plays_per_track"] = df["play_count"] / df["song_count_lib"].replace(0, np.nan)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    df = df.merge(reviews, how="left", on=["album", "album_artist"])

    df["decade"] = (df["year_of_release"] // 10 * 10).astype(int)

    eligible = (
        df["date_added"] < (today - pd.Timedelta(days=365))
    ) & (
        df["album_complete"].fillna(0) <= 1
    )
    df.loc[eligible, "your_rating"] = (
        df.loc[eligible, "engagement"].rank(pct=True) * 10
    ).round(1)

    return df


def slept_on(albums):
    return albums[
        (albums["pitchfork_score"] >= 8.5) & (albums["plays_per_track"] < 5)
    ].sort_values("pitchfork_score", ascending=False)


def hidden_gems(rated):
    return rated[
        (rated["your_rating"] >= rated["your_rating"].quantile(0.75))
        & (rated["pitchfork_score"].fillna(0) < 7.5)
    ].sort_values("your_rating", ascending=False)


def worth_revisiting(rated, songs):
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=730)
    last_played = (
        songs[songs["play_date"].notna()]
        .groupby(["album", "album_artist"])["play_date"]
        .max()
        .reset_index()
        .rename(columns={"play_date": "last_played"})
    )
    df = rated.merge(last_played, on=["album", "album_artist"], how="left")
    return df[
        (df["your_rating"] >= df["your_rating"].quantile(0.6))
        & (df["last_played"] < cutoff)
    ].sort_values("your_rating", ascending=False)
