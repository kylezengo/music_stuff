#!/usr/bin/env python
import json
import pandas as pd
import streamlit as st
from pathlib import Path

from config import LIBRARY_XML, REVIEWS_DIR
from library import load_library
from reviews import load_reviews
from ratings import build_albums, slept_on, hidden_gems, worth_revisiting
from matching import fuzzy_match_reviews
from playlists import load_playlist_index, _PLAYLIST_DIR

st.set_page_config(page_title="Music", layout="wide", page_icon="🎵")

_DISPLAY_NAMES = {
    "album": "Album", "album_artist": "Artist", "your_rating": "Your Rating",
    "pitchfork_score": "Pitchfork", "play_count": "Plays", "skip_count": "Skips",
    "genre_clean": "Genre", "decade": "Decade", "plays_per_track": "Plays/Track",
    "last_played": "Last Played",
}


@st.cache_data
def load_data():
    songs = load_library(LIBRARY_XML)
    reviews = load_reviews(REVIEWS_DIR)
    albums = build_albums(songs, reviews)
    albums, _ = fuzzy_match_reviews(albums, reviews)
    revisit = worth_revisiting(albums[albums["your_rating"].notna()], songs)
    return songs, albums, revisit


def fmt_hours(ms_series):
    return (ms_series / 3_600_000).round(0).astype(int)


with st.spinner("Loading library..."):
    songs, albums, revisit_df = load_data()

rated = albums[albums["your_rating"].notna()]

st.title("🎵 Music")
tab_overview, tab_albums, tab_discover, tab_playlists = st.tabs(
    ["Overview", "Albums", "Discover", "Playlists"]
)

# ── Overview ──────────────────────────────────────────────────────────────────

with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Songs", f"{len(songs):,}")
    c2.metric("Albums", f"{len(albums):,}")
    c3.metric("Pitchfork Match", f"{albums['pitchfork_score'].notna().mean():.0%}")
    c4.metric("Rated Albums", f"{len(rated):,}")

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Listen Time by Genre")
        genre_chart = (
            songs[songs["genre_clean"].notna()]
            .groupby("genre_clean")["total_listen_time"]
            .sum()
            .pipe(fmt_hours)
            .sort_values(ascending=False)
            .head(12)
            .rename("Hours")
            .reset_index()
            .rename(columns={"genre_clean": "Genre"})
            .set_index("Genre")
        )
        st.bar_chart(genre_chart)

    with col2:
        st.subheader("Songs by Decade")
        decade_chart = (
            songs[(songs["decade"] > 0) & (songs["decade"] <= 2030)]
            .groupby("decade")["song_name"]
            .count()
            .rename("Songs")
            .reset_index()
            .rename(columns={"decade": "Decade"})
            .set_index("Decade")
        )
        st.bar_chart(decade_chart)

    st.subheader("Top Artists by Listen Time")
    top_artists = (
        songs.groupby("artist")["total_listen_time"]
        .sum()
        .pipe(fmt_hours)
        .sort_values(ascending=False)
        .head(15)
        .rename("Hours")
        .reset_index()
        .rename(columns={"artist": "Artist"})
        .set_index("Artist")
    )
    st.bar_chart(top_artists)


# ── Albums ────────────────────────────────────────────────────────────────────

with tab_albums:
    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        genre_opts = ["All"] + sorted(albums["genre_clean"].dropna().unique().tolist())
        genre_filter = st.selectbox("Genre", genre_opts)

    with col2:
        sort_by = st.selectbox(
            "Sort by",
            ["your_rating", "pitchfork_score", "play_count"],
            format_func=lambda x: {
                "your_rating": "Your Rating",
                "pitchfork_score": "Pitchfork Score",
                "play_count": "Play Count",
            }[x],
        )

    with col3:
        rated_only = st.checkbox("Rated only", value=True)

    df = rated if rated_only else albums
    if genre_filter != "All":
        df = df[df["genre_clean"] == genre_filter]
    df = df.sort_values(sort_by, ascending=False)

    album_cols = ["album", "album_artist", "your_rating", "pitchfork_score",
                  "play_count", "skip_count", "genre_clean", "decade"]
    st.dataframe(
        df[album_cols].rename(columns=_DISPLAY_NAMES),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Your Rating": st.column_config.NumberColumn(format="%.1f"),
            "Pitchfork": st.column_config.NumberColumn(format="%.1f"),
        },
    )


# ── Discover ──────────────────────────────────────────────────────────────────

with tab_discover:
    d1, d2, d3 = st.tabs(["Slept On", "Hidden Gems", "Worth Revisiting"])

    with d1:
        st.subheader("Critic's Picks You've Slept On")
        st.caption("Pitchfork ≥ 8.5, fewer than 5 plays per track")
        cols = ["album", "album_artist", "pitchfork_score", "plays_per_track", "play_count", "genre_clean"]
        st.dataframe(
            slept_on(albums)[cols].rename(columns=_DISPLAY_NAMES),
            use_container_width=True, hide_index=True,
            column_config={"Plays/Track": st.column_config.NumberColumn(format="%.1f")},
        )

    with d2:
        st.subheader("Your Hidden Gems")
        st.caption("Your top 25%, Pitchfork < 7.5 or no review")
        cols = ["album", "album_artist", "your_rating", "pitchfork_score", "play_count", "genre_clean"]
        st.dataframe(
            hidden_gems(rated)[cols].rename(columns=_DISPLAY_NAMES),
            use_container_width=True, hide_index=True,
            column_config={"Your Rating": st.column_config.NumberColumn(format="%.1f")},
        )

    with d3:
        st.subheader("Worth Revisiting")
        st.caption("Top-rated albums not played in 2+ years")
        revisit = revisit_df[["album", "album_artist", "your_rating", "last_played", "play_count", "genre_clean"]].copy()
        revisit["last_played"] = pd.to_datetime(revisit["last_played"]).dt.strftime("%Y-%m")
        st.dataframe(
            revisit.rename(columns=_DISPLAY_NAMES),
            use_container_width=True, hide_index=True,
            column_config={"Your Rating": st.column_config.NumberColumn(format="%.1f")},
        )


# ── Playlists ─────────────────────────────────────────────────────────────────

with tab_playlists:
    playlist_files = sorted(_PLAYLIST_DIR.glob("playlist_*.csv")) if _PLAYLIST_DIR.exists() else []

    if not playlist_files:
        st.info("No playlists found. Run `python playlists.py` to generate them.")
    else:
        index = load_playlist_index()
        labels = [index[f.name]["name"] if f.name in index else f.stem for f in playlist_files]
        sil_map = {f.name: index[f.name]["silhouette"] for f in playlist_files if f.name in index}

        if sil_map:
            order = sorted(range(len(playlist_files)), key=lambda i: sil_map.get(playlist_files[i].name, 0), reverse=True)
            labels = [labels[i] for i in order]
            playlist_files = [playlist_files[i] for i in order]

        selected_label = st.selectbox("Playlist", labels)
        selected_file = playlist_files[labels.index(selected_label)]

        @st.cache_data
        def _load_playlist(path):
            return pd.read_csv(path)

        pl = _load_playlist(str(selected_file))
        sil = sil_map.get(selected_file.name)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Songs", len(pl))
        c2.metric("Avg Plays", f"{pl['play_count'].mean():.0f}")
        p4k_mean = pl["pitchfork_score"].mean()
        c3.metric("Avg Pitchfork", f"{p4k_mean:.1f}" if pd.notna(p4k_mean) else "N/A")
        c4.metric("Cohesion", f"{sil:.2f}" if sil is not None else "N/A",
                  help="Silhouette score: how well-defined this cluster is (0–1, higher = tighter)")

        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("Genre Mix")
            st.bar_chart(
                pl["genre_clean"].value_counts()
                .rename("Songs").reset_index()
                .rename(columns={"genre_clean": "Genre"})
                .set_index("Genre")
            )

        with col2:
            st.subheader("Tracks")
            st.dataframe(
                pl[["song_name", "artist", "album", "genre_clean", "play_count", "pitchfork_score"]]
                .rename(columns={"song_name": "Song", "artist": "Artist", "album": "Album",
                                 "genre_clean": "Genre", "play_count": "Plays", "pitchfork_score": "Pitchfork"}),
                use_container_width=True, hide_index=True,
                column_config={"Pitchfork": st.column_config.NumberColumn(format="%.1f")},
            )
