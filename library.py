import urllib.parse
import xml.etree.ElementTree as ET
import pandas as pd

from config import GENRE_CLEANUP, GENRE_NULLS

_COL_MAP = {
    "Track ID": "track_id",
    "Name": "song_name",
    "Play Count": "play_count",
    "Skip Count": "skip_count",
    "Album": "album",
    "Artist": "artist",
    "Album Artist": "album_artist",
    "Genre": "genre",
    "Kind": "kind",
    "Persistent ID": "persistent_id",
    "Year": "year_of_release",
    "Play Date UTC": "play_date",
    "Skip Date": "skip_date",
    "Release Date": "release_date",
    "Date Modified": "date_modified",
    "Date Added": "date_added",
    "Disc Count": "disc_count",
    "Disc Number": "disc_number",
    "Track Count": "track_count",
    "Track Number": "track_number",
    "Total Time": "total_time",
    "Location": "file_path",
}


def _parse_track(elem):
    items = list(elem)
    d = {}
    for i in range(0, len(items) - 1, 2):
        if items[i].tag == "key":
            d[items[i].text] = items[i + 1].text
    return d


def _playlist_track_ids(playlists, name):
    for pl in playlists:
        items = list(pl)
        for j, el in enumerate(items):
            if el.tag == "key" and el.text == "Name" and j + 1 < len(items) and items[j + 1].text == name:
                for k, el2 in enumerate(items):
                    if el2.tag == "key" and el2.text == "Playlist Items" and k + 1 < len(items):
                        return [
                            int(tid.text)
                            for track_dict in items[k + 1].findall("dict")
                            for tid in track_dict.findall("integer")
                        ]
    return []


def load_library(xml_path, playlist_names=None):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    top = root.findall("dict")[0]

    tracks_dict = playlist_array = None
    for item in list(top):
        if item.tag == "dict" and tracks_dict is None:
            tracks_dict = item
        elif item.tag == "array" and playlist_array is None:
            playlist_array = item
        if tracks_dict is not None and playlist_array is not None:
            break

    playlists = playlist_array.findall("dict") if playlist_array is not None else []
    raw = [_parse_track(t) for t in tracks_dict.findall("dict")]

    df = pd.DataFrame(raw)
    if "Genre" in df.columns:
        df = df[df["Genre"] != "Podcast"].copy()

    present = [k for k in _COL_MAP if k in df.columns]
    df = df[present].rename(columns={k: _COL_MAP[k] for k in present})

    if "file_path" in df.columns:
        df["file_path"] = df["file_path"].apply(
            lambda x: urllib.parse.unquote(x.replace("file://", "")) if pd.notna(x) else None
        )

    for col in ["track_id", "play_count", "skip_count", "year_of_release",
                "disc_count", "disc_number", "track_count", "track_number", "total_time"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["play_date", "skip_date", "release_date", "date_modified", "date_added"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

    for col in ["play_count", "skip_count"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = df[col].fillna(0).astype(int)

    if "year_of_release" not in df.columns:
        df["year_of_release"] = 0
    df["year_of_release"] = df["year_of_release"].fillna(0).astype(int)

    if "album_artist" not in df.columns:
        df["album_artist"] = df.get("artist")
    df["album_artist"] = df["album_artist"].fillna(df["artist"])

    total_time = df["total_time"].fillna(0)
    df["total_listen_time"] = df["play_count"] * total_time
    df["playMinusSkip"] = df["play_count"] - df["skip_count"]
    df["playMinusSkip_time"] = df["playMinusSkip"] * total_time
    df["decade"] = (df["year_of_release"] // 10 * 10).astype(int)

    df["genre_clean"] = df["genre"].copy() if "genre" in df.columns else None
    df.loc[df["genre_clean"].isin(GENRE_NULLS), "genre_clean"] = None
    df["genre_clean"] = df["genre_clean"].replace(GENRE_CLEANUP)

    if playlist_names:
        for name in playlist_names:
            ids = set(_playlist_track_ids(playlists, name))
            df[f"in_{name.lower().replace(' ', '_')}"] = df["track_id"].isin(ids)

    return df
