import json
from collections import defaultdict
from difflib import get_close_matches

from config import FUZZY_CACHE


def _key(artist, album):
    return f"{str(artist).strip()}_{str(album).strip()}"


def _norm(s):
    return str(s).lower().strip().rstrip(".")


def fuzzy_match_reviews(df_albums, reviews, threshold=0.85):
    reviews = reviews.copy()
    reviews["_key_norm"] = (
        reviews["album_artist"].fillna("").str.strip() + "_" + reviews["album"].fillna("").str.strip()
    ).str.lower().str.rstrip(".")

    review_lookup = reviews.drop_duplicates("_key_norm").set_index("_key_norm")
    pitchfork_cols = [c for c in reviews.columns if c.startswith("pitchfork_")]

    artist_keys = defaultdict(list)
    for k in review_lookup.index:
        artist_keys[k.split("_", maxsplit=1)[0]].append(k)
    all_keys = list(review_lookup.index)

    # Load cache: maps library_key → matched review_key (or "" for no match)
    cache = {}
    if FUZZY_CACHE.exists():
        with open(FUZZY_CACHE, encoding="utf-8") as f:
            cache = json.load(f)

    unmatched = df_albums[df_albums["pitchfork_score"].isna()]
    result = df_albums.copy()
    matched = new_computations = 0

    for row in unmatched.itertuples():
        idx = row.Index
        key = _norm(_key(getattr(row, "album_artist", "") or "", getattr(row, "album", "") or ""))

        if key in cache:
            match_key = cache[key] or None
        else:
            artist_norm = key.split("_")[0]
            candidates = artist_keys.get(artist_norm, [])
            close = get_close_matches(key, candidates, n=1, cutoff=threshold) if candidates else []
            if not close:
                close = get_close_matches(key, all_keys, n=1, cutoff=threshold)
            match_key = close[0] if close else None
            cache[key] = match_key or ""
            new_computations += 1

        if not match_key or match_key not in review_lookup.index:
            continue

        matched_row = review_lookup.loc[match_key]
        result.loc[idx, pitchfork_cols] = matched_row[pitchfork_cols].values
        matched += 1

    if new_computations > 0:
        FUZZY_CACHE.parent.mkdir(parents=True, exist_ok=True)
        with open(FUZZY_CACHE, "w", encoding="utf-8") as f:
            json.dump(cache, f)

    return result, matched
