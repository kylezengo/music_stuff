"""Extract mood dimensions from Pitchfork review text via keyword scoring.

Produces four 0-1 scores per album (where a review exists):
  mood_energy      high = intense/driving/explosive, low = ambient/hushed/sparse
  mood_valence     high = joyful/euphoric/bright,    low = bleak/haunting/mournful
  mood_danceability high = groovy/infectious/rhythmic, low = formless/abstract/droning
  mood_acousticness high = acoustic/organic/intimate,  low = electronic/synthetic/processed
"""

import re
import numpy as np
import pandas as pd

MOOD_COLS = ["mood_energy", "mood_valence", "mood_danceability", "mood_acousticness"]

_KEYWORDS = {
    "mood_energy": {
        "high": [
            "relentless", "explosive", "frenetic", "frantic", "ferocious", "furious",
            "kinetic", "propulsive", "blistering", "thunderous", "roaring", "visceral",
            "driving", "urgent", "intense", "pummeling", "careening", "hurtling",
            "cathartic", "anthemic", "euphoric", "soaring", "scorching", "acrobatic",
            "whiplash", "aggressive", "punishing", "chaotic", "relentlessly",
        ],
        "low": [
            "languid", "meditative", "hushed", "sparse", "understated", "contemplative",
            "introspective", "minimal", "minimalist", "ambient", "drone", "serene",
            "tranquil", "gentle", "quiet", "subdued", "slow-burning", "unhurried",
            "restrained", "delicate", "fragile", "pastoral", "beatific", "plaintive",
        ],
    },
    "mood_valence": {
        "high": [
            "joyful", "euphoric", "celebratory", "exuberant", "uplifting", "playful",
            "warm", "optimistic", "bright", "cheerful", "buoyant", "sunny", "hopeful",
            "triumphant", "jubilant", "whimsical", "breezy", "lighthearted", "fun",
            "infectious", "irresistible", "giddy", "ebullient",
        ],
        "low": [
            "bleak", "dark", "melancholic", "somber", "haunting", "desolate", "forlorn",
            "mournful", "grim", "sinister", "ominous", "brooding", "oppressive",
            "despairing", "nihilistic", "dystopian", "tragic", "grief", "sorrow",
            "wistful", "longing", "elegiac", "anguish", "torpor", "depressive",
            "desolation", "dread",
        ],
    },
    "mood_danceability": {
        "high": [
            "danceable", "groove", "groovy", "funky", "rhythmic", "hypnotic", "pulsating",
            "infectious", "irresistible", "club", "rave", "party", "syncopated",
            "head-nodding", "foot-tapping", "floor", "beat", "bounce", "swagger",
        ],
        "low": [
            "formless", "droning", "drone", "sprawling", "meandering", "static",
            "abstract", "atonal", "noise", "texture", "textural", "shapeless",
        ],
    },
    "mood_acousticness": {
        "high": [
            "acoustic", "guitar", "piano", "vocal", "organic", "stripped", "unplugged",
            "intimate", "chamber", "strings", "fingerpicking", "voice", "folk",
            "orchestral", "live", "handmade", "analog", "wooden",
        ],
        "low": [
            "synthesizer", "synth", "electronic", "digital", "programmed", "drum machine",
            "samples", "sampling", "glitch", "glitchy", "processed", "distorted",
            "industrial", "techno", "house", "modular", "computerized",
        ],
    },
}


def _compile(words):
    return re.compile(r"\b(?:" + "|".join(re.escape(w) for w in words) + r")\b")


# Pre-compile all patterns once at module load
_PATTERNS = {
    col: (_compile(words["high"]), _compile(words["low"]))
    for col, words in _KEYWORDS.items()
}


def _score(text, pos_pat, neg_pat):
    if not isinstance(text, str) or not text:
        return np.nan
    t = text.lower()
    pos = len(pos_pat.findall(t))
    neg = len(neg_pat.findall(t))
    return (pos - neg) / (pos + neg + 1)


def extract_mood(reviews):
    """Return reviews with mood_* columns added (0-1, normalized across corpus)."""
    df = reviews.copy()
    text_col = next((c for c in df.columns if "review" in c and df[c].dtype == object), None)
    if text_col is None:
        return df

    for col, (pos_pat, neg_pat) in _PATTERNS.items():
        raw = df[text_col].apply(lambda t: _score(t, pos_pat, neg_pat))
        valid = raw.dropna()
        if len(valid) > 0:
            lo, hi = valid.min(), valid.max()
            df[col] = (raw - lo) / (hi - lo) if hi > lo else raw
        else:
            df[col] = raw

    return df
