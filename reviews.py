from pathlib import Path
import pandas as pd

_ARTIST_CORRECTIONS = {
    "Run the Jewels": "Run The Jewels",
    "Gojira": "GOJIRA",
    "Nicolás Jaar": "Nicolas Jaar",
    "Jeffrey Lewis & the Jitters": "Jeffrey Lewis",
    "King Gizzard & the Lizard Wizard": "King Gizzard & The Lizard Wizard",
}

_ALBUM_CORRECTIONS = {
    "Indigo Child": "Indigo Child - EP",
    "Keep it Like a Secret": "Keep It Like a Secret",
    "Hounds of Love": "Hounds of Love (Remastered)",
    "Instrumental Tape 2": "Instrumental Mixtape 2",
    "The Things We Do to Find People Who Feel Like Us": "The Things We Do To Find People Who Feel Like Us",
    "How You Sell Soul to a Souless People Who Sold Their Soul": "How You Sell Soul to a Soulless People Who Sold Their Soul",
    "How You Sell Soul to a Souless People Who Sold Their Soul???": "How You Sell Soul to a Soulless People Who Sold Their Soul?",
    "No One's First and You're Next": "No One's First, and You're Next",
    "Lift Your Skinny Fists like Antennas to Heaven": "Lift Your Skinny Fists Like Antennas to Heaven",
    "Allelujah! Don't Bend! Ascend!": "'Allelujah! Don't Bend! Ascend!",
    "Hurry Up, We're Dreaming": "Hurry Up, We're Dreaming.",
    "Digital Shades Vol. 1": "Digital Shades, Vol. 1",
    "Good News For People Who Love Bad News": "Good News for People Who Love Bad News",
}


def load_reviews(reviews_dir):
    dfs = [pd.read_csv(f) for f in Path(reviews_dir).glob("*.csv")]
    if not dfs:
        raise FileNotFoundError(f"No CSV files found in {reviews_dir}")
    df = pd.concat(dfs).reset_index(drop=True)

    df["artist"] = df["artist"].replace(_ARTIST_CORRECTIONS)
    df["album"] = df["album"].replace(_ALBUM_CORRECTIONS)
    df["album"] = df["album"].str.strip()

    df.columns = "pitchfork_" + df.columns
    df = df.rename(columns={"pitchfork_artist": "album_artist", "pitchfork_album": "album"})
    df = df.drop_duplicates(subset=["album", "album_artist"])

    return df
