# music_stuff

Personal music rating and playlist generation system that combines iTunes listening history with Pitchfork reviews.

## What it does

- Computes a personal engagement score per album based on your play/skip behavior
- Matches your library against ~29k Pitchfork reviews
- Surfaces insights: your top albums, critic picks you've ignored, hidden gems, albums worth revisiting
- Generates mood-based playlists using k-means clustering on genre, listening behavior, and review text sentiment
- Exports playlists as M3U files for import into Music.app

## Setup

1. Export your iTunes library: **Music.app → File → Library → Export Library** → save to `data/library/`
2. Run the Pitchfork scraper to collect reviews: `python scraper.py`
3. Run the main report: `python main.py`
4. Generate playlists: `python playlists.py`
5. Launch the web app: `streamlit run app.py`

## Commands

```bash
python main.py                          # ratings report
python playlists.py                     # generate playlists (k=8 default)
python playlists.py --k 6 --size 50    # custom k and playlist size
python export_playlist.py               # list available playlists
python export_playlist.py "Alternative" # export one playlist as M3U
python export_playlist.py --all         # export all playlists
python scraper.py                       # scrape new Pitchfork reviews
streamlit run app.py                    # web UI
```

## Data

All data lives in `data/` (gitignored):
- `data/library/` — iTunes XML exports (drop new exports here, latest is auto-selected)
- `data/reviews/` — Pitchfork review CSVs
- `data/playlists/` — generated playlist CSVs
- `data/m3u/` — exported M3U files for Music.app
