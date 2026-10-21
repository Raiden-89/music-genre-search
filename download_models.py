import os
import requests

# Directory where models are saved
MODEL_DIR = "models"

# Essentia model URLs (EffNet + classification head + metadata)
EMBEDDING_MODEL_URL = (
    "https://essentia.upf.edu/models/feature-extractors/discogs-effnet/"
    "discogs-effnet-bs64-1.pb"
)

GENRE_MODEL_URL = (
    "https://essentia.upf.edu/models/classification-heads/genre_discogs400/"
    "genre_discogs400-discogs-effnet-1.pb"
)

META_URL = (
    "https://essentia.upf.edu/models/classification-heads/genre_discogs400/"
    "genre_discogs400-discogs-effnet-1.json"
)


def download_file(url: str, destination: str):
    """Downloads a file from URL if it does not already exist."""
    if os.path.exists(destination):
        print(f"[SKIP] Already exists: {destination}")
        return

    print(f"[DOWNLOAD] {url}")
    try:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(destination, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        print(f"[OK] Saved to: {destination}")
    except Exception as e:
        print(f"[ERROR] Cannot download {url}")
        print(f"        Details: {e}")


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    files_to_download = [
        (EMBEDDING_MODEL_URL, os.path.join(MODEL_DIR, "discogs-effnet-bs64-1.pb")),
        (GENRE_MODEL_URL, os.path.join(MODEL_DIR, "genre_discogs400-discogs-effnet-1.pb")),
        (META_URL, os.path.join(MODEL_DIR, "metadata.json")),
    ]

    for url, dest in files_to_download:
        download_file(url, dest)


if __name__ == "__main__":
    main()
