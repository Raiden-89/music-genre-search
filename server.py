import os
import json
import shutil
import uuid

import numpy as np
import requests
import tempfile
import essentia.standard as es
from fastapi import FastAPI, UploadFile, File
from typing import Optional

from fastapi import HTTPException  # at the top of the file, with the other imports
from fastapi.staticfiles import StaticFiles
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# FastAPI instance
app = FastAPI(
    title="Music Genre Embeddings API",
    description=(
        "API to analyze MP3 tracks, extract genre distributions "
        "and save genre embeddings (400D, top-10 non-zero) to Qdrant."
    ),
)




# =========================
#   CONFIG OFFLINE / ONLINE
# =========================
# True = use only local models in ./models; False = download them if missing
FLAG_OFFLINE = os.getenv("OFFLINE_MODE", "true").lower() == "true"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)  # optional but convenient

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# --- MODEL URLS (used only in online mode) ---
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

# Local paths to model files
embedding_model_path = os.path.join(MODEL_DIR, "discogs-effnet-bs64-1.pb")
genre_model_path = os.path.join(MODEL_DIR, "genre_discogs400-discogs-effnet-1.pb")
meta_path = os.path.join(MODEL_DIR, "metadata.json")

TOP_K_GENRES = 10  # Number of genres to keep in the vector (others set to 0)


def download_file(url: str, destination: str):
    """
    Downloads a file from URL if it does not already exist.
    Only used when FLAG_OFFLINE = False.
    """
    if os.path.exists(destination):
        return

    print(f"Downloading {url} -> {destination} ...")
    try:
        r = requests.get(url, stream=True, timeout=60)
        r.raise_for_status()
        with open(destination, "wb") as f:
            for chunk in r.iter_content(1024):
                if chunk:
                    f.write(chunk)
        print("Done.")
    except Exception as e:
        print(f"Download error {url}: {e}")


# =========================
#   MODEL LOADING
# =========================

if FLAG_OFFLINE:
    # OFFLINE mode: expects files to already be in ./models
    for path, name in [
        (embedding_model_path, "EffNet embedding"),
        (genre_model_path, "genre classification head"),
        (meta_path, "genre metadata"),
    ]:
        if not os.path.exists(path):
            print(f"[ERROR] OFFLINE mode but missing file {name}: {path}")
else:
    # ONLINE mode: download if missing
    download_file(EMBEDDING_MODEL_URL, embedding_model_path)
    download_file(GENRE_MODEL_URL, genre_model_path)
    download_file(META_URL, meta_path)

# -------------------------------------------------
# Load the list of classes (genre names)
# -------------------------------------------------
classes = []
if os.path.exists(meta_path):
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
            classes = meta.get("classes", [])
    except Exception as e:
        print(f"Cannot read metadata.json: {e}")
else:
    print("[WARN] metadata.json not found, classes will be empty.")

num_classes = None  # will be determined at first inference

# -------------------------------------------------
# Initialize Essentia models once at startup
# -------------------------------------------------
embedding_model = None
genre_model = None

try:
    # EffNet embedding model (Erdos + internal mel)
    embedding_model = es.TensorflowPredictEffnetDiscogs(
        graphFilename=embedding_model_path,
        output="PartitionedCall:1",  # embedding output [T, emb_dim]
    )

    # Classification head (400 Discogs genres)
    genre_model = es.TensorflowPredict2D(
        graphFilename=genre_model_path,
        input="serving_default_model_Placeholder",
        output="PartitionedCall:0",  # output [T, num_classes] or [1, T, num_classes]
    )

    print("--- MODELS LOADED SUCCESSFULLY ---")

except Exception as e:
    print(f"Error during TensorFlow model initialization: {e}")


# =========================
#     QDRANT CONFIG
# =========================

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "tracks_genre_embeddings")

qdrant_client: Optional[QdrantClient] = None

try:
    qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    print(f"Connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
except Exception as e:
    print(f"[WARN] Cannot connect to Qdrant: {e}")


def ensure_collection(vector_size: int):
    """
    Creates the Qdrant collection (if needed) with size = number of genres.
    """
    if qdrant_client is None:
        return

    try:
        qdrant_client.get_collection(QDRANT_COLLECTION)
        # NOTE: you could also verify that the existing dimension
        # matches vector_size; if not, handle accordingly.
    except Exception:
        print(f"Creating Qdrant collection '{QDRANT_COLLECTION}' with dim={vector_size}")
        qdrant_client.recreate_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,  # cosine similarity on the genre vector
            ),
        )


# =========================
#  UTILITY FUNCTIONS
# =========================

def process_audio_file(temp_filename: str):
    """
    Loads audio and computes:
    - EffNet embeddings [T, emb_dim]
    - genre activations [T, num_classes]
    Returns (embeddings, avg_activations).
    """
    if embedding_model is None or genre_model is None:
        raise RuntimeError(
            "Essentia/TensorFlow models not initialized. "
            "Verify that essentia-tensorflow is installed correctly."
        )

    # Load audio (mono, 16 kHz)
    loader = es.MonoLoader(
        filename=temp_filename,
        sampleRate=16000,
        resampleQuality=4,
    )
    audio = loader()

    # EffNet embeddings [T, emb_dim] (not used directly for Qdrant,
    # but could be useful in the future for other features)
    embeddings = embedding_model(audio)

    # Genre activations [T, num_classes] or [1, T, num_classes]
    predictions = genre_model(embeddings)
    activations = np.array(predictions)

    # Normalize shape to [T, num_classes]
    if activations.ndim == 3:
        activations = activations[0]
    elif activations.ndim > 3:
        activations = activations.reshape(-1, activations.shape[-1])

    if activations.ndim != 2:
        raise RuntimeError(f"Unexpected model output shape: {activations.shape}")

    # Temporal average -> vector [num_classes]
    avg_activations = np.mean(activations, axis=0)

    # Ensure num_classes is set
    global num_classes
    if num_classes is None:
        num_classes = int(avg_activations.shape[0])

    return embeddings, avg_activations


def build_genre_vector_and_topk(avg_activations: np.ndarray):
    """
    Given the average activation vector [num_classes],
    builds:
    - a dense 400D (or num_classes) vector with only TOP_K_GENRES non-zero values
    - a dict {genre: percentage} for the top-K
    """
    if avg_activations.ndim != 1:
        raise ValueError("avg_activations must be 1D")

    # Clamp negative values to 0
    avg_activations = np.maximum(avg_activations, 0.0)

    # Sort classes by descending score
    indices = np.argsort(avg_activations)[::-1]
    top_indices = indices[:TOP_K_GENRES]

    dense_vector = np.zeros_like(avg_activations, dtype=float)

    # Copy scores for top-K only
    dense_vector[top_indices] = avg_activations[top_indices]

    # Build the top-K genre dict
    results = {}
    for i in top_indices:
        if i < len(classes):
            genre_name = classes[i]
        else:
            genre_name = f"genre_{i}"
        # Not normalizing to sum 1; score * 100 as per original logic
        results[genre_name] = round(float(avg_activations[i]) * 100, 2)

    return dense_vector, results


# =========================
#        ENDPOINT /analyze
# =========================
@app.post("/analyze")
async def analyze_genre(file: UploadFile = File(...)):
    """
    Analyzes the MP3 track:
    - computes the genre distribution (400 classes)
    - builds a dense 400D genre vector:
      only the 10 genres with the highest magnitude are non-zero; the rest are 0.
    Does NOT save anything to Qdrant, only returns the result.
    """
    ext = os.path.splitext(file.filename or "")[1] or ".mp3"
    temp_filename = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4().hex}{ext}")

    try:
        # Save the audio file temporarily to disk
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Feature extraction
        _, avg_activations = process_audio_file(temp_filename)

        # Build genre vector + top-K
        genre_vector, top_genres = build_genre_vector_and_topk(avg_activations)

        return {
            "filename": file.filename,
            "top_genres": top_genres,
            "genre_vector": genre_vector.tolist(),  # optional, useful for debug/analysis
        }

    except Exception as e:
        print(f"INTERNAL ERROR /analyze: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)


# =========================
#        ENDPOINT /index
# =========================
@app.post("/index")
async def index_track(
    file: UploadFile = File(...),
    track_id: Optional[str] = None,
):
    """
    Analyzes the MP3 track and SAVES the genre embedding to Qdrant.
    - dense 400D vector (num_classes) with only the top-10 genres non-zero.
    - payload with filename and top_genres.
    """
    if qdrant_client is None:
        return {"error": "Qdrant not available; cannot index tracks."}

    ext = os.path.splitext(file.filename or "")[1] or ".mp3"
    temp_filename = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4().hex}{ext}")

    try:
        # Save the audio file temporarily
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Feature extraction
        _, avg_activations = process_audio_file(temp_filename)

        # Build genre vector + top-K
        genre_vector, top_genres = build_genre_vector_and_topk(avg_activations)

        vector_size = int(genre_vector.shape[0])
        ensure_collection(vector_size)

        # Track ID in Qdrant (if not provided, use UUID4)
        if track_id is None:
            track_id = str(uuid.uuid4())

        point = PointStruct(
            id=track_id,
            vector=genre_vector.tolist(),
            payload={
                "filename": file.filename,
                "top_genres": top_genres,
            },
        )

        qdrant_client.upsert(
            collection_name=QDRANT_COLLECTION,
            points=[point],
        )

        return {
            "track_id": track_id,
            "filename": file.filename,
            "top_genres": top_genres,
            "message": "Track successfully indexed in Qdrant.",
        }

    except Exception as e:
        print(f"INTERNAL ERROR /index: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)


# =========================
#        ENDPOINT /similar
# =========================
@app.post("/similar")
async def similar_tracks(file: UploadFile = File(...), limit: int = 10):
    """
    Analyzes the MP3 track and returns:
    - the top genres of the query track
    - the 'limit' most similar tracks already stored in Qdrant
      (similarity computed on the dense 400D genre vector with top-10 non-zero).
    """
    if embedding_model is None or genre_model is None:
        return {"error": "Essentia/TensorFlow models not initialized."}

    if qdrant_client is None:
        return {"error": "Qdrant not available; cannot search for similar tracks."}

    ext = os.path.splitext(file.filename or "")[1] or ".mp3"
    temp_filename = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4().hex}{ext}")

    try:
        # Save the audio file temporarily
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Feature extraction
        _, avg_activations = process_audio_file(temp_filename)

        # Build the genre vector for the query
        genre_vector, top_genres = build_genre_vector_and_topk(avg_activations)

        vector_size = int(genre_vector.shape[0])
        ensure_collection(vector_size)

        # Query Qdrant
        response = qdrant_client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=genre_vector.tolist(),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

        points = response.points or []

        similar = []
        for p in points:
            similar.append(
                {
                    "id": p.id,
                    "score": p.score,  # higher = more similar
                    "filename": (p.payload or {}).get("filename"),
                    "top_genres": (p.payload or {}).get("top_genres"),
                }
            )

        return {
            "filename": file.filename,
            "top_genres": top_genres,
            "similar_tracks": similar,
        }

    except Exception as e:
        print(f"INTERNAL ERROR /similar: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

# =========================
#        ENDPOINT /debug
# =========================
@app.get("/debug/{track_id}")
async def debug_track(track_id: str, with_vector: bool = True):
    """
    Returns what is stored in Qdrant for a given track:
    - full payload (filename, top_genres, etc.)
    - embedding vector (optional, controlled by with_vector)
    """
    if qdrant_client is None:
        raise HTTPException(status_code=500, detail="Qdrant not available.")

    try:
        points = qdrant_client.retrieve(
            collection_name=QDRANT_COLLECTION,
            ids=[track_id],
            with_payload=True,
            with_vectors=with_vector,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Qdrant error: {e}")

    if not points:
        raise HTTPException(status_code=404, detail=f"Track {track_id} not found.")

    p = points[0]
    payload = p.payload or {}

    response = {
        "id": p.id,
        "filename": payload.get("filename"),
        "top_genres": payload.get("top_genres"),
        "payload": payload,
    }

    if with_vector and p.vector is not None:
        # p.vector is a list of floats
        response["vector_dim"] = len(p.vector)
        response["vector_preview"] = p.vector[:20]  # first 20 values, to avoid bloat
        response["vector_full"] = p.vector          # full vector if needed

    return response
