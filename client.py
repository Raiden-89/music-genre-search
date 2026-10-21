import requests
import json
import os
import sys
import argparse

BASE_URL = "http://127.0.0.1:8000"


def call_endpoint(endpoint: str, audio_file_path: str, extra_params=None):
    if extra_params is None:
        extra_params = {}

    url = f"{BASE_URL}/{endpoint}"

    if not os.path.exists(audio_file_path):
        print(f"[ERROR] File not found: {audio_file_path}")
        sys.exit(1)

    print(f"Calling {url}")
    print(f"File: {audio_file_path}")

    try:
        with open(audio_file_path, "rb") as f:
            files = {"file": f}
            response = requests.post(url, files=files, params=extra_params)

        print(f"Status Code: {response.status_code}")
        print("SERVER RESPONSE:")

        try:
            data = response.json()
            print(json.dumps(data, indent=2, ensure_ascii=False))
        except Exception:
            # If not valid JSON, print raw text
            print(response.text)

    except Exception as e:
        print(f"Client script error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="CLI client for Music Genre Embeddings API (FastAPI + Qdrant)"
    )
    parser.add_argument(
        "command",
        choices=["analyze", "index", "similar"],
        help="Operation to perform: analyze | index | similar",
    )
    parser.add_argument(
        "audio_file",
        help="Path to the MP3/WAV audio file to analyze",
    )
    parser.add_argument(
        "--track-id",
        help="Track ID to use in Qdrant (only for 'index')",
        default=None,
    )
    parser.add_argument(
        "--limit",
        help="Maximum number of similar tracks to return (only for 'similar')",
        type=int,
        default=10,
    )

    args = parser.parse_args()

    if args.command == "analyze":
        call_endpoint("analyze", args.audio_file)

    elif args.command == "index":
        extra = {}
        if args.track_id:
            extra["track_id"] = args.track_id
        call_endpoint("index", args.audio_file, extra_params=extra)

    elif args.command == "similar":
        extra = {"limit": args.limit}
        call_endpoint("similar", args.audio_file, extra_params=extra)


if __name__ == "__main__":
    main()
