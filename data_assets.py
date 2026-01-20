from __future__ import annotations

from pathlib import Path
from typing import Tuple
import zipfile

import requests

from app_config import DATA_CACHE_DIR, OUTDOOR_TEMPS_DIR, OUTDOOR_TEMPS_GDRIVE_FILE_ID


def _has_outdoor_temps() -> bool:
    if not OUTDOOR_TEMPS_DIR.exists():
        return False
    return any(OUTDOOR_TEMPS_DIR.rglob("*.parquet"))


def _download_file_from_google_drive(file_id: str, destination: Path) -> Tuple[bool, str]:
    url = "https://drive.google.com/uc?export=download"

    def get_confirm_token(response) -> str:
        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                return value
        return ""

    try:
        session = requests.Session()
        response = session.get(url, params={"id": file_id}, stream=True, timeout=60)
        token = get_confirm_token(response)
        if token:
            response = session.get(url, params={"id": file_id, "confirm": token}, stream=True, timeout=60)

        if response.status_code != 200:
            return False, f"Download failed (status {response.status_code})"

        destination.parent.mkdir(parents=True, exist_ok=True)
        with open(destination, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        return True, "ok"
    except Exception as e:
        return False, f"Download error: {e}"


def ensure_outdoor_temps_dataset() -> Tuple[bool, str]:
    """
    Ensure the 3_Outdoor_temps dataset exists locally. Downloads and unzips if missing.
    """
    if _has_outdoor_temps():
        return True, "Already available"

    zip_path = DATA_CACHE_DIR / "3_Outdoor_temps.zip"
    ok, msg = _download_file_from_google_drive(OUTDOOR_TEMPS_GDRIVE_FILE_ID, zip_path)
    if not ok:
        return False, msg

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(DATA_CACHE_DIR)
    except Exception as e:
        return False, f"Unzip error: {e}"

    if _has_outdoor_temps():
        return True, "Downloaded and extracted"
    return False, "Dataset not found after extraction"
