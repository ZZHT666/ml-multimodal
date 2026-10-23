"""Locate model artifacts: a local path, a gs:// URI, or Vertex AI's AIP_STORAGE_URI.

On Vertex AI, `gcloud ai models upload --artifact-uri gs://.../model/` makes
Vertex copy that folder and pass its location to the container as the
AIP_STORAGE_URI environment variable. Locally you just point MODEL_PATH /
INDEX_PATH at files under outputs/.
"""
import hashlib
import os
from pathlib import Path

CKPT_NAME = "fusion_model.pt"
INDEX_NAME = "gallery_index.pt"


def _download_gcs(uri: str, dest_dir: Path) -> Path:
    from google.cloud import storage  # imported lazily: only needed on GCP

    bucket_name, _, blob_name = uri[len("gs://"):].partition("/")
    dest = dest_dir / Path(blob_name).name
    if not dest.exists():
        dest_dir.mkdir(parents=True, exist_ok=True)
        storage.Client().bucket(bucket_name).blob(blob_name).download_to_filename(str(dest))
    return dest


def resolve(path_or_uri: str, cache_dir: str = "/tmp/artifacts") -> Path:
    if path_or_uri.startswith("gs://"):
        return _download_gcs(path_or_uri, Path(cache_dir))
    path = Path(path_or_uri)
    if not path.exists():
        raise FileNotFoundError(f"artifact not found: {path}")
    return path


def default_location(env_var: str, filename: str) -> str:
    """Explicit env var > Vertex AI's AIP_STORAGE_URI > local outputs/."""
    if os.environ.get(env_var):
        return os.environ[env_var]
    if os.environ.get("AIP_STORAGE_URI"):
        return os.environ["AIP_STORAGE_URI"].rstrip("/") + "/" + filename
    return str(Path("outputs") / filename)


def file_md5(path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()
