"""In-memory CF movie embedding cache loaded from MinIO parquet."""

from __future__ import annotations

import tempfile
import numpy as np
import pyarrow.parquet as pq

from pathlib import Path
from numpy.typing import NDArray
from botocore.client import BaseClient
from common.features.schema import CF_EMBEDDING_DIM
from common.schemas.cf_artifact_manifest import CfArtifactManifest
from common.storage.s3 import cf_movie_embeddings_object_key, download_file
from common.storage.cf_artifact_reader import load_cf_artifact_manifest, resolve_cf_version


class CfEmbeddingCache:
    """
    Hold movie CF embeddings in memory for fast online inference lookups.

    Do this by:
    1. Resolving the CF artifact version that matches the deployed hybrid model.
    2. Downloading movie_cf_embeddings.parquet from MinIO once at startup.
    3. Serving vector lookups by movie_id during /recommend requests.
    """

    def __init__(self, embeddings: dict[int, NDArray[np.float32]], cf_version: str) -> None:
        """
        Create a cache from a preloaded embedding map.

        ============================ Arguments ============================
        embeddings: Mapping of movie_id to 64-d CF vectors.
        cf_version: CF artifact version these embeddings came from.
        """
        self.cf_version = cf_version
        self._embeddings = embeddings

    @classmethod
    def load(cls, client: BaseClient, bucket: str, *, cf_version_override: str | None, \
                expected_cf_version: str | None = None) -> CfEmbeddingCache:
        """
        Download and load the CF embedding parquet from MinIO.

        Do this by:
        1. Resolving the CF artifact version from settings or MinIO.
        2. Verifying the manifest status is complete.
        3. Parsing movie_id and cf_embedding columns into a dict.

        ============================ Arguments ============================
        client: The boto3 S3 client.
        bucket: MinIO bucket name.
        cf_version_override: Optional explicit CF version from settings.
        expected_cf_version: CF version required by the loaded hybrid model.

        ============================ Returns ============================
        A populated CfEmbeddingCache instance.
        """
        # Resolve the CF artifact version from settings or MinIO.
        cf_version = resolve_cf_version(client, bucket, cf_version_override)
        # If the expected CF version is provided and it does not match the resolved CF version, raise an error.
        if expected_cf_version and cf_version != expected_cf_version:
            raise ValueError(
                f"CF version mismatch: model expects {expected_cf_version}, cache loaded {cf_version}"
            )

        # Load the CF artifact manifest.
        manifest = load_cf_artifact_manifest(client, bucket, cf_version)
        
        # If the manifest status is not complete, raise an error.
        if manifest.status != "complete":
            raise ValueError(f"CF artifact {cf_version} is not complete (status={manifest.status})")

        # Load the CF embeddings parquet file.
        embeddings = _load_embeddings_parquet(client, bucket, manifest)
        # Return a new CfEmbeddingCache instance.
        return cls(embeddings=embeddings, cf_version=cf_version)

    def has_movie(self, movie_id: int) -> bool:
        """Return whether one movie id has a cached CF embedding."""
        return movie_id in self._embeddings

    def get(self, movie_id: int) -> NDArray[np.float32]:
        """
        Return the CF embedding for one movie or a zero vector when missing.

        ============================ Arguments ============================
        movie_id: Catalog movie id.

        ============================ Returns ============================
        CF embedding vector with shape (64,).
        """
        embedding = self._embeddings.get(movie_id)
        if embedding is None:
            return np.zeros(CF_EMBEDDING_DIM, dtype=np.float32)
        return embedding

    def __len__(self) -> int:
        """Return how many movie embeddings are cached."""
        return len(self._embeddings)


def _load_embeddings_parquet(client: BaseClient, bucket: str, 
                                manifest: CfArtifactManifest) -> dict[int, NDArray[np.float32]]:
    """
    Download one CF embeddings parquet file and parse it into a dict.

    ============================ Arguments ============================
    client: The boto3 S3 client.
    bucket: MinIO bucket name.
    manifest: Complete CF artifact manifest.

    ============================ Returns ============================
    Mapping of movie_id to CF embedding vectors.
    """
    # Create a temporary directory to download the parquet file to.
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a local path for the parquet file.
        local_path = Path(temp_dir) / "movie_cf_embeddings.parquet"
        # Download the parquet file from MinIO.
        download_file(client, bucket, cf_movie_embeddings_object_key(manifest.cf_version), local_path)
        # Read the parquet file into a pyarrow table.
        table = pq.read_table(local_path)

    # Get the movie ids and CF embeddings from the table.
    movie_ids = table.column("movie_id").to_pylist()
    cf_embeddings = table.column("cf_embedding").to_pylist()

    # Create a dictionary to store the embeddings.
    embeddings: dict[int, NDArray[np.float32]] = {}
    for movie_id, vector in zip(movie_ids, cf_embeddings, strict=True):
        # Convert the vector to a numpy array.
        array = np.asarray(vector, dtype=np.float32)
        # If the array shape is not (64,), raise an error.
        if array.shape != (CF_EMBEDDING_DIM,):
            raise ValueError(f"CF embedding for movie {movie_id} has invalid shape {array.shape}")
        # Add the embedding to the dictionary.
        embeddings[int(movie_id)] = array
    # Return the dictionary of embeddings.
    return embeddings
