"""Sentence-transformer model loader for embedder-api."""

from __future__ import annotations

from sentence_transformers import SentenceTransformer

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingModel:
    """
    Load MiniLM once and encode text batches into vectors.

    Do this by:
    1. Loading the sentence-transformer model at startup.
    2. Encoding batches with normalize_embeddings enabled for cosine search.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        """
        Load the embedding model into memory.

        Do this by:
            1. Loading the sentence-transformer model at startup.
            2. Getting the embedding dimension.
            
        ============================ Arguments ============================
        model_name: Hugging Face model id to load.
        """
        # Store the model name and load the model.
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        
        # Get the embedding dimension.
        dimension = self._model.get_embedding_dimension()
        if dimension is None:
            raise RuntimeError(f"Could not determine embedding dimension for {model_name}")
        self.dimension = int(dimension)

    def encode(self, texts: list[str]) -> list[list[float]]:
        """
        Encode a batch of texts into embedding vectors.

        ============================ Arguments ============================
        texts: Input strings to embed.

        ============================ Returns ============================
        One normalized float vector per input text.
        """
        vectors = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [vector.tolist() for vector in vectors]
