"""Local Sentence Transformers adapter with query/document separation."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer


class LocalSentenceEmbedder:
    """Load and validate a local Hugging Face sentence embedding model."""

    def __init__(
        self,
        model_id: str,
        *,
        expected_dimension: int,
        normalize: bool = True,
        device: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.expected_dimension = expected_dimension
        self.normalize = normalize
        self.model = SentenceTransformer(
            model_id,
            device=device,
            # The published config stores this as a list while recent
            # Transformers versions expect a mapping. The model's custom
            # pre-tokenizer is a simple Split, not the affected Mistral regex.
            tokenizer_kwargs={"extra_special_tokens": {}},
        )
        actual_dimension = self.model.get_sentence_embedding_dimension()
        if actual_dimension != expected_dimension:
            raise ValueError(
                f"Expected {expected_dimension} dimensions from {model_id}, got {actual_dimension}"
            )

    def _validate(self, vectors: np.ndarray) -> np.ndarray:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.expected_dimension:
            raise ValueError(f"Unexpected embedding shape: {vectors.shape}")
        if not np.isfinite(vectors).all():
            raise ValueError("Embedding output contains NaN or infinite values")
        if self.normalize:
            norms = np.linalg.norm(vectors, axis=1)
            if not np.allclose(norms, 1.0, atol=1e-4):
                raise ValueError("Embedding output is not L2-normalized")
        return vectors

    def encode_documents(self, texts: Sequence[str], *, batch_size: int) -> np.ndarray:
        """Encode document chunks with the model's document-specific method when available."""

        encoder = getattr(self.model, "encode_document", self.model.encode)
        vectors = encoder(
            list(texts),
            batch_size=batch_size,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return self._validate(vectors)

    def encode_queries(self, texts: Sequence[str], *, batch_size: int = 32) -> np.ndarray:
        """Encode retrieval queries with the model's query-specific method when available."""

        encoder = getattr(self.model, "encode_query", self.model.encode)
        vectors = encoder(
            list(texts),
            batch_size=batch_size,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return self._validate(vectors)
