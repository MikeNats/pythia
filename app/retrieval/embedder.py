"""Local embedding model — loaded once, turns text into vectors."""

from fastembed import TextEmbedding


class Embedder:
    def __init__(self) -> None:
        self._model = TextEmbedding()  # BAAI/bge-small-en-v1.5 → 384-dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self._model.embed(texts)]


_embedder: Embedder | None = None


def get_embedder() -> Embedder:
    """Singleton — load the model once, reuse it."""
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder
