import logging
from pathlib import Path

from sentence_transformers import SentenceTransformer, util


logger = logging.getLogger(__name__)


def chunk_text(
    text,
    chunk_size=100,
    overlap=20,
):
    words = text.split()

    if not words:
        return []

    if chunk_size <= 0:
        raise ValueError(
            "Chunk size must be greater than zero."
        )

    if overlap < 0:
        raise ValueError(
            "Overlap cannot be negative."
        )

    if overlap >= chunk_size:
        raise ValueError(
            "Overlap must be smaller than chunk size."
        )

    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size

        chunk = " ".join(
            words[start:end]
        )

        chunks.append(chunk)

        if end >= len(words):
            break

        start += chunk_size - overlap

    return chunks


def load_text_documents(path):
    knowledge_path = Path(path)

    if not knowledge_path.exists():
        raise FileNotFoundError(
            f"Knowledge path not found: {knowledge_path}"
        )

    if knowledge_path.is_file():
        if knowledge_path.suffix.lower() != ".txt":
            raise ValueError(
                "Knowledge file must be a .txt file."
            )

        files = [knowledge_path]

    elif knowledge_path.is_dir():
        files = sorted(
            knowledge_path.glob("*.txt")
        )

        if not files:
            raise ValueError(
                "Knowledge folder contains no .txt files."
            )

    else:
        raise ValueError(
            "Knowledge path must be a file or folder."
        )

    documents = []

    for file_path in files:
        content = file_path.read_text(
            encoding="utf-8"
        ).strip()

        if content:
            documents.append(
                {
                    "source": file_path.name,
                    "text": content,
                }
            )

    if not documents:
        raise ValueError(
            "No usable knowledge documents were found."
        )

    return documents


class RAGStore:
    def __init__(
        self,
        knowledge_path,
        model_name=(
            "sentence-transformers/"
            "all-MiniLM-L6-v2"
        ),
        chunk_size=100,
        overlap=20,
    ):
        self.knowledge_path = Path(
            knowledge_path
        )

        self.model_name = model_name
        self.chunk_size = chunk_size
        self.overlap = overlap

        logger.info(
            "Loading embedding model %s",
            self.model_name,
        )

        self.embedding_model = (
            SentenceTransformer(
                self.model_name
            )
        )

        self.chunks = (
            self._prepare_chunks()
        )

        self.embeddings = (
            self._create_embeddings()
        )

        logger.info(
            "RAG store ready with %s chunks.",
            len(self.chunks),
        )

    def _prepare_chunks(self):
        documents = load_text_documents(
            self.knowledge_path
        )

        chunks = []

        for document in documents:
            document_chunks = chunk_text(
                document["text"],
                chunk_size=self.chunk_size,
                overlap=self.overlap,
            )

            for chunk in document_chunks:
                chunks.append(
                    {
                        "source": document[
                            "source"
                        ],
                        "text": chunk,
                    }
                )

        if not chunks:
            raise ValueError(
                "No knowledge chunks were created."
            )

        return chunks

    def _create_embeddings(self):
        texts = [
            chunk["text"]
            for chunk in self.chunks
        ]

        return self.embedding_model.encode(
            texts,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )

    def retrieve(
        self,
        query,
        top_k=3,
    ):
        if not isinstance(query, str):
            raise TypeError(
                "Query must be a string."
            )

        query = query.strip()

        if not query:
            raise ValueError(
                "Query cannot be empty."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        query_embedding = (
            self.embedding_model.encode(
                query,
                convert_to_tensor=True,
                normalize_embeddings=True,
            )
        )

        similarities = util.cos_sim(
            query_embedding,
            self.embeddings,
        )[0]

        result_count = min(
            top_k,
            len(self.chunks),
        )

        matches = similarities.topk(
            k=result_count
        )

        results = []

        for score, index in zip(
            matches.values,
            matches.indices,
        ):
            chunk = self.chunks[
                index.item()
            ]

            results.append(
                {
                    "source": chunk[
                        "source"
                    ],
                    "text": chunk["text"],
                    "score": float(
                        score.item()
                    ),
                }
            )

        logger.info(
            "Retrieved %s chunks for query.",
            len(results),
        )

        return results
