import torch
import pytest

from rag_store import (
    RAGStore,
    chunk_text,
    load_text_documents,
)


def test_chunk_text_creates_chunks():
    text = " ".join(
        f"word{i}"
        for i in range(250)
    )

    chunks = chunk_text(
        text,
        chunk_size=100,
        overlap=20,
    )

    assert len(chunks) == 3


def test_chunks_include_overlap():
    words = [
        f"word{i}"
        for i in range(150)
    ]

    text = " ".join(words)

    chunks = chunk_text(
        text,
        chunk_size=100,
        overlap=20,
    )

    first_chunk = chunks[0].split()
    second_chunk = chunks[1].split()

    assert first_chunk[-20:] == second_chunk[:20]


def test_load_single_txt_file(tmp_path):
    file_path = (
        tmp_path / "knowledge.txt"
    )

    file_path.write_text(
        "Messages asking for bank "
        "details may be suspicious."
    )

    documents = load_text_documents(
        file_path
    )

    assert len(documents) == 1

    assert (
        documents[0]["source"]
        == "knowledge.txt"
    )


def test_load_folder_of_txt_files(
    tmp_path,
):
    first = (
        tmp_path / "first.txt"
    )

    second = (
        tmp_path / "second.txt"
    )

    first.write_text(
        "First knowledge document."
    )

    second.write_text(
        "Second knowledge document."
    )

    documents = load_text_documents(
        tmp_path
    )

    assert len(documents) == 2


def test_rejects_non_txt_file(
    tmp_path,
):
    file_path = (
        tmp_path / "knowledge.csv"
    )

    file_path.write_text(
        "message,test"
    )

    with pytest.raises(
        ValueError,
        match=".txt",
    ):
        load_text_documents(
            file_path
        )


def test_rag_retrieves_relevant_chunk(
    tmp_path,
    monkeypatch,
):
    phishing_file = (
        tmp_path / "phishing.txt"
    )

    prize_file = (
        tmp_path / "prizes.txt"
    )

    phishing_file.write_text(
        "Never send your bank details "
        "or password through suspicious SMS messages."
    )

    prize_file.write_text(
        "Unexpected prize messages may "
        "claim that you have won a reward."
    )

    class FakeEmbeddingModel:
        def encode(
            self,
            text,
            convert_to_tensor=True,
            normalize_embeddings=True,
        ):
            def vector(value):
                value = value.lower()

                if (
                    "bank" in value
                    or "password" in value
                ):
                    return [1.0, 0.0]

                return [0.0, 1.0]

            if isinstance(text, list):
                return torch.tensor(
                    [
                        vector(item)
                        for item in text
                    ]
                )

            return torch.tensor(
                vector(text)
            )

    monkeypatch.setattr(
        "rag_store.SentenceTransformer",
        lambda model_name: (
            FakeEmbeddingModel()
        ),
    )

    store = RAGStore(
        tmp_path
    )

    results = store.retrieve(
        "Someone asked for my bank password",
        top_k=1,
    )

    assert len(results) == 1

    assert (
        results[0]["source"]
        == "phishing.txt"
    )

    assert (
        "bank details"
        in results[0]["text"]
    )
