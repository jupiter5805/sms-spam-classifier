import pytest

from rag_store import (
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

    first_chunk = (
        chunks[0].split()
    )

    second_chunk = (
        chunks[1].split()
    )

    assert first_chunk[-20:] == (
        second_chunk[:20]
    )


def test_load_single_txt_file(
    tmp_path,
):
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
