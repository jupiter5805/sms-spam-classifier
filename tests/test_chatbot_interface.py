import csv

import pytest

from chatbot_interface import load_inputs_from_file


def test_loads_messages_from_txt(tmp_path):
    file_path = tmp_path / "messages.txt"

    file_path.write_text(
        "Hello there\n"
        "Win a free prize now\n"
        "\n"
        "See you tonight\n"
    )

    result = load_inputs_from_file(
        file_path
    )

    assert result == [
        "Hello there",
        "Win a free prize now",
        "See you tonight",
    ]


def test_loads_messages_from_csv(tmp_path):
    file_path = tmp_path / "messages.csv"

    with open(
        file_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["message"],
        )

        writer.writeheader()

        writer.writerow(
            {
                "message": "Hello there",
            }
        )

        writer.writerow(
            {
                "message": "Claim your prize",
            }
        )

    result = load_inputs_from_file(
        file_path
    )

    assert result == [
        "Hello there",
        "Claim your prize",
    ]


def test_rejects_unsupported_file_type(
    tmp_path,
):
    file_path = tmp_path / "messages.json"

    file_path.write_text(
        "{}"
    )

    with pytest.raises(
        ValueError,
        match="Unsupported file type",
    ):
        load_inputs_from_file(
            file_path
        )


def test_rejects_missing_file():
    with pytest.raises(
        FileNotFoundError,
    ):
        load_inputs_from_file(
            "does_not_exist.txt"
        )


def test_csv_requires_message_column(
    tmp_path,
):
    file_path = tmp_path / "messages.csv"

    file_path.write_text(
        "name,value\n"
        "test,123\n"
    )

    with pytest.raises(
        ValueError,
        match="message, sms, or text",
    ):
        load_inputs_from_file(
            file_path
        )
