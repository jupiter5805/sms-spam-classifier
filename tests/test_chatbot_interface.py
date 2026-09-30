import csv
from unittest.mock import Mock

import pytest

from chatbot_interface import (
    display_result,
    load_inputs_from_file,
    process_message,
)


def test_loads_messages_from_txt(
    tmp_path,
):
    file_path = (
        tmp_path / "messages.txt"
    )

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


def test_loads_messages_from_csv(
    tmp_path,
):
    file_path = (
        tmp_path / "messages.csv"
    )

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
    file_path = (
        tmp_path / "messages.json"
    )

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
    file_path = (
        tmp_path / "messages.csv"
    )

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


def test_process_message_combines_models():
    classifier = Mock()
    language_model = Mock()
    rag_store = Mock()
    history_logger = Mock()

    language_model.extract_sms.return_value = (
        "Congratulations, claim your prize."
    )

    classifier.classify_with_confidence.return_value = {
        "label": "spam",
        "confidence": 0.96,
    }

    rag_store.retrieve.return_value = [
        {
            "source": "phishing.txt",
            "text": (
                "Never share personal "
                "information by SMS."
            ),
            "score": 0.91,
        }
    ]

    language_model.generate_response.return_value = (
        "This message looks like spam. "
        "Be cautious."
    )

    result = process_message(
        user_input=(
            "Can you check this message? "
            "Congratulations, claim your prize."
        ),
        classifier=classifier,
        language_model=language_model,
        style="friendly",
        history_logger=history_logger,
        rag_store=rag_store,
    )

    assert (
        result["classification"]
        == "spam"
    )

    assert (
        result["confidence"]
        == 0.96
    )

    assert (
        result["retrieved_context"][0][
            "source"
        ]
        == "phishing.txt"
    )

    assert (
        "spam"
        in result["response"].lower()
    )

    classifier.classify_with_confidence.assert_called_once()

    rag_store.retrieve.assert_called_once()

    language_model.generate_response.assert_called_once()


def test_display_result_format(
    capsys,
):
    result = {
        "input": "Test message",
        "extracted_sms": (
            "Claim your free prize"
        ),
        "classification": "spam",
        "confidence": 0.95,
        "retrieved_context": [
            {
                "source": "phishing.txt",
                "text": (
                    "Suspicious message."
                ),
                "score": 0.91,
            }
        ],
        "response": (
            "This looks like spam."
        ),
    }

    display_result(
        result
    )

    output = (
        capsys.readouterr().out
    )

    assert (
        "Classification: SPAM"
        in output
    )

    assert (
        "Confidence: 95.00%"
        in output
    )

    assert (
        "phishing.txt"
        in output
    )

    assert (
        "This looks like spam."
        in output
    )
