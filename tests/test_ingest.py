import pandas as pd

from ingest import clean_dataset


def test_clean_dataset_standardises_labels():
    dataframe = pd.DataFrame(
        {
            "label": [" HAM ", "SPAM"],
            "message": ["Hello", "You won a prize"],
        }
    )

    result = clean_dataset(dataframe)

    assert result["label"].tolist() == ["ham", "spam"]


def test_clean_dataset_removes_duplicate_rows():
    dataframe = pd.DataFrame(
        {
            "label": ["ham", "ham"],
            "message": ["Hello there", "Hello there"],
        }
    )

    result = clean_dataset(dataframe)

    assert len(result) == 1


def test_clean_dataset_removes_invalid_labels():
    dataframe = pd.DataFrame(
        {
            "label": ["ham", "unknown", "spam"],
            "message": ["Hello", "Test message", "Win now"],
        }
    )

    result = clean_dataset(dataframe)

    assert result["label"].tolist() == ["ham", "spam"]


def test_clean_dataset_removes_empty_messages():
    dataframe = pd.DataFrame(
        {
            "label": ["ham", "spam"],
            "message": ["Hello", "   "],
        }
    )

    result = clean_dataset(dataframe)

    assert len(result) == 1
    assert result.iloc[0]["message"] == "Hello"


def test_clean_dataset_removes_missing_values():
    dataframe = pd.DataFrame(
        {
            "label": ["ham", None, "spam"],
            "message": ["Hello", "Missing label", None],
        }
    )

    result = clean_dataset(dataframe)

    assert len(result) == 1
    assert result.iloc[0]["label"] == "ham"
