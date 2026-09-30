import pickle

import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sms_classifier import SMSClassifier


@pytest.fixture
def trained_model_path(tmp_path):
    messages = [
        "Hello see you later",
        "Are we still meeting tonight",
        "Can you call me when you arrive",
        "Congratulations you won a free prize",
        "Claim your cash reward now",
        "Urgent you have won money",
    ]

    labels = [
        "ham",
        "ham",
        "ham",
        "spam",
        "spam",
        "spam",
    ]

    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(),
            ),
            (
                "classifier",
                LogisticRegression(
                    random_state=42
                ),
            ),
        ]
    )

    model.fit(
        messages,
        labels,
    )

    model_path = tmp_path / "test_model.pkl"

    with open(model_path, "wb") as file:
        pickle.dump(model, file)

    return model_path


def test_classifier_loads_model(trained_model_path):
    classifier = SMSClassifier(
        trained_model_path
    )

    assert classifier.model is not None


def test_classifier_returns_valid_classification(
    trained_model_path,
):
    classifier = SMSClassifier(
        trained_model_path
    )

    result = classifier.classify_message(
        "You have won a free cash prize"
    )

    assert result in {
        "ham",
        "spam",
    }


def test_classifier_returns_confidence(
    trained_model_path,
):
    classifier = SMSClassifier(
        trained_model_path
    )

    result = classifier.classify_with_confidence(
        "Claim your free prize now"
    )

    assert result["label"] in {
        "ham",
        "spam",
    }

    assert 0 <= result["confidence"] <= 1


def test_classifier_rejects_empty_message(
    trained_model_path,
):
    classifier = SMSClassifier(
        trained_model_path
    )

    with pytest.raises(
        ValueError,
        match="Message cannot be empty",
    ):
        classifier.classify_message("   ")


def test_classifier_rejects_non_string_input(
    trained_model_path,
):
    classifier = SMSClassifier(
        trained_model_path
    )

    with pytest.raises(
        TypeError,
        match="Message must be a string",
    ):
        classifier.classify_message(123)


def test_classifier_raises_for_missing_model():
    with pytest.raises(
        FileNotFoundError,
        match="Model file not found",
    ):
        SMSClassifier(
            "models/does_not_exist.pkl"
        )
