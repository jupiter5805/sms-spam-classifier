import logging
import os
import pickle
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


BASE_DIR = Path(__file__).resolve().parent

DEFAULT_DATA_SOURCE = (
    BASE_DIR / "data" / "processed" / "cleaned_dataset.csv"
)

MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "trained_model.pkl"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def load_dataset(data_source):
    logger.info("Loading cleaned dataset from %s", data_source)

    dataframe = pd.read_csv(data_source)

    required_columns = {"label", "message"}

    if not required_columns.issubset(dataframe.columns):
        raise ValueError(
            "Dataset must contain label and message columns."
        )

    logger.info("Loaded %s rows.", len(dataframe))

    return dataframe


def split_dataset(dataframe):
    features = dataframe["message"]
    labels = dataframe["label"]

    X_train, X_validation, y_train, y_validation = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    logger.info(
        "Training set: %s rows. Validation set: %s rows.",
        len(X_train),
        len(X_validation),
    )

    return X_train, X_validation, y_train, y_validation


def create_models():
    logistic_regression = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    naive_bayes = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                MultinomialNB(),
            ),
        ]
    )

    return {
        "Logistic Regression": logistic_regression,
        "Multinomial Naive Bayes": naive_bayes,
    }


def evaluate_model(model_name, y_validation, predictions):
    accuracy = accuracy_score(
        y_validation,
        predictions,
    )

    precision = precision_score(
        y_validation,
        predictions,
        pos_label="spam",
        zero_division=0,
    )

    recall = recall_score(
        y_validation,
        predictions,
        pos_label="spam",
        zero_division=0,
    )

    f1 = f1_score(
        y_validation,
        predictions,
        pos_label="spam",
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_validation,
        predictions,
        labels=["ham", "spam"],
    )

    report = classification_report(
        y_validation,
        predictions,
        zero_division=0,
    )

    logger.info("%s results", model_name)
    logger.info("Accuracy: %.4f", accuracy)
    logger.info("Precision: %.4f", precision)
    logger.info("Recall: %.4f", recall)
    logger.info("F1 Score: %.4f", f1)
    logger.info("Confusion Matrix:\n%s", matrix)
    logger.info("Classification Report:\n%s", report)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def save_model(model):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    with open(MODEL_PATH, "wb") as file:
        pickle.dump(model, file)

    logger.info(
        "Trained model saved to %s",
        MODEL_PATH,
    )


def train_models(data_source):
    dataframe = load_dataset(data_source)

    X_train, X_validation, y_train, y_validation = split_dataset(
        dataframe
    )

    models = create_models()

    best_model = None
    best_model_name = None
    best_metrics = None

    for model_name, model in models.items():
        logger.info("Training %s.", model_name)

        model.fit(
            X_train,
            y_train,
        )

        predictions = model.predict(
            X_validation
        )

        metrics = evaluate_model(
            model_name,
            y_validation,
            predictions,
        )

        if best_metrics is None or metrics["f1"] > best_metrics["f1"]:
            best_model = model
            best_model_name = model_name
            best_metrics = metrics

    logger.info(
        "Best model: %s with F1 Score %.4f",
        best_model_name,
        best_metrics["f1"],
    )

    save_model(best_model)

    return best_model


if __name__ == "__main__":
    data_source = os.getenv(
        "CLEANED_DATA_SOURCE",
        str(DEFAULT_DATA_SOURCE),
    )

    train_models(data_source)
