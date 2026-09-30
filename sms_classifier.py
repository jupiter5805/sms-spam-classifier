import logging
import pickle
from pathlib import Path


logger = logging.getLogger(__name__)


class SMSClassifier:
    def __init__(self, model_path):
        self.model_path = Path(model_path)
        self.model = self._load_model()

    def _load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}"
            )

        logger.info("Loading model from %s", self.model_path)

        with open(self.model_path, "rb") as file:
            model = pickle.load(file)

        if not hasattr(model, "predict"):
            raise TypeError("Loaded object is not a valid classifier.")

        logger.info("Model loaded successfully.")

        return model

    def validate_message(self, message):
        if not isinstance(message, str):
            raise TypeError("Message must be a string.")

        cleaned_message = message.strip()

        if not cleaned_message:
            raise ValueError("Message cannot be empty.")

        return cleaned_message

    def classify_message(self, message):
        cleaned_message = self.validate_message(message)

        prediction = self.model.predict(
            [cleaned_message]
        )[0]

        logger.info(
            "Message classified as %s",
            prediction,
        )

        return prediction

    def classify_with_confidence(self, message):
        cleaned_message = self.validate_message(message)

        prediction = self.model.predict(
            [cleaned_message]
        )[0]

        confidence = None

        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(
                [cleaned_message]
            )[0]

            classes = list(self.model.classes_)
            prediction_index = classes.index(prediction)

            confidence = float(
                probabilities[prediction_index]
            )

        logger.info(
            "Message classified as %s",
            prediction,
        )

        return {
            "label": prediction,
            "confidence": confidence,
        }
