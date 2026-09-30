import logging
import os
from pathlib import Path

from sms_classifier import SMSClassifier


BASE_DIR = Path(__file__).resolve().parent

DEFAULT_MODEL_PATH = (
    BASE_DIR / "models" / "trained_model.pkl"
)


logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def display_result(result):
    label = result["label"].upper()
    confidence = result["confidence"]

    if confidence is not None:
        confidence_percentage = confidence * 100

        print(
            f"[Result] {label} "
            f"({confidence_percentage:.2f}% confidence)"
        )
    else:
        print(f"[Result] {label}")

    if label == "SPAM":
        print("This message looks like spam.")
    else:
        print("This message looks like a legitimate message.")


def run_interface():
    model_path = os.getenv(
        "MODEL_PATH",
        str(DEFAULT_MODEL_PATH),
    )

    try:
        classifier = SMSClassifier(model_path)
    except (FileNotFoundError, TypeError) as error:
        print(f"Unable to load classifier: {error}")
        return

    print()
    print("SMS Spam Classifier")
    print("-------------------")
    print("Enter an SMS message to classify.")
    print("Type 'exit' or 'quit' to close the program.")

    while True:
        try:
            message = input("\nEnter SMS message: ")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting SMS Spam Classifier.")
            break

        if message.strip().lower() in {
            "exit",
            "quit",
        }:
            print("Exiting SMS Spam Classifier.")
            break

        try:
            result = classifier.classify_with_confidence(
                message
            )

            display_result(result)

        except (ValueError, TypeError) as error:
            print(f"Invalid input: {error}")


if __name__ == "__main__":
    run_interface()
