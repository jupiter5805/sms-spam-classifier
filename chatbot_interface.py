import logging
import os
from pathlib import Path

from language_model import TinyLlamaAssistant
from sms_classifier import SMSClassifier


BASE_DIR = Path(__file__).resolve().parent

DEFAULT_MODEL_PATH = (
    BASE_DIR / "models" / "trained_model.pkl"
)


logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def run_chatbot():
    model_path = os.getenv(
        "MODEL_PATH",
        str(DEFAULT_MODEL_PATH),
    )

    print()
    print("SMS Spam Assistant")
    print("------------------")
    print("Loading models. The first run may download TinyLlama.")

    try:
        classifier = SMSClassifier(model_path)
        language_model = TinyLlamaAssistant()
    except Exception as error:
        print(f"Unable to start chatbot: {error}")
        return

    print()
    print("Assistant: Hi! Send me an SMS or describe one")
    print("and I'll check whether it looks like spam.")
    print("Type 'exit' or 'quit' to close the chatbot.")

    while True:
        try:
            user_input = input("\nYou: ").strip()

        except (KeyboardInterrupt, EOFError):
            print("\nAssistant: Goodbye!")
            break

        if user_input.lower() in {
            "exit",
            "quit",
        }:
            print("Assistant: Goodbye!")
            break

        if not user_input:
            print(
                "Assistant: Please enter a message "
                "for me to analyse."
            )
            continue

        try:
            extracted_sms = language_model.extract_sms(
                user_input
            )

            result = classifier.classify_with_confidence(
                extracted_sms
            )

            response = language_model.generate_response(
                original_input=user_input,
                extracted_sms=extracted_sms,
                classification=result["label"],
                confidence=result["confidence"],
            )

            print()
            print(f"Extracted SMS: {extracted_sms}")
            print(
                f"Classification: "
                f"{result['label'].upper()}"
            )

            if result["confidence"] is not None:
                print(
                    f"Confidence: "
                    f"{result['confidence'] * 100:.2f}%"
                )

            print(f"\nAssistant: {response}")

        except (ValueError, TypeError) as error:
            print(f"Assistant: {error}")

        except Exception as error:
            logging.exception(
                "Unexpected chatbot error."
            )

            print(
                "Assistant: Something went wrong while "
                "processing that message."
            )


if __name__ == "__main__":
    run_chatbot()
