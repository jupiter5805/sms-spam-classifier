import argparse
import csv
import logging
import os
from pathlib import Path

from language_model import STYLE_PROMPTS, TinyLlamaAssistant
from sms_classifier import SMSClassifier


BASE_DIR = Path(__file__).resolve().parent

DEFAULT_MODEL_PATH = (
    BASE_DIR / "models" / "trained_model.pkl"
)

LOG_DIR = BASE_DIR / "logs"
HISTORY_PATH = LOG_DIR / "chat_history.log"


def configure_history_logger():
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    history_logger = logging.getLogger(
        "chat_history"
    )

    history_logger.setLevel(logging.INFO)
    history_logger.propagate = False

    if not history_logger.handlers:
        handler = logging.FileHandler(
            HISTORY_PATH,
            encoding="utf-8",
        )

        formatter = logging.Formatter(
            "%(asctime)s | %(message)s"
        )

        handler.setFormatter(formatter)
        history_logger.addHandler(handler)

    return history_logger


def load_inputs_from_file(file_path):
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {path}"
        )

    if path.suffix.lower() == ".txt":
        with open(path, "r", encoding="utf-8") as file:
            messages = [
                line.strip()
                for line in file
                if line.strip()
            ]

        if not messages:
            raise ValueError(
                "The text file contains no messages."
            )

        return messages

    if path.suffix.lower() == ".csv":
        with open(
            path,
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as file:
            reader = csv.DictReader(file)

            if not reader.fieldnames:
                raise ValueError(
                    "The CSV file has no columns."
                )

            available_columns = {
                column.lower(): column
                for column in reader.fieldnames
            }

            message_column = None

            for candidate in [
                "message",
                "sms",
                "text",
            ]:
                if candidate in available_columns:
                    message_column = available_columns[
                        candidate
                    ]
                    break

            if message_column is None:
                raise ValueError(
                    "CSV must contain a message, sms, or text column."
                )

            messages = []

            for row in reader:
                value = row.get(
                    message_column,
                    "",
                ).strip()

                if value:
                    messages.append(value)

        if not messages:
            raise ValueError(
                "The CSV file contains no messages."
            )

        return messages

    raise ValueError(
        "Unsupported file type. Use a .txt or .csv file."
    )


def process_message(
    user_input,
    classifier,
    language_model,
    style,
    history_logger,
):
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
        style=style,
    )

    confidence = result["confidence"]

    if confidence is not None:
        confidence_percentage = confidence * 100
        confidence_text = (
            f"{confidence_percentage:.2f}%"
        )
    else:
        confidence_text = "Not available"

    history_logger.info(
        "INPUT=%r | EXTRACTED=%r | "
        "CLASSIFICATION=%s | CONFIDENCE=%s | "
        "STYLE=%s | RESPONSE=%r",
        user_input,
        extracted_sms,
        result["label"],
        confidence_text,
        style,
        response,
    )

    return {
        "input": user_input,
        "extracted_sms": extracted_sms,
        "classification": result["label"],
        "confidence": confidence,
        "response": response,
    }


def display_result(result):
    print()
    print(
        f"Extracted SMS: "
        f"{result['extracted_sms']}"
    )

    print(
        f"Classification: "
        f"{result['classification'].upper()}"
    )

    if result["confidence"] is not None:
        print(
            f"Confidence: "
            f"{result['confidence'] * 100:.2f}%"
        )

    print(
        f"\nAssistant: {result['response']}"
    )


def run_file_mode(
    file_path,
    classifier,
    language_model,
    style,
    history_logger,
):
    messages = load_inputs_from_file(
        file_path
    )

    print(
        f"Loaded {len(messages)} message(s) "
        f"from {file_path}."
    )

    for index, message in enumerate(
        messages,
        start=1,
    ):
        print()
        print(
            f"Message {index}/{len(messages)}"
        )
        print("-" * 30)

        try:
            result = process_message(
                message,
                classifier,
                language_model,
                style,
                history_logger,
            )

            display_result(result)

        except Exception as error:
            history_logger.exception(
                "Failed to process input %r",
                message,
            )

            print(
                f"Unable to process message: "
                f"{error}"
            )


def run_interactive_mode(
    classifier,
    language_model,
    style,
    history_logger,
):
    print()
    print("SMS Spam Assistant")
    print("------------------")
    print(
        f"Response style: {style}"
    )
    print(
        "Send me an SMS or describe one "
        "and I'll check whether it looks like spam."
    )
    print(
        "Type 'exit' or 'quit' to close the chatbot."
    )

    while True:
        try:
            user_input = input(
                "\nYou: "
            ).strip()

        except (
            KeyboardInterrupt,
            EOFError,
        ):
            print(
                "\nAssistant: Goodbye!"
            )
            break

        if user_input.lower() in {
            "exit",
            "quit",
        }:
            print(
                "Assistant: Goodbye!"
            )
            break

        if not user_input:
            print(
                "Assistant: Please enter "
                "a message to analyse."
            )
            continue

        try:
            result = process_message(
                user_input,
                classifier,
                language_model,
                style,
                history_logger,
            )

            display_result(result)

        except Exception as error:
            history_logger.exception(
                "Failed to process input %r",
                user_input,
            )

            print(
                "Assistant: I couldn't process "
                f"that message. {error}"
            )


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Natural language SMS spam classifier"
        )
    )

    parser.add_argument(
        "--file",
        help=(
            "Optional .txt or .csv file "
            "containing messages to classify"
        ),
    )

    parser.add_argument(
        "--style",
        choices=list(
            STYLE_PROMPTS.keys()
        ),
        default="friendly",
        help="Assistant response style",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    model_path = os.getenv(
        "MODEL_PATH",
        str(DEFAULT_MODEL_PATH),
    )

    history_logger = (
        configure_history_logger()
    )

    print(
        "Loading classifier and language model..."
    )

    try:
        classifier = SMSClassifier(
            model_path
        )

        language_model = (
            TinyLlamaAssistant()
        )

    except Exception as error:
        history_logger.exception(
            "Application startup failed."
        )

        print(
            f"Unable to start chatbot: {error}"
        )

        return

    if args.file:
        try:
            run_file_mode(
                args.file,
                classifier,
                language_model,
                args.style,
                history_logger,
            )

        except (
            FileNotFoundError,
            ValueError,
        ) as error:
            print(
                f"Unable to read input file: "
                f"{error}"
            )

        return

    run_interactive_mode(
        classifier,
        language_model,
        args.style,
        history_logger,
    )


if __name__ == "__main__":
    main()
