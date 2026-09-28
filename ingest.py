import logging
import os
import zipfile
from pathlib import Path

import pandas as pd
import requests


BASE_DIR = Path(__file__).resolve().parent

DATASET_URL = os.getenv(
    "DATASET_URL",
    "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip",
)

RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

ZIP_PATH = RAW_DIR / "sms_spam_collection.zip"
RAW_FILE_PATH = RAW_DIR / "SMSSpamCollection"
CLEANED_FILE_PATH = PROCESSED_DIR / "cleaned_dataset.csv"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def download_dataset(force=False):
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if ZIP_PATH.exists() and not force:
        logger.info("Dataset archive already exists. Skipping download.")
        return ZIP_PATH

    logger.info("Downloading dataset from UCI")

    response = requests.get(DATASET_URL, timeout=30)
    response.raise_for_status()

    ZIP_PATH.write_bytes(response.content)

    logger.info("Dataset downloaded successfully.")

    return ZIP_PATH


def extract_dataset(force=False):
    if RAW_FILE_PATH.exists() and not force:
        logger.info("Raw dataset already exists. Skipping extraction.")
        return RAW_FILE_PATH

    logger.info("Extracting dataset.")

    with zipfile.ZipFile(ZIP_PATH, "r") as archive:
        archive.extractall(RAW_DIR)

    logger.info("Dataset extracted successfully.")

    return RAW_FILE_PATH


def load_dataset(file_path):
    logger.info("Loading dataset.")

    dataframe = pd.read_csv(
        file_path,
        sep="\t",
        names=["label", "message"],
        header=None,
    )

    logger.info("Loaded %s rows.", len(dataframe))

    return dataframe


def clean_dataset(dataframe):
    logger.info("Cleaning dataset.")

    cleaned = dataframe.copy()

    cleaned.columns = cleaned.columns.str.strip().str.lower()

    cleaned = cleaned.dropna(subset=["label", "message"])

    cleaned["label"] = cleaned["label"].astype(str).str.strip().str.lower()
    cleaned["message"] = cleaned["message"].astype(str).str.strip()

    cleaned = cleaned[cleaned["label"].isin(["ham", "spam"])]
    cleaned = cleaned[cleaned["message"] != ""]
    cleaned = cleaned.drop_duplicates()

    cleaned = cleaned.reset_index(drop=True)

    logger.info("Cleaning complete. %s rows remain.", len(cleaned))

    return cleaned


def save_dataset(dataframe):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    dataframe.to_csv(CLEANED_FILE_PATH, index=False)

    logger.info("Cleaned dataset saved to %s", CLEANED_FILE_PATH)

    return CLEANED_FILE_PATH


def run_pipeline(force_download=False):
    download_dataset(force=force_download)
    raw_file = extract_dataset(force=force_download)
    dataframe = load_dataset(raw_file)
    cleaned_dataframe = clean_dataset(dataframe)
    save_dataset(cleaned_dataframe)

    logger.info("Data ingestion pipeline completed successfully.")


if __name__ == "__main__":
    run_pipeline()
