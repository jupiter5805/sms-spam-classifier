# SMS Spam Classifier

A machine learning project that builds an end-to-end natural language classification pipeline for identifying SMS messages as spam or legitimate messages.

The project will combine a traditional machine learning classifier with a language model and Retrieval-Augmented Generation (RAG) to create a natural language interface.

## Dataset

This project uses the SMS Spam Collection dataset from the UCI Machine Learning Repository.

The dataset contains SMS messages labelled as:

- `ham` - legitimate SMS messages
- `spam` - unwanted or spam messages

The raw dataset is downloaded automatically by the ingestion pipeline and is not stored in this GitHub repository.

## Project Structure

```text
sms-spam-classifier/
├── data/
│   ├── raw/
│   └── processed/
├── tests/
│   └── test_ingest.py
├── .gitignore
├── ingest.py
├── README.md
└── requirements.txt
```

## Setup

Clone the repository and enter the project directory.

```bash
git clone https://github.com/jupiter5805/sms-spam-classifier.git
cd sms-spam-classifier
```

Create a virtual environment.

```bash
python3 -m venv .venv
```

Activate the virtual environment.

```bash
source .venv/bin/activate
```

Install the dependencies.

```bash
pip install -r requirements.txt
```

## Data Ingestion

Run the ingestion pipeline with:

```bash
python ingest.py
```

The ingestion pipeline:

1. Downloads the SMS Spam Collection dataset if it is not already available locally
2. Extracts the downloaded archive
3. Loads the dataset into a Pandas DataFrame
4. Standardises labels and column names
5. Removes missing, invalid, empty and duplicate records
6. Saves the cleaned dataset locally

The processed dataset is saved as:

```text
data/processed/cleaned_dataset.csv
```

Raw and processed datasets are excluded from Git so they can be reproduced using the ingestion pipeline instead of being stored in the repository.

The dataset source can also be changed using the `DATASET_URL` environment variable, helping keep the ingestion process configurable for future deployment.

## Testing

Unit tests are written using `pytest`.

Run the test suite with:

```bash
pytest -v
```

Current tests verify that the cleaning process:

- Standardises SMS labels
- Removes duplicate records
- Removes invalid labels
- Removes empty messages
- Removes missing values

## Future Development

The project will be extended to include:

- Machine learning classifier training
- Model evaluation
- A prediction interface
- Natural language model integration
- Retrieval-Augmented Generation
- Automated testing
- Potential AWS deployment

## Dataset Attribution

This project uses the SMS Spam Collection dataset from the UCI Machine Learning Repository.

Almeida, T. A., Hidalgo, J. M. G., and Yamakami, A. (2011). Contributions to the Study of SMS Spam Filtering: New Collection and Results.

The dataset is used for educational machine learning development.
## Model Training

The project includes a machine learning training pipeline built using scikit-learn.

SMS messages are converted into numerical features using TF-IDF vectorisation.

The cleaned dataset is split into training and validation sets using an 80/20 stratified split.

Two classification models are trained and evaluated:

- Logistic Regression
- Multinomial Naive Bayes

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Classification Report

The model with the highest F1 score for spam detection is automatically selected and saved.

Run the training pipeline with:

```bash
python train_model.py