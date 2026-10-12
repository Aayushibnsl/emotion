"""
Emotion Detection from Text  --  model training script.

Machine learning pipeline
-------------------------
    CSV  ->  clean text  ->  TF-IDF  ->  Logistic Regression  ->  saved .pkl files

Run this once from the project folder:

    python train_model.py

It creates three files inside the `models/` folder:
    tfidf_vectorizer.pkl  - the fitted TF-IDF vectorizer
    emotion_model.pkl     - the trained Logistic Regression model
    model_info.json       - evaluation scores, shown in the Streamlit app
"""

import json
import os
import sys
from datetime import datetime

import joblib
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

from preprocessing import clean_text

# --------------------------------------------------------------------------
# Settings  (everything you may want to tweak lives here)
# --------------------------------------------------------------------------
DATA_PATH = os.path.join("data", "emotion_dataset.csv")
MODELS_DIR = "models"
VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
MODEL_PATH = os.path.join(MODELS_DIR, "emotion_model.pkl")
INFO_PATH = os.path.join(MODELS_DIR, "model_info.json")

TEXT_COLUMN = "content"      # the input feature  (X)
LABEL_COLUMN = "sentiment"   # the target         (y)

TEST_SIZE = 0.2              # 80% training / 20% testing
RANDOM_STATE = 42            # fixed seed so results are reproducible


# --------------------------------------------------------------------------
# Step 1  --  load the dataset
# --------------------------------------------------------------------------
def load_dataset(path=DATA_PATH):
    """Read the CSV file and make sure it has the columns we need."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.\n"
            "Please download 'Emotion Detection from Text' by Pashupati Gupta from\n"
            "https://www.kaggle.com/datasets/pashupatigupta/emotion-detection-from-text\n"
            f"and save the CSV as '{path}'."
        )

    df = pd.read_csv(path)

    missing = [c for c in (TEXT_COLUMN, LABEL_COLUMN) if c not in df.columns]
    if missing:
        raise ValueError(
            f"The dataset is missing the column(s): {missing}.\n"
            f"Columns found: {list(df.columns)}\n"
            f"This project expects a '{TEXT_COLUMN}' column (the tweet text) and a "
            f"'{LABEL_COLUMN}' column (the emotion label)."
        )

    print(f"[1/6] Loaded {len(df):,} rows from {path}")
    print(f"      Columns: {list(df.columns)}")
    return df


# --------------------------------------------------------------------------
# Step 2  --  clean the data
# --------------------------------------------------------------------------
def prepare_data(df):
    """Drop missing rows and duplicates, then clean the text of every tweet."""
    rows_before = len(df)

    # Remove rows where the text or the label is missing
    df = df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN]).copy()

    # Remove repeated tweets (the same text should not be counted twice, and
    # duplicated text with two different labels would confuse the model)
    df = df.drop_duplicates(subset=[TEXT_COLUMN])

    # Apply the shared cleaning function to every tweet
    df["clean_text"] = df[TEXT_COLUMN].apply(clean_text)

    # A few tweets are only emoji / digits, so they become empty after cleaning
    df = df[df["clean_text"].str.len() > 0]

    print(f"[2/6] Cleaned data: {rows_before:,} rows -> {len(df):,} usable rows")
    print(f"      Emotion classes found: {df[LABEL_COLUMN].nunique()}")
    print(f"      {sorted(df[LABEL_COLUMN].unique())}")
    return df


# --------------------------------------------------------------------------
# Step 3  --  split into training and testing sets
# --------------------------------------------------------------------------
def split_data(df):
    """Separate X and y, then split them 80/20.

    `stratify=y` keeps the same emotion proportions in both halves, which
    matters here because the dataset is quite imbalanced.
    """
    X = df["clean_text"]        # input feature  = the cleaned tweet
    y = df[LABEL_COLUMN]        # target         = the emotion label

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"[3/6] Train: {len(X_train):,} rows   |   Test: {len(X_test):,} rows")
    return X_train, X_test, y_train, y_test


# --------------------------------------------------------------------------
# Step 4  --  TF-IDF + Logistic Regression
# --------------------------------------------------------------------------
def train_model(X_train, y_train):
    """Convert text to numbers with TF-IDF, then fit Logistic Regression."""
    # TF-IDF turns each tweet into a vector of word importance scores.
    #   ngram_range=(1, 2) -> single words AND two-word phrases ("not happy")
    #   min_df=2           -> ignore words that appear in only one tweet
    #   sublinear_tf=True  -> dampen very frequent words (uses 1 + log(tf))
    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    print(f"[4/6] TF-IDF built {X_train_tfidf.shape[1]:,} features")

    # Logistic Regression handles all 13 classes at once (multinomial softmax)
    # and gives us probabilities, which the app shows as a confidence score.
    model = LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE)
    model.fit(X_train_tfidf, y_train)
    print("[5/6] Logistic Regression trained")

    return vectorizer, model


# --------------------------------------------------------------------------
# Step 5  --  evaluate the model
# --------------------------------------------------------------------------
def evaluate_model(vectorizer, model, X_test, y_test):
    """Score the model on the unseen test set and print a small report."""
    X_test_tfidf = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_tfidf)

    # 'weighted' averaging accounts for how many samples each class has,
    # 'macro' treats every emotion equally (harsher on rare emotions).
    scores = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_weighted": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
    }

    print("\n" + "=" * 58)
    print("MODEL EVALUATION (on the 20% test set)")
    print("=" * 58)
    print(f"Accuracy            : {scores['accuracy']:.4f}")
    print(f"Precision (weighted): {scores['precision_weighted']:.4f}")
    print(f"Recall    (weighted): {scores['recall_weighted']:.4f}")
    print(f"F1-score  (weighted): {scores['f1_weighted']:.4f}")
    print(f"F1-score  (macro)   : {scores['f1_macro']:.4f}")

    print("\nPer-emotion report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    labels = sorted(y_test.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    print("Confusion matrix (rows = actual, columns = predicted):")
    print(pd.DataFrame(cm, index=labels, columns=labels).to_string())
    print("=" * 58 + "\n")

    report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    return scores, labels, cm, report


# --------------------------------------------------------------------------
# Step 6  --  save everything so the app never has to retrain
# --------------------------------------------------------------------------
def save_artifacts(vectorizer, model, scores, labels, cm, report, n_train, n_test):
    """Save the vectorizer, the model and a small JSON of evaluation scores."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(model, MODEL_PATH)

    info = {
        "trained_at": datetime.now().strftime("%d %B %Y, %H:%M"),
        "dataset": "Emotion Detection from Text (Pashupati Gupta, Kaggle)",
        "n_train": int(n_train),
        "n_test": int(n_test),
        "n_features": int(len(vectorizer.vocabulary_)),
        "labels": list(labels),
        "scores": scores,
        "confusion_matrix": cm.tolist(),
        "per_class_report": report,
    }
    with open(INFO_PATH, "w") as f:
        json.dump(info, f, indent=2)

    print(f"[6/6] Saved:\n      {VECTORIZER_PATH}\n      {MODEL_PATH}\n      {INFO_PATH}")


# --------------------------------------------------------------------------
# The full pipeline in one function (the app can call this too)
# --------------------------------------------------------------------------
def train_and_save():
    """Run every step above, from raw CSV to saved model files."""
    df = load_dataset()
    df = prepare_data(df)
    X_train, X_test, y_train, y_test = split_data(df)
    vectorizer, model = train_model(X_train, y_train)
    scores, labels, cm, report = evaluate_model(vectorizer, model, X_test, y_test)
    save_artifacts(vectorizer, model, scores, labels, cm, report, len(X_train), len(X_test))
    return scores


def main():
    print("\nTraining the Emotion Detection model...\n")
    try:
        scores = train_and_save()
    except (FileNotFoundError, ValueError) as error:
        # Friendly message instead of a raw Python traceback
        print("\n[!] Training stopped:\n")
        print(error)
        sys.exit(1)

    print(f"Done! Test accuracy: {scores['accuracy'] * 100:.2f}%")
    print("Now run:  streamlit run app.py\n")


if __name__ == "__main__":
    main()
