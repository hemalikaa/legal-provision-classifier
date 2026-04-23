"""
tfidf_baseline.py
=================
Classical TF-IDF + Logistic Regression baseline for legal provision
classification on the LEDGAR dataset.

This serves as the non-neural lower bound in our comparison. TF-IDF
represents each provision as a sparse vector of weighted word frequencies,
which Logistic Regression then classifies into one of 100 categories.

Why include this baseline:
  - Establishes the minimum acceptable performance for any neural approach
  - Requires no GPU, no API calls, runs in under 2 minutes
  - Directly comparable to published classical baselines in legal NLP
  - Demonstrates the value added by fine-tuning and LLM-based approaches

Expected results on LEDGAR test set:
  Micro-F1: ~0.70-0.73  |  Macro-F1: ~0.55-0.60
  (significantly below BERT fine-tuned: 0.8715 / 0.7862)
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
from datasets import DatasetDict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def prepare_data(
    dataset: DatasetDict,
    split: str,
) -> Tuple[List[str], List[int]]:
    """Extract texts and labels from a dataset split.

    Args:
        dataset: HuggingFace DatasetDict with LEDGAR splits.
        split: One of 'train', 'validation', 'test'.

    Returns:
        Tuple of (list of provision texts, list of integer label IDs).
    """
    texts = [example["text"] for example in dataset[split]]
    labels = [example["label"] for example in dataset[split]]
    logger.info("Prepared %d examples from '%s' split", len(texts), split)
    return texts, labels


def train_tfidf(
    dataset: DatasetDict,
    config: Dict,
    label_names: List[str],
) -> Tuple[TfidfVectorizer, LogisticRegression]:
    """Train TF-IDF vectorizer and Logistic Regression classifier on LEDGAR.

    TF-IDF (Term Frequency-Inverse Document Frequency) converts each
    provision into a sparse numerical vector where each dimension represents
    a word, weighted by how often it appears in that provision relative to
    how common it is across all provisions.

    Logistic Regression then learns a linear decision boundary in this
    high-dimensional space to separate the 100 provision categories.

    Args:
        dataset: HuggingFace DatasetDict with LEDGAR splits.
        config: Project configuration dictionary.
        label_names: List of 100 provision category names.

    Returns:
        Tuple of (fitted TfidfVectorizer, fitted LogisticRegression).
    """
    seed = config["project"]["seed"]

    # Prepare training data
    train_texts, train_labels = prepare_data(dataset, "train")

    # TF-IDF vectorizer settings
    # max_features=50000: keep the 50K most informative terms
    # ngram_range=(1,2): use both single words and bigrams (e.g. "governing law")
    # sublinear_tf=True: apply log normalization to term frequencies
    logger.info("Fitting TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"\b[a-zA-Z][a-zA-Z]+\b",
    )
    X_train = vectorizer.fit_transform(train_texts)
    logger.info(
        "TF-IDF matrix shape: %d provisions x %d features",
        X_train.shape[0],
        X_train.shape[1],
    )

    # Logistic Regression with L2 regularization
    # C=5.0: regularization strength (higher = less regularization)
    # max_iter=1000: enough iterations to converge on 100-class problem
    # solver='saga': efficient for large sparse datasets
    logger.info("Training Logistic Regression classifier...")
    clf = LogisticRegression(
        C=5.0,
        max_iter=1000,
        solver="saga",
        multi_class="multinomial",
        random_state=seed,
        n_jobs=-1,
        verbose=1,
    )
    clf.fit(X_train, train_labels)
    logger.info("Logistic Regression training complete.")

    # Save model artifacts
    output_dir = Path("models/tfidf_baseline")
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, output_dir / "vectorizer.joblib")
    joblib.dump(clf, output_dir / "classifier.joblib")
    logger.info("TF-IDF model saved to %s", output_dir)

    return vectorizer, clf


def evaluate_tfidf(
    dataset: DatasetDict,
    label_names: List[str],
    vectorizer: TfidfVectorizer = None,
    clf: LogisticRegression = None,
    model_dir: str = "models/tfidf_baseline",
) -> Dict:
    """Evaluate TF-IDF + LR classifier on the LEDGAR test set.

    Args:
        dataset: HuggingFace DatasetDict with LEDGAR splits.
        label_names: List of 100 provision category names.
        vectorizer: Fitted TfidfVectorizer. If None, loads from model_dir.
        clf: Fitted LogisticRegression. If None, loads from model_dir.
        model_dir: Path to saved model artifacts.

    Returns:
        Dictionary containing micro-F1, macro-F1, and classification report.
    """
    # Load from disk if not provided
    if vectorizer is None or clf is None:
        logger.info("Loading TF-IDF model from %s", model_dir)
        vectorizer = joblib.load(Path(model_dir) / "vectorizer.joblib")
        clf = joblib.load(Path(model_dir) / "classifier.joblib")

    test_texts, test_labels = prepare_data(dataset, "test")

    logger.info("Transforming test set with TF-IDF...")
    X_test = vectorizer.transform(test_texts)

    logger.info("Running predictions on test set...")
    predictions = clf.predict(X_test)

    micro_f1 = f1_score(test_labels, predictions, average="micro", zero_division=0)
    macro_f1 = f1_score(test_labels, predictions, average="macro", zero_division=0)

    report = classification_report(
        test_labels,
        predictions,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )

    logger.info(
        "TF-IDF Test Results — Micro-F1: %.4f | Macro-F1: %.4f",
        micro_f1,
        macro_f1,
    )

    return {
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "classification_report": report,
        "predictions": predictions.tolist(),
        "labels": test_labels,
    }


def load_tfidf_model(
    model_dir: str = "models/tfidf_baseline",
) -> Tuple[TfidfVectorizer, LogisticRegression]:
    """Load saved TF-IDF model for inference in Streamlit app.

    Args:
        model_dir: Path to saved model artifacts.

    Returns:
        Tuple of (TfidfVectorizer, LogisticRegression).

    Raises:
        FileNotFoundError: If model files do not exist.
    """
    vectorizer_path = Path(model_dir) / "vectorizer.joblib"
    clf_path = Path(model_dir) / "classifier.joblib"

    if not vectorizer_path.exists() or not clf_path.exists():
        raise FileNotFoundError(
            f"TF-IDF model not found at '{model_dir}'. "
            "Run tfidf_baseline.py first to train it."
        )

    vectorizer = joblib.load(vectorizer_path)
    clf = joblib.load(clf_path)
    logger.info("TF-IDF model loaded from %s", model_dir)
    return vectorizer, clf


def classify_single(
    text: str,
    vectorizer: TfidfVectorizer,
    clf: LogisticRegression,
    label_names: List[str],
) -> Dict:
    """Classify a single provision using the TF-IDF model.

    Used by the Streamlit app for real-time inference.

    Args:
        text: Contract provision text.
        vectorizer: Fitted TfidfVectorizer.
        clf: Fitted LogisticRegression.
        label_names: List of provision category names.

    Returns:
        Dictionary with predicted category and confidence score.
    """
    X = vectorizer.transform([text])
    pred_id = clf.predict(X)[0]
    proba = clf.predict_proba(X)[0]
    confidence = float(proba[pred_id])

    return {
        "prediction": label_names[pred_id],
        "confidence": round(confidence, 4),
    }


if __name__ == "__main__":
    from data_loader import load_config, load_ledgar

    cfg = load_config()
    ds, labels = load_ledgar(cfg)

    vectorizer, clf = train_tfidf(ds, cfg, labels)
    results = evaluate_tfidf(ds, labels, vectorizer, clf)
    logger.info(
        "Micro-F1: %.4f | Macro-F1: %.4f",
        results["micro_f1"],
        results["macro_f1"],
    )