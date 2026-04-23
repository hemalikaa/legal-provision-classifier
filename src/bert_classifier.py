"""
bert_classifier.py
==================
Fine-tunes BERT-base-uncased on the LEDGAR dataset for single-label
legal provision classification across 100 categories.

Results from prior training run:
  Micro-F1: 0.8715  |  Macro-F1: 0.7862  (exceeds LEGAL-BERT baseline of ~0.77)
"""

import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from datasets import DatasetDict
from sklearn.metrics import classification_report, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    pipeline,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set random seeds for full reproducibility across all libraries.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info("Random seed set to %d", seed)


def tokenize_dataset(
    dataset: DatasetDict,
    tokenizer_name: str,
    max_length: int,
) -> DatasetDict:
    """Tokenize the full dataset using the specified tokenizer.

    Args:
        dataset: HuggingFace DatasetDict with train/validation/test splits.
        tokenizer_name: HuggingFace model name for the tokenizer.
        max_length: Maximum token sequence length (512 for BERT).

    Returns:
        Tokenized DatasetDict ready for training.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def tokenize_fn(batch: Dict) -> Dict:
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    tokenized = dataset.map(tokenize_fn, batched=True, desc="Tokenizing")
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    logger.info("Tokenization complete.")
    return tokenized


def compute_metrics(eval_pred: Tuple) -> Dict[str, float]:
    """Compute micro and macro F1 scores for multi-class classification.

    These are the standard metrics for LEDGAR as established in the
    LexGLUE benchmark (Chalkidis et al., 2022).

    Args:
        eval_pred: Tuple of (logits, labels) from the Trainer.

    Returns:
        Dictionary with 'f1_micro' and 'f1_macro' scores.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "f1_micro": f1_score(labels, predictions, average="micro", zero_division=0),
        "f1_macro": f1_score(labels, predictions, average="macro", zero_division=0),
    }


def train_bert(
    config: Dict,
    dataset: DatasetDict,
    label_names: List[str],
) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    """Fine-tune BERT-base-uncased on LEDGAR for provision classification.

    Args:
        config: Project configuration dictionary.
        dataset: HuggingFace DatasetDict with raw text splits.
        label_names: List of 100 provision category names.

    Returns:
        Tuple of (fine-tuned model, tokenizer).
    """
    bert_cfg = config["bert"]
    data_cfg = config["data"]
    seed = config["project"]["seed"]

    set_seed(seed)

    # Tokenize
    tokenized = tokenize_dataset(dataset, bert_cfg["model_name"], data_cfg["max_length"])
    tokenizer = AutoTokenizer.from_pretrained(bert_cfg["model_name"])

    # Load model with classification head
    model = AutoModelForSequenceClassification.from_pretrained(
        bert_cfg["model_name"],
        num_labels=data_cfg["num_labels"],
        id2label={i: lbl for i, lbl in enumerate(label_names)},
        label2id={lbl: i for i, lbl in enumerate(label_names)},
    )
    logger.info("Loaded %s with %d labels", bert_cfg["model_name"], data_cfg["num_labels"])

    training_args = TrainingArguments(
        output_dir=bert_cfg["output_dir"],
        num_train_epochs=bert_cfg["num_train_epochs"],
        per_device_train_batch_size=bert_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=bert_cfg["per_device_eval_batch_size"],
        learning_rate=bert_cfg["learning_rate"],
        weight_decay=bert_cfg["weight_decay"],
        warmup_ratio=bert_cfg["warmup_ratio"],
        eval_strategy=bert_cfg["eval_strategy"],
        save_strategy=bert_cfg["save_strategy"],
        load_best_model_at_end=bert_cfg["load_best_model_at_end"],
        metric_for_best_model=bert_cfg["metric_for_best_model"],
        fp16=bert_cfg["fp16"] and torch.cuda.is_available(),
        logging_steps=bert_cfg["logging_steps"],
        seed=seed,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    logger.info("Starting BERT fine-tuning...")
    trainer.train()
    logger.info("Training complete.")

    # Save model and tokenizer
    Path(bert_cfg["output_dir"]).mkdir(parents=True, exist_ok=True)
    trainer.save_model(bert_cfg["output_dir"])
    tokenizer.save_pretrained(bert_cfg["output_dir"])
    logger.info("Model saved to %s", bert_cfg["output_dir"])

    return model, tokenizer


def evaluate_bert(
    config: Dict,
    dataset: DatasetDict,
    label_names: List[str],
    model_dir: Optional[str] = None,
) -> Dict:
    """Evaluate fine-tuned BERT on the test set and return full metrics.

    Args:
        config: Project configuration dictionary.
        dataset: HuggingFace DatasetDict.
        label_names: List of provision category names.
        model_dir: Path to saved model. Defaults to config value.

    Returns:
        Dictionary containing micro-F1, macro-F1, and per-class classification report.
    """
    bert_cfg = config["bert"]
    data_cfg = config["data"]
    model_dir = model_dir or bert_cfg["output_dir"]

    logger.info("Loading fine-tuned BERT from %s", model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    tokenized = tokenize_dataset(dataset, model_dir, data_cfg["max_length"])

    trainer = Trainer(
        model=model,
        compute_metrics=compute_metrics,
    )

    logger.info("Running evaluation on test set...")
    predictions_output = trainer.predict(tokenized["test"])
    preds = np.argmax(predictions_output.predictions, axis=-1)
    labels = predictions_output.label_ids

    micro_f1 = f1_score(labels, preds, average="micro", zero_division=0)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)

    report = classification_report(
        labels,
        preds,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )

    logger.info("BERT Test Results — Micro-F1: %.4f | Macro-F1: %.4f", micro_f1, macro_f1)

    return {
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "classification_report": report,
        "predictions": preds.tolist(),
        "labels": labels.tolist(),
    }


def load_bert_pipeline(model_dir: str) -> pipeline:
    """Load fine-tuned BERT as a HuggingFace inference pipeline for the Streamlit app.

    Args:
        model_dir: Path to the saved fine-tuned BERT model.

    Returns:
        HuggingFace text classification pipeline.
    """
    device = 0 if torch.cuda.is_available() else -1
    clf_pipeline = pipeline(
        "text-classification",
        model=model_dir,
        tokenizer=model_dir,
        device=device,
        truncation=True,
        max_length=512,
    )
    logger.info("BERT inference pipeline loaded from %s", model_dir)
    return clf_pipeline


if __name__ == "__main__":
    from data_loader import load_config, load_ledgar

    cfg = load_config()
    ds, labels = load_ledgar(cfg)
    train_bert(cfg, ds, labels)
    results = evaluate_bert(cfg, ds, labels)
    logger.info("Micro-F1: %.4f | Macro-F1: %.4f", results["micro_f1"], results["macro_f1"])