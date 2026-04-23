"""
evaluate.py
===========
Unified evaluation script comparing all five classification approaches:
  1. TF-IDF + Logistic Regression (classical baseline)
  2. Zero-shot Llama-3.3-70B via Groq
  3. Chain-of-Thought Llama-3.3-70B via Groq
  4. RAG Few-shot Llama-3.3-70B + ChromaDB via Groq
  5. Fine-tuned BERT-base (Micro-F1: 0.8715, Macro-F1: 0.7862)

Published baseline for comparison:
  LEGAL-BERT (Chalkidis et al., 2022): Macro-F1 ~0.77

Also evaluates the risk detector against a hand-labelled gold set
and generates all plots and tables for the final report.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score, precision_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def save_results(results: Dict, output_path: str) -> None:
    """Save evaluation results to a JSON file.

    Args:
        results: Dictionary of evaluation metrics and predictions.
        output_path: Path to save the JSON file.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    serialisable = {
        k: v.tolist() if isinstance(v, np.ndarray) else v
        for k, v in results.items()
        if k != "classification_report"
    }
    with open(output_path, "w") as f:
        json.dump(serialisable, f, indent=2)
    logger.info("Results saved to %s", output_path)


def plot_f1_comparison(
    tfidf_results: Dict,
    zeroshot_results: Dict,
    cot_results: Dict,
    rag_results: Dict,
    bert_results: Dict,
    output_dir: str,
) -> None:
    """Bar chart comparing Micro-F1 and Macro-F1 across all five approaches.

    Args:
        tfidf_results: TF-IDF baseline results dict.
        zeroshot_results: Zero-shot results dict.
        cot_results: Chain-of-Thought results dict.
        rag_results: RAG results dict.
        bert_results: BERT results dict.
        output_dir: Directory to save the plot.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    models = [
        "TF-IDF\n+ LR",
        "Zero-shot\nLlama-3.3-70B",
        "CoT\nLlama-3.3-70B",
        "RAG Few-shot\nLlama-3.3-70B",
        "BERT\n(Fine-tuned)",
    ]
    micro_f1 = [
        tfidf_results["micro_f1"],
        zeroshot_results["micro_f1"],
        cot_results["micro_f1"],
        rag_results["micro_f1"],
        bert_results["micro_f1"],
    ]
    macro_f1 = [
        tfidf_results["macro_f1"],
        zeroshot_results["macro_f1"],
        cot_results["macro_f1"],
        rag_results["macro_f1"],
        bert_results["macro_f1"],
    ]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width / 2, micro_f1, width, label="Micro-F1", color="#2563EB", alpha=0.85)
    bars2 = ax.bar(x + width / 2, macro_f1, width, label="Macro-F1", color="#16A34A", alpha=0.85)

    for bar in bars1:
        ax.annotate(
            f"{bar.get_height():.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )
    for bar in bars2:
        ax.annotate(
            f"{bar.get_height():.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )

    # LEGAL-BERT published baseline
    ax.axhline(
        y=0.77,
        color="red",
        linestyle="--",
        linewidth=1.2,
        label="LEGAL-BERT Baseline (Macro-F1 ~0.77)",
    )

    ax.set_xlabel("Classification Approach", fontsize=12)
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_title(
        "Legal Provision Classification: F1 Score Comparison\nLEDGAR Dataset (100 Categories)",
        fontsize=13,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "f1_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("F1 comparison plot saved.")


def plot_top_class_f1(
    bert_results: Dict,
    label_names: List[str],
    output_dir: str,
    top_n: int = 20,
) -> None:
    """Horizontal bar chart of per-class F1 for top-N categories (BERT).

    Args:
        bert_results: BERT results dict with classification_report.
        label_names: List of all label names.
        output_dir: Directory to save the plot.
        top_n: Number of top categories to display.
    """
    report = bert_results.get("classification_report", {})
    if not report:
        logger.warning("No classification report available.")
        return

    class_f1 = {
        label: report[label]["f1-score"]
        for label in label_names
        if label in report
    }
    sorted_labels = sorted(class_f1, key=class_f1.get, reverse=True)[:top_n]
    scores = [class_f1[l] for l in sorted_labels]

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = [
        "#2563EB" if s >= 0.9 else "#F59E0B" if s >= 0.75 else "#EF4444"
        for s in scores
    ]
    ax.barh(sorted_labels[::-1], scores[::-1], color=colors[::-1], alpha=0.85)
    ax.set_xlabel("F1 Score", fontsize=12)
    ax.set_title(
        f"Top {top_n} Provision Categories by F1 Score (BERT Fine-tuned)",
        fontsize=13,
    )
    ax.set_xlim(0, 1.05)
    ax.axvline(x=0.9, color="green", linestyle="--", linewidth=1, alpha=0.7, label="F1 = 0.90")
    ax.legend(fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "per_class_f1.png"), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Per-class F1 plot saved.")


def plot_risk_distribution(risk_results: List[Dict], output_dir: str) -> None:
    """Pie chart showing distribution of risk levels across assessed provisions.

    Args:
        risk_results: List of risk assessment dictionaries.
        output_dir: Directory to save the plot.
    """
    risk_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
    for r in risk_results:
        risk_counts[r["risk_level"]] += 1

    labels = list(risk_counts.keys())
    sizes = list(risk_counts.values())
    colors = ["#16A34A", "#F59E0B", "#EF4444"]
    explode = (0, 0.05, 0.1)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.pie(
        sizes,
        explode=explode,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 12},
    )
    ax.set_title("Contract Provision Risk Level Distribution\n(LEDGAR Test Set)", fontsize=13)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "risk_distribution.png"), dpi=150, bbox_inches="tight"
    )
    plt.close()
    logger.info("Risk distribution plot saved.")


def evaluate_risk_detector(
    risk_detector,
    gold_set: List[Dict],
    label_names: List[str],
) -> Dict:
    """Evaluate risk detector against a hand-labelled gold set.

    The gold set contains provisions manually labelled as LOW, MEDIUM,
    or HIGH risk. Precision measures how often our detector's HIGH risk
    flags are correct.

    Args:
        risk_detector: Initialised RiskDetector instance.
        gold_set: List of dicts with 'text', 'category', 'true_risk_level'.
        label_names: List of provision category names.

    Returns:
        Dictionary with precision, recall, and per-class metrics.
    """
    true_labels = []
    pred_labels = []

    for item in gold_set:
        result = risk_detector.assess(item["text"], item["category"])
        true_labels.append(item["true_risk_level"])
        pred_labels.append(result["risk_level"])

    risk_levels = ["LOW", "MEDIUM", "HIGH"]
    label2id = {l: i for i, l in enumerate(risk_levels)}

    true_ids = [label2id[l] for l in true_labels]
    pred_ids = [label2id[l] for l in pred_labels]

    precision = precision_score(true_ids, pred_ids, average="weighted", zero_division=0)
    micro_f1 = f1_score(true_ids, pred_ids, average="micro", zero_division=0)

    report = classification_report(
        true_labels,
        pred_labels,
        labels=risk_levels,
        output_dict=True,
        zero_division=0,
    )

    logger.info(
        "Risk Detector — Precision: %.4f | Micro-F1: %.4f", precision, micro_f1
    )

    return {
        "precision": round(precision, 4),
        "micro_f1": round(micro_f1, 4),
        "classification_report": report,
        "true_labels": true_labels,
        "pred_labels": pred_labels,
    }


def generate_comparison_table(
    tfidf_results: Dict,
    zeroshot_results: Dict,
    cot_results: Dict,
    rag_results: Dict,
    bert_results: Dict,
    output_dir: str,
) -> pd.DataFrame:
    """Generate and save a comparison table of all approaches vs baselines.

    Args:
        tfidf_results: TF-IDF results.
        zeroshot_results: Zero-shot results.
        cot_results: CoT results.
        rag_results: RAG results.
        bert_results: BERT results.
        output_dir: Directory to save the CSV.

    Returns:
        DataFrame with the full comparison table.
    """
    data = {
        "Approach": [
            "LEGAL-BERT (Chalkidis et al., 2022) — Published Baseline",
            "TF-IDF + Logistic Regression (Ours)",
            "Zero-shot Llama-3.3-70B (Ours)",
            "Chain-of-Thought Llama-3.3-70B (Ours)",
            "RAG Few-shot Llama-3.3-70B (Ours)",
            "BERT-base Fine-tuned (Ours)",
        ],
        "Micro-F1": [
            "~0.8700",
            f"{tfidf_results['micro_f1']:.4f}",
            f"{zeroshot_results['micro_f1']:.4f}",
            f"{cot_results['micro_f1']:.4f}",
            f"{rag_results['micro_f1']:.4f}",
            f"{bert_results['micro_f1']:.4f}",
        ],
        "Macro-F1": [
            "~0.7700",
            f"{tfidf_results['macro_f1']:.4f}",
            f"{zeroshot_results['macro_f1']:.4f}",
            f"{cot_results['macro_f1']:.4f}",
            f"{rag_results['macro_f1']:.4f}",
            f"{bert_results['macro_f1']:.4f}",
        ],
        "Training Required": ["Yes", "Yes", "No", "No", "No", "Yes"],
        "Approach Type": [
            "Domain-adapted BERT",
            "Classical ML",
            "Zero-shot Prompting",
            "CoT Prompting",
            "RAG + Few-shot",
            "BERT Fine-tuning",
        ],
    }
    df = pd.DataFrame(data)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = os.path.join(output_dir, "comparison_table.csv")
    df.to_csv(output_path, index=False)
    logger.info("Comparison table saved to %s", output_path)
    return df


def run_full_evaluation(
    config: Dict,
    dataset,
    label_names: List[str],
    groq_api_key: str,
    gold_set: Optional[List[Dict]] = None,
    bert_model_dir: Optional[str] = None,
    skip_bert_train: bool = True,
) -> Dict:
    """Run complete evaluation of all five approaches and the risk detector.

    Args:
        config: Project configuration dictionary.
        dataset: HuggingFace DatasetDict.
        label_names: List of provision category names.
        groq_api_key: Groq API key.
        gold_set: Hand-labelled provisions for risk detector evaluation.
        bert_model_dir: Path to saved BERT model.
        skip_bert_train: If True, load BERT from disk instead of retraining.

    Returns:
        Dictionary with results for all approaches.
    """
    from data_loader import get_sample, load_reference_embeddings
    from bert_classifier import evaluate_bert, train_bert
    from tfidf_baseline import evaluate_tfidf, train_tfidf
    from zeroshot_classifier import ZeroShotClassifier
    from cot_classifier import CoTClassifier
    from rag_pipeline import RAGClassifier
    from risk_detector import RiskDetector

    results_dir = config["evaluation"]["results_dir"]
    plots_dir = config["evaluation"]["plots_dir"]
    Path(plots_dir).mkdir(parents=True, exist_ok=True)

    # --- TF-IDF ---
    logger.info("=== Evaluating TF-IDF + Logistic Regression ===")
    tfidf_vec, tfidf_clf = train_tfidf(dataset, config, label_names)
    tfidf_results = evaluate_tfidf(dataset, label_names, tfidf_vec, tfidf_clf)
    save_results(tfidf_results, os.path.join(results_dir, "tfidf_results.json"))

    # --- BERT ---
    logger.info("=== Evaluating BERT ===")
    if not skip_bert_train:
        train_bert(config, dataset, label_names)
    bert_results = evaluate_bert(config, dataset, label_names, bert_model_dir)
    save_results(bert_results, os.path.join(results_dir, "bert_results.json"))

    # --- Zero-shot ---
    logger.info("=== Evaluating Zero-shot Llama ===")
    zs_clf = ZeroShotClassifier(config, label_names, groq_api_key)
    zs_samples = get_sample(dataset, "test", config["evaluation"]["zeroshot_sample_size"])
    zeroshot_results = zs_clf.classify_batch(zs_samples, label_names)
    save_results(zeroshot_results, os.path.join(results_dir, "zeroshot_results.json"))

    # --- CoT ---
    logger.info("=== Evaluating Chain-of-Thought Llama ===")
    cot_clf = CoTClassifier(config, label_names, groq_api_key)
    cot_samples = get_sample(dataset, "test", config["evaluation"]["zeroshot_sample_size"])
    cot_results = cot_clf.classify_batch(cot_samples)
    save_results(cot_results, os.path.join(results_dir, "cot_results.json"))

    # --- RAG ---
    logger.info("=== Evaluating RAG Few-shot Llama ===")
    rag_clf = RAGClassifier(config, label_names, groq_api_key)
    train_samples = get_sample(dataset, "train", 5000, seed=config["project"]["seed"])
    rag_clf.build_index(train_samples)
    rag_samples = get_sample(dataset, "test", config["rag"]["eval_sample_size"])
    rag_results = rag_clf.classify_batch(rag_samples)
    save_results(rag_results, os.path.join(results_dir, "rag_results.json"))

    # --- Risk Detector ---
    logger.info("=== Evaluating Risk Detector ===")
    ref_embeddings = load_reference_embeddings(
        config["data"]["reference_embeddings_path"], label_names
    )
    detector = RiskDetector(config, ref_embeddings)

    # Batch risk assessment on test samples
    test_samples = get_sample(dataset, "test", 100)
    risk_texts = [s["text"] for s in test_samples]
    risk_categories = [label_names[s["label"]] for s in test_samples]
    risk_results = detector.assess_batch(risk_texts, risk_categories)
    save_results(
        {"risk_assessments": risk_results},
        os.path.join(results_dir, "risk_results.json"),
    )

    # Gold set evaluation if provided
    risk_eval = {}
    if gold_set:
        logger.info("=== Evaluating Risk Detector on Gold Set ===")
        risk_eval = evaluate_risk_detector(detector, gold_set, label_names)
        save_results(risk_eval, os.path.join(results_dir, "risk_gold_eval.json"))

    # --- Plots and Tables ---
    logger.info("=== Generating Plots and Comparison Table ===")
    plot_f1_comparison(
        tfidf_results, zeroshot_results, cot_results, rag_results, bert_results, plots_dir
    )
    plot_top_class_f1(bert_results, label_names, plots_dir)
    plot_risk_distribution(risk_results, plots_dir)
    df = generate_comparison_table(
        tfidf_results, zeroshot_results, cot_results, rag_results, bert_results, results_dir
    )

    logger.info("\n%s", df.to_string(index=False))
    logger.info("Full evaluation complete. Results saved to %s", results_dir)

    return {
        "tfidf": tfidf_results,
        "zeroshot": zeroshot_results,
        "cot": cot_results,
        "rag": rag_results,
        "bert": bert_results,
        "risk": risk_results,
        "risk_eval": risk_eval,
    }


if __name__ == "__main__":
    from data_loader import load_config, load_ledgar

    cfg = load_config()
    ds, labels = load_ledgar(cfg)

    run_full_evaluation(
        config=cfg,
        dataset=ds,
        label_names=labels,
        groq_api_key=os.environ["GROQ_API_KEY"],
        bert_model_dir=cfg["bert"]["output_dir"],
        skip_bert_train=True,
    )