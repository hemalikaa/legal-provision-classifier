"""
data_loader.py
==============
Handles loading and preprocessing of the LEDGAR dataset from HuggingFace
and building reference embeddings for risk detection.

Reference embeddings are computed as the centroid (average) embedding of
all training provisions per category. These serve as the 'standard' for
each provision type during risk assessment — no manually written templates
are used.
"""

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml
from datasets import DatasetDict, load_dataset
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml") -> Dict:
    """Load project configuration from YAML file.

    Args:
        config_path: Path to the config YAML file.

    Returns:
        Dictionary containing all configuration parameters.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logger.info("Config loaded from %s", config_path)
    return config


def load_ledgar(config: Dict) -> Tuple[DatasetDict, List[str]]:
    """Load the LEDGAR dataset from HuggingFace and return dataset + label names.

    LEDGAR is a single-label multi-class dataset of 60,000 contract provisions
    from SEC EDGAR filings, annotated across 100 provision categories.

    Args:
        config: Project configuration dictionary.

    Returns:
        Tuple of (DatasetDict with train/validation/test splits, list of label names).

    Raises:
        ValueError: If the dataset cannot be loaded.
    """
    data_cfg = config["data"]
    logger.info("Loading LEDGAR dataset from HuggingFace: %s", data_cfg["dataset_name"])

    dataset = load_dataset(data_cfg["dataset_name"], data_cfg["dataset_config"])
    label_names: List[str] = dataset["train"].features["label"].names

    logger.info(
        "Dataset loaded — Train: %d | Val: %d | Test: %d | Labels: %d",
        len(dataset["train"]),
        len(dataset["validation"]),
        len(dataset["test"]),
        len(label_names),
    )
    return dataset, label_names


def get_sample(
    dataset: DatasetDict,
    split: str,
    n: int,
    seed: int = 42,
) -> List[Dict]:
    """Return a reproducible random sample from a dataset split.

    Args:
        dataset: HuggingFace DatasetDict.
        split: One of 'train', 'validation', 'test'.
        n: Number of samples to return.
        seed: Random seed for reproducibility.

    Returns:
        List of sample dictionaries with 'text' and 'label' keys.
    """
    random.seed(seed)
    split_data = dataset[split]
    indices = random.sample(range(len(split_data)), min(n, len(split_data)))
    samples = [split_data[i] for i in indices]
    logger.info("Sampled %d examples from '%s' split", len(samples), split)
    return samples


def build_reference_embeddings(
    dataset: DatasetDict,
    label_names: List[str],
    embedding_model: str,
    output_path: str,
    max_per_category: int = 100,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Build and save reference embeddings for risk detection.

    For each of the 100 provision categories, we compute the centroid
    (average) embedding of up to max_per_category training provisions.
    This centroid represents what a 'standard' provision of that type
    looks like — derived entirely from real LEDGAR data, not manually
    written templates.

    During risk detection, a new provision's embedding is compared against
    its category centroid using cosine similarity. Low similarity indicates
    the provision deviates significantly from standard language for that
    category, which warrants review.

    Args:
        dataset: HuggingFace DatasetDict containing LEDGAR splits.
        label_names: List of 100 provision category names.
        embedding_model: Name of the sentence-transformer model to use.
        output_path: Path to save the reference embeddings as a .npz file.
        max_per_category: Maximum number of training provisions to average
            per category. More provisions = more stable centroid.
        seed: Random seed for reproducibility when sampling per category.

    Returns:
        Dictionary mapping category name to its centroid embedding (np.ndarray).
    """
    random.seed(seed)
    logger.info("Loading sentence transformer: %s", embedding_model)
    model = SentenceTransformer(embedding_model)

    # Group training provisions by category
    logger.info("Grouping training provisions by category...")
    category_texts: Dict[str, List[str]] = {label: [] for label in label_names}
    for example in dataset["train"]:
        label = label_names[example["label"]]
        category_texts[label].append(example["text"])

    # Compute centroid embedding per category
    reference_embeddings: Dict[str, np.ndarray] = {}

    logger.info(
        "Computing centroid embeddings for %d categories (up to %d provisions each)...",
        len(label_names),
        max_per_category,
    )

    for label in label_names:
        texts = category_texts[label]

        if not texts:
            logger.warning("No training examples found for category '%s'", label)
            # Fallback: encode the label name itself as a proxy
            reference_embeddings[label] = model.encode(label, normalize_embeddings=True)
            continue

        # Sample up to max_per_category provisions
        sampled = random.sample(texts, min(max_per_category, len(texts)))

        # Encode and compute centroid
        embeddings = model.encode(sampled, normalize_embeddings=True, show_progress_bar=False)
        centroid = np.mean(embeddings, axis=0)

        # Re-normalise centroid so cosine similarity remains valid
        centroid = centroid / np.linalg.norm(centroid)
        reference_embeddings[label] = centroid

    # Save to disk as compressed numpy archive
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **{
        label.replace(" ", "_"): emb
        for label, emb in reference_embeddings.items()
    })
    logger.info(
        "Reference embeddings saved to %s (%d categories)",
        output_path,
        len(reference_embeddings),
    )
    return reference_embeddings


def load_reference_embeddings(
    output_path: str,
    label_names: List[str],
) -> Dict[str, np.ndarray]:
    """Load pre-computed reference embeddings from disk.

    Args:
        output_path: Path to the saved .npz file.
        label_names: List of provision category names.

    Returns:
        Dictionary mapping category name to centroid embedding.

    Raises:
        FileNotFoundError: If embeddings file does not exist.
    """
    path = Path(output_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Reference embeddings not found at '{output_path}'. "
            "Run data_loader.py first to build them."
        )
    data = np.load(output_path)
    embeddings = {
        label: data[label.replace(" ", "_")]
        for label in label_names
        if label.replace(" ", "_") in data
    }
    logger.info(
        "Loaded reference embeddings for %d categories from %s",
        len(embeddings),
        output_path,
    )
    return embeddings


if __name__ == "__main__":
    cfg = load_config()
    dataset, label_names = load_ledgar(cfg)
    build_reference_embeddings(
        dataset=dataset,
        label_names=label_names,
        embedding_model=cfg["risk"]["embedding_model"],
        output_path=cfg["data"]["reference_embeddings_path"],
        max_per_category=cfg["data"]["max_per_category"],
        seed=cfg["project"]["seed"],
    )
    logger.info("Data setup complete.")