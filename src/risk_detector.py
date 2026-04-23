"""
risk_detector.py
================
Contract risk detection using cosine similarity between a provision's
embedding and the centroid embedding of its predicted category.

Centroid embeddings are computed from LEDGAR training provisions — they
represent what a 'typical' provision of each category looks like based
on real SEC contract data, not manually written templates.

Risk levels:
  LOW    — provision closely matches typical language (similarity >= 0.85)
  MEDIUM — some deviation from typical language  (similarity >= 0.65)
  HIGH   — significant deviation from typical language (similarity < 0.65)

Limitation: These thresholds and centroids are derived from SEC-filed
commercial contracts only. Results may not generalise to other contract
types such as employment agreements or residential leases.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class RiskDetector:
    """Detects contract risk by comparing provisions to LEDGAR category centroids.

    For each provision, we embed it using a sentence-transformer model and
    compute cosine similarity against the centroid embedding of its predicted
    category. The centroid is the average embedding of up to 100 training
    provisions from that category in LEDGAR.

    Low similarity means the provision's language deviates significantly
    from what is typical for that category in real SEC contracts.

    Attributes:
        config: Project configuration dictionary.
        reference_embeddings: Dict mapping category name to centroid embedding.
        model: Sentence transformer for encoding new provisions.
        high_threshold: Cosine similarity threshold for LOW risk.
        medium_threshold: Cosine similarity threshold for MEDIUM risk.
    """

    def __init__(
        self,
        config: Dict,
        reference_embeddings: Dict[str, np.ndarray],
    ) -> None:
        """Initialise the risk detector with pre-computed reference embeddings.

        Args:
            config: Project configuration dictionary.
            reference_embeddings: Dict mapping category name to centroid
                embedding computed from LEDGAR training provisions.
        """
        self.config = config
        self.reference_embeddings = reference_embeddings
        self.high_threshold = config["risk"]["similarity_threshold_high"]
        self.medium_threshold = config["risk"]["similarity_threshold_medium"]

        logger.info("Loading sentence transformer: %s", config["risk"]["embedding_model"])
        self.model = SentenceTransformer(config["risk"]["embedding_model"])
        logger.info(
            "RiskDetector ready with %d category reference embeddings.",
            len(reference_embeddings),
        )

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two normalised embedding vectors.

        Since both vectors are L2-normalised, cosine similarity equals
        the dot product directly.

        Args:
            a: First normalised embedding vector.
            b: Second normalised embedding vector.

        Returns:
            Cosine similarity score in range [-1, 1].
        """
        return float(np.dot(a, b))

    def _get_risk_level(self, similarity: float) -> str:
        """Map cosine similarity score to a risk level.

        Args:
            similarity: Cosine similarity between provision and category centroid.

        Returns:
            Risk level string: 'LOW', 'MEDIUM', or 'HIGH'.
        """
        if similarity >= self.high_threshold:
            return "LOW"
        elif similarity >= self.medium_threshold:
            return "MEDIUM"
        else:
            return "HIGH"

    def assess(self, text: str, predicted_category: str) -> Dict:
        """Assess the risk level of a single contract provision.

        Embeds the provision and computes cosine similarity against the
        centroid embedding of its predicted category. Returns the risk
        level, similarity score, and a plain-language explanation.

        Args:
            text: Contract provision text.
            predicted_category: Predicted or known provision category name.

        Returns:
            Dictionary with keys:
                - similarity (float): cosine similarity score
                - risk_level (str): 'LOW', 'MEDIUM', or 'HIGH'
                - category (str): category assessed against
                - explanation (str): plain-language explanation
        """
        # Fallback to 'General' if category has no reference embedding
        category = (
            predicted_category
            if predicted_category in self.reference_embeddings
            else "General"
        )

        provision_embedding = self.model.encode(text, normalize_embeddings=True)
        reference_embedding = self.reference_embeddings[category]
        similarity = self._cosine_similarity(provision_embedding, reference_embedding)
        risk_level = self._get_risk_level(similarity)

        explanations = {
            "LOW": (
                f"This provision closely matches typical {category} language "
                f"found in SEC contracts (similarity: {similarity:.3f}). "
                f"No significant deviations detected."
            ),
            "MEDIUM": (
                f"This provision shows moderate deviation from typical {category} "
                f"language in SEC contracts (similarity: {similarity:.3f}). "
                f"Legal review is recommended."
            ),
            "HIGH": (
                f"This provision significantly deviates from typical {category} "
                f"language in SEC contracts (similarity: {similarity:.3f}). "
                f"Immediate legal review is strongly advised."
            ),
        }

        return {
            "similarity": round(similarity, 4),
            "risk_level": risk_level,
            "category": category,
            "explanation": explanations[risk_level],
        }

    def assess_batch(
        self,
        texts: List[str],
        predicted_categories: List[str],
    ) -> List[Dict]:
        """Assess risk for a batch of provisions.

        Args:
            texts: List of contract provision texts.
            predicted_categories: List of predicted category names.

        Returns:
            List of risk assessment dictionaries.
        """
        results = []
        for text, category in zip(texts, predicted_categories):
            result = self.assess(text, category)
            results.append(result)

        risk_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
        for r in results:
            risk_counts[r["risk_level"]] += 1

        logger.info(
            "Batch risk assessment — LOW: %d | MEDIUM: %d | HIGH: %d",
            risk_counts["LOW"],
            risk_counts["MEDIUM"],
            risk_counts["HIGH"],
        )
        return results

    def get_riskiest_provisions(
        self,
        texts: List[str],
        predicted_categories: List[str],
        n: int = 5,
    ) -> List[Dict]:
        """Return the n provisions with the lowest similarity (highest risk).

        Args:
            texts: List of contract provision texts.
            predicted_categories: List of predicted category names.
            n: Number of highest-risk provisions to return.

        Returns:
            List of n risk assessment dicts sorted by similarity ascending.
        """
        assessments = self.assess_batch(texts, predicted_categories)
        for i, assessment in enumerate(assessments):
            assessment["text"] = texts[i]
        return sorted(assessments, key=lambda x: x["similarity"])[:n]


if __name__ == "__main__":
    from data_loader import (
        get_sample,
        load_config,
        load_ledgar,
        load_reference_embeddings,
    )

    cfg = load_config()
    ds, labels = load_ledgar(cfg)
    ref_embeddings = load_reference_embeddings(
        cfg["data"]["reference_embeddings_path"], labels
    )

    detector = RiskDetector(cfg, ref_embeddings)
    samples = get_sample(ds, "test", 10)

    for sample in samples[:3]:
        category = labels[sample["label"]]
        result = detector.assess(sample["text"], category)
        logger.info(
            "Category: %s | Risk: %s | Similarity: %.4f",
            result["category"],
            result["risk_level"],
            result["similarity"],
        )