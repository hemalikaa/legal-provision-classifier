"""
zeroshot_classifier.py
======================
Zero-shot legal provision classification using Llama-3.3-70B-Versatile
via the Groq API. No fine-tuning — tests raw LLM capability on the task.

Baseline result: Micro-F1 ~0.47 on 200 test samples.
"""

import logging
import time
from typing import Dict, List, Optional

from groq import Groq

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Prompt template for zero-shot classification
ZEROSHOT_PROMPT = """You are a legal contract expert. Your task is to classify a contract provision into exactly one of the following {n_labels} categories.

Categories:
{label_list}

CONTRACT PROVISION:
\"\"\"{text}\"\"\"

Instructions:
- Read the provision carefully.
- Select the single most appropriate category from the list above.
- Respond with ONLY the exact category name, nothing else. No explanation, no punctuation.

Category:"""


class ZeroShotClassifier:
    """Zero-shot legal provision classifier using Groq-hosted Llama 3.3-70B.

    Attributes:
        client: Groq API client.
        model: Name of the Groq-hosted model to use.
        config: Project configuration dictionary.
        label_names: List of 100 LEDGAR provision categories.
        label_set: Set of label names for fast membership checking.
    """

    def __init__(self, config: Dict, label_names: List[str], api_key: str) -> None:
        """Initialise the zero-shot classifier.

        Args:
            config: Project configuration dictionary.
            label_names: List of provision category names from LEDGAR.
            api_key: Groq API key.
        """
        self.config = config
        self.label_names = label_names
        self.label_set = set(label_names)
        self.client = Groq(api_key=api_key)
        self.model = config["groq"]["model"]
        self.temperature = config["groq"]["temperature"]
        self.max_tokens = config["groq"]["max_tokens"]
        self.request_delay = config["groq"]["request_delay_seconds"]
        self.max_retries = config["groq"]["max_retries"]
        self.retry_delay = config["groq"]["retry_delay_seconds"]
        logger.info("ZeroShotClassifier initialised with model: %s", self.model)

    def _build_prompt(self, text: str) -> str:
        """Build the zero-shot classification prompt.

        Args:
            text: Contract provision text to classify.

        Returns:
            Formatted prompt string.
        """
        label_list = "\n".join(f"- {lbl}" for lbl in self.label_names)
        return ZEROSHOT_PROMPT.format(
            n_labels=len(self.label_names),
            label_list=label_list,
            text=text[:1500],  # truncate very long provisions
        )

    def _call_api(self, prompt: str) -> Optional[str]:
        """Call the Groq API with retry logic on rate limit errors.

        Args:
            prompt: The formatted classification prompt.

        Returns:
            Raw text response from the model, or None on failure.
        """
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                logger.warning(
                    "API call failed (attempt %d/%d): %s", attempt + 1, self.max_retries, e
                )
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
        return None

    def _parse_response(self, response: Optional[str]) -> str:
        """Parse model response and map to a valid label.

        Handles cases where the model returns extra text by checking if
        any known label is contained in the response.

        Args:
            response: Raw model response string.

        Returns:
            Matched label name, or 'General' as fallback.
        """
        if not response:
            return "General"

        # Exact match first
        if response in self.label_set:
            return response

        # Case-insensitive match
        response_lower = response.lower()
        for label in self.label_names:
            if label.lower() == response_lower:
                return label

        # Substring match (model may include extra words)
        for label in self.label_names:
            if label.lower() in response_lower:
                return label

        logger.debug("Could not parse response '%s', defaulting to 'General'", response)
        return "General"

    def classify(self, text: str) -> str:
        """Classify a single contract provision zero-shot.

        Args:
            text: Contract provision text.

        Returns:
            Predicted provision category name.
        """
        prompt = self._build_prompt(text)
        response = self._call_api(prompt)
        prediction = self._parse_response(response)
        time.sleep(self.request_delay)
        return prediction

    def classify_batch(
        self,
        samples: List[Dict],
        label_names: List[str],
    ) -> Dict:
        """Classify a batch of samples and return predictions + metrics.

        Args:
            samples: List of dicts with 'text' and 'label' keys.
            label_names: Full list of label names for metric computation.

        Returns:
            Dictionary with predictions, true labels, and F1 scores.
        """
        from sklearn.metrics import f1_score

        predictions: List[str] = []
        true_labels: List[int] = []
        label2id = {lbl: i for i, lbl in enumerate(label_names)}

        logger.info("Starting zero-shot classification of %d samples...", len(samples))

        for i, sample in enumerate(samples):
            pred_label = self.classify(sample["text"])
            predictions.append(pred_label)
            true_labels.append(sample["label"])

            if (i + 1) % 20 == 0:
                logger.info("Progress: %d/%d samples classified", i + 1, len(samples))

        # Convert string predictions to integer IDs
        pred_ids = [label2id.get(p, 0) for p in predictions]

        micro_f1 = f1_score(true_labels, pred_ids, average="micro", zero_division=0)
        macro_f1 = f1_score(true_labels, pred_ids, average="macro", zero_division=0)

        logger.info(
            "Zero-shot Results — Micro-F1: %.4f | Macro-F1: %.4f", micro_f1, macro_f1
        )

        return {
            "predictions": predictions,
            "pred_ids": pred_ids,
            "true_labels": true_labels,
            "micro_f1": micro_f1,
            "macro_f1": macro_f1,
        }


if __name__ == "__main__":
    import os

    from data_loader import get_sample, load_config, load_ledgar

    cfg = load_config()
    ds, labels = load_ledgar(cfg)
    samples = get_sample(ds, "test", cfg["evaluation"]["zeroshot_sample_size"])

    classifier = ZeroShotClassifier(cfg, labels, api_key=os.environ["GROQ_API_KEY"])
    results = classifier.classify_batch(samples, labels)
    logger.info("Micro-F1: %.4f | Macro-F1: %.4f", results["micro_f1"], results["macro_f1"])