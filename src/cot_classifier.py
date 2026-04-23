"""
cot_classifier.py
=================
Chain-of-Thought (CoT) legal provision classification using
Llama-3.3-70B-Versatile via the Groq API.

Unlike zero-shot classification which asks for a direct answer, CoT
prompting instructs the model to reason step by step before committing
to a category. This tests whether structured reasoning improves over
direct zero-shot classification on a 100-class legal task.

Key improvement over M3 implementation:
  - Robust JSON parsing with multiple fallback strategies
  - Explicit JSON-only instruction in system prompt
  - Response cleaning before parsing (strips markdown fences)
  - Falls back to zero-shot style parsing if JSON fails entirely
  - Reasoning is captured and stored for qualitative analysis

Expected results: Micro-F1 ~0.50-0.58 (above zero-shot ~0.47,
below RAG which provides concrete examples rather than just reasoning)
"""

import json
import logging
import re
import time
from typing import Dict, List, Optional, Tuple

from groq import Groq

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# System prompt — instructs model to respond ONLY in JSON
COT_SYSTEM_PROMPT = """You are a legal contract expert. You must respond ONLY with a valid JSON object. Do not include any text before or after the JSON. Do not use markdown code fences."""

# CoT user prompt — asks model to reason before classifying
COT_PROMPT = """Classify the following contract provision into exactly one of the {n_labels} categories below.

Think step by step:
1. Identify the key legal concept in the provision
2. Consider which categories it could belong to
3. Eliminate unlikely categories
4. State your final answer

Categories:
{label_list}

Contract provision:
\"\"\"{text}\"\"\"

Respond with ONLY this JSON object, nothing else:
{{"reasoning": "<your step by step thinking in 2-3 sentences>", "label": "<exact category name from the list>", "confidence": "<high|medium|low>"}}"""


class CoTClassifier:
    """Chain-of-Thought legal provision classifier using Groq-hosted Llama 3.3-70B.

    Prompts the model to reason through the classification step by step
    before producing a final answer. Captures reasoning for qualitative
    analysis in the report.

    Key difference from ZeroShotClassifier: the model explicitly reasons
    before answering, which can improve accuracy on ambiguous provisions
    where the category is not immediately obvious from surface features.

    Attributes:
        config: Project configuration dictionary.
        label_names: List of 100 LEDGAR provision categories.
        label_set: Set of label names for fast membership checking.
        client: Groq API client.
        model: Groq model name.
    """

    def __init__(self, config: Dict, label_names: List[str], api_key: str) -> None:
        """Initialise the CoT classifier.

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
        logger.info("CoTClassifier initialised with model: %s", self.model)

    def _build_prompt(self, text: str) -> str:
        """Build the CoT classification prompt.

        Args:
            text: Contract provision text to classify.

        Returns:
            Formatted prompt string.
        """
        label_list = "\n".join(f"- {lbl}" for lbl in self.label_names)
        return COT_PROMPT.format(
            n_labels=len(self.label_names),
            label_list=label_list,
            text=text[:1500],
        )

    def _call_api(self, prompt: str) -> Optional[str]:
        """Call Groq API with system prompt and retry logic.

        Uses a system prompt that strictly instructs JSON-only output,
        which was the main cause of parsing failures in M3.

        Args:
            prompt: Formatted CoT classification prompt.

        Returns:
            Raw text response from the model, or None on failure.
        """
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": COT_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                logger.warning(
                    "API call failed (attempt %d/%d): %s",
                    attempt + 1,
                    self.max_retries,
                    e,
                )
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
        return None

    def _clean_response(self, response: str) -> str:
        """Clean raw model response before JSON parsing.

        Handles common issues:
        - Markdown code fences: ```json ... ```
        - Leading/trailing whitespace
        - Occasional preamble text before the JSON object

        Args:
            response: Raw model response string.

        Returns:
            Cleaned string ready for JSON parsing.
        """
        # Remove markdown code fences
        response = re.sub(r"```json\s*", "", response)
        response = re.sub(r"```\s*", "", response)
        response = response.strip()

        # If response contains text before the JSON object, extract just the JSON
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            return json_match.group(0)

        return response

    def _parse_label(self, label_str: str) -> str:
        """Map a label string from the model to a valid LEDGAR category.

        Args:
            label_str: Raw label string from model response.

        Returns:
            Matched label name, or 'General' as fallback.
        """
        if not label_str:
            return "General"

        label_str = label_str.strip()

        # Exact match
        if label_str in self.label_set:
            return label_str

        # Case-insensitive match
        label_lower = label_str.lower()
        for label in self.label_names:
            if label.lower() == label_lower:
                return label

        # Substring match
        for label in self.label_names:
            if label.lower() in label_lower:
                return label

        logger.debug("Could not match label '%s', defaulting to 'General'", label_str)
        return "General"

    def _parse_response(self, response: Optional[str]) -> Tuple[str, str, str]:
        """Parse CoT response into label, reasoning, and confidence.

        Uses multiple fallback strategies to handle malformed responses —
        the main improvement over the M3 implementation which crashed on
        any JSON parsing error.

        Strategy:
          1. Clean response (remove markdown fences, extract JSON)
          2. Parse as JSON and extract 'label', 'reasoning', 'confidence'
          3. If JSON fails, try to extract label using regex
          4. If regex fails, fall back to substring matching on full response

        Args:
            response: Raw model response string.

        Returns:
            Tuple of (predicted_label, reasoning, confidence).
        """
        if not response:
            return "General", "", "low"

        cleaned = self._clean_response(response)

        # Strategy 1: proper JSON parsing
        try:
            parsed = json.loads(cleaned)
            label = self._parse_label(parsed.get("label", ""))
            reasoning = parsed.get("reasoning", "")
            confidence = parsed.get("confidence", "low")
            return label, reasoning, confidence
        except json.JSONDecodeError:
            logger.debug("JSON parsing failed, trying regex fallback")

        # Strategy 2: regex extraction of label field
        label_match = re.search(r'"label"\s*:\s*"([^"]+)"', response)
        if label_match:
            label = self._parse_label(label_match.group(1))
            reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]+)"', response)
            reasoning = reasoning_match.group(1) if reasoning_match else ""
            return label, reasoning, "low"

        # Strategy 3: treat full response as plain text and substring match
        logger.debug("Regex fallback failed, using plain text matching")
        label = self._parse_label(response)
        return label, "", "low"

    def classify(self, text: str) -> Dict:
        """Classify a single provision using Chain-of-Thought prompting.

        Args:
            text: Contract provision text.

        Returns:
            Dictionary with 'prediction', 'reasoning', 'confidence',
            and 'raw_response'.
        """
        prompt = self._build_prompt(text)
        raw_response = self._call_api(prompt)
        label, reasoning, confidence = self._parse_response(raw_response)
        time.sleep(self.request_delay)

        return {
            "prediction": label,
            "reasoning": reasoning,
            "confidence": confidence,
            "raw_response": raw_response,
        }

    def classify_batch(self, samples: List[Dict]) -> Dict:
        """Classify a batch of samples and compute evaluation metrics.

        Args:
            samples: List of dicts with 'text' and 'label' keys.

        Returns:
            Dictionary with predictions, true labels, F1 scores,
            and sample reasoning examples for qualitative analysis.
        """
        from sklearn.metrics import f1_score

        predictions: List[str] = []
        true_labels: List[int] = []
        reasoning_examples: List[Dict] = []
        label2id = {lbl: i for i, lbl in enumerate(self.label_names)}

        logger.info("Starting CoT classification of %d samples...", len(samples))

        for i, sample in enumerate(samples):
            result = self.classify(sample["text"])
            predictions.append(result["prediction"])
            true_labels.append(sample["label"])

            # Store first 10 reasoning examples for qualitative analysis
            if len(reasoning_examples) < 10:
                reasoning_examples.append({
                    "text": sample["text"][:300],
                    "true_label": self.label_names[sample["label"]],
                    "predicted_label": result["prediction"],
                    "reasoning": result["reasoning"],
                    "confidence": result["confidence"],
                    "correct": result["prediction"] == self.label_names[sample["label"]],
                })

            if (i + 1) % 20 == 0:
                logger.info("Progress: %d/%d samples classified", i + 1, len(samples))

        pred_ids = [label2id.get(p, 0) for p in predictions]

        micro_f1 = f1_score(true_labels, pred_ids, average="micro", zero_division=0)
        macro_f1 = f1_score(true_labels, pred_ids, average="macro", zero_division=0)

        logger.info(
            "CoT Results — Micro-F1: %.4f | Macro-F1: %.4f", micro_f1, macro_f1
        )

        return {
            "predictions": predictions,
            "pred_ids": pred_ids,
            "true_labels": true_labels,
            "micro_f1": micro_f1,
            "macro_f1": macro_f1,
            "reasoning_examples": reasoning_examples,
        }


if __name__ == "__main__":
    import os

    from data_loader import get_sample, load_config, load_ledgar

    cfg = load_config()
    ds, labels = load_ledgar(cfg)
    samples = get_sample(ds, "test", cfg["evaluation"]["zeroshot_sample_size"])

    classifier = CoTClassifier(cfg, labels, api_key=os.environ["GROQ_API_KEY"])
    results = classifier.classify_batch(samples)

    logger.info(
        "Micro-F1: %.4f | Macro-F1: %.4f",
        results["micro_f1"],
        results["macro_f1"],
    )

    # Print sample reasoning examples
    logger.info("\n--- Sample CoT Reasoning Examples ---")
    for ex in results["reasoning_examples"][:3]:
        logger.info(
            "True: %s | Predicted: %s | Correct: %s\nReasoning: %s\n",
            ex["true_label"],
            ex["predicted_label"],
            ex["correct"],
            ex["reasoning"],
        )