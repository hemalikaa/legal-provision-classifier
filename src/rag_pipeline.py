"""
rag_pipeline.py
===============
Few-shot Retrieval-Augmented Generation pipeline for legal provision
classification. Uses ChromaDB to store training examples as embeddings
and retrieves the top-K most similar provisions as few-shot demonstrations
for Llama-3.3-70B via Groq.

This approach retrieves semantically similar examples at inference time,
giving the LLM concrete in-context demonstrations rather than relying
purely on zero-shot instruction following.
"""

import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chromadb
from chromadb.utils import embedding_functions
from groq import Groq

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Few-shot RAG prompt template
RAG_PROMPT = """You are a legal contract expert specialising in contract provision classification.

Below are {k} example contract provisions with their correct categories:

{examples}

Now classify the following contract provision into exactly one of these {n_labels} categories:
{label_list}

CONTRACT PROVISION TO CLASSIFY:
\"\"\"{text}\"\"\"

Instructions:
- Use the examples above as reference for how provisions map to categories.
- Select the single most appropriate category.
- Respond with ONLY the exact category name. No explanation, no punctuation.

Category:"""


class RAGClassifier:
    """Few-shot RAG classifier for legal provision classification.

    Embeds training provisions into ChromaDB, then at inference time
    retrieves the top-K most similar examples to use as few-shot
    demonstrations for Llama-3.3-70B via Groq.

    Attributes:
        config: Project configuration dictionary.
        label_names: List of 100 LEDGAR provision categories.
        client: ChromaDB persistent client.
        collection: ChromaDB collection storing training embeddings.
        groq_client: Groq API client.
    """

    def __init__(self, config: Dict, label_names: List[str], api_key: str) -> None:
        """Initialise the RAG classifier.

        Args:
            config: Project configuration dictionary.
            label_names: List of provision category names from LEDGAR.
            api_key: Groq API key.
        """
        self.config = config
        self.label_names = label_names
        self.label_set = set(label_names)
        self.label2id = {lbl: i for i, lbl in enumerate(label_names)}
        self.groq_client = Groq(api_key=api_key)
        self.model = config["groq"]["model"]
        self.temperature = config["groq"]["temperature"]
        self.max_tokens = config["groq"]["max_tokens"]
        self.request_delay = config["groq"]["request_delay_seconds"]
        self.max_retries = config["groq"]["max_retries"]
        self.retry_delay = config["groq"]["retry_delay_seconds"]
        self.top_k = config["rag"]["top_k"]

        # Sentence-transformer embedding function for ChromaDB
        self.embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=config["rag"]["embedding_model"]
        )

        # Initialise ChromaDB persistent client
        persist_dir = config["rag"]["chroma_persist_dir"]
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection_name = config["rag"]["collection_name"]
        logger.info("RAGClassifier initialised. ChromaDB path: %s", persist_dir)

    def build_index(self, train_samples: List[Dict], batch_size: int = 500) -> None:
        """Build ChromaDB index from training samples.

        Embeds each training provision and stores it with its label
        as metadata. Existing collections are deleted and rebuilt for
        a clean index.

        Args:
            train_samples: List of dicts with 'text' and 'label' keys.
            batch_size: Number of documents to add per ChromaDB call.
        """
        # Delete existing collection to rebuild cleanly
        try:
            self.client.delete_collection(self.collection_name)
            logger.info("Deleted existing collection '%s'", self.collection_name)
        except Exception:
            pass

        self.collection = self.client.create_collection(
            name=self.collection_name,
            embedding_function=self.embed_fn,
            metadata={"hnsw:space": "cosine"},
        )

        logger.info("Building ChromaDB index from %d training samples...", len(train_samples))

        for i in range(0, len(train_samples), batch_size):
            batch = train_samples[i : i + batch_size]
            self.collection.add(
                documents=[s["text"] for s in batch],
                metadatas=[{"label": self.label_names[s["label"]]} for s in batch],
                ids=[f"doc_{i + j}" for j, _ in enumerate(batch)],
            )
            logger.info(
                "Indexed batch %d/%d (%d docs)",
                i // batch_size + 1,
                (len(train_samples) + batch_size - 1) // batch_size,
                len(batch),
            )

        logger.info("ChromaDB index built with %d documents.", self.collection.count())

    def load_index(self) -> None:
        """Load existing ChromaDB collection (skip rebuild if already exists).

        Raises:
            ValueError: If the collection does not exist.
        """
        try:
            self.collection = self.client.get_collection(
                name=self.collection_name,
                embedding_function=self.embed_fn,
            )
            logger.info(
                "Loaded ChromaDB collection '%s' (%d docs)",
                self.collection_name,
                self.collection.count(),
            )
        except Exception as e:
            raise ValueError(
                f"Collection '{self.collection_name}' not found. "
                "Run build_index() first."
            ) from e

    def _retrieve_examples(self, text: str) -> List[Tuple[str, str]]:
        """Retrieve top-K similar provisions from ChromaDB.

        Args:
            text: Query provision text.

        Returns:
            List of (provision_text, label_name) tuples.
        """
        results = self.collection.query(
            query_texts=[text],
            n_results=self.top_k,
            include=["documents", "metadatas"],
        )
        examples = [
            (doc, meta["label"])
            for doc, meta in zip(
                results["documents"][0], results["metadatas"][0]
            )
        ]
        return examples

    def _build_prompt(self, text: str, examples: List[Tuple[str, str]]) -> str:
        """Build few-shot RAG prompt with retrieved examples.

        Args:
            text: Contract provision to classify.
            examples: List of (example_text, label) tuples from retrieval.

        Returns:
            Formatted prompt string.
        """
        example_str = "\n\n".join(
            f"Example {i+1}:\nProvision: \"{ex_text[:300]}...\"\nCategory: {ex_label}"
            for i, (ex_text, ex_label) in enumerate(examples)
        )
        label_list = "\n".join(f"- {lbl}" for lbl in self.label_names)
        return RAG_PROMPT.format(
            k=len(examples),
            examples=example_str,
            n_labels=len(self.label_names),
            label_list=label_list,
            text=text[:1500],
        )

    def _call_api(self, prompt: str) -> Optional[str]:
        """Call Groq API with retry logic.

        Args:
            prompt: Formatted classification prompt.

        Returns:
            Model response text, or None on failure.
        """
        for attempt in range(self.max_retries):
            try:
                response = self.groq_client.chat.completions.create(
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

        Args:
            response: Raw model response.

        Returns:
            Matched label name, or 'General' as fallback.
        """
        if not response:
            return "General"
        if response in self.label_set:
            return response
        response_lower = response.lower()
        for label in self.label_names:
            if label.lower() == response_lower:
                return label
        for label in self.label_names:
            if label.lower() in response_lower:
                return label
        return "General"

    def classify(self, text: str) -> Dict:
        """Classify a single provision using RAG few-shot prompting.

        Args:
            text: Contract provision text.

        Returns:
            Dictionary with 'prediction', 'retrieved_examples', and 'raw_response'.
        """
        examples = self._retrieve_examples(text)
        prompt = self._build_prompt(text, examples)
        raw_response = self._call_api(prompt)
        prediction = self._parse_response(raw_response)
        time.sleep(self.request_delay)

        return {
            "prediction": prediction,
            "retrieved_examples": examples,
            "raw_response": raw_response,
        }

    def classify_batch(self, samples: List[Dict]) -> Dict:
        """Classify a batch of samples and compute evaluation metrics.

        Args:
            samples: List of dicts with 'text' and 'label' keys.

        Returns:
            Dictionary with predictions, true labels, and F1 scores.
        """
        from sklearn.metrics import f1_score

        predictions: List[str] = []
        true_labels: List[int] = []

        logger.info("Starting RAG classification of %d samples...", len(samples))

        for i, sample in enumerate(samples):
            result = self.classify(sample["text"])
            predictions.append(result["prediction"])
            true_labels.append(sample["label"])

            if (i + 1) % 20 == 0:
                logger.info("Progress: %d/%d samples classified", i + 1, len(samples))

        pred_ids = [self.label2id.get(p, 0) for p in predictions]

        micro_f1 = f1_score(true_labels, pred_ids, average="micro", zero_division=0)
        macro_f1 = f1_score(true_labels, pred_ids, average="macro", zero_division=0)

        logger.info(
            "RAG Results — Micro-F1: %.4f | Macro-F1: %.4f", micro_f1, macro_f1
        )

        return {
            "predictions": predictions,
            "pred_ids": pred_ids,
            "true_labels": true_labels,
            "micro_f1": micro_f1,
            "macro_f1": macro_f1,
        }


if __name__ == "__main__":
    from data_loader import get_sample, load_config, load_ledgar

    cfg = load_config()
    ds, labels = load_ledgar(cfg)

    rag = RAGClassifier(cfg, labels, api_key=os.environ["GROQ_API_KEY"])

    # Build index from a subset of training data (use full set for best performance)
    train_samples = get_sample(ds, "train", 5000, seed=cfg["project"]["seed"])
    rag.build_index(train_samples)

    # Evaluate on test samples
    test_samples = get_sample(ds, "test", cfg["rag"]["eval_sample_size"])
    results = rag.classify_batch(test_samples)
    logger.info("Micro-F1: %.4f | Macro-F1: %.4f", results["micro_f1"], results["macro_f1"])