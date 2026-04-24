"""
streamlit_app.py
================
Legal Provision Classification & Contract Risk Analyzer
Interactive demo comparing all 4 classification approaches side by side.
"""

import os
import sys
import time
from pathlib import Path

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Legal Provision Analyzer",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .main-title {
        font-family: 'Playfair Display', serif;
        font-size: 2.4rem;
        font-weight: 700;
        color: #f1f5f9;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-family: 'DM Sans', sans-serif;
        color: #94a3b8;
        font-size: 1rem;
        margin-bottom: 2rem;
    }

    .section-header {
        font-family: 'Playfair Display', serif;
        font-size: 1.2rem;
        font-weight: 700;
        color: #f1f5f9;
        border-bottom: 2px solid #334155;
        padding-bottom: 0.4rem;
        margin-bottom: 1rem;
    }

    .risk-low {
        background: linear-gradient(135deg, #064e3b, #065f46);
        border-left: 5px solid #059669;
        padding: 14px 18px;
        border-radius: 8px;
        margin: 8px 0;
        color: #d1fae5 !important;
    }

    .risk-medium {
        background: linear-gradient(135deg, #78350f, #92400e);
        border-left: 5px solid #d97706;
        padding: 14px 18px;
        border-radius: 8px;
        margin: 8px 0;
        color: #fef3c7 !important;
    }

    .risk-high {
        background: linear-gradient(135deg, #7f1d1d, #991b1b);
        border-left: 5px solid #dc2626;
        padding: 14px 18px;
        border-radius: 8px;
        margin: 8px 0;
        color: #fee2e2 !important;
    }

    .approach-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px;
        margin: 6px 0;
        font-family: 'DM Mono', monospace;
        font-size: 0.85rem;
        color: #f1f5f9 !important;
    }

    .approach-card.agree {
        border-left: 4px solid #059669;
    }

    .approach-card.disagree {
        border-left: 4px solid #dc2626;
    }

    .consensus-box {
        background: #1e3a5f;
        border: 1px solid #2563eb;
        border-radius: 10px;
        padding: 14px 18px;
        margin: 12px 0;
        font-size: 0.95rem;
        color: #f1f5f9 !important;
    }

    .retrieved-example {
        background: #1e293b;
        border-radius: 8px;
        padding: 10px 14px;
        margin: 6px 0;
        font-size: 0.82rem;
        font-family: 'DM Mono', monospace;
        border-left: 3px solid #475569;
        color: #f1f5f9 !important;
    }

    .stTextArea textarea {
        font-family: 'DM Mono', monospace;
        font-size: 0.9rem;
        border-radius: 8px;
    }

    .stButton > button {
        background: linear-gradient(135deg, #2563eb, #1d4ed8);
        color: white;
        font-family: 'DM Sans', sans-serif;
        font-weight: 600;
        font-size: 1rem;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        width: 100%;
        cursor: pointer;
        transition: all 0.2s;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #1d4ed8, #1e40af);
        transform: translateY(-1px);
    }

    .metric-label {
        font-size: 0.75rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }

    .metric-value {
        font-family: 'Playfair Display', serif;
        font-size: 1.6rem;
        font-weight: 700;
        color: #f1f5f9;
    }
</style>
""", unsafe_allow_html=True)


# ─── Sample Provisions ────────────────────────────────────────────────────────
SAMPLE_PROVISIONS = {
    "Governing Laws": "This Agreement shall be governed by and construed in accordance with the laws of the State of Delaware, without giving effect to any choice of law or conflict of law provisions thereof.",
    "Confidentiality": "Each party agrees to keep confidential all non-public information disclosed by the other party in connection with this Agreement and to use such information solely for the purposes contemplated herein.",
    "Indemnification (Unusual)": "The receiving party shall indemnify and hold harmless the disclosing party from any and all claims arising from the receiving party's use of the confidential information, including any consequential, indirect, or punitive damages, without limitation.",
    "Entire Agreement": "This Agreement constitutes the entire agreement between the parties with respect to the subject matter hereof and supersedes all prior agreements, representations and understandings of the parties.",
    "Severability": "If any provision of this Agreement is held to be invalid or unenforceable, the remaining provisions shall continue in full force and effect and shall be interpreted to give maximum effect to the intent of the parties.",
    "Ambiguous (Test Disagreement)": "Each party represents and warrants that it has the full right, power and authority to enter into this Agreement and that the execution and performance of this Agreement will not conflict with or violate any law, regulation or agreement to which it is a party.",
}


# ─── Load Models ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading classification models...")
def load_models():
    from data_loader import load_config, load_ledgar, load_reference_embeddings
    from tfidf_baseline import load_tfidf_model
    from risk_detector import RiskDetector

    cfg = load_config("config/config.yaml")
    _, label_names = load_ledgar(cfg)

    # TF-IDF
    try:
        vectorizer, clf = load_tfidf_model()
        tfidf_loaded = True
    except Exception:
        vectorizer, clf = None, None
        tfidf_loaded = False

    # Risk detector
    try:
        ref_embeddings = load_reference_embeddings(
            cfg["data"]["reference_embeddings_path"], label_names
        )
        detector = RiskDetector(cfg, ref_embeddings)
        risk_loaded = True
    except Exception:
        detector = None
        risk_loaded = False

    return cfg, label_names, vectorizer, clf, detector, tfidf_loaded, risk_loaded


@st.cache_resource(show_spinner="Loading RAG index...")
def load_rag(_cfg, _label_names):
    from rag_pipeline import RAGClassifier
    groq_key = os.environ.get("GROQ_API_KEY", "")
    if not groq_key:
        return None
    try:
        rag = RAGClassifier(_cfg, _label_names, api_key=groq_key)
        rag.load_index()
        return rag
    except Exception:
        return None


# ─── Classification Functions ─────────────────────────────────────────────────
def classify_tfidf(text, vectorizer, clf, label_names):
    if vectorizer is None:
        return None, None, None, None
    try:
        start = time.time()
        X = vectorizer.transform([text])
        pred_id = clf.predict(X)[0]
        proba = clf.predict_proba(X)[0]
        elapsed = time.time() - start
        top5_idx = proba.argsort()[-5:][::-1]
        top5 = [(label_names[i], round(float(proba[i]) * 100, 1)) for i in top5_idx]
        return label_names[pred_id], round(float(proba[pred_id]) * 100, 1), elapsed, top5
    except Exception:
        return None, None, None, None


def classify_zeroshot(text, cfg, label_names, api_key):
    try:
        from zeroshot_classifier import ZeroShotClassifier
        start = time.time()
        clf = ZeroShotClassifier(cfg, label_names, api_key=api_key)
        result = clf.classify(text)
        elapsed = time.time() - start
        return result, elapsed
    except Exception:
        return None, None


def classify_cot(text, cfg, label_names, api_key):
    try:
        from cot_classifier import CoTClassifier
        start = time.time()
        clf = CoTClassifier(cfg, label_names, api_key=api_key)
        result = clf.classify(text)
        elapsed = time.time() - start
        return result, elapsed
    except Exception:
        return None, None


def classify_rag(text, rag_clf):
    try:
        start = time.time()
        result = rag_clf.classify(text)
        elapsed = time.time() - start
        return result, elapsed
    except Exception:
        return None, None


def get_consensus(predictions):
    valid = [p for p in predictions if p is not None]
    if not valid:
        return "No valid predictions", False
    from collections import Counter
    counts = Counter(valid)
    most_common, count = counts.most_common(1)[0]
    if count == len(valid):
        return f"✅ All {len(valid)} approaches agree — **{most_common}** — high confidence classification.", True
    elif count >= len(valid) * 0.75:
        return f"🟡 Majority ({count}/{len(valid)}) predict **{most_common}**. Some disagreement detected — this may be an ambiguous provision.", True
    else:
        return f"🔴 Approaches disagree significantly. Predictions: {dict(counts)}. This provision may span multiple categories.", False


# ─── Main App ─────────────────────────────────────────────────────────────────
def main():
    st.markdown('<div class="main-title">⚖️ Legal Provision Analyzer</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Compare 4 AI approaches to classify contract provisions and assess deviation risk</div>', unsafe_allow_html=True)

    cfg, label_names, vectorizer, clf, detector, tfidf_loaded, risk_loaded = load_models()

    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        groq_key = st.text_input(
            "Groq API Key",
            value=os.environ.get("GROQ_API_KEY", ""),
            type="password",
            help="Required for LLM-based approaches"
        )

        st.markdown("---")
        st.markdown("### 🔬 Approaches")
        use_tfidf = st.checkbox("TF-IDF + Logistic Regression", value=True, disabled=not tfidf_loaded)
        use_zeroshot = st.checkbox("Zero-shot Llama-3.3-70B", value=bool(groq_key))
        use_cot = st.checkbox("Chain-of-Thought Llama-3.3-70B", value=bool(groq_key))
        use_rag = st.checkbox("RAG Few-shot Llama-3.3-70B", value=False)

        st.markdown("---")
        st.markdown("### 📊 Model Performance")
        perf_data = {
            "Approach": ["TF-IDF", "Zero-shot", "CoT", "RAG", "BERT"],
            "Micro-F1": [0.8726, 0.5150, 0.6500, 0.6800, 0.8715],
        }
        st.dataframe(pd.DataFrame(perf_data), hide_index=True, use_container_width=True)

        st.markdown("---")
        st.markdown("### 📝 Sample Provisions")
        sample_choice = st.selectbox("Load a sample", ["— Select —"] + list(SAMPLE_PROVISIONS.keys()))

    # Load RAG
    rag_clf = None
    if use_rag and groq_key:
        with st.spinner("Loading RAG index..."):
            rag_clf = load_rag(cfg, tuple(label_names))

    # Input
    st.markdown('<div class="section-header">Contract Provision Input</div>', unsafe_allow_html=True)

    default_text = ""
    if sample_choice != "— Select —":
        default_text = SAMPLE_PROVISIONS[sample_choice]

    provision_text = st.text_area(
        "Paste a contract provision below:",
        value=default_text,
        height=150,
        placeholder="e.g. This Agreement shall be governed by the laws of the State of Delaware...",
        label_visibility="collapsed",
    )

    analyze_clicked = st.button("⚡ Analyze Provision", use_container_width=True)

    # Results
    if analyze_clicked and provision_text.strip():
        st.markdown("---")
        st.markdown('<div class="section-header">Classification Results</div>', unsafe_allow_html=True)

        results = {}
        top5_data = None
        rag_examples = None

        with st.spinner("Running classification approaches..."):

            if use_tfidf and tfidf_loaded:
                pred, conf, elapsed, top5 = classify_tfidf(provision_text, vectorizer, clf, label_names)
                results["TF-IDF + LR"] = {
                    "prediction": pred,
                    "confidence": f"{conf}%",
                    "time": f"{elapsed:.3f}s",
                    "extra": None
                }
                top5_data = top5

            if use_zeroshot and groq_key:
                pred, elapsed = classify_zeroshot(provision_text, cfg, label_names, groq_key)
                results["Zero-shot Llama"] = {
                    "prediction": pred,
                    "confidence": "—",
                    "time": f"{elapsed:.1f}s" if elapsed else "—",
                    "extra": None
                }

            if use_cot and groq_key:
                result, elapsed = classify_cot(provision_text, cfg, label_names, groq_key)
                if result:
                    results["CoT Llama"] = {
                        "prediction": result["prediction"],
                        "confidence": result.get("confidence", "—").title(),
                        "time": f"{elapsed:.1f}s" if elapsed else "—",
                        "extra": result.get("reasoning", "")
                    }

            if use_rag and rag_clf:
                result, elapsed = classify_rag(provision_text, rag_clf)
                if result:
                    results["RAG Llama"] = {
                        "prediction": result["prediction"],
                        "confidence": "—",
                        "time": f"{elapsed:.1f}s" if elapsed else "—",
                        "extra": None
                    }
                    rag_examples = result.get("retrieved_examples", [])

        # Consensus
        predictions = [v["prediction"] for v in results.values() if v["prediction"]]
        majority_prediction = max(set(predictions), key=predictions.count) if predictions else None
        consensus_msg, agree = get_consensus(predictions)

        st.markdown(f'<div class="consensus-box">{consensus_msg}</div>', unsafe_allow_html=True)

        col1, col2 = st.columns([3, 2])

        with col1:
            st.markdown("**Approach Comparison**")
            for approach, data in results.items():
                pred = data["prediction"] or "Failed"
                is_majority = pred == majority_prediction
                card_class = "agree" if is_majority else "disagree"
                icon = "✅" if is_majority else "❌"
                st.markdown(
                    f'<div class="approach-card {card_class}">'
                    f'<strong>{icon} {approach}</strong><br>'
                    f'Prediction: <strong>{pred}</strong> &nbsp;|&nbsp; '
                    f'Confidence: {data["confidence"]} &nbsp;|&nbsp; '
                    f'Time: {data["time"]}'
                    f'</div>',
                    unsafe_allow_html=True
                )

            cot_result = results.get("CoT Llama", {})
            if cot_result.get("extra"):
                with st.expander("💭 CoT Reasoning"):
                    st.write(cot_result["extra"])

        with col2:
            if top5_data:
                st.markdown("**TF-IDF Top 5 Category Confidence**")
                labels = [x[0] for x in top5_data]
                scores = [x[1] for x in top5_data]
                colors = ["#2563eb" if i == 0 else "#475569" for i in range(len(labels))]

                fig, ax = plt.subplots(figsize=(5, 3))
                fig.patch.set_facecolor('#0f172a')
                ax.set_facecolor('#1e293b')
                bars = ax.barh(labels[::-1], scores[::-1], color=colors[::-1], height=0.5)
                ax.set_xlabel("Confidence (%)", fontsize=9, color='#94a3b8')
                ax.set_xlim(0, max(scores) * 1.25)
                for bar, score in zip(bars, scores[::-1]):
                    ax.text(
                        bar.get_width() + 0.5,
                        bar.get_y() + bar.get_height() / 2,
                        f'{score}%', va='center', fontsize=8, color='#f1f5f9'
                    )
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_color('#334155')
                ax.spines['left'].set_color('#334155')
                ax.tick_params(labelsize=8, colors='#94a3b8')
                ax.xaxis.label.set_color('#94a3b8')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

        # Risk Assessment
        st.markdown("---")
        st.markdown('<div class="section-header">Risk Assessment</div>', unsafe_allow_html=True)

        if risk_loaded and majority_prediction and detector:
            risk_result = detector.assess(provision_text, majority_prediction)
            risk_level = risk_result["risk_level"]
            similarity = risk_result["similarity"]
            explanation = risk_result["explanation"]

            risk_class = {"LOW": "risk-low", "MEDIUM": "risk-medium", "HIGH": "risk-high"}[risk_level]
            risk_icon = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}[risk_level]

            rcol1, rcol2, rcol3 = st.columns(3)
            with rcol1:
                st.markdown('<div class="metric-label">Risk Level</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{risk_icon} {risk_level}</div>', unsafe_allow_html=True)
            with rcol2:
                st.markdown('<div class="metric-label">Similarity Score</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{similarity:.3f}</div>', unsafe_allow_html=True)
            with rcol3:
                st.markdown('<div class="metric-label">Category Assessed</div>', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="metric-value" style="font-size:1rem; margin-top:0.4rem">{majority_prediction}</div>',
                    unsafe_allow_html=True
                )

            st.markdown(f'<div class="{risk_class}">{explanation}</div>', unsafe_allow_html=True)
            st.progress(similarity, text=f"Similarity to standard {majority_prediction} language")

        # RAG Examples
        if rag_examples:
            st.markdown("---")
            st.markdown('<div class="section-header">RAG Retrieved Examples</div>', unsafe_allow_html=True)
            st.caption("These are the 5 most similar LEDGAR training provisions used as few-shot examples for RAG classification.")
            for i, (ex_text, ex_label) in enumerate(rag_examples):
                st.markdown(
                    f'<div class="retrieved-example">'
                    f'<strong>Example {i+1} → {ex_label}</strong><br>'
                    f'{ex_text[:250]}{"..." if len(ex_text) > 250 else ""}'
                    f'</div>',
                    unsafe_allow_html=True
                )

    elif analyze_clicked and not provision_text.strip():
        st.warning("Please enter a contract provision to analyze.")

    # Footer
    st.markdown("---")
    st.caption("Legal Provision Classification & Contract Risk Analyzer | IE 7374 Generative AI | Northeastern University Vancouver | Spring 2026")
    st.caption("Dataset: LEDGAR via LexGLUE (Chalkidis et al., 2022) | Model: Llama-3.3-70B via Groq | Embeddings: sentence-transformers/all-MiniLM-L6-v2")


if __name__ == "__main__":
    main()