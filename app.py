
import json
from pathlib import Path

import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from google import genai

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "processed_text"

st.set_page_config(
    page_title="Sarkari Sathi",
    page_icon="📚",
    layout="wide"
)

@st.cache_resource
def load_project_data():
    with open(
        DATA_DIR / "sarkari_sathi_metadata.json",
        "r",
        encoding="utf-8"
    ) as f:
        metadata = json.load(f)

    embeddings = np.load(
        DATA_DIR / "sarkari_sathi_embeddings.npy"
    )

    model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    return metadata, embeddings, model


def get_search_terms(question):
    stop_words = {
        "के", "कति", "कस्तो", "कस्ता", "कसरी", "बारे",
        "मा", "को", "का", "कि", "र", "ले", "लाई",
        "the", "what", "how", "is", "are", "in", "of",
        "for", "and", "to", "was", "were"
    }

    return [
        word.strip(".,?!:;()[]{}\"'")
        for word in question.lower().split()
        if len(word.strip(".,?!:;()[]{}\"'")) > 1
        and word.strip(".,?!:;()[]{}\"'") not in stop_words
    ]


def expand_search_terms(terms):
    corrections = {
        "स्थानीय": ["थिानीय"],
        "तहमा": ["थिानीय िह", "थिानीय"],
        "परामर्श": ["परामशि", "परामिि"],
        "खरिद": ["खररद"],
        "मर्मत": ["ममिि", "ममित"],
        "प्रतिस्पर्धा": ["प्रतिथपधाि", "प्रतिस्पधाि"],
        "सोझै": ["सोझै खररद"],
    }

    expanded = list(terms)

    for term in terms:
        expanded.extend(corrections.get(term, []))

    return list(dict.fromkeys(expanded))


def get_relevant_evidence(
    question, metadata, embeddings, embedding_model, top_k=5
):
    terms = expand_search_terms(get_search_terms(question))

    query_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True
    )[0]

    similarities = np.dot(embeddings, query_embedding)
    results = []

    for i, item in enumerate(metadata):
        text = item.get("text", "")

        matched_terms = [
            term for term in set(terms)
            if len(term) > 1 and term in text
        ]

        score = float(similarities[i]) + 0.15 * len(matched_terms)

        if "सोझै खररद" in text or "सोझै खरिद" in text:
            score += 0.25

        if "प्रतिथपधाि नगराई" in text or "प्रतिस्पधाि नगराई" in text:
            score += 0.25

        results.append({
            "score": score,
            "page": item.get("page", "Unknown"),
            "text": text
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


st.title("📚 Sarkari Sathi")
st.subheader("AI Assistant for Nepal Government Reports")
st.write(
    "Ask questions about government reports in Nepali or English. "
    "Answers include supporting PDF page references."
)

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input(
        "Gemini API key",
        type="password",
        help="Your key is used for this session and is not saved by this app."
    )
    st.caption("Your API key is not written to the project files.")

try:
    metadata, embeddings, embedding_model = load_project_data()
    st.success("Government report search system loaded.")
except Exception as exc:
    st.error(f"Could not load project data: {exc}")
    st.stop()

question = st.text_area(
    "Enter your question",
    placeholder="उदाहरण: स्थानीय तहले सोझै खरिद गर्दा कति खर्च गरेका छन्?",
    height=100
)

if st.button("Ask Sarkari Sathi", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")
    elif not api_key.strip():
        st.warning("Enter your Gemini API key in the sidebar.")
    else:
        with st.spinner("Searching reports and preparing an answer..."):
            try:
                evidence = get_relevant_evidence(
                    question,
                    metadata,
                    embeddings,
                    embedding_model,
                    top_k=5
                )

                context = "\n\n".join(
                    f"PDF page {item['page']}:\n{item['text']}"
                    for item in evidence
                )

                prompt = f"""
You are Sarkari Sathi, an assistant for Nepal government reports.

Answer in the same language as the question.
Use only the evidence supplied below.
Do not invent facts, amounts, rules, or conclusions.
If the evidence does not answer the question, say that the
available evidence does not establish the answer.
Cite the relevant PDF page numbers.
Distinguish combined expenditure from expenditure on an individual category.

Question:
{question}

Evidence:
{context}
"""

                client = genai.Client(api_key=api_key)
                response = client.models.generate_content(
                    model="gemini-3.5-flash-lite",
                    contents=prompt
                )

                st.subheader("Answer")
                st.markdown(response.text or "No answer was returned.")

                with st.expander("View retrieved evidence"):
                    for item in evidence:
                        st.markdown(
                            f"**PDF page {item['page']}** "
                            f"(retrieval score: {item['score']:.3f})"
                        )
                        st.text(item["text"])
                        st.divider()

            except Exception as exc:
                st.error(f"Something went wrong: {exc}")
