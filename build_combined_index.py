
import json
from pathlib import Path

import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "processed_text"

PDF_SOURCES = [
    {
        "file": "6Ph-महालेखापरीक्षकको वार्षिक प्रतिवेदन, २०८१.pdf",
        "source": "OAG Annual Report 2081",
    },
    {
        "file": "20250629144539_CFS 2023-2024_fmeywfy.pdf",
        "source": "FCGO Consolidated Financial Statement 2023/24",
    },
]

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def split_text(text):
    text = " ".join(text.split())
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - CHUNK_OVERLAP

    return chunks


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    metadata = []

    for pdf_info in PDF_SOURCES:
        pdf_path = BASE_DIR / pdf_info["file"]

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        print(f"\nProcessing: {pdf_info['source']}")
        reader = PdfReader(str(pdf_path))
        print("PDF pages:", len(reader.pages))

        empty_pages = 0

        for page_index, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""

            if not page_text.strip():
                empty_pages += 1
                continue

            for chunk_index, chunk in enumerate(split_text(page_text), start=1):
                metadata.append({
                    "chunk_id": (
                        f"{pdf_info['source'].replace(' ', '_')}"
                        f"_page_{page_index + 1}_chunk_{chunk_index}"
                    ),
                    "page": page_index + 1,
                    "source": pdf_info["source"],
                    "source_file": pdf_info["file"],
                    "text": chunk,
                })

        print("Pages without extractable text:", empty_pages)

    if not metadata:
        raise RuntimeError("No text was extracted from the PDFs.")

    print("\nTotal chunks:", len(metadata))

    model = SentenceTransformer(MODEL_NAME)
    texts = [item["text"] for item in metadata]

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    with open(DATA_DIR / "sarkari_sathi_metadata_combined.json",
              "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    with open(DATA_DIR / "sarkari_sathi_chunks_combined.jsonl",
              "w", encoding="utf-8") as f:
        for item in metadata:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    np.save(DATA_DIR / "sarkari_sathi_embeddings_combined.npy", embeddings)

    print("\nCombined index created.")
    print("Metadata records:", len(metadata))
    print("Embedding shape:", embeddings.shape)
    print("Original index files were not overwritten.")


if __name__ == "__main__":
    main()
