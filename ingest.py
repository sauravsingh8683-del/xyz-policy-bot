import os
import pdfplumber
import pickle
from rank_bm25 import BM25Okapi
import re


def extract_text_from_pdf(filepath):
    text = ""
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def simple_tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())

def chunk_text(filename, text):
    lines = text.strip().split("\n")
    chunks = []
    current_chunk_lines = []

    for line in lines:
        if line.strip() and line.strip()[0].isdigit() and ". " in line[:5]:
            if current_chunk_lines:
                chunk_text_value = "\n".join(current_chunk_lines).strip()
                if chunk_text_value:
                    chunks.append(f"[{filename}]\n{chunk_text_value}")
            current_chunk_lines = [line]
        else:
            current_chunk_lines.append(line)

    if current_chunk_lines:
        chunk_text_value = "\n".join(current_chunk_lines).strip()
        if chunk_text_value:
            chunks.append(f"[{filename}]\n{chunk_text_value}")

    return chunks


if __name__ == "__main__":
    all_chunks = []
    all_sources = []

    docs_folder = "Documents" if os.path.exists("Documents") and len([f for f in os.listdir("Documents") if f.endswith(".pdf")]) > 0 else "."
    for filename in os.listdir(docs_folder):
        if filename.endswith(".pdf"):
            filepath = os.path.join(docs_folder, filename)
            print(f"Reading: {filename}")
            text = extract_text_from_pdf(filepath)
            chunks = chunk_text(filename, text)
            for c in chunks:
                all_chunks.append(c)
                all_sources.append(filename)

    print(f"\nTotal chunks created: {len(all_chunks)}")
    tokenized_chunks = [simple_tokenize(c) for c in all_chunks]
    bm25 = BM25Okapi(tokenized_chunks)

    os.makedirs("index", exist_ok=True)
    with open("index/store.pkl", "wb") as f:
        pickle.dump({
            "chunks": all_chunks,
            "sources": all_sources,
            "bm25": bm25
        }, f)

    print("Index saved to index/store.pkl")
