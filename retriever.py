import os
import pickle
from ingest import build_index

def load_index():
    index_path = "index/store.pkl"

    # Agar index nahi hai toh pehle bana le
    if not os.path.exists(index_path):
        os.makedirs("index", exist_ok=True)
        print("Index nahi mila, naya bana raha hu...")
        build_index()

    with open(index_path, "rb") as f:
        return pickle.load(f)

def search_index(query, index_data, top_k=3):
    # tera purana search wala code yahan rahega
    # agar rank_bm25 use kar raha hai toh:
    from rank_bm25 import BM25Okapi
    bm25 = index_data['bm25']
    docs = index_data['docs']
    tokenized_query = query.split(" ")
    scores = bm25.get_scores(tokenized_query)
    top_n = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [docs[i] for i in top_n]
