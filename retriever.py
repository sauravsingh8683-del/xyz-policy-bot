import os
import pickle

def load_index():
    path = "index/store.pkl"
    if not os.path.exists(path):
        from ingest import build_index
        return build_index()
    with open(path, "rb") as f:
        return pickle.load(f)

def search(query, data, top_k=3):
    from ingest import simple_tokenize
    tokenized_query = simple_tokenize(query)
    scores = data["bm25"].get_scores(tokenized_query)
    top_n = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [(data["chunks"][i], data["sources"][i]) for i in top_n]
