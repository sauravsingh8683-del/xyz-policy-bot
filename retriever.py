import pickle
from ingest import simple_tokenize

def load_index():
    with open("index/store.pkl", "rb") as f:
        data = pickle.load(f)
    return data

def search(query, data, top_k=3):
    bm25 = data["bm25"]
    chunks = data["chunks"]
    sources = data["sources"]

    tokenized_query = simple_tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    results = []
    for i in ranked_indices:
        results.append({
            "chunk": chunks[i],
            "source": sources[i],
            "score": scores[i]
        })
    return results
if __name__ == "__main__":
    data = load_index()
    print("Policy Assistant Ready! ('exit' likh ke band kar sakte ho)\n")

    while True:
        query = input("Apna sawaal poocho: ")
        if query.lower() in ("exit", "quit"):
            break

        results = search(query, data)

        if not results:
            print("Koi relevant policy nahi mili is sawaal ke liye.\n")
            continue

        for r in results:
            print("\n----")
            print(f"Source: {r['source']} (score: {r['score']:.2f})")
            print(r['chunk'])
        print()
