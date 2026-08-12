import os
from groq import Groq
from retriever import load_index, search
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = "Tum XYZ Corp ke internal policy assistant ho. Neeche diye gaye policy excerpts ka use karke hi jawab do. Agar jawab context me nahi hai to bol do information nahi hai. Hamesha source batao."

def ask(query):
    data = load_index()
    results = search(query, data, top_k=4)
    context = "\n\n".join(f"Source: {r['source']}\n{r['chunk']}" for r in results)

    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nSawaal: {query}"}
        ]
    )
    return completion.choices[0].message.content

if __name__ == "__main__":
    print("Policy Assistant Ready (Groq Free)!")
    while True:
        q = input("\nTu puch (exit likh ke band kar): ")
        if q.lower() in ["exit", "quit"]: break
        print("\nBot:", ask(q))