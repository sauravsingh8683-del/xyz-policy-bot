from fastapi import FastAPI
from pydantic import BaseModel
from retriever import load_index, search
from groq import Groq
import os

app = FastAPI()
data = load_index()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

class Q(BaseModel):
    question: str

@app.get("/")
def home():
    return {"status": "RAG API OK"}

@app.post("/ask")
def ask(q: Q):
    results = search(q.question, data, top_k=2)
    context = "\n".join([r[0] for r in results])
    prompt = f"Tu XYZ Corp ka HR assistant hai. Sirf is context se jawaab de, short me: {context}\nSawaal: {q.question}"
    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role":"user","content":prompt}]
    )
    return {"answer": res.choices[0].message.content}
