import streamlit as st
from groq import Groq
from retriever import load_index, search
import os
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

st.set_page_config(page_title="XYZ Corp Policy Bot", page_icon="📄")
st.title("📄 XYZ Corp - Policy Assistant")
st.caption("Saare HR, WFH, Leave policies yahan pucho")

query = st.text_input("Apna sawaal likho (Ex: WFH kitne din milta hai?)")

if query:
    data = load_index()
    results = search(query, data, top_k=3)

    if results:
        context = "\n\n".join(f"Source: {r['source']}\n{r['chunk']}" for r in results)

        # LLM se jawab
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "Tum XYZ Corp ke policy assistant ho. Context ka use karke short aur clear jawab do. Source ka naam bhi batao."},
                {"role": "user", "content": f"Context:\n{context}\n\nSawaal: {query}"}
            ]
        )
        st.success(completion.choices[0].message.content)

        st.divider()
        st.subheader("Top Sources:")
        for r in results:
            st.code(f"{r['source']} | Score: {r['score']:.2f}\n{r['chunk'][:400]}...")
    else:
        st.warning("Isse related koi policy nahi mili.")
