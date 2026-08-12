import streamlit as st
from groq import Groq
from retriever import load_index, search

st.set_page_config(page_title="XYZ Policy Assistant")
st.title("📄 XYZ Corp - Policy Assistant")
st.write("Saare HR, WFH, Leave policies yahan pucho")

client = Groq(api_key=st.secrets["GROQ_API_KEY"])

@st.cache_resource
def get_data():
    return load_index()

data = get_data()

query = st.text_input("Apna sawaal likho (Ex: WFH kitne din milta hai?)")

if query:
    results = search(query, data, top_k=3)
    context = "\n\n".join([r[0] for r in results])
    prompt = f"Context: {context}\n\nSawaal: {query}\nJawaab policy ke basis par do."
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    st.write(response.choices[0].message.content)
