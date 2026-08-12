import streamlit as st
import os
from groq import Groq
from retriever import load_index, search

st.set_page_config(page_title="XYZ Policy Assistant")
st.title("📄 XYZ Corp - Policy Assistant")
st.write("Saare HR, WFH, Leave policies yahan pucho")

# Groq client
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# Index load (agar nahi hai toh retriever khud bana lega)
@st.cache_resource
def get_data():
    return load_index()

data = get_data()

query = st.text_input("Apna sawaal likho (Ex: WFH kitne din milta hai?)")

if query:
    results = search(query, data, top_k=3)

    context = "\n\n".join([r[0] for r in results])

    prompt = f"""
    Tum XYZ Corp ke policy assistant ho. Neeche diye gaye policy documents ke basis par hi jawaab do.
    Agar jawaab document me nahi hai to bolo 'Is baare me policy me jaankari nahi hai'.

    Context:
    {context}

    Sawaal: {query}
    Jawaab Hindi/English mix me simple bhasha me do.
    """

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    st.write(response.choices[0].message.content)

    with st.expander("Source dekho"):
        for chunk, src in results:
            st.write(f"**{src}**")
            st.write(chunk[:500])
