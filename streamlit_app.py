import streamlit as st

from src.summarizer import ExtractiveSummarizer
from src.abstractive_summarizer import AbstractiveSummarizer

st.set_page_config(page_title="Text Summarizer", layout="centered")

st.title("🧠 Text Summarization App")
st.write("Supports **Extractive (TF-IDF)** and **Abstractive (Transformer)** summarization")

# -------- User Inputs -------- #
text = st.text_area("Enter text to summarize", height=250)

method = st.selectbox(
    "Choose summarization method",
    ["Extractive", "Abstractive"]
)

num_sentences = st.slider(
    "Number of sentences (Extractive only)",
    min_value=1,
    max_value=10,
    value=3
)

# -------- Action -------- #
if st.button("Generate Summary"):

    if not text.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Generating summary..."):

            if method == "Extractive":
                summarizer = ExtractiveSummarizer(top_n=num_sentences)
                summary = summarizer.generate_summary(text)

            else:
                summarizer = AbstractiveSummarizer()
                summary = summarizer.generate_summary(text)

        st.subheader("📄 Summary")
        st.success(summary)
