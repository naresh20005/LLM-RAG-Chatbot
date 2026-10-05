import streamlit as st

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# Page configuration
st.set_page_config(
    page_title="College RAG Chatbot",
    page_icon="🎓"
)

st.title("🎓 College RAG Chatbot")
st.write("Ask questions about the college information.")


# Load embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Load FAISS vector database
vectorstore = FAISS.load_local(
    "vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)


# User question
question = st.text_input("Ask your question:")


if question:

    # Search similar documents
    results = vectorstore.similarity_search(
        question,
        k=3
    )

    st.subheader("Retrieved Information")

    for result in results:
        st.write(result.page_content)