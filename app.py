import streamlit as st
from rag_pipeline import rag_chain, retriever

st.set_page_config(page_title="🎬 Movie RAG Chatbot", layout="wide")

st.title("🎬 Movie Script RAG — Chat Assistant")
st.write("Ask questions about characters, scenes, genres, movies, and dialogue tone.")

query = st.text_input("Ask a question about any movie scene or character:")

if st.button("Analyze"):
    if not query.strip():
        st.warning("Please type a question.")
    else:
        with st.spinner("Searching movie scripts..."):
            answer = rag_chain.invoke(query)
            docs = retriever.get_relevant_documents(query)

        st.subheader("🎯 Answer")
        st.write(answer)

        st.subheader("📚 Retrieved Context")
        for i, d in enumerate(docs, start=1):
            with st.expander(f"Result {i} — {d.metadata.get('movie', 'Unknown Movie')}"):
                st.write("**Character:**", d.metadata.get("character", "Unknown"))
                st.write("**Genres:**", d.metadata.get("genres", "Unknown"))
                st.write("**Rating:**", d.metadata.get("rating", "N/A"))
                st.write("**Year:**", d.metadata.get("year", "N/A"))
                st.markdown("---")
                st.write(d.page_content)
