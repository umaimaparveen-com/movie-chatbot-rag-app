import os
import pandas as pd
import ast

from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ============================================================
# LOAD DATASET (Assuming CSVs are in /data directory)
# ============================================================

DATA_PATH = "data/cornell/"

lines_df = pd.read_csv(f"data/movie_lines.csv.gz")
characters_df = pd.read_csv(f"{DATA_PATH}/movie_characters.csv")
titles_df = pd.read_csv(f"{DATA_PATH}/movie_titles.csv")

lines_df.columns = lines_df.columns.str.strip()
characters_df.columns = characters_df.columns.str.strip()
titles_df.columns = titles_df.columns.str.strip()

# ============================================================
# MERGE TABLES
# ============================================================

merged = (
    lines_df
    .merge(characters_df, on=["characterID", "movieID"], how="left")
    .merge(titles_df, on="movieID", how="left")
)

# ============================================================
# CREATE DOCUMENTS
# ============================================================

documents = []
for _, row in merged.iterrows():
    if not isinstance(row["text"], str) or row["text"].strip() == "":
        continue

    doc = Document(
        page_content=row["text"],
        metadata={
            "movie": row.get("movieName", "Unknown"),
            "character": row.get("characterName", "Unknown"),
            "genres": row.get("genres", "Unknown"),
            "rating": row.get("rating"),
            "year": row.get("releaseYear"),
        }
    )
    documents.append(doc)

# ============================================================
# SPLIT INTO CHUNKS
# ============================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)

chunks = text_splitter.split_documents(documents)

# ============================================================
# CREATE VECTOR DB
# ============================================================

embedding_model = HuggingFaceEmbeddings(model_name="intfloat/e5-small-v2")

vector_db = Chroma(
    collection_name="movie_scripts",
    embedding_function=embedding_model,
    persist_directory="store/movie_scripts"
)

batch_size = 2000
for i in range(0, len(chunks), batch_size):
    vector_db.add_documents(chunks[i:i+batch_size])

retriever = vector_db.as_retriever(search_kwargs={"k": 4})

# ============================================================
# LLM + PROMPT
# ============================================================

llm = ChatOllama(model="llama3", temperature=0)

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a movie-script analysis assistant.

Use only the context provided below to answer the question.  
If needed, reconstruct scenes, summarize tone, or explain meaning.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""
)

def format_docs(docs):
    return "\n\n".join(
        f"Text: {d.page_content}\nMetadata: {d.metadata}" for d in docs
    )

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
