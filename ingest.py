import tempfile
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from db import Documents, DocumentInformationChunks, Tags, DocumentTags, db
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from google import genai
from google.genai import types
from os import getenv


splitter = RecursiveCharacterTextSplitter(
    chunk_size = 800,
    chunk_overlap = 150,
    separators = ["\n\n", "\n", ".", " ", ""],
)

client = genai.Client(
    api_key=getenv("GOOGLE_API_KEY"),
    http_options={"api_version": "v1"},
)

def get_embeddings(texts: list[str]) -> list:
    result = client.models.embed_content(
        model="models/gemini-embedding-001",
        contents=[texts],
        config=types.EmbedContentConfig(task_type="retrieval_document"),
    )
    return [e.values for e in result.embeddings]

def load_file(content:bytes, filename:str) -> list[str]:
    """Load raw bytes into LangChain documents and return text chunks."""
    suffix = "." + filename.split(".")[-1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        if suffix ==".pdf":
            loader = PyPDFLoader(tmp_path)
        else:
            loader = TextLoader(tmp_path, encoding="utf-8")
        
        docs = loader.load()
        chunks = splitter.split_documents(docs)
        return [c.page_content for c in chunks]
    finally:
        os.unlink(tmp_path)

def ingest_document(content:bytes, filename:str, tag_names:list[str] = []) -> dict:
    """
    Full ingestion pipeline:
    1. Parse file → chunks
    2. Embed chunks via Gemini
    3. Store in PostgreSQL with pgvector
    4. Attach tags if provided
    """
    chunks = load_file(content, filename)
    if not chunks:
        raise ValueError("No text could be extracted from the document.")
    
    vectors = get_embeddings(chunks)

    with db.atomic():
        doc = Documents.create(name=filename, status="processing")

        DocumentInformationChunks.insert_many([
            {
                "document": doc.id,
                "chunk": chunk,
                "embedding": vector,
                "chunk_index": idx,
            }
            for idx, (chunk, vector) in enumerate(zip(chunks, vectors))
        ]).execute()

        for tag_name in tag_names:
            tag, _ = Tags.get_or_create(name=tag_name.strip().lower())
            DocumentTags.get_or_create(document=doc, tag=tag)

        doc.status = "processed"
        doc.save()

    return {"doc_id": doc.id, "chunks": len(chunks)}
