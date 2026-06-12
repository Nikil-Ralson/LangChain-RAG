from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import Optional
import uvicorn

from ingest import ingest_document
from rag import answer_question
from db import initialize_db, Tags


@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_db()
    yield


app = FastAPI(title="RAG API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    session_id: str
    question: str
    doc_id: Optional[int] = None   # None = search across all documents


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]


@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    tags: str = Form(default=""),   # comma-separated e.g. "tax,2024,personal"
):
    """Upload and ingest a document into the vector store."""
    allowed = {".pdf", ".txt", ".md"}
    suffix = "." + file.filename.split(".")[-1].lower()
    if suffix not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Allowed: {allowed}")

    tag_names = [t.strip() for t in tags.split(",") if t.strip()] if tags else []

    content = await file.read()
    result = ingest_document(content, file.filename, tag_names=tag_names)

    return {
        "message": f"Ingested {result['chunks']} chunks from '{file.filename}'",
        "doc_id": result["doc_id"],
        "tags": tag_names,
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Ask a question and get a RAG-powered answer."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    result = answer_question(req.session_id, req.question, doc_id=req.doc_id)
    return ChatResponse(answer=result["answer"], sources=result["sources"])


@app.get("/tags")
async def list_tags():
    """Return all available tags for the frontend selector."""
    tags = Tags.select()
    return [{"id": t.id, "name": t.name} for t in tags]


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
