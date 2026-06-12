from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from db import DocumentInformationChunks, Conversations, Messages, db
from google import genai
from google.genai import types
from os import getenv

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided document context.

Rules:
- Answer only from the context provided below.
- If the context doesn't contain enough information, say "I don't have enough information in the uploaded documents to answer this."
- Be concise and precise.
- Cite which part of the context you used when relevant.

Context:
{context}"""

TOP_K = 5
MAX_HISTORY = 6  

client = genai.Client(
    api_key=getenv("GOOGLE_API_KEY"),
    http_options={"api_version": "v1"},
)

def embed_query(text: str) -> list:
    result = client.models.embed_content(
        model="models/gemini-embedding-001",
        contents=[text],
        config=types.EmbedContentConfig(task_type="retrieval_query"),
    )
    return result.embeddings[0].values

def retrieve_chunks(question:str, doc_id:int = None) -> list:
    """Embed the question and find the top-K most similar chunks via cosine distance.
    Optionally filter by a specific document."""
    q_vector = embed_query(question)

    query = (
        DocumentInformationChunks.select()
        .order_by(DocumentInformationChunks.embedding.cosine_distance(q_vector))
    )

    if doc_id is not None:
        query = query.where(DocumentInformationChunks.document == doc_id)
    return list(query.limit(TOP_K))

def get_or_create_conversation(session_id:str) -> Conversations:
    conv, _ = Conversations.get_or_create(session_id=session_id)
    return conv

def load_history(conversation:Conversations) -> list:
    """Load the last N messages as LangChain message objects."""
    messages = (
        Messages.select()
        .where(Messages.conversation == conversation)
        .order_by(Messages.created_at.desc())
        .limit(MAX_HISTORY)
    )
    history = []
    for msg in reversed(list(messages)):
        if msg.role == "user":
            history.append(HumanMessage(content=msg.content))
        else:
            history.append(AIMessage(content=msg.content))
    return history

def save_turn(conversation:Conversations, question: str, answer:str):
    with db.atomic():
        Messages.create(conversation=conversation, role="user", content=question)
        Messages.create(conversation=conversation, role="assistant", content=answer)

def answer_question(session_id:str,question:str, doc_id:int = None) -> dict:
    """
    Full RAG pipeline:
    1. Retrieve relevant chunks (optionally scoped to one document)
    2. Load conversation history
    3. Build prompt → call Gemini
    4. Save turn to DB
    """

    #1. Retrieve relevant chunks
    chunks = retrieve_chunks(question, doc_id=doc_id)
    if not chunks:
        return {
            "answer": "No documents have been ingested yet. Please upload a document first.",
            "sources": [],
        }
    context = "\n\n---\n\n".join(c.chunk for c in chunks)
    sources = list({c.document.name for c in chunks})

    #Load conversation history
    conversation = get_or_create_conversation(session_id)
    history = load_history(conversation)

    # 3. Build prompt and call LLM
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])

    chain = prompt | llm
    response = chain.invoke({
        "context": context,
        "history": history,
        "question": question,
    })

    answer = response.content

    # 4. Persist turn
    save_turn(conversation, question, answer)

    return {"answer": answer, "sources": sources}
