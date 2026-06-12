from dotenv import load_dotenv
load_dotenv()

from os import getenv
from datetime import datetime
from pgvector.peewee import VectorField
from peewee import (
    PostgresqlDatabase, Model, TextField, DateTimeField,
    ForeignKeyField, AutoField, IntegerField
)

db = PostgresqlDatabase(
    getenv("POSTGRES_DB_NAME"),
    user=getenv("POSTGRES_DB_USER"),
    password=getenv("POSTGRES_DB_PASSWORD"),
    host=getenv("POSTGRES_DB_HOST"),
    port=int(getenv("POSTGRES_DB_PORT", 5432)),
    sslmode=getenv("POSTGRES_SSL_MODE", "require")
)

class BaseModel(Model):
    class Meta:
        database = db

class Documents(BaseModel):
    id = AutoField()
    name = TextField()
    status = TextField(default="pending")
    created_at = DateTimeField(default=datetime.now)
    class Meta:
        db_table = "documents"

class Tags(BaseModel):
    id = AutoField()
    name = TextField(unique=True)
    class Meta:
        db_table = "tags"

class DocumentTags(BaseModel):
    document = ForeignKeyField(Documents, backref = "documents_tags", on_delete="CASCADE")
    tag = ForeignKeyField(Tags, backref="document_tags", on_delete="CASCADE")
    class Meta:
        db_table = "document_tags"

class DocumentInformationChunks(BaseModel):
    id = AutoField()
    document = ForeignKeyField(Documents, backref="chunks", on_delete="CASCADE")
    chunk = TextField()
    embedding = VectorField(3072)
    chunk_index = IntegerField(default=0)
    created_at = DateTimeField(default=datetime.now)
    class Meta:
        db_table = "document_information_chunks"

class Conversations(BaseModel):
    id = AutoField()
    session_id = TextField(unique=True)
    created_at = DateTimeField(default=datetime.now)
    class Meta:
        db_table = "conversations"

class Messages(BaseModel):
    id = AutoField()
    conversation = ForeignKeyField(Conversations, backref="messages", on_delete="CASCADE")
    role = TextField()  
    content = TextField()
    created_at = DateTimeField(default=datetime.now)
    class Meta:
        db_table = "messages"

def get_db():
    if db.is_closed():
        db.connect()
    return db

def initialize_db():
    get_db()

    db.execute_sql("CREATE EXTENSION IF NOT EXISTS vector")
    db.create_tables([Documents, Tags, DocumentTags, DocumentInformationChunks, Conversations, Messages], safe=True)

    try:
        db.execute_sql("""
            CREATE INDEX IF NOT EXISTS document_information_chunks_embedding_index
            ON document_information_chunks
            USING hnsw (embedding vector_cosine_ops)
        """)
        print("HNSW index ready.")
    except Exception as e:
        print(f"⚠ Could not create HNSW index: {e}")

    print("Database initialized.")