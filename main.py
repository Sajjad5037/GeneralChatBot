# backend/main.py

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import Column, Integer, Text, DateTime, func, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import os
import io
from PyPDF2 import PdfReader

# ----------------------------
# Database Setup
# ----------------------------
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:password@localhost:5432/chatbot_db"
)
print(f"[DEBUG] Connecting to database at: {DATABASE_URL}")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ----------------------------
# Models
# ----------------------------
class KnowledgeBase(Base):
    __tablename__ = "knowledgebases"
    id = Column(Integer, primary_key=True, index=True)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

# Create tables
print("[DEBUG] Creating database tables if they don't exist...")
Base.metadata.create_all(bind=engine)

# ----------------------------
# FastAPI App
# ----------------------------
app = FastAPI(title="Chatbot KB Backend")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://class-management-system-new.web.app"],  # frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Dependency to get DB session
# ----------------------------
def get_db():
    db = SessionLocal()
    try:
        print("[DEBUG] Opening new database session")
        yield db
    finally:
        print("[DEBUG] Closing database session")
        db.close()

# ----------------------------
# Upload PDF and Save Knowledge Base
# ----------------------------
@app.post("/api/knowledge-base/upload")
async def upload_pdf(
    user_id: int = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    print(f"[DEBUG] Received upload request: user_id={user_id}, filename={file.filename}, content_type={file.content_type}")
    
    # Extract text from PDF
    try:
        file_bytes = await file.read()
        print(f"[DEBUG] Read {len(file_bytes)} bytes from uploaded file")
        reader = PdfReader(io.BytesIO(file_bytes))
        text = ""
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            print(f"[DEBUG] Page {i+1}: extracted {len(page_text)} characters")
            text += page_text
    except Exception as e:
        print(f"[ERROR] Failed to read PDF: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to read PDF: {e}")

    if not text.strip():
        print("[WARNING] PDF contains no readable text")
        raise HTTPException(status_code=400, detail="PDF contains no readable text")

    # Check if KB already exists for this user
    kb = db.query(KnowledgeBase).filter(KnowledgeBase.user_id == user_id).first()
    if kb:
        kb.content = text
        print(f"[DEBUG] Overwriting existing knowledge base: id={kb.id}")
    else:
        kb = KnowledgeBase(user_id=user_id, content=text)
        db.add(kb)
        print("[DEBUG] Creating new knowledge base")

    db.commit()
    db.refresh(kb)
    print(f"[DEBUG] Knowledge base saved: id={kb.id}, content_length={len(text)}")

    return {"knowledge_base_id": kb.id, "message": "PDF content saved successfully."}

# ----------------------------
# Get All Knowledge Bases (Public Chatbot)
# ----------------------------
@app.get("/api/knowledge-base")
def get_all_kbs(db: Session = Depends(get_db)):
    print("[DEBUG] Fetching all knowledge bases from database...")
    kbs = db.query(KnowledgeBase).all()
    print(f"[DEBUG] Retrieved {len(kbs)} knowledge base entries")
    for i, kb in enumerate(kbs, start=1):
        print(f"[DEBUG] KB {i}: id={kb.id}, length={len(kb.content)}")
    return {"knowledge_base": [kb.content for kb in kbs]}

# ----------------------------
# Root endpoint
# ----------------------------
@app.get("/")
def root():
    print("[DEBUG] Root endpoint accessed")
    return {"message": "Chatbot KB Backend is running"}
