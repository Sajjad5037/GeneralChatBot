# backend/main.py

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import Column, Integer, Text, DateTime, func, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import os
import io
from PyPDF2 import PdfReader
from openai import OpenAI


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
    user_id = Column(Integer, nullable=False)  # associate with doctor
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
    allow_origins=[
        "https://class-management-system-new.web.app",
        "https://chat-for-me-ai-login.vercel.app"
    ],
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
    user_id: int = Form(...),  # Read doctor/user ID from frontend
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

    # Overwrite existing KB for the user if it exists
    kb = db.query(KnowledgeBase).filter(KnowledgeBase.user_id == user_id).first()
    if kb:
        print(f"[DEBUG] Overwriting existing knowledge base for user_id={user_id}, kb_id={kb.id}")
        kb.content = text
    else:
        kb = KnowledgeBase(user_id=user_id, content=text)
        db.add(kb)

    db.commit()
    db.refresh(kb)
    print(f"[DEBUG] Knowledge base saved: id={kb.id}, user_id={kb.user_id}, content_length={len(text)}")

    return {"knowledge_base_id": kb.id, "message": "PDF content saved successfully."}
# ----------------------------
#  (Public Chatbot)
# ----------------------------
#for clinics, saloon etc
@app.post("/api/chat")
def chat(message: str = Body(...), user_id: int = Body(...), db: Session = Depends(get_db)):
    print(f"[DEBUG] Received chat request: user_id={user_id}, message='{message}'")

    # Fetch KB for this doctor
    kb = db.query(KnowledgeBase).filter(KnowledgeBase.user_id == user_id).first()
    if not kb:
        print(f"[WARNING] No knowledge base found for user_id={user_id}")
        return {"reply": "Sorry, I have no knowledge to answer this yet."}

    print(f"[DEBUG] Knowledge base retrieved: id={kb.id}, content_length={len(kb.content)}")

    # Build prompt using doctor's KB
    prompt = f"You are Dr. {user_id}. Answer the question based on the knowledge below.\n\nKnowledge:\n{kb.content}\n\nUser: {message}"
    print(f"[DEBUG] Prompt length: {len(prompt)} characters")

    try:
        # Call OpenAI GPT-4.0-mini
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=500
        )

        bot_reply = response.choices[0].message.content
        print(f"[DEBUG] Bot reply length: {len(bot_reply)} characters")

        return {"reply": bot_reply}

    except Exception as e:
        print(f"[ERROR] OpenAI API call failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate reply from OpenAI")

# for general bussinesses
@app.post("/api/chat-business")
def chat(message: str = Body(...), username: str = Body(...), db: Session = Depends(get_db)):
    print(f"[DEBUG] Received chat request: username='{username}', message='{message}'")

    # Fetch doctor by username
    doctor = db.query(Doctor).filter(Doctor.username == username).first()
    if not doctor:
        print(f"[WARNING] No doctor found with username='{username}'")
        raise HTTPException(status_code=404, detail="Doctor not found")

    print(f"[DEBUG] Doctor found: id={doctor.id}, name={doctor.name}")

    # Fetch KB for this doctor
    kb = db.query(KnowledgeBase).filter(KnowledgeBase.user_id == doctor.id).first()
    if not kb:
        print(f"[WARNING] No knowledge base found for user_id={doctor.id}")
        return {"reply": "Sorry, I have no knowledge to answer this yet."}

    print(f"[DEBUG] Knowledge base retrieved: id={kb.id}, content_length={len(kb.content)}")

    # Build prompt using doctor's KB
    prompt = f"You are Dr. {doctor.name}. Answer the question based on the knowledge below.\n\nKnowledge:\n{kb.content}\n\nUser: {message}"
    print(f"[DEBUG] Prompt length: {len(prompt)} characters")

    try:
        # Call OpenAI GPT-4.0-mini
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=500
        )

        bot_reply = response.choices[0].message.content
        print(f"[DEBUG] Bot reply length: {len(bot_reply)} characters")

        return {"reply": bot_reply}

    except Exception as e:
        print(f"[ERROR] OpenAI API call failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate reply from OpenAI")

        
# ----------------------------
# Root endpoint
# ----------------------------
@app.get("/")
def root():
    print("[DEBUG] Root endpoint accessed")
    return {"message": "Chatbot KB Backend is running"}
