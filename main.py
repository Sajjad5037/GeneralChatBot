# backend/main.py

import hashlib


from passlib.context import CryptContext

from pydantic import BaseModel

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, Body, Request
import json
from fastapi.responses import JSONResponse,PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text,Column, Integer, Text, DateTime, func, create_engine,String, Boolean


from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import os
import io
from PyPDF2 import PdfReader
from openai import OpenAI
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests
import re


whatsapp_access_token = os.getenv("WHATSAPP_ACCESS_TOKEN")
whatsapp_phone_number_id = os.getenv("WHATSAPP_PHONE_NUMBER_ID")
webhook_verify_token = os.getenv("webhook_verify_token")
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")



client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
vector_stores = {} 

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

class ChatbotLink(Base):
    __tablename__ = "chatbot_links"

    id = Column(Integer, primary_key=True, index=True)
    doctor_id = Column(Integer, ForeignKey("doctors.id"))
    public_token = Column(String, unique=True, index=True)

    require_password = Column(Boolean, default=False)
    access_password = Column(String, nullable=True)  # hashed password
    

class Session(Base):
    __tablename__ = "sessions"  # table name changed

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    message = Column(Text, nullable=True)
    public_token = Column(String, nullable=False, unique=True)
    session_token = Column(String, nullable=False, unique=True)
    specialization = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

class SessionData(BaseModel):
    id: int | None = None  # optional if updating
    name: str
    message: str | None = None
    public_token: str
    session_token: str
    specialization: str | None = None

class WhatsAppKnowledgeBase(Base):
    __tablename__ = "WhatsAppknowledgebases"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False)        # associate with doctor
    phone_number = Column(String(15), nullable=False)  # store phone number
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


@app.post("/chatbot/settings")
def update_chatbot_settings(payload: dict, db: Session = Depends(get_db)):
    print("\n========== DEBUG: /chatbot/settings CALLED ==========")
    print("Incoming payload:", payload)

    session_token = payload.get("session_token")
    public_token = payload.get("public_token")
    require_password = payload.get("require_password")
    raw_password = payload.get("password")

    print("Parsed values:")
    print(" session_token =", session_token)
    print(" public_token =", public_token)
    print(" require_password =", require_password)
    print(" raw_password =", raw_password)

    # ---------------- VERIFY ADMIN SESSION ----------------
    session = db.query(SessionModel).filter(
        SessionModel.session_token == session_token
    ).first()

    print("DB session fetched:", session)

    if not session:
        print("ERROR: Invalid session token → access denied")
        raise HTTPException(status_code=401, detail="Invalid session")

    # ---------------- FETCH CHATBOT RECORD ----------------
    bot = db.query(ChatbotLink).filter(
        ChatbotLink.public_token == public_token
    ).first()

    print("DB chatbot link fetched:", bot)

    if not bot:
        print("ERROR: No chatbot link found for this public_token")
        raise HTTPException(status_code=404, detail="Chatbot link not found")

    # ---------------- UPDATE PASSWORD SETTINGS ----------------
    print("Updating chatbot settings...")

    bot.require_password = require_password
    print("require_password updated to:", require_password)

    if require_password:
        print("Password protection enabled")

        if not raw_password:
            print("ERROR: Admin enabled password but did not send password")
            raise HTTPException(
                status_code=400,
                detail="Password value required when require_password is true"
            )

        hashed_pw = pwd_context.hash(raw_password)
        print("Generated hashed password:", hashed_pw)

        bot.access_password = hashed_pw
    else:
        print("Password protection disabled → clearing existing password")
        bot.access_password = None

    db.commit()
    print("DB commit successful")
    print("========== /chatbot/settings END ==========\n")

    return {"message": "Chatbot access settings updated successfully"}

@app.get("/chatbot/init/{public_token}")
def chatbot_init(public_token: str, db: Session = Depends(get_db)):
    print("\n========== DEBUG: /chatbot/init CALLED ==========")
    print("Incoming public_token:", public_token)

    bot = db.query(ChatbotLink).filter(
        ChatbotLink.public_token == public_token
    ).first()

    print("DB chatbot fetched:", bot)

    if not bot:
        print("ERROR: Chatbot not found for this public_token")
        raise HTTPException(status_code=404, detail="Chatbot not found")

    print("require_password =", bot.require_password)

    if bot.require_password:
        print("Chatbot IS password protected → returning requiresPassword=True")
        print("========== /chatbot/init END ==========\n")
        return {"requiresPassword": True}

    print("Chatbot is NOT password protected → returning requiresPassword=False")
    print("========== /chatbot/init END ==========\n")
    return {
        "requiresPassword": False,
        "publicToken": public_token,
    }

@app.post("/chatbot/validate-password")
def validate_password(payload: dict, db: Session = Depends(get_db)):
    print("\n========== DEBUG: /chatbot/validate-password CALLED ==========")
    print("Incoming payload:", payload)

    public_token = payload.get("public_token")
    entered_password = payload.get("password")

    print("Parsed values:")
    print(" public_token =", public_token)
    print(" entered_password =", entered_password)

    bot = db.query(ChatbotLink).filter(
        ChatbotLink.public_token == public_token
    ).first()

    print("DB chatbot fetched:", bot)

    if not bot:
        print("ERROR: Chatbot not found")
        raise HTTPException(status_code=404, detail="Chatbot not found")

    print("require_password =", bot.require_password)
    print("stored hashed password =", bot.access_password)

    # If NOT password protected → automatically allow
    if not bot.require_password:
        print("Chatbot doesn't require password → auto-allow access")
        token = str(uuid4())
        print("Generated chatAccessToken:", token)
        print("========== /chatbot/validate-password END ==========\n")
        return {"valid": True, "chatAccessToken": token}

    # Password IS required → verify
    if not entered_password:
        print("ERROR: No password entered while required")
        return {"valid": False}

    print("Verifying entered password...")
    is_valid = pwd_context.verify(entered_password, bot.access_password)

    print("Password match result:", is_valid)

    if not is_valid:
        print("ERROR: Password is incorrect")
        print("========== /chatbot/validate-password END ==========\n")
        return {"valid": False}

    # Correct password → generate session token
    chat_token = str(uuid4())
    print("Password correct → granted access")
    print("Generated chatAccessToken:", chat_token)

    print("========== /chatbot/validate-password END ==========\n")
    return {"valid": True, "chatAccessToken": chat_token}





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

    # ----- Recreate temporary vector store -----
    if user_id in vector_stores:
        print(f"[DEBUG] Deleting existing temporary vector store for user_id={user_id}")
        del vector_stores[user_id]

    chunks = chunk_text(kb.content, chunk_size=500, overlap=50)
    embeddings = embed_texts(chunks)
    vector_stores[user_id] = {"chunks": chunks, "embeddings": np.array(embeddings)}
    print(f"[DEBUG] New vector store created for user_id={user_id} with {len(chunks)} chunks")

    return {"knowledge_base_id": kb.id, "message": "PDF content saved successfully and vector store rebuilt."}



@app.post("/save-doctor")
async def save_doctor(session_data: SessionData, db: Session = Depends(get_db)):
    """
    Save a new session record (or update if id exists).
    """
    if session_data.id:
        # Update existing record
        db_session = db.query(SessionModel).filter(SessionModel.id == session_data.id).first()
        if not db_session:
            raise HTTPException(status_code=404, detail="Session not found")
        db_session.name = session_data.name
        db_session.message = session_data.message
        db_session.public_token = session_data.public_token
        db_session.session_token = session_data.session_token
        db_session.specialization = session_data.specialization
    else:
        # Create new session record
        db_session = SessionModel(
            name=session_data.name,
            message=session_data.message,
            public_token=session_data.public_token,
            session_token=session_data.session_token,
            specialization=session_data.specialization,
        )
        db.add(db_session)

    db.commit()
    db.refresh(db_session)
    return {"message": "Session saved successfully", "session_id": db_session.id}
        




# --- Dependency ---
@app.post("/api/whatsapp-knowledge-base/upload")
async def upload_pdf(
    user_id: int = Form(...),            # Required (sent by frontend)
    file: UploadFile = File(...),        # Required (sent by frontend)
    db: Session = Depends(get_db)
):
    print(f"[DEBUG] Upload request: user_id={user_id}, filename={file.filename}")

    # ----- Extract PDF text -----
    try:
        file_bytes = await file.read()
        print(f"[DEBUG] Read {len(file_bytes)} bytes from file")

        reader = PdfReader(io.BytesIO(file_bytes))
        text = ""

        for i, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            print(f"[DEBUG] Page {i+1}: extracted {len(page_text)} characters")
            text += page_text

    except Exception as e:
        print(f"[ERROR] PDF parsing error: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to read PDF: {e}")

    if not text.strip():
        print("[WARNING] PDF has no readable content")
        raise HTTPException(status_code=400, detail="Uploaded PDF contains no readable text")

    print(f"[DEBUG] Final extracted text length: {len(text)} characters")

    # ----- Find existing KB or create new -----
    kb = db.query(WhatsAppKnowledgeBase).filter(
        WhatsAppKnowledgeBase.user_id == user_id
    ).first()

    if kb:
        print(f"[DEBUG] Updating existing KB (id={kb.id})")
        kb.content = text
    else:
        print("[DEBUG] Creating new KB entry")
        kb = WhatsAppKnowledgeBase(
            user_id=user_id,
            content=text
        )
        db.add(kb)

    db.commit()
    db.refresh(kb)

    print(f"[DEBUG] KB saved: id={kb.id}, user_id={kb.user_id}")

    return {
        "knowledge_base_id": kb.id,
        "message": "PDF knowledge base uploaded successfully."
    }


"""

@app.post("/api/whatsapp-knowledge-base/upload")
async def upload_pdf(
    user_id: int = Form(...),            # doctor/user ID
    phone_number: str = Form(...),       # phone number from frontend
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    print(f"[DEBUG] Received upload request: user_id={user_id}, phone_number={phone_number}, filename={file.filename}, content_type={file.content_type}")
    
    
    # --- Extract text from PDF ---
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
    print(f"[DEBUG] Total extracted text length: {len(text)} characters")

    # --- Check if a KB already exists for this user + phone number ---
    kb = db.query(WhatsAppKnowledgeBase).filter(
        WhatsAppKnowledgeBase.user_id == user_id,
        WhatsAppKnowledgeBase.phone_number == phone_number
    ).first()

    if kb:
        print(f"[DEBUG] Overwriting existing WhatsApp KB: kb_id={kb.id}")
        kb.content = text
    else:
        print("[DEBUG] Creating new WhatsApp KB entry")
        kb = WhatsAppKnowledgeBase(
            user_id=user_id,
            phone_number=phone_number,
            content=text
        )
        db.add(kb)

    # --- Commit to database ---
    db.commit()
    db.refresh(kb)
    print(f"[DEBUG] WhatsApp knowledge base saved: id={kb.id}, user_id={kb.user_id}, phone_number={kb.phone_number}, content_length={len(text)}")

    return {"knowledge_base_id": kb.id, "message": "PDF content saved successfully."}
"""


"""
# ----------------------------
#  (Public Chatbot)
# ----------------------------
# for clinics, salon etc
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
"""


def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def embed_texts(texts):
    """Return list of embeddings for a list of texts using OpenAI embeddings"""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [np.array(e.embedding) for e in response.data]

# for clinics, salon etc
@app.post("/api/chat")
def chat(message: str = Body(...), user_id: int = Body(...), db: Session = Depends(get_db)):
    print("==============================================")
    print("[DEBUG] Entered /api/chat endpoint")
    print(f"[DEBUG] user_id={user_id}, message='{message}'")
    print(f"[DEBUG] db injected: {db}")
    print(f"[DEBUG] db type: {type(db)}")

    if db is None:
        print("[ERROR] Database session is None! FastAPI did not inject a session.")
        raise HTTPException(status_code=500, detail="Database session is None")

    # --- Test DB connection properly ---
    try:
        _ = db.execute(text("SELECT 1")).fetchone()  # wrap raw SQL in text()
        print("[DEBUG] Database connection test: OK ✅")
    except Exception as e:
        print(f"[ERROR] Database connection test failed: {e}")
        raise HTTPException(status_code=500, detail="Database session invalid or closed")

    # --- Fetch KB for this doctor ---
    try:
        print("[DEBUG] Querying KnowledgeBase for user_id:", user_id)
        kb = db.query(KnowledgeBase).filter(KnowledgeBase.user_id == user_id).first()
        print("[DEBUG] Query executed successfully.")
    except Exception as e:
        print(f"[ERROR] Query to KnowledgeBase failed: {e}")
        raise HTTPException(status_code=500, detail=f"Database query failed: {e}")

    if not kb:
        print(f"[WARNING] No knowledge base found for user_id={user_id}")
        return {"reply": "Sorry, I have no knowledge to answer this yet."}

    print(f"[DEBUG] Knowledge base retrieved: id={kb.id}, content_length={len(kb.content)}")

    # --- Build prompt using doctor's KB ---
    prompt = f"You are Dr. {user_id}. Answer the question concisely based on the knowledge below.\n\nKnowledge:\n{kb.content}\n\nUser: {message}\n\nInstructions: Provide a brief summary in 2-3 sentences. Avoid long paragraphs."

    print(f"[DEBUG] Prompt length: {len(prompt)} characters")

    # --- Call OpenAI API ---
    try:
        print("[DEBUG] Sending prompt to OpenAI API...")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=500
        )

        bot_reply = response.choices[0].message.content
        print(f"[DEBUG] Bot reply length: {len(bot_reply)} characters")

        print("[DEBUG] Returning successful response ✅")
        return {"reply": bot_reply}

    except Exception as e:
        print(f"[ERROR] OpenAI API call failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate reply from OpenAI")


# ---------------------------------------------------------------
# OLD CHAT ENDPOINT — NOW DISABLED (COMMENTED OUT)
# ---------------------------------------------------------------

"""
@app.post("/api/chat-whatsapp")
def chat(
    message: str = Body(...),
    user_id: int = Body(...),
    db: Session = Depends(get_db)
):
    print(f"[DEBUG] Received chat request: user_id={user_id}, message='{message}'")

    # Fetch KB for this doctor
    kb = db.query(WhatsAppKnowledgeBase).filter(WhatsAppKnowledgeBase.user_id == user_id).first()
    if not kb:
        print(f"[WARNING] No knowledge base found for user_id={user_id}")
        return {"reply": "Sorry, I have no knowledge to answer this yet."}

    # Compute hash of current KB content
    kb_hash = hashlib.md5(kb.content.encode("utf-8")).hexdigest()

    # --- Build or rebuild vector store if it doesn't exist or KB changed ---
    if (user_id not in vector_stores) or (vector_stores[user_id].get("kb_hash") != kb_hash):
        chunks = chunk_text(kb.content, chunk_size=500, overlap=50)
        embeddings = embed_texts(chunks)
        vector_stores[user_id] = {
            "chunks": chunks,
            "embeddings": np.array(embeddings),
            "kb_hash": kb_hash
        }
        print(f"[DEBUG] Vector store created/rebuilt for user_id={user_id} with {len(chunks)} chunks")

    store = vector_stores[user_id]

    # --- Embed the user query ---
    query_embedding = np.array(embed_texts([message])[0])

    # --- Compute similarities ---
    sims = cosine_similarity([query_embedding], store["embeddings"])[0]
    top_idx = sims.argmax()  # get the most similar chunk
    relevant_chunk = store["chunks"][top_idx]
    print(f"[DEBUG] Top chunk index: {top_idx}, similarity: {sims[top_idx]:.4f}")

    # --- Build prompt using only relevant chunk ---
    prompt = f'''You are Dr. {user_id}. Answer the question concisely based on the knowledge below.

    Knowledge:
    {relevant_chunk}
    
    User: {message}
    
    Instructions: Provide a brief, 1–2 sentence answer. Avoid long explanations.'''

    print(f"[DEBUG] Prompt length: {len(prompt)} characters")

    try:
        # Call OpenAI GPT-4o-mini
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
"""
@app.post("/api/chat-whatsapp")
def chat_whatsapp(
    payload: dict = Body(...),
    db: Session = Depends(get_db)
):
    print("\n==================== DEBUG: /api/chat-whatsapp CALLED ====================")
    print("Incoming payload:", payload)

    # Extract incoming fields
    message = payload.get("message")
    public_token = payload.get("public_token")
    chat_access_token = payload.get("chat_access_token")

    print(f"Parsed parameters:\n message='{message}'\n public_token={public_token}\n chat_access_token={chat_access_token}")

    if not message or not public_token:
        print("[ERROR] Missing message or public_token")
        raise HTTPException(status_code=400, detail="message and public_token are required")

    # ---------------------------
    # STEP 1 — Fetch ChatbotLink
    # ---------------------------
    bot = db.query(ChatbotLink).filter(ChatbotLink.public_token == public_token).first()
    print("Fetched ChatbotLink:", bot)

    if not bot:
        print("[ERROR] No ChatbotLink found for public_token:", public_token)
        raise HTTPException(status_code=404, detail="Invalid chatbot link")

    # ---------------------------
    # STEP 2 — Validate Password If Required
    # ---------------------------
    print("Chatbot requires_password =", bot.require_password)

    if bot.require_password:
        print("Password protection is enabled")

        if not chat_access_token:
            print("[ERROR] chat_access_token missing")
            raise HTTPException(status_code=401, detail="Password required")

        print("chat_access_token provided → allowing access")
    else:
        print("Chatbot does NOT require password → allowing public access")

    # ---------------------------
    # STEP 3 — Fetch doctor’s knowledge base
    # ---------------------------
    print("Fetching knowledge base for doctor_id =", bot.doctor_id)

    kb = db.query(WhatsAppKnowledgeBase).filter(WhatsAppKnowledgeBase.user_id == bot.doctor_id).first()

    if not kb:
        print(f"[WARNING] No KB found for doctor_id={bot.doctor_id}")
        return {"reply": "Sorry, no knowledge base is available yet."}

    print("KB fetched successfully. KB size:", len(kb.content), "characters")

    # Compute hash of the KB
    kb_hash = hashlib.md5(kb.content.encode("utf-8")).hexdigest()
    print("KB hash:", kb_hash)

    # ---------------------------
    # STEP 4 — Build or reuse vector store
    # ---------------------------
    if (bot.doctor_id not in vector_stores) or (vector_stores[bot.doctor_id]["kb_hash"] != kb_hash):
        print("Vector store missing or outdated → rebuilding...")

        chunks = chunk_text(kb.content, chunk_size=500, overlap=50)
        embeddings = embed_texts(chunks)

        vector_stores[bot.doctor_id] = {
            "chunks": chunks,
            "embeddings": np.array(embeddings),
            "kb_hash": kb_hash
        }

        print(f"[DEBUG] Vector store rebuilt with {len(chunks)} chunks")
    else:
        print("[DEBUG] Using cached vector store")

    store = vector_stores[bot.doctor_id]

    # ---------------------------
    # STEP 5 — Embed user query
    # ---------------------------
    query_embedding = np.array(embed_texts([message])[0])
    print("Query embedding shape:", query_embedding.shape)

    sims = cosine_similarity([query_embedding], store["embeddings"])[0]
    top_idx = sims.argmax()
    print(f"[DEBUG] Top chunk index = {top_idx}, similarity = {sims[top_idx]:.4f}")

    relevant_chunk = store["chunks"][top_idx]

    # ---------------------------
    # STEP 6 — Build prompt
    # ---------------------------
    prompt = f"""
You are an AI assistant for doctor ID {bot.doctor_id}.
Answer concisely (1–2 sentences) using ONLY the knowledge below.

Knowledge:
{relevant_chunk}

User: {message}
    """.strip()

    print(f"[DEBUG] Prompt length = {len(prompt)} characters")

    # ---------------------------
    # STEP 7 — Call OpenAI
    # ---------------------------
    try:
        print("[DEBUG] Sending prompt to OpenAI...")

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=300
        )

        bot_reply = response.choices[0].message.content
        print("[DEBUG] OpenAI response length:", len(bot_reply))

        print("==================== /api/chat-whatsapp END ====================\n")
        return {"reply": bot_reply}

    except Exception as e:
        print("[ERROR] OpenAI API failed:", e)
        print("==================== /api/chat-whatsapp ERROR END ====================\n")
        raise HTTPException(status_code=500, detail="Failed to generate AI reply")
       

#IMPLEMENTING ENDPOINTS FOR WHATS APP CHATBOT
def get_relevant_context(kb_text: str, user_query: str, top_k: int = 3) -> str:
    """
    Create a temporary vector store from kb_text and retrieve relevant context
    for the user query.
    
    :param kb_text: Full text from WhatsAppKnowledgeBase for a user
    :param user_query: The incoming user message
    :param top_k: Number of most relevant chunks to retrieve
    :return: Concatenated relevant context string
    """
    if not kb_text.strip():
        return ""  # no KB content available
    
    # --- 1. Split KB text into chunks ---
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,   # adjust based on token limits
        chunk_overlap=50
    )
    chunks = text_splitter.split_text(kb_text)
    
    # --- 2. Embed the chunks ---
    embeddings = OpenAIEmbeddings()
    
    # --- 3. Create temporary FAISS vector store ---
    vector_store = FAISS.from_texts(chunks, embeddings)
    
    # --- 4. Retrieve top_k relevant chunks ---
    results = vector_store.similarity_search(user_query, k=top_k)
    
    # Combine retrieved chunks into a single string
    context = "\n".join([r.page_content for r in results])
    return context

# Endpoint to get doctor ID from sessionToken
@app.get("/get-doctor-id/{session_token}")
async def get_doctor_id(session_token: str):
    doctor = next((d for d in doctors if d["sessionToken"] == session_token), None)
    if doctor:
        return {"doctor_id": doctor["id"]}
    raise HTTPException(status_code=404, detail="Doctor not found for this session token")

# Endpoint to get doctor name from doctor ID
@app.get("/get-doctor-name/{doctor_id}")
async def get_doctor_name(doctor_id: int):
    doctor = next((d for d in doctors if d["id"] == doctor_id), None)
    if doctor:
        return {"doctor_name": doctor["name"]}
    raise HTTPException(status_code=404, detail="Doctor not found for this ID")

@app.api_route("/webhook", methods=["GET", "POST"])
async def webhook(request: Request):
    # -------------------- GET: Verification --------------------
    if request.method == "GET":
        mode = request.query_params.get("hub.mode")
        token = request.query_params.get("hub.verify_token")
        challenge = request.query_params.get("hub.challenge")
        print(f"[DEBUG] GET verification request: mode={mode}, token={token}, challenge={challenge}")

        if mode == "subscribe" and token == webhook_verify_token:
            print("[DEBUG] Webhook verified successfully!")
            return PlainTextResponse(content=challenge, status_code=200)

        print("[WARNING] Webhook verification failed")
        return PlainTextResponse(content="Webhook verification failed", status_code=403)

    # -------------------- POST: Handle incoming messages --------------------
    elif request.method == "POST":
        try:
            data = await request.json()
            print("[DEBUG] Received webhook payload:", json.dumps(data, indent=2))

            # Extract message
            entry = data.get("entry", [])[0]
            change = entry.get("changes", [])[0]
            value = change.get("value", {})
            messages = value.get("messages", [])

            if not messages:
                return JSONResponse(content={"status": "no message"}, status_code=200)

            message = messages[0]
            from_number = message.get("from")
            user_text = message.get("text", {}).get("body", "")
            phone_number_id = value.get("metadata", {}).get("phone_number_id")

            if not user_text:
                return JSONResponse(content={"status": "empty message"}, status_code=200)

            print(f"[DEBUG] Message received from {from_number}: {user_text}")

            # --- Open DB session safely ---
            with SessionLocal() as db:
                # Fetch KB based on chatbot number
                display_number = value["metadata"]["display_phone_number"]  # "+1 555 140 8854"

                kb_entries = db.query(WhatsAppKnowledgeBase)\
                               .filter(WhatsAppKnowledgeBase.phone_number == display_number)\
                               .all()
                kb_text = "\n".join([kb.content for kb in kb_entries]) if kb_entries else ""

            if not kb_text.strip():
                print(f"[WARNING] No knowledge base content for chatbot {display_number}")
                return JSONResponse(content={"reply": "Sorry, I have no knowledge to answer this yet."}, status_code=200)

            # --- Build temporary vector store per user ---
            if from_number not in vector_stores:
                chunks = chunk_text(kb_text, chunk_size=500, overlap=50)
                embeddings = embed_texts(chunks)
                vector_stores[from_number] = {"chunks": chunks, "embeddings": np.array(embeddings)}
                print(f"[DEBUG] Vector store created for {from_number} with {len(chunks)} chunks")

            store = vector_stores[from_number]

            # Embed user query & find most similar chunk
            query_embedding = np.array(embed_texts([user_text])[0])
            sims = cosine_similarity([query_embedding], store["embeddings"])[0]
            top_idx = sims.argmax()
            relevant_chunk = store["chunks"][top_idx]
            print(f"[DEBUG] Top chunk index: {top_idx}, similarity: {sims[top_idx]:.4f}")

            # Build prompt
            prompt = f"You are an AI assistant. Answer the question based on the knowledge below.\n\nKnowledge:\n{relevant_chunk}\n\nUser: {user_text}"
            print(f"[DEBUG] Prompt length: {len(prompt)} characters")

            # Generate AI reply
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=500
            )
            bot_reply = response.choices[0].message.content
            print(f"[DEBUG] Bot reply length: {len(bot_reply)} characters")

            # Send reply to WhatsApp
            api_url = f"https://graph.facebook.com/v17.0/{phone_number_id}/messages"
            headers = {
                "Authorization": f"Bearer {whatsapp_access_token}",
                "Content-Type": "application/json"
            }
            payload = {
                "messaging_product": "whatsapp",
                "to": from_number,
                "type": "text",
                "text": {"body": bot_reply}
            }
            resp = requests.post(api_url, headers=headers, json=payload)
            print("[DEBUG] WhatsApp API response:", resp.json())

            return JSONResponse(content={"status": "message processed"}, status_code=200)

        except Exception as e:
            print("[ERROR] Error processing webhook:", e)
            return JSONResponse(content={"error": str(e)}, status_code=500)


"""
@app.api_route("/webhook", methods=["GET", "POST"])
async def webhook(request: Request):
    if request.method == "GET":
        # Verification
        mode = request.query_params.get("hub.mode")
        token = request.query_params.get("hub.verify_token")
        challenge = request.query_params.get("hub.challenge")
        print("GET verification request:", mode, token, challenge)

        if mode == "subscribe" and token == webhook_verify_token:
            print("Webhook verified successfully!")
            return PlainTextResponse(content=challenge, status_code=200)

        print("Webhook verification failed")
        return PlainTextResponse(content="Webhook verification failed", status_code=403)

    elif request.method == "POST":
        try:
            webhook_payload = await request.json()
            print("Received webhook payload:", webhook_payload)
            # Handle incoming messages here
            return JSONResponse(content={"status": "received"}, status_code=200)
        except Exception as e:
            print("Error processing webhook:", e)
            return JSONResponse(content={"error": str(e)}, status_code=500)


@app.api_route("/webhook", methods=["GET", "POST"])
async def webhook(request: Request):
    if request.method == "GET":
        # --- Webhook verification ---
        mode = request.query_params.get("hub.mode")
        token = request.query_params.get("hub.verify_token")
        challenge = request.query_params.get("hub.challenge")
        
        if mode == "subscribe" and token == webhook_verify_token:
            print("Webhook verified successfully!")
            return PlainTextResponse(content=challenge, status_code=200)
        
        print("Webhook verification failed")
        return PlainTextResponse(content="Webhook verification failed", status_code=403)

    elif request.method == "POST":
        try:
            data = await request.json()
            print("Received webhook payload:", json.dumps(data, indent=2))

            # --- Extract message ---
            entry = data.get("entry", [])[0]
            change = entry.get("changes", [])[0]
            value = change.get("value", {})
            messages = value.get("messages", [])

            if not messages:
                return JSONResponse(content={"status": "no message"}, status_code=200)

            message = messages[0]
            from_number = message["from"]
            user_text = message.get("text", {}).get("body", "")
            phone_number_id = value["metadata"]["phone_number_id"]

            if not user_text:
                return JSONResponse(content={"status": "empty message"}, status_code=200)

            print(f"Message received from {from_number}: {user_text}")

            # --- Chatbot personality: Sajjad’s personal assistant ---
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an AI assistant created by Sajjad Ali Noor, "
                            "a full-stack developer from Lahore with expertise in Python, FastAPI, "
                            "and building intelligent systems such as chatbot integrations and "
                            "clinic management tools. "
                            "You represent Sajjad professionally — answer politely, explain technical things clearly, "
                            "and reflect his calm, thoughtful tone. "
                            "If users ask about Sajjad, tell them he’s a developer focused on AI-powered web apps, "
                            "problem-solving, and backend design."
                        ),
                    },
                    {"role": "user", "content": user_text}
                ],
                temperature=0.3,
                max_tokens=500
            )

            bot_reply = completion.choices[0].message.content.strip()
            print(f"AI Reply: {bot_reply}")

            # --- Send reply back to WhatsApp ---
            api_url = f"https://graph.facebook.com/v17.0/{phone_number_id}/messages"
            headers = {
                "Authorization": f"Bearer {whatsapp_access_token}",
                "Content-Type": "application/json"
            }
            payload = {
                "messaging_product": "whatsapp",
                "to": from_number,
                "type": "text",
                "text": {"body": bot_reply}
            }

            resp = requests.post(api_url, headers=headers, json=payload)
            print("WhatsApp API response:", resp.json())

            return JSONResponse(content={"status": "message processed"}, status_code=200)

        except Exception as e:
            print("Error processing webhook:", e)
            return JSONResponse(content={"error": str(e)}, status_code=500)
"""

def send_whatsapp_message(recipient_number, message_text):
    api_url = f"https://graph.facebook.com/v22.0/{whatsapp_phone_number_id}/messages"
    payload = {
        "messaging_product": "whatsapp",
        "to": recipient_number,
        "type": "text",
        "text": {"body": message_text}
    }
    headers = {
        "Authorization": f"Bearer {whatsapp_access_token}",
        "Content-Type": "application/json"
    }
    response = requests.post(api_url, headers=headers, data=json.dumps(payload))
    print("WhatsApp API response:", response.json())
    return response

# --- Optional route to manually send messages ---
@app.post("/send_message")
async def manual_send_message(request: Request):
    request_data = await request.json()
    recipient_number = request_data.get("to")
    message_body = request_data.get("body", "Hello from WhatsApp Demo!")

    resp = send_whatsapp_message(recipient_number, message_body)
    return JSONResponse(content={"response": resp.json()})
        
# ----------------------------
# Root endpoint
# ----------------------------
@app.get("/")
def root():
    print("[DEBUG] Root endpoint accessed")
    return {"message": "Chatbot KB Backend is running"}
