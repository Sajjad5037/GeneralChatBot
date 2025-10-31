# backend/main.py

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, Body, Request
import json
from fastapi.responses import JSONResponse,PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import Column, Integer, Text, DateTime, func, create_engine,String
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


from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from sqlalchemy.orm import Session
from PyPDF2 import PdfReader
import io

app = FastAPI()

# --- Dependency ---
def get_db():
    # your DB session creation logic here
    pass

# --- Database model example ---
# class WhatsAppKnowledgeBase(Base):
#     id: int
#     user_id: int
#     chatbot_number: str
#     content: str

@app.post("/api/whatsapp-knowledge-base/upload")
async def upload_pdf(
    user_id: int = Form(...),            # client/user ID
    chatbot_number: str = Form(...),     # the WhatsApp number acting as chatbot
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    print(f"[DEBUG] Received upload request: user_id={user_id}, chatbot_number={chatbot_number}, filename={file.filename}, content_type={file.content_type}")

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

    # --- Check if a KB already exists for this client + chatbot number ---
    kb = db.query(WhatsAppKnowledgeBase).filter(
        WhatsAppKnowledgeBase.user_id == user_id,
        WhatsAppKnowledgeBase.chatbot_number == chatbot_number
    ).first()

    if kb:
        print(f"[DEBUG] Overwriting existing WhatsApp KB: kb_id={kb.id}")
        kb.content = text
    else:
        print("[DEBUG] Creating new WhatsApp KB entry")
        kb = WhatsAppKnowledgeBase(
            user_id=user_id,
            chatbot_number=chatbot_number,
            content=text
        )
        db.add(kb)

    # --- Commit to database ---
    db.commit()
    db.refresh(kb)
    print(f"[DEBUG] WhatsApp knowledge base saved: id={kb.id}, user_id={kb.user_id}, chatbot_number={kb.chatbot_number}, content_length={len(text)}")

    return {"knowledge_base_id": kb.id, "message": "PDF content saved successfully."}



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

@app.post("/api/chat")
def chat(message: str = Body(...), user_id: int = Body(...), db: Session = Depends(get_db)):
    print(f"[DEBUG] Received chat request: user_id={user_id}, message='{message}'")

    # Fetch KB for this doctor
    kb = db.query(KnowledgeBase).filter(KnowledgeBase.user_id == user_id).first()
    if not kb:
        print(f"[WARNING] No knowledge base found for user_id={user_id}")
        return {"reply": "Sorry, I have no knowledge to answer this yet."}

    # --- Build temporary vector store if it doesn't exist ---
    if user_id not in vector_stores:
        chunks = chunk_text(kb.content, chunk_size=500, overlap=50)
        embeddings = embed_texts(chunks)
        vector_stores[user_id] = {"chunks": chunks, "embeddings": np.array(embeddings)}
        print(f"[DEBUG] Vector store created for user_id={user_id} with {len(chunks)} chunks")

    store = vector_stores[user_id]

    # --- Embed the user query ---
    query_embedding = np.array(embed_texts([message])[0])

    # --- Compute similarities ---
    sims = cosine_similarity([query_embedding], store["embeddings"])[0]
    top_idx = sims.argmax()  # get the most similar chunk
    relevant_chunk = store["chunks"][top_idx]
    print(f"[DEBUG] Top chunk index: {top_idx}, similarity: {sims[top_idx]:.4f}")

    # --- Build prompt using only relevant chunk ---
    prompt = f"You are Dr. {user_id}. Answer the question based on the knowledge below.\n\nKnowledge:\n{relevant_chunk}\n\nUser: {message}"
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
                               .filter(WhatsAppKnowledgeBase.chatbot_number == display_number)\
                               .all()
                kb_text = "\n".join([kb.content for kb in kb_entries]) if kb_entries else ""

            if not kb_text.strip():
                print(f"[WARNING] No knowledge base content for chatbot {phone_number_id}")
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
