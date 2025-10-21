# backend/main.py

from fastapi import FastAPI, HTTPException, Depends, Form
from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime, func, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import uuid
import os

# ----------------------------
# Database Setup
# ----------------------------
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/chatbot_db")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ----------------------------
# Models
# ----------------------------
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    email = Column(String(150), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

class KnowledgeBase(Base):
    __tablename__ = "knowledgebases"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"))
    content = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

class ChatbotToken(Base):
    __tablename__ = "chatbottokens"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"))
    token = Column(String(100), unique=True, nullable=False, default=lambda: str(uuid.uuid4()))
    created_at = Column(DateTime(timezone=True), server_default=func.now())

# Create tables
Base.metadata.create_all(bind=engine)

# ----------------------------
# FastAPI App
# ----------------------------
app = FastAPI(title="Chatbot KB Backend")

# ----------------------------
# Dependency to get DB session
# ----------------------------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ----------------------------
# User registration endpoint
# ----------------------------
@app.post("/api/register")
def register_user(name: str = Form(...), email: str = Form(...), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == email).first()
    if user:
        raise HTTPException(status_code=400, detail="Email already registered")
    new_user = User(name=name, email=email)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"id": new_user.id, "name": new_user.name, "email": new_user.email}

# ----------------------------
# Save Knowledge Base endpoint
# ----------------------------
@app.post("/api/knowledge-base")
def save_knowledge_base(user_id: int = Form(...), content: str = Form(...), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Save Knowledge Base
    kb = KnowledgeBase(user_id=user_id, content=content)
    db.add(kb)

    # Generate token if not exists
    token = db.query(ChatbotToken).filter(ChatbotToken.user_id == user_id).first()
    if not token:
        token = ChatbotToken(user_id=user_id, token=str(uuid.uuid4()))
        db.add(token)

    db.commit()
    db.refresh(kb)
    db.refresh(token)

    return {"knowledge_base_id": kb.id, "chatbot_token": token.token}

# ----------------------------
# Get Knowledge Base by token (Public Chatbot)
# ----------------------------
@app.get("/api/public-chatbot/{token}")
def get_kb_by_token(token: str, db: Session = Depends(get_db)):
    token_record = db.query(ChatbotToken).filter(ChatbotToken.token == token).first()
    if not token_record:
        raise HTTPException(status_code=404, detail="Chatbot not found")
    
    kb = db.query(KnowledgeBase).filter(KnowledgeBase.user_id == token_record.user_id).all()
    content = [k.content for k in kb]
    return {"user_id": token_record.user_id, "knowledge_base": content}

# ----------------------------
# Optional: Root endpoint
# ----------------------------
@app.get("/")
def root():
    return {"message": "Chatbot KB Backend is running"}
