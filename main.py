import os
from datetime import datetime, timedelta
from uuid import uuid4
import base64
import io
from typing import List

import requests
from fastapi import FastAPI, Depends, HTTPException, status, Request, Response, Form
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import JSONResponse, StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr
from sqlalchemy import create_engine, Column, String, DateTime, JSON, or_
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from starlette.middleware.sessions import SessionMiddleware
from starlette.middleware.cors import CORSMiddleware
from PIL import Image
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# FastAPI app setup
app = FastAPI(title="ChatApp API")

# Database setup
SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Security setup
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Middleware
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Models
class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    api_key = Column(String, unique=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class ChatHistory(Base):
    __tablename__ = "chat_history"

    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    user_id = Column(String, index=True)
    content = Column(JSON)

# Pydantic models
class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: str | None = None

class ChatMessage(BaseModel):
    user_input: str

class ImageGeneration(BaseModel):
    prompt: str
    width: int = 512
    height: int = 512

class UserUpdate(BaseModel):
    email: EmailStr | None = None
    password: str | None = None

# Helper functions
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def authenticate_user(db, username: str, password: str):
    user = db.query(User).filter(User.username == username).first()
    if not user or not verify_password(password, user.hashed_password):
        return False
    return user

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    db = next(get_db())
    user = db.query(User).filter(User.username == token_data.username).first()
    if user is None:
        raise credentials_exception
    return user

def get_chat_history(db: Session, user_id: str):
    return db.query(ChatHistory).filter(ChatHistory.user_id == user_id).first()

def update_chat_history(db: Session, user_id: str, new_message: dict):
    chat_history = get_chat_history(db, user_id)
    if chat_history:
        chat_history.content.append(new_message)
    else:
        chat_history = ChatHistory(user_id=user_id, content=[new_message])
        db.add(chat_history)
    db.commit()

def send_request(user_input: str, history: List[dict], max_tokens: int = 1500):
    data = {
        "messages": history + [{"role": "user", "content": user_input}],
        "max_tokens": max_tokens
    }
    headers = {
        "Content-Type": "application/json"
    }
    API_ENDPOINT = "http://127.0.0.1:5000/v1/chat/completions"  # Replace with your actual API endpoint
    try:
        response = requests.post(API_ENDPOINT, json=data, headers=headers)
        response.raise_for_status()
        response_data = response.json()
        assistant_response = response_data['choices'][0]['message']['content']
        return assistant_response
    except requests.RequestException as e:
        print(f"An error occurred: {e}")
        return None

# Routes
@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    db = next(get_db())
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/register", status_code=status.HTTP_201_CREATED)
async def register(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(
        or_(User.username == user.username, User.email == user.email)
    ).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username or email already registered")
    hashed_password = get_password_hash(user.password)
    api_key = os.urandom(32).hex()
    new_user = User(username=user.username, email=user.email, hashed_password=hashed_password, api_key=api_key)
    db.add(new_user)
    db.commit()
    return {"message": "User created successfully"}

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("landing.html", {"request": request})

@app.get("/chat")
async def chat(request: Request, current_user: User = Depends(get_current_user)):
    return templates.TemplateResponse("chat.html", {"request": request, "user": current_user})

@app.post("/api/v1/chat")
async def chat_api(
    message: ChatMessage,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    chat_history = get_chat_history(db, current_user.id)
    history = chat_history.content if chat_history else []
    
    assistant_response = send_request(message.user_input, history)
    if assistant_response:
        user_message = {"role": "user", "content": message.user_input}
        assistant_message = {"role": "assistant", "content": assistant_response}
        
        update_chat_history(db, current_user.id, user_message)
        update_chat_history(db, current_user.id, assistant_message)
        
        return JSONResponse(content={"response": assistant_response})
    else:
        raise HTTPException(status_code=500, detail="Failed to get response from the API")

@app.post("/api/v1/image")
async def generate_image(
    image_params: ImageGeneration,
    current_user: User = Depends(get_current_user)
):
    API_URLS = ["http://127.0.0.1:7860", "http://127.0.0.1:7861"]

    payload = {
        "prompt": image_params.prompt,
        "steps": 25,
        "width": image_params.width,
        "height": image_params.height,
    }

    for api_url in API_URLS:
        try:
            response = requests.post(url=f'{api_url}/sdapi/v1/txt2img', json=payload)
            if response.status_code == 200 and 'images' in response.json():
                image_data_base64 = response.json()['images'][0]
                if ',' in image_data_base64:
                    base64_data = image_data_base64.split(',', 1)[1]
                else:
                    base64_data = image_data_base64

                image_data = base64.b64decode(base64_data)
                image = Image.open(io.BytesIO(image_data))
                temp_img = io.BytesIO()
                image.save(temp_img, format="PNG")
                temp_img.seek(0)
                return StreamingResponse(temp_img, media_type="image/png")
        except requests.RequestException as e:
            print(f"Error contacting API at {api_url}: {e}")

    raise HTTPException(status_code=500, detail="Failed to get a valid response from both APIs")

@app.get("/api/v1/user/profile")
async def get_user_profile(current_user: User = Depends(get_current_user)):
    return {
        "username": current_user.username,
        "email": current_user.email,
        "created_at": current_user.created_at
    }

@app.put("/api/v1/user/profile")
async def update_user_profile(
    user_update: UserUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if user_update.email:
        current_user.email = user_update.email
    if user_update.password:
        current_user.hashed_password = get_password_hash(user_update.password)
    db.commit()
    return {"message": "Profile updated successfully"}

@app.get("/api/v1/chat/history")
async def get_chat_history_api(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    chat_history = get_chat_history(db, current_user.id)
    return {"history": chat_history.content if chat_history else []}

@app.delete("/api/v1/chat/history")
async def delete_chat_history(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    chat_history = get_chat_history(db, current_user.id)
    if chat_history:
        db.delete(chat_history)
        db.commit()
    return {"message": "Chat history deleted successfully"}

# Error handling
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": exc.detail},
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"message": "Validation error", "details": exc.errors()},
    )

# Middleware for HTTPS redirect
@app.middleware("http")
async def https_redirect(request: Request, call_next):
    if request.url.scheme == "http" and not os.getenv("DEBUG"):
        url = request.url.replace(scheme="https")
        return RedirectResponse(url, status_code=status.HTTP_301_MOVED_PERMANENTLY)
    return await call_next(request)

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# OpenAPI customization
app.title = "ChatApp API"
app.description = "API for ChatApp with chat and image generation capabilities"
app.version = "1.0.0"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("DEBUG", "false").lower() == "true",
        workers=int(os.getenv("WORKERS", 1)),
        log_level=os.getenv("LOG_LEVEL", "info"),
    )