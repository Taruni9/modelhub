from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .database import Base, engine
from .api.models import router as models_router
from .api.datasets import router as datasets_router


# Create database tables
Base.metadata.create_all(bind=engine)


app = FastAPI(
    title="Cloud-Based ML Model Serving Platform",
    description=(
        "Platform for uploading, validating, "
        "managing and serving machine learning models."
    ),
    version="1.0.0"
)


# Allow React frontend to communicate with FastAPI
app.add_middleware(
    CORSMiddleware,
   allow_origins=[
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
    "http://localhost:3000",
    "http://127.0.0.1:3000"
],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# Register model API
app.include_router(models_router)
app.include_router(datasets_router)


@app.get("/")
def root():
    return {
        "message": "ML Model Serving Platform API",
        "status": "running",
        "version": "1.0.0"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }