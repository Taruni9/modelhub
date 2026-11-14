import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env from backend directory if present
env_file = BASE_DIR / ".env"
if env_file.is_file():
    load_dotenv(env_file)
else:
    load_dotenv()

APP_NAME = os.getenv(
    "APP_NAME",
    "ML Model Serving Platform"
)

# Upload directory: anchored to backend/uploads by default
raw_upload_dir = os.getenv("UPLOAD_DIR", "uploads")
if os.path.isabs(raw_upload_dir):
    UPLOAD_DIR = raw_upload_dir
else:
    UPLOAD_DIR = str(BASE_DIR / raw_upload_dir)

# Database URL: anchored to backend/ml_platform.db by default
raw_db_url = os.getenv("DATABASE_URL")
if raw_db_url:
    DATABASE_URL = raw_db_url
else:
    db_path = str(BASE_DIR / "ml_platform.db")
    DATABASE_URL = f"sqlite:///{db_path}"

MAX_FILE_SIZE_MB = int(
    os.getenv("MAX_FILE_SIZE_MB", "100")
)

# Cloud Storage (AWS S3) Configuration
STORAGE_BACKEND = os.getenv("STORAGE_BACKEND", "local").lower()
AWS_S3_BUCKET_NAME = os.getenv("AWS_S3_BUCKET_NAME", "")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")

# JWT Authentication Configuration
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "modelcloud-super-secret-jwt-key-2026-auth")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = int(os.getenv("JWT_EXPIRATION_HOURS", "72"))
