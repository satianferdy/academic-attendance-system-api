import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
BASE_DIR = Path(__file__).resolve().parent.parent
env_path = BASE_DIR / '.env'

# Load .env file if exists
if env_path.exists():
    load_dotenv(env_path)

class Config:
    """Application configuration class"""
    # Server settings
    DEBUG = os.getenv('DEBUG', 'False') == 'True'
    SECRET_KEY = os.getenv('SECRET_KEY')  # No default - must be set in .env
    API_KEY = os.getenv('API_KEY')  # No default - must be set in .env
    
    # Database settings
    DB_HOST = os.getenv('DB_HOST', 'localhost')
    DB_PORT = os.getenv('DB_PORT', '3306')
    DB_USER = os.getenv('DB_USER', 'root')
    DB_PASSWORD = os.getenv('DB_PASSWORD', '')
    DB_NAME = os.getenv('DB_NAME', 'academic_attendance_system_app')
    SQLALCHEMY_DATABASE_URI = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    
    # Face recognition settings
    FACE_RECOGNITION_MODEL_PATH = os.getenv('FACE_RECOGNITION_MODEL_PATH', 'models/facenet_keras.h5')
    FACE_IMAGE_SIZE = (160, 160)  # FaceNet requires 160x160 images
    FACE_DETECTION_CONFIDENCE = 0.95  # Minimum confidence for face detection
    FACE_RECOGNITION_THRESHOLD = 0.7  # Threshold for face recognition (lower = more strict)
    
    # Storage settings
    UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'storage/faces')
    MAX_CONTENT_LENGTH = 5 * 1024 * 1024  # 5MB max size for uploaded images
    
class TestConfig(Config):
    """Test configuration class"""
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    Testing = True