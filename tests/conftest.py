import pytest
from flask import Flask
from unittest.mock import MagicMock, patch
import numpy as np
from PIL import Image
import io
import os

from app.config import Config
from app.core.dependencies import Container, TestContainer
from app.models.database import Base, Database
from app import create_app
from tests.mocks import MockFaceDetector, MockFaceEmbedder, MockDatabase

@pytest.fixture(scope="session")
def test_config():
    class TestConfig(Config):
        SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
        FACE_RECOGNITION_MODEL_PATH = 'dummy_model.h5'
        UPLOAD_FOLDER = 'test_storage'
        DEBUG = True
        API_KEY = 'test-key'
        TESTING = True
    return TestConfig()

@pytest.fixture(scope="function")
def test_app(test_config):
    app = create_app(testing=True)
    app.config.from_object(test_config)
    
    # Create test container dengan mock overrides
    container = TestContainer()
    app.container = container
    
    container.wire(modules=['app.api.routes'])
    
    yield app

@pytest.fixture(scope="function")
def client(test_app):
    return test_app.test_client()

@pytest.fixture(scope="function")
def database(test_app):
    db = Database(test_app.config['SQLALCHEMY_DATABASE_URI'])
    engine = db.engine
    Base.metadata.create_all(bind=engine)
    yield db
    Base.metadata.drop_all(bind=engine)
    db.close_session()

@pytest.fixture
def mock_face_detector():
    detector = MagicMock()
    detector.detect_face.return_value = (
        Image.new('RGB', (160, 160)), 
        [0, 0, 100, 100]
    )
    return detector

@pytest.fixture
def mock_face_embedder():
    embedder = MagicMock()
    embedder.get_embedding.return_value = np.random.randn(128)
    return embedder

@pytest.fixture
def sample_image():
    image = Image.new('RGB', (500, 500), color='red')
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)
    return img_byte_arr

@pytest.fixture
def mock_face_recognition_service():
    """Create a mock face recognition service with default responses."""
    service = MagicMock()
    
    # Default successful responses
    service.process_face.return_value = {
        'status': 'success',
        'data': {
            'nim': '12345',
            'embedding': [0.1] * 128,
            'face_image': 'base64_encoded_image',
            'face_box': {'x': 0, 'y': 0, 'width': 100, 'height': 100},
            'image_info': {'format': 'JPEG', 'size': 1000},
            'quality_metrics': {'blur_score': 50.0}
        }
    }
    
    service.verify_face.return_value = {
        'status': 'success',
        'message': 'Face verified successfully',
        'student_id': 1,
        'nim': '12345',
        'similarity': 0.9
    }
    
    service.validate_quality.return_value = {
        'status': 'success',
        'data': {
            'quality_metrics': {'blur_score': 75.0}
        }
    }
    
    return service

@pytest.fixture
def mock_face_embedding_service(test_config):
    """Create mock face embedding service following established patterns."""
    service = MagicMock()
    
    # Default successful responses
    service.get_embedding.return_value = np.random.randn(128)
    service.preprocess_image.return_value = np.zeros((1, 160, 160, 3), dtype=np.float32)
    service.model = MagicMock()
    service.image_size = test_config.FACE_IMAGE_SIZE
    
    return service

@pytest.fixture
def mock_face_detection_service(test_config):
    """Create mock face detection service following established patterns."""
    service = MagicMock()
    
    # Default successful response
    service.detect_face.return_value = (
        Image.new('RGB', (160, 160)), 
        [0, 0, 100, 100]
    )
    service.min_confidence = test_config.FACE_DETECTION_CONFIDENCE
    service.detector = MagicMock()
    
    return service

@pytest.fixture(autouse=True)
def cleanup_upload_folder(test_config):
    yield
    if os.path.exists(test_config.UPLOAD_FOLDER):
        for f in os.listdir(test_config.UPLOAD_FOLDER):
            os.remove(os.path.join(test_config.UPLOAD_FOLDER, f))