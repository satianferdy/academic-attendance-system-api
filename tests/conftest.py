import pytest
from flask import Flask
from unittest.mock import MagicMock, patch
import numpy as np
from PIL import Image
import io
import os

from app.config import Config
from app.core.dependencies import Container
from app.models.database import Base, Database
from app import create_app  # <-- Impor create_app dari aplikasi utama


@pytest.fixture(scope="session")
def test_config():
    class TestConfig(Config):
        SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
        FACE_RECOGNITION_MODEL_PATH = 'dummy_model.h5'
        UPLOAD_FOLDER = 'test_storage'
        DEBUG = True
        API_KEY = 'test-key'
    return TestConfig()


@pytest.fixture(scope="module")
def test_app(test_config):
    # Gunakan create_app untuk menginisialisasi aplikasi Flask
    app = create_app(testing=True)
    app.config.from_object(test_config)
    
    # Setup dependency injection container
    container = Container()
    app.container = container
    
    yield app


@pytest.fixture(scope="module")
def client(test_app):
    # Kembalikan test client dari aplikasi Flask
    return test_app.test_client()


@pytest.fixture(scope="module")
def database(test_app):
    # Gunakan database dari aplikasi Flask
    db = Database(test_app.config['SQLALCHEMY_DATABASE_URI'])
    engine = db.engine
    
    # Buat semua tabel
    Base.metadata.create_all(bind=engine)
    
    yield db
    
    # Hapus semua tabel setelah tes selesai
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


@pytest.fixture(autouse=True)
def cleanup_upload_folder(test_config):
    yield
    # Cleanup test storage after each test
    if os.path.exists(test_config.UPLOAD_FOLDER):
        for f in os.listdir(test_config.UPLOAD_FOLDER):
            os.remove(os.path.join(test_config.UPLOAD_FOLDER, f))