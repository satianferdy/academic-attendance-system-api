import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from PIL import Image
from app.services.face_embedding import FaceNetEmbedding, FaceEmbeddingError

@patch('tensorflow.keras.models.load_model')
def test_face_embedding_model_loading(mock_load, test_config):
    mock_model = MagicMock()
    mock_load.return_value = mock_model
    
    embedder = FaceNetEmbedding(test_config.FACE_RECOGNITION_MODEL_PATH)
    assert embedder.model is mock_model

@patch('tensorflow.keras.models.load_model')
def test_face_embedding_preprocessing(mock_load):
    mock_model = MagicMock()
    mock_load.return_value = mock_model
    embedder = FaceNetEmbedding('dummy_path')
    image = Image.new('RGB', (160, 160))
    processed = embedder.preprocess_image(image)
    assert processed.shape == (1, 160, 160, 3)

@patch('tensorflow.keras.models.load_model')
def test_face_embedding_generation(mock_load):
    mock_model = MagicMock()
    mock_model.predict.return_value = [np.random.randn(128)]
    mock_load.return_value = mock_model
    
    embedder = FaceNetEmbedding('dummy_path')
    embedding = embedder.get_embedding(Image.new('RGB', (160, 160)))
    assert embedding.shape == (128,)
    assert np.isclose(np.linalg.norm(embedding), 1.0, atol=1e-6)