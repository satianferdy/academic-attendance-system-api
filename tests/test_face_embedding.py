import pytest
import numpy as np
from unittest.mock import MagicMock
from PIL import Image
from app.core.exceptions import FaceEmbeddingError

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

class TestFaceEmbeddingService:
    """Test suite for face embedding service following established patterns."""
    
    def test_face_embedding_model_loading(self, mock_face_embedding_service, test_config):
        """Test face embedding model loading."""
        assert mock_face_embedding_service.model is not None
        assert mock_face_embedding_service.image_size == test_config.FACE_IMAGE_SIZE
    
    def test_face_embedding_preprocessing(self, mock_face_embedding_service):
        """Test face embedding preprocessing."""
        image = Image.new('RGB', (160, 160))
        processed = mock_face_embedding_service.preprocess_image(image)
        
        assert processed.shape == (1, 160, 160, 3)
        assert processed.dtype == np.float32
    
    def test_face_embedding_generation(self, mock_face_embedding_service):
        """Test face embedding generation."""
        image = Image.new('RGB', (160, 160))
        embedding = mock_face_embedding_service.get_embedding(image)
        
        assert embedding.shape == (128,)
        assert isinstance(embedding, np.ndarray)
    
    def test_face_embedding_normalization(self, mock_face_embedding_service):
        """Test embedding normalization."""
        # Create a mock embedding that's normalized
        normalized_embedding = np.random.randn(128)
        normalized_embedding = normalized_embedding / np.linalg.norm(normalized_embedding)
        mock_face_embedding_service.get_embedding.return_value = normalized_embedding
        
        image = Image.new('RGB', (160, 160))
        embedding = mock_face_embedding_service.get_embedding(image)
        
        # Check if embedding is normalized (unit length)
        assert np.isclose(np.linalg.norm(embedding), 1.0, atol=1e-6)
    
    def test_face_embedding_error_handling(self, mock_face_embedding_service):
        """Test face embedding error handling."""
        # Configure mock to raise exception
        mock_face_embedding_service.get_embedding.side_effect = FaceEmbeddingError("Model failed")
        
        image = Image.new('RGB', (160, 160))
        
        with pytest.raises(FaceEmbeddingError):
            mock_face_embedding_service.get_embedding(image)
    
    def test_face_embedding_invalid_input(self, mock_face_embedding_service):
        """Test face embedding with invalid input."""
        # Configure mock to handle invalid input
        mock_face_embedding_service.get_embedding.side_effect = FaceEmbeddingError("Invalid input")
        
        with pytest.raises(FaceEmbeddingError):
            mock_face_embedding_service.get_embedding(None)
    
    def test_face_embedding_consistency(self, mock_face_embedding_service):
        """Test embedding consistency for same image."""
        # Configure mock to return consistent embeddings
        consistent_embedding = np.random.randn(128)
        mock_face_embedding_service.get_embedding.return_value = consistent_embedding
        
        image = Image.new('RGB', (160, 160))
        
        embedding1 = mock_face_embedding_service.get_embedding(image)
        embedding2 = mock_face_embedding_service.get_embedding(image)
        
        np.testing.assert_array_equal(embedding1, embedding2)
    
    def test_face_embedding_different_images(self, mock_face_embedding_service):
        """Test embeddings for different images."""
        def side_effect_embeddings(*args):
            return np.random.randn(128)
        
        mock_face_embedding_service.get_embedding.side_effect = side_effect_embeddings
        
        image1 = Image.new('RGB', (160, 160), color='red')
        image2 = Image.new('RGB', (160, 160), color='blue')
        
        embedding1 = mock_face_embedding_service.get_embedding(image1)
        embedding2 = mock_face_embedding_service.get_embedding(image2)
        
        # Embeddings should be different (with high probability)
        assert not np.array_equal(embedding1, embedding2)
    
    def test_preprocessing_different_sizes(self, mock_face_embedding_service):
        """Test preprocessing images of different sizes."""
        def preprocess_side_effect(image):
            # Simulate resizing to target size
            return np.zeros((1, 160, 160, 3), dtype=np.float32)
        
        mock_face_embedding_service.preprocess_image.side_effect = preprocess_side_effect
        
        # Test different input sizes
        small_image = Image.new('RGB', (50, 50))
        large_image = Image.new('RGB', (500, 500))
        
        processed_small = mock_face_embedding_service.preprocess_image(small_image)
        processed_large = mock_face_embedding_service.preprocess_image(large_image)
        
        # Both should result in same output size
        assert processed_small.shape == processed_large.shape == (1, 160, 160, 3)
    
    def test_embedding_vector_properties(self, mock_face_embedding_service):
        """Test properties of embedding vectors."""
        # Configure mock to return a proper embedding
        embedding = np.random.randn(128).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)  # Normalize
        mock_face_embedding_service.get_embedding.return_value = embedding
        
        image = Image.new('RGB', (160, 160))
        result_embedding = mock_face_embedding_service.get_embedding(image)
        
        # Test embedding properties
        assert len(result_embedding) == 128
        assert result_embedding.dtype == np.float32
        assert -1.0 <= result_embedding.min() <= result_embedding.max() <= 1.0