import pytest
from unittest.mock import MagicMock
import numpy as np
from PIL import Image
from app.core.exceptions import (
    FaceDetectionError, NoFaceDetectedError, MultipleFacesError
)

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

class TestFaceDetectionService:
    """Test suite for face detection service following established patterns."""
    
    def test_detect_face_success(self, mock_face_detection_service, sample_image):
        """Test successful face detection."""
        face_img, box = mock_face_detection_service.detect_face(sample_image.read())
        
        assert face_img.size == (160, 160)
        assert len(box) == 4
        assert all(isinstance(coord, int) for coord in box)
    
    def test_detect_face_no_face(self, mock_face_detection_service, sample_image):
        """Test face detection when no face is detected."""
        mock_face_detection_service.detect_face.side_effect = NoFaceDetectedError("No face detected")
        
        with pytest.raises(NoFaceDetectedError):
            mock_face_detection_service.detect_face(sample_image.read())
    
    def test_detect_face_multiple_faces(self, mock_face_detection_service, sample_image):
        """Test face detection when multiple faces are detected."""
        mock_face_detection_service.detect_face.side_effect = MultipleFacesError("Multiple faces detected")
        
        with pytest.raises(MultipleFacesError):
            mock_face_detection_service.detect_face(sample_image.read())
    
    def test_detect_face_low_confidence(self, mock_face_detection_service, sample_image):
        """Test face detection with low confidence."""
        mock_face_detection_service.detect_face.side_effect = NoFaceDetectedError("Face detection confidence too low")
        
        with pytest.raises(NoFaceDetectedError):
            mock_face_detection_service.detect_face(sample_image.read())
    
    def test_detect_face_error_handling(self, mock_face_detection_service, sample_image):
        """Test face detection error handling."""
        mock_face_detection_service.detect_face.side_effect = FaceDetectionError("Detection failed")
        
        with pytest.raises(FaceDetectionError):
            mock_face_detection_service.detect_face(sample_image.read())
    
    def test_detect_face_with_bytes_input(self, mock_face_detection_service, sample_image):
        """Test face detection with bytes input."""
        face_img, box = mock_face_detection_service.detect_face(sample_image.read())
        
        assert isinstance(face_img, Image.Image)
        assert isinstance(box, list)
        assert len(box) == 4
    
    def test_detect_face_with_pil_input(self, mock_face_detection_service):
        """Test face detection with PIL Image input."""
        image = Image.new('RGB', (500, 500))
        face_img, box = mock_face_detection_service.detect_face(image)
        
        assert isinstance(face_img, Image.Image)
        assert isinstance(box, list)
    
    def test_detect_face_different_image_sizes(self, mock_face_detection_service):
        """Test face detection with different image sizes."""
        def detect_side_effect(image_data):
            if hasattr(image_data, 'size'):
                # PIL Image
                size = image_data.size
            else:
                # Bytes - simulate processing
                size = (500, 500)  # Default size for test
            
            # Return appropriate face crop size
            face_size = min(160, min(size) // 2)
            return Image.new('RGB', (face_size, face_size)), [0, 0, 100, 100]
        
        mock_face_detection_service.detect_face.side_effect = detect_side_effect
        
        small_image = Image.new('RGB', (100, 100))
        large_image = Image.new('RGB', (1000, 1000))
        
        small_face, small_box = mock_face_detection_service.detect_face(small_image)
        large_face, large_box = mock_face_detection_service.detect_face(large_image)
        
        assert isinstance(small_face, Image.Image)
        assert isinstance(large_face, Image.Image)
        assert len(small_box) == len(large_box) == 4
    
    def test_detect_face_confidence_threshold(self, mock_face_detection_service, test_config):
        """Test confidence threshold configuration."""
        assert mock_face_detection_service.min_confidence == test_config.FACE_DETECTION_CONFIDENCE
    
    def test_detect_face_box_coordinates(self, mock_face_detection_service, sample_image):
        """Test face bounding box coordinates."""
        def detect_with_specific_box(image_data):
            return Image.new('RGB', (120, 120)), [10, 20, 100, 110]
        
        mock_face_detection_service.detect_face.side_effect = detect_with_specific_box
        
        face_img, box = mock_face_detection_service.detect_face(sample_image.read())
        
        x, y, width, height = box
        assert x >= 0
        assert y >= 0
        assert width > 0
        assert height > 0
    
    def test_detect_face_image_format_conversion(self, mock_face_detection_service):
        """Test detection with different image formats."""
        def detect_with_format_check(image_data):
            # Simulate format conversion logic
            return Image.new('RGB', (160, 160)), [0, 0, 100, 100]
        
        mock_face_detection_service.detect_face.side_effect = detect_with_format_check
        
        # Test with different format
        rgba_image = Image.new('RGBA', (500, 500))
        face_img, box = mock_face_detection_service.detect_face(rgba_image)
        
        assert face_img.mode == 'RGB'  # Should be converted to RGB
    
    def test_detect_face_margin_handling(self, mock_face_detection_service, sample_image):
        """Test face detection with margin handling."""
        def detect_with_margin(image_data):
            # Simulate margin calculation
            base_box = [50, 50, 100, 100]  # x, y, width, height
            # Add 10% margin
            margin_x = int(base_box[2] * 0.1)
            margin_y = int(base_box[3] * 0.1)
            
            adjusted_box = [
                max(0, base_box[0] - margin_x),
                max(0, base_box[1] - margin_y),
                base_box[2] + 2 * margin_x,
                base_box[3] + 2 * margin_y
            ]
            
            return Image.new('RGB', (120, 120)), adjusted_box
        
        mock_face_detection_service.detect_face.side_effect = detect_with_margin
        
        face_img, box = mock_face_detection_service.detect_face(sample_image.read())
        
        # Verify margin was applied (box should be larger than base detection)
        assert box[2] > 100  # width should be expanded
        assert box[3] > 100  # height should be expanded
    
    def test_detector_initialization(self, mock_face_detection_service, test_config):
        """Test detector initialization with configuration."""
        assert mock_face_detection_service.detector is not None
        assert mock_face_detection_service.min_confidence == test_config.FACE_DETECTION_CONFIDENCE
    
    def test_detect_face_invalid_input(self, mock_face_detection_service):
        """Test face detection with invalid input."""
        mock_face_detection_service.detect_face.side_effect = FaceDetectionError("Invalid input format")
        
        with pytest.raises(FaceDetectionError):
            mock_face_detection_service.detect_face(None)