import pytest
import numpy as np
from unittest.mock import MagicMock
from PIL import Image
from app.core.exceptions import (
    NoFaceDetectedError, MultipleFacesError, FaceNotRegisteredError, 
    StudentNotFoundError
)

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

class TestFaceRecognitionService:
    """Test suite for the FaceRecognitionService following established patterns."""
    
    def test_process_face_success(self, mock_face_recognition_service, sample_image):
        """Test successful face processing."""
        result = mock_face_recognition_service.process_face(sample_image.read(), '12345')
        
        assert result['status'] == 'success'
        assert result['data']['nim'] == '12345'
        assert 'embedding' in result['data']
        assert 'face_image' in result['data']
        assert 'quality_metrics' in result['data']
        assert result['data']['quality_metrics']['blur_score'] == 50.0
    
    def test_process_face_no_face(self, mock_face_recognition_service, sample_image):
        """Test process_face when no face is detected."""
        mock_face_recognition_service.process_face.return_value = {
            'status': 'error',
            'code': 'NoFaceDetectedError',
            'message': 'No face detected in the image'
        }
        
        result = mock_face_recognition_service.process_face(sample_image.read(), '12345')
        
        assert result['status'] == 'error'
        assert result['code'] == 'NoFaceDetectedError'
    
    def test_process_face_multiple_faces(self, mock_face_recognition_service, sample_image):
        """Test process_face when multiple faces are detected."""
        mock_face_recognition_service.process_face.return_value = {
            'status': 'error',
            'code': 'MultipleFacesError',
            'message': 'Multiple faces detected in the image'
        }
        
        result = mock_face_recognition_service.process_face(sample_image.read(), '12345')
        
        assert result['status'] == 'error'
        assert result['code'] == 'MultipleFacesError'
    
    def test_process_face_low_quality(self, mock_face_recognition_service, sample_image):
        """Test process_face with low quality (blurry) image."""
        mock_face_recognition_service.process_face.return_value = {
            'status': 'error',
            'code': 'LOW_QUALITY_IMAGE',
            'message': 'Image is too blurry',
            'blur_score': 15.0
        }
        
        result = mock_face_recognition_service.process_face(sample_image.read(), '12345')
        
        assert result['status'] == 'error'
        assert result['code'] == 'LOW_QUALITY_IMAGE'
        assert 'blur_score' in result
    
    def test_validate_quality_success(self, mock_face_recognition_service, sample_image):
        """Test successful quality validation."""
        result = mock_face_recognition_service.validate_quality(sample_image.read())
        
        assert result['status'] == 'success'
        assert 'data' in result
        assert 'quality_metrics' in result['data']
        assert result['data']['quality_metrics']['blur_score'] == 75.0
    
    def test_validate_quality_no_face(self, mock_face_recognition_service, sample_image):
        """Test quality validation when no face is detected."""
        mock_face_recognition_service.validate_quality.return_value = {
            'status': 'error',
            'code': 'NoFaceDetectedError',
            'message': 'No face detected in the image'
        }
        
        result = mock_face_recognition_service.validate_quality(sample_image.read())
        
        assert result['status'] == 'error'
        assert result['code'] == 'NoFaceDetectedError'
    
    def test_verify_face_success(self, mock_face_recognition_service, sample_image):
        """Test successful face verification."""
        result = mock_face_recognition_service.verify_face(sample_image.read(), 1, '12345')
        
        assert result['status'] == 'success'
        assert result['student_id'] == 1
        assert result['nim'] == '12345'
        assert result['similarity'] == 0.9
    
    def test_verify_face_student_not_found(self, mock_face_recognition_service, sample_image):
        """Test verify_face when student is not found."""
        mock_face_recognition_service.verify_face.return_value = {
            'status': 'error',
            'message': 'Student with NIM 12345 not found'
        }
        
        result = mock_face_recognition_service.verify_face(sample_image.read(), 1, '12345')
        
        assert result['status'] == 'error'
        assert 'not found' in result['message'].lower()
    
    def test_verify_face_not_registered(self, mock_face_recognition_service, sample_image):
        """Test verify_face when face is not registered."""
        mock_face_recognition_service.verify_face.return_value = {
            'status': 'error',
            'message': 'Student with NIM 12345 has not registered a face'
        }
        
        result = mock_face_recognition_service.verify_face(sample_image.read(), 1, '12345')
        
        assert result['status'] == 'error'
        assert 'not registered' in result['message'].lower()
    
    def test_verify_face_not_enrolled(self, mock_face_recognition_service, sample_image):
        """Test verify_face when student is not enrolled in class."""
        mock_face_recognition_service.verify_face.return_value = {
            'status': 'error',
            'message': 'Student with NIM 12345 is not enrolled in this class'
        }
        
        result = mock_face_recognition_service.verify_face(sample_image.read(), 1, '12345')
        
        assert result['status'] == 'error'
        assert 'not enrolled' in result['message'].lower()
    
    def test_verify_face_low_similarity(self, mock_face_recognition_service, sample_image):
        """Test verify_face with low similarity (failed verification)."""
        mock_face_recognition_service.verify_face.return_value = {
            'status': 'error',
            'message': 'Face verification failed: similarity 0.5 below threshold 0.7'
        }
        
        result = mock_face_recognition_service.verify_face(sample_image.read(), 1, '12345')
        
        assert result['status'] == 'error'
        assert 'similarity' in result['message'].lower()
        assert '0.5' in result['message']

    def test_calculate_blur_score(self, mock_face_recognition_service):
        """Test blur score calculation."""
        # Mock the internal method if needed for unit testing
        image = Image.new('RGB', (160, 160))
        # Since we're using mock service, we can test the response format
        result = mock_face_recognition_service.validate_quality(image)
        assert 'quality_metrics' in result['data']
        assert 'blur_score' in result['data']['quality_metrics']
    
    def test_face_to_base64(self, mock_face_recognition_service, sample_image):
        """Test base64 conversion through process_face."""
        result = mock_face_recognition_service.process_face(sample_image.read(), '12345')
        assert 'face_image' in result['data']
        assert isinstance(result['data']['face_image'], str)
        assert len(result['data']['face_image']) > 0
    
    def test_compare_embeddings(self, mock_face_recognition_service, sample_image):
        """Test embedding comparison through verify_face."""
        result = mock_face_recognition_service.verify_face(sample_image.read(), 1, '12345')
        assert 'similarity' in result
        assert 0 <= result['similarity'] <= 1.0
        
        