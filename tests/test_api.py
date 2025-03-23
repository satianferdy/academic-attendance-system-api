import pytest
from unittest.mock import patch, MagicMock
import numpy as np
import json
import io
from PIL import Image

def test_health_check(client):
    """Test the health check endpoint."""
    response = client.get('/api/health')
    assert response.status_code == 200
    assert response.json == {
        'status': 'success', 
        'message': 'Face recognition service is running'
    }

# Process Face Tests
def test_process_face_missing_api_key(client, sample_image):
    """Test process-face endpoint without API key."""
    response = client.post('/api/process-face', data={
        'image': (sample_image, 'test.jpg'),
        'nim': '12345'
    })
    assert response.status_code == 401
    assert 'Unauthorized access' in response.json['message']

@patch('app.services.face_recognition.FaceRecognitionService.process_face')
def test_process_face_success(mock_process, client, sample_image):
    """Test successful face processing."""
    # Create mock return value
    mock_process.return_value = {
        'status': 'success',
        'data': {
            'nim': '12345',
            'embedding': [0.1] * 128,
            'face_image': 'base64_encoded_image',
            'face_box': {
                'x': 0,
                'y': 0,
                'width': 100,
                'height': 100
            },
            'image_info': {
                'format': 'JPEG',
                'size': 1000
            },
            'quality_metrics': {
                'blur_score': 50.5
            }
        }
    }
    
    # Make request
    response = client.post('/api/process-face', 
        data={
            'image': (sample_image, 'test.jpg'),
            'nim': '12345'
        },
        headers={'X-API-Key': 'test-key'}
    )
    
    # Assert response
    assert response.status_code == 200
    assert response.json['status'] == 'success'
    assert 'data' in response.json
    assert response.json['data']['nim'] == '12345'
    assert 'embedding' in response.json['data']
    assert 'quality_metrics' in response.json['data']

def test_process_face_invalid_input(client):
    """Test process-face with missing image."""
    response = client.post('/api/process-face', 
        data={'nim': '12345'},
        headers={'X-API-Key': 'test-key'}
    )
    assert response.status_code == 400
    assert 'image' in response.json['errors']

@patch('app.services.face_recognition.FaceRecognitionService.process_face')
def test_process_face_low_quality(mock_process, client, sample_image):
    """Test process-face with low quality image."""
    mock_process.return_value = {
        'status': 'error',
        'code': 'LOW_QUALITY_IMAGE',
        'message': 'Image is too blurry',
        'blur_score': 15.0
    }
    
    response = client.post('/api/process-face', 
        data={
            'image': (sample_image, 'test.jpg'),
            'nim': '12345'
        },
        headers={'X-API-Key': 'test-key'}
    )
    
    assert response.status_code == 400
    assert response.json['status'] == 'error'
    assert response.json['code'] == 'LOW_QUALITY_IMAGE'

@patch('app.services.face_recognition.FaceRecognitionService.process_face')
def test_process_face_no_face_detected(mock_process, client, sample_image):
    """Test process-face when no face is detected."""
    mock_process.return_value = {
        'status': 'error',
        'code': 'NoFaceDetectedError',
        'message': 'No face detected in the image'
    }
    
    response = client.post('/api/process-face', 
        data={
            'image': (sample_image, 'test.jpg'),
            'nim': '12345'
        },
        headers={'X-API-Key': 'test-key'}
    )
    
    assert response.status_code == 400
    assert response.json['status'] == 'error'
    assert response.json['code'] == 'NoFaceDetectedError'

# Verify Face Tests
def test_verify_face_missing_api_key(client, sample_image):
    """Test verify-face endpoint without API key."""
    response = client.post('/api/verify-face', data={
        'image': (sample_image, 'test.jpg'),
        'class_id': 1,
        'nim': '12345'
    })
    assert response.status_code == 401
    assert 'Unauthorized access' in response.json['message']

@patch('app.services.face_recognition.FaceRecognitionService.verify_face')
def test_verify_face_success(mock_verify, client, sample_image):
    """Test successful face verification."""
    # Mock the verify_face method to return a success response
    mock_verify.return_value = {
        'status': 'success',
        'message': 'Face verified successfully',
        'student_id': 1,
        'nim': '12345',
        'similarity': 0.95
    }
    
    response = client.post(
        '/api/verify-face',
        data={
            'image': (sample_image, 'test.jpg'),
            'class_id': 1,
            'nim': '12345'
        },
        headers={'X-API-Key': 'test-key'}
    )
    assert response.status_code == 200
    assert response.json['status'] == 'success'
    assert response.json['student_id'] == 1
    assert response.json['similarity'] == 0.95

def test_verify_face_invalid_input(client):
    """Test verify-face with missing parameters."""
    response = client.post('/api/verify-face', 
        data={'nim': '12345'},
        headers={'X-API-Key': 'test-key'}
    )
    assert response.status_code == 400
    assert 'image' in response.json['errors']
    
    response = client.post('/api/verify-face', 
        data={
            'image': (io.BytesIO(b'test'), 'test.jpg'),
            'nim': '12345'
        },
        headers={'X-API-Key': 'test-key'}
    )
    assert response.status_code == 400
    assert 'class_id' in response.json['errors']

@patch('app.services.face_recognition.FaceRecognitionService.verify_face')
def test_verify_face_verification_failed(mock_verify, client, sample_image):
    """Test face verification failure."""
    mock_verify.return_value = {
        'status': 'error',
        'message': 'Face verification failed: similarity 0.65 below threshold 0.70'
    }
    
    response = client.post(
        '/api/verify-face',
        data={
            'image': (sample_image, 'test.jpg'),
            'class_id': 1,
            'nim': '12345'
        },
        headers={'X-API-Key': 'test-key'}
    )
    assert response.status_code == 400
    assert response.json['status'] == 'error'
    assert 'similarity' in response.json['message']

@patch('app.services.face_recognition.FaceRecognitionService.verify_face')
def test_verify_face_not_enrolled(mock_verify, client, sample_image):
    """Test verification for student not enrolled in class."""
    mock_verify.return_value = {
        'status': 'error',
        'message': 'Student with NIM 12345 is not enrolled in this class'
    }
    
    response = client.post(
        '/api/verify-face',
        data={
            'image': (sample_image, 'test.jpg'),
            'class_id': 1,
            'nim': '12345'
        },
        headers={'X-API-Key': 'test-key'}
    )
    assert response.status_code == 400
    assert response.json['status'] == 'error'
    assert 'not enrolled' in response.json['message']

# Validate Quality Tests
def test_validate_quality_missing_api_key(client, sample_image):
    """Test validate-quality endpoint without API key."""
    response = client.post('/api/validate-quality', data={
        'image': (sample_image, 'test.jpg')
    })
    assert response.status_code == 401
    assert 'Unauthorized access' in response.json['message']

@patch('app.services.face_recognition.FaceRecognitionService.validate_quality')
def test_validate_quality_success(mock_validate, client, sample_image):
    """Test successful quality validation."""
    mock_validate.return_value = {
        'status': 'success',
        'data': {
            'quality_metrics': {
                'blur_score': 75.5
            }
        }
    }
    
    response = client.post(
        '/api/validate-quality',
        data={
            'image': (sample_image, 'test.jpg')
        },
        headers={'X-API-Key': 'test-key'}
    )
    assert response.status_code == 200
    assert response.json['status'] == 'success'
    assert 'data' in response.json
    assert 'quality_metrics' in response.json['data']
    assert 'blur_score' in response.json['data']['quality_metrics']

def test_validate_quality_invalid_input(client):
    """Test validate-quality with missing image."""
    response = client.post('/api/validate-quality', 
        data={},
        headers={'X-API-Key': 'test-key'}
    )
    assert response.status_code == 400
    assert 'image' in response.json['errors']

@patch('app.services.face_recognition.FaceRecognitionService.validate_quality')
def test_validate_quality_no_face(mock_validate, client, sample_image):
    """Test quality validation when no face is detected."""
    mock_validate.return_value = {
        'status': 'error',
        'code': 'NoFaceDetectedError',
        'message': 'No face detected in the image'
    }
    
    response = client.post(
        '/api/validate-quality',
        data={
            'image': (sample_image, 'test.jpg')
        },
        headers={'X-API-Key': 'test-key'}
    )
    assert response.status_code == 400
    assert response.json['status'] == 'error'
    assert response.json['code'] == 'NoFaceDetectedError'

# Face Recognition Service Tests
@pytest.fixture
def face_recognition_service(mock_face_detector, mock_face_embedder, database):
    """Create a FaceRecognitionService instance for testing."""
    from app.services.face_recognition import FaceRecognitionService
    return FaceRecognitionService(
        face_detector=mock_face_detector,
        face_embedder=mock_face_embedder,
        recognition_threshold=0.7,
        db=database
    )

def test_face_service_calculate_blur_score(face_recognition_service):
    """Test the _calculate_blur_score method."""
    # Create a test image with known blur characteristics
    image = Image.new('RGB', (160, 160))
    blur_score = face_recognition_service._calculate_blur_score(image)
    assert isinstance(blur_score, float)

def test_face_service_compare_embeddings(face_recognition_service):
    """Test the _compare_embeddings method."""
    # Create two random embeddings
    embedding1 = np.random.randn(128)
    embedding1 = embedding1 / np.linalg.norm(embedding1)
    
    # Same embedding should have similarity 1.0
    similarity = face_recognition_service._compare_embeddings(embedding1, embedding1)
    assert similarity > 0.99
    
    # Different embeddings should have lower similarity
    embedding2 = np.random.randn(128)
    embedding2 = embedding2 / np.linalg.norm(embedding2)
    similarity = face_recognition_service._compare_embeddings(embedding1, embedding2)
    assert 0 <= similarity <= 1.0

def test_face_service_process_face(face_recognition_service, sample_image):
    """Test the process_face method directly."""
    # Setup mocks for embedding
    face_recognition_service.face_embedder.get_embedding.return_value = np.ones(128)
    
    # Override _calculate_blur_score to return a good value
    face_recognition_service._calculate_blur_score = MagicMock(return_value=50.0)
    
    # Call the method
    result = face_recognition_service.process_face(sample_image.read(), '12345')
    
    # Assertions
    assert result['status'] == 'success'
    assert result['data']['nim'] == '12345'
    assert 'embedding' in result['data']
    assert 'face_image' in result['data']
    assert 'quality_metrics' in result['data']
    assert result['data']['quality_metrics']['blur_score'] == 50.0

def test_face_service_validate_quality(face_recognition_service, sample_image):
    """Test the validate_quality method directly."""
    # Override _calculate_blur_score to return a good value
    face_recognition_service._calculate_blur_score = MagicMock(return_value=75.0)
    
    # Call the method
    result = face_recognition_service.validate_quality(sample_image.read())
    
    # Assertions
    assert result['status'] == 'success'
    assert 'quality_metrics' in result['data']
    assert result['data']['quality_metrics']['blur_score'] == 75.0