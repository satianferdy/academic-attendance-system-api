import pytest
from unittest.mock import MagicMock
import numpy as np
import json
import io
from PIL import Image

@pytest.fixture
def mock_face_service(test_app):
    """Override face recognition service dengan MagicMock."""
    mock_service = MagicMock()
    
    # Set default return values
    mock_service.process_face.return_value = {
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
    
    mock_service.verify_face.return_value = {
        'status': 'success',
        'message': 'Face verified successfully',
        'student_id': 1,
        'nim': '12345',
        'similarity': 0.95
    }
    
    mock_service.validate_quality.return_value = {
        'status': 'success',
        'data': {
            'quality_metrics': {
                'blur_score': 75.5
            }
        }
    }
    
    # Override di container
    test_app.container.face_recognition_service.override(mock_service)
    yield mock_service
    # Reset setelah test
    test_app.container.face_recognition_service.reset_override()

def test_health_check(client):
    """Test the health check endpoint."""
    response = client.get('/api/health')
    assert response.status_code == 200
    data = response.get_json()  # Fixed: use get_json() instead of .json
    assert data == {
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
    data = response.get_json()
    assert 'Unauthorized access' in data['message']

def test_process_face_success(client, sample_image, mock_face_service):
    """Test successful face processing."""
    response = client.post('/api/process-face', 
        data={
            'image': (sample_image, 'test.jpg'),
            'nim': '12345'
        },
        headers={'X-API-Key': 'test-key'}
    )
    
    assert response.status_code == 200
    data = response.get_json()
    assert data['status'] == 'success'
    assert 'data' in data
    assert data['data']['nim'] == '12345'
    assert 'embedding' in data['data']
    assert 'quality_metrics' in data['data']

def test_process_face_invalid_input(client):
    """Test process-face with missing image."""
    response = client.post('/api/process-face', 
        data={'nim': '12345'},
        headers={'X-API-Key': 'test-key'}
    )
    assert response.status_code == 400
    data = response.get_json()
    assert 'image' in data['errors']

def test_process_face_low_quality(client, sample_image, mock_face_service):
    """Test process-face with low quality image."""
    mock_face_service.process_face.return_value = {
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
    data = response.get_json()
    assert data['status'] == 'error'
    assert data['code'] == 'LOW_QUALITY_IMAGE'

def test_process_face_no_face_detected(client, sample_image, mock_face_service):
    """Test process-face when no face is detected."""
    mock_face_service.process_face.return_value = {
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
    data = response.get_json()
    assert data['status'] == 'error'
    assert data['code'] == 'NoFaceDetectedError'

# Verify Face Tests
def test_verify_face_missing_api_key(client, sample_image):
    """Test verify-face endpoint without API key."""
    response = client.post('/api/verify-face', data={
        'image': (sample_image, 'test.jpg'),
        'class_id': 1,
        'nim': '12345'
    })
    assert response.status_code == 401
    data = response.get_json()
    assert 'Unauthorized access' in data['message']

def test_verify_face_success(client, sample_image, mock_face_service):
    """Test successful face verification."""
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
    data = response.get_json()
    assert data['status'] == 'success'
    assert data['student_id'] == 1
    assert data['similarity'] == 0.95

def test_verify_face_invalid_input(client):
    """Test verify-face with missing parameters."""
    response = client.post('/api/verify-face', 
        data={'nim': '12345'},
        headers={'X-API-Key': 'test-key'}
    )
    assert response.status_code == 400
    data = response.get_json()
    assert 'image' in data['errors']
    
    # Create a proper BytesIO object for the image test
    test_image = io.BytesIO(b'test')
    response = client.post('/api/verify-face', 
        data={
            'image': (test_image, 'test.jpg'),
            'nim': '12345'
        },
        headers={'X-API-Key': 'test-key'}
    )
    assert response.status_code == 400
    data = response.get_json()
    assert 'class_id' in data['errors']

def test_verify_face_verification_failed(client, sample_image, mock_face_service):
    """Test face verification failure."""
    mock_face_service.verify_face.return_value = {
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
    data = response.get_json()
    assert data['status'] == 'error'
    assert 'similarity' in data['message']

def test_verify_face_not_enrolled(client, sample_image, mock_face_service):
    """Test verification for student not enrolled in class."""
    mock_face_service.verify_face.return_value = {
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
    data = response.get_json()
    assert data['status'] == 'error'
    assert 'not enrolled' in data['message']

# Validate Quality Tests
def test_validate_quality_missing_api_key(client, sample_image):
    """Test validate-quality endpoint without API key."""
    response = client.post('/api/validate-quality', data={
        'image': (sample_image, 'test.jpg')
    })
    assert response.status_code == 401
    data = response.get_json()
    assert 'Unauthorized access' in data['message']

def test_validate_quality_success(client, sample_image, mock_face_service):
    """Test successful quality validation."""
    response = client.post(
        '/api/validate-quality',
        data={
            'image': (sample_image, 'test.jpg')
        },
        headers={'X-API-Key': 'test-key'}
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data['status'] == 'success'
    assert 'data' in data
    assert 'quality_metrics' in data['data']
    assert 'blur_score' in data['data']['quality_metrics']

def test_validate_quality_invalid_input(client):
    """Test validate-quality with missing image."""
    response = client.post('/api/validate-quality', 
        data={},
        headers={'X-API-Key': 'test-key'}
    )
    assert response.status_code == 400
    data = response.get_json()
    assert 'image' in data['errors']

def test_validate_quality_no_face(client, sample_image, mock_face_service):
    """Test quality validation when no face is detected."""
    mock_face_service.validate_quality.return_value = {
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
    data = response.get_json()
    assert data['status'] == 'error'
    assert data['code'] == 'NoFaceDetectedError'