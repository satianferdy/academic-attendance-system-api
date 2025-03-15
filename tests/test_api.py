import pytest
from unittest.mock import patch
import io
from PIL import Image

def test_health_check(client):
    response = client.get('/api/health')
    assert response.status_code == 200
    assert response.json == {'status': 'success', 'message': 'Face recognition service is running'}

def test_register_face_missing_api_key(client, sample_image):
    response = client.post('/api/register-face', data={
        'image': (sample_image, 'test.jpg'),
        'nim': '12345'
    })
    assert response.status_code == 401
    assert 'Unauthorized access' in response.json['message']

@patch('app.services.face_recognition.FaceRecognitionService.register_face')
def test_register_face_success(mock_register, client, sample_image):
    mock_register.return_value = {'status': 'success', 'message': 'Face registered'}
    response = client.post('/api/register-face', 
        data={
            'image': (sample_image, 'test.jpg'),
            'nim': '12345'
        },
        headers={'X-API-Key': 'test-key'}
    )
    assert response.status_code == 200
    assert response.json['status'] == 'success'

def test_register_face_invalid_input(client):
    response = client.post('/api/register-face', 
        data={'nim': '12345'},
        headers={'X-API-Key': 'test-key'}
    )
    assert response.status_code == 400
    assert 'image' in response.json['errors']
    
@patch('app.services.face_recognition.FaceRecognitionService.verify_face')
def test_verify_face_success(mock_verify, client, sample_image):
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