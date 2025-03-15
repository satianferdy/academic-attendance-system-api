import pytest
from unittest.mock import MagicMock, patch
import numpy as np
import io 
from PIL import Image
from app.services.face_detection import MTCNNFaceDetector, FaceDetectionError

@pytest.fixture
def mtcnn_detector():
    return MTCNNFaceDetector(min_confidence=0.95)

def test_detect_face_no_face(mtcnn_detector):
    detector = mtcnn_detector
    detector.detector.detect_faces = MagicMock(return_value=[])
    image = Image.new('RGB', (500, 500))
    
    with pytest.raises(FaceDetectionError):
        detector.detect_face(image)

def test_detect_face_multiple_faces(mtcnn_detector):
    detector = mtcnn_detector
    detector.detector.detect_faces = MagicMock(return_value=[
        {'box': [0,0,100,100], 'confidence': 0.99},
        {'box': [100,100,100,100], 'confidence': 0.98}
    ])
    
    with pytest.raises(FaceDetectionError):
        detector.detect_face(Image.new('RGB', (500, 500)))

@patch('app.services.face_detection.MTCNN')
def test_detect_face_success(mock_mtcnn):
    mock_detector = mock_mtcnn.return_value
    mock_detector.detect_faces.return_value = [
        {'box': [0, 0, 100, 100], 'confidence': 0.99}
    ]

    detector = MTCNNFaceDetector()
    
    # Buat gambar dan konversi ke bytes
    image = Image.new('RGB', (500, 500))
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)
    
    # Gunakan bytes sebagai input
    face_img, box = detector.detect_face(img_byte_arr.getvalue())
    assert face_img.size == (120, 120)