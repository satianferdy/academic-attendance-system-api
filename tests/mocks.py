from app.services.interfaces import *
import numpy as np
from PIL import Image
from unittest.mock import MagicMock

class MockFaceDetector(FaceDetectorInterface):
    def detect_face(self, image_data):
        return Image.new('RGB', (160, 160)), [0, 0, 100, 100]

class MockFaceEmbedder(FaceEmbedderInterface):
    def get_embedding(self, face_image):
        return np.random.randn(128)

class MockFaceRecognitionService(FaceRecognitionServiceInterface):
    def process_face(self, image_data, nim):
        return {
            'status': 'success',
            'data': {
                'nim': nim,
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
        
    def verify_face(self, image_data, class_id, nim):
        return {
            'status': 'success',
            'message': 'Face verified successfully',
            'student_id': 1,
            'nim': nim,
            'similarity': 0.95
        }
        
    def validate_quality(self, image_data):
        return {
            'status': 'success',
            'data': {
                'quality_metrics': {
                    'blur_score': 75.5
                }
            }
        }

class MockDatabase:
    def __init__(self, *args, **kwargs):
        pass
        
    def get_session(self):
        return MagicMock()
    
    def engine(self):
        return MagicMock()
        
    def close_session(self):
        pass