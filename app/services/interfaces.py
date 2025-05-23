# app/services/interfaces.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
from PIL import Image
import numpy as np

class FaceDetectorInterface(ABC):
    @abstractmethod
    def detect_face(self, image_data) -> Tuple[Image.Image, list]:
        pass

class FaceEmbedderInterface(ABC):
    @abstractmethod
    def get_embedding(self, face_image: Image.Image) -> np.ndarray:
        pass

class FaceRecognitionServiceInterface(ABC):
    @abstractmethod
    def process_face(self, image_data: bytes, nim: str) -> Dict[str, Any]:
        pass
        
    @abstractmethod
    def verify_face(self, image_data, class_id: int, nim: str) -> Dict[str, Any]:
        pass
        
    @abstractmethod
    def validate_quality(self, image_data: bytes) -> Dict[str, Any]:
        pass