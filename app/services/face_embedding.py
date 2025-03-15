import numpy as np
import tensorflow as tf
from PIL import Image
from app.core.exceptions import FaceEmbeddingError

class FaceEmbedderInterface:
    """Interface for face embedding services."""
    
    def get_embedding(self, face_image):
        raise NotImplementedError("Subclasses must implement get_embedding")

class FaceNetEmbedding(FaceEmbedderInterface):
    """FaceNet-based face embedding implementation."""
    
    def __init__(self, model_path, image_size=(160, 160)):
        self.image_size = image_size
        self.model = self._load_model(model_path)
        
    def _load_model(self, model_path):

        try:
            # Avoid TensorFlow warnings
            tf.keras.utils.disable_interactive_logging()
            
            # Load the model
            model = tf.keras.models.load_model(model_path, compile=False)
            
            # Warm up the model
            dummy_input = np.zeros((1, self.image_size[0], self.image_size[1], 3), dtype=np.float32)
            _ = model.predict(dummy_input)
            
            return model
        except Exception as e:
            raise FaceEmbeddingError(f"Failed to load FaceNet model: {str(e)}")
    
    def preprocess_image(self, face_image):

        # Resize image to the required dimensions
        face_image = face_image.resize(self.image_size)
        
        # Convert to array and normalize
        face_array = np.array(face_image, dtype=np.float32)
        
        # Normalize pixel values to [-1, 1]
        face_array = (face_array - 127.5) / 128.0
        
        # Add batch dimension
        face_array = np.expand_dims(face_array, axis=0)
        
        return face_array
    
    def get_embedding(self, face_image):

        try:
            # Preprocess the face image
            preprocessed_face = self.preprocess_image(face_image)
            
            # Generate embedding
            embedding = self.model.predict(preprocessed_face)[0]
            
            # Normalize embedding to unit length
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
        except Exception as e:
            raise FaceEmbeddingError(f"Failed to extract face embedding: {str(e)}")