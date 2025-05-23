import numpy as np
from mtcnn import MTCNN
from PIL import Image
import io
from app.core.exceptions import FaceDetectionError, MultipleFacesError, NoFaceDetectedError
from app.services.interfaces import FaceDetectorInterface

# class FaceDetectorInterface:
#     """Interface for face detection services."""
    
#     def detect_face(self, image_data):
#         raise NotImplementedError("Subclasses must implement detect_face")

class MTCNNFaceDetector(FaceDetectorInterface):
    """MTCNN-based face detector implementation."""
    
    def __init__(self, min_confidence=0.95):
        self.detector = MTCNN(
            min_face_size=30,
            steps_threshold=[min_confidence, min_confidence, min_confidence]
        )
        self.min_confidence = min_confidence
        
    def detect_face(self, image_data):
        try:
            # Convert image data to PIL Image
            if isinstance(image_data, bytes):
                img = Image.open(io.BytesIO(image_data))
            else:
                img = Image.open(image_data)
                
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
                
            # Convert to numpy array for MTCNN
            img_array = np.array(img)
            
            # Detect faces
            faces = self.detector.detect_faces(img_array)
            
            # Check if faces were detected
            if not faces:
                raise NoFaceDetectedError("No face detected in the image")
                
            # Check if multiple faces were detected
            if len(faces) > 1:
                raise MultipleFacesError("Multiple faces detected in the image")
            
            # Get the detected face
            face = faces[0]
            
            # Check confidence
            if face['confidence'] < self.min_confidence:
                raise NoFaceDetectedError(f"Face detection confidence too low: {face['confidence']}")
            
            # Extract face box
            x, y, width, height = face['box']
            
            # Add a margin to the face box (10% of face dimensions)
            margin_x = int(width * 0.1)
            margin_y = int(height * 0.1)
            
            # Calculate new box coordinates with margins
            x = max(0, x - margin_x)
            y = max(0, y - margin_y)
            width = min(img.width - x, width + 2 * margin_x)
            height = min(img.height - y, height + 2 * margin_y)
            
            # Crop the face
            face_img = img.crop((x, y, x + width, y + height))
            
            return face_img, face['box']
            
        except (NoFaceDetectedError, MultipleFacesError) as e:
            # Re-raise these exceptions as they are expected
            raise
        except Exception as e:
            # Convert any other exceptions to FaceDetectionError
            raise FaceDetectionError(f"Face detection failed: {str(e)}")