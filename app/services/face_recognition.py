import os
import numpy as np
import cv2
import io
import base64
from datetime import datetime
import uuid
from PIL import Image
from app.core.exceptions import (
    FaceRecognitionException, FaceVerificationError, 
    StudentNotFoundError, FaceNotRegisteredError, DatabaseError,
    NoFaceDetectedError, MultipleFacesError, FaceDetectionError
)

class FaceRecognitionServiceInterface:
    """Interface for face recognition services."""
    
    def verify_face(self, image_data, class_id, nim):
        """Verify face against stored embeddings"""
        raise NotImplementedError("Subclasses must implement verify_face")
        
    def process_face(self, image_data, nim):
        """Process face image and extract features"""
        raise NotImplementedError("Subclasses must implement process_face")
        
    def validate_quality(self, image_data):
        """Validate face image quality and return embedding"""
        raise NotImplementedError("Subclasses must implement validate_quality")

class FaceRecognitionService(FaceRecognitionServiceInterface):
    """Implementation of face recognition service."""
    
    def __init__(self, face_detector, face_embedder, recognition_threshold, db):

        self.face_detector = face_detector
        self.face_embedder = face_embedder
        self.recognition_threshold = recognition_threshold
        self.db = db
    
    def _get_student_by_nim(self, session, nim):
        from app.models.database import Student
        
        student = session.query(Student).filter(Student.nim == nim).first()
        if not student:
            raise StudentNotFoundError(f"Student with NIM {nim} not found")
        return student
    
    
    def _compare_embeddings(self, embedding1, embedding2):
        # Calculate Euclidean distance
        distance = np.linalg.norm(embedding1 - embedding2)
        
        # Convert distance to similarity (1 - normalized distance)
        # Typical threshold for FaceNet: ~0.7-0.8 for similarity
        similarity = 1.0 - min(distance / 2.0, 1.0)
        
        return similarity
    
    def _calculate_blur_score(self, face_image):

        gray = cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    
    def _face_to_base64(self, face_image, quality=90):

        img_byte_arr = io.BytesIO()
        face_image.save(img_byte_arr, format='JPEG', quality=quality)
        return base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
    
    def process_face(self, image_data, nim):

        try:
            # Detect face
            face_image, face_box = self.face_detector.detect_face(image_data)
            
            # Extract embedding
            embedding = self.face_embedder.get_embedding(face_image)
            embedding = embedding / np.linalg.norm(embedding)
            
            # Calculate blur score
            blur_score = self._calculate_blur_score(face_image)
            
            # Check blur threshold
            if blur_score < 10:
                return {
                    'status': 'error',
                    'code': 'LOW_QUALITY_IMAGE',
                    'message': 'Image is too blurry',
                    'blur_score': blur_score
                }
                
            # Convert face image to base64
            base64_image = self._face_to_base64(face_image)
            
            # Return successful response
            return {
                'status': 'success',
                'data': {
                    'nim': nim,
                    'embedding': embedding.tolist(),
                    'face_image': base64_image,
                    'face_box': {
                        'x': int(face_box[0]),
                        'y': int(face_box[1]),
                        'width': int(face_box[2]),
                        'height': int(face_box[3])
                    },
                    'image_info': {
                        'format': 'JPEG',
                        'size': len(base64_image)
                    },
                    'quality_metrics': {
                        'blur_score': blur_score
                    }
                }
            }
            
        except (NoFaceDetectedError, MultipleFacesError, FaceDetectionError) as e:
            return {
                'status': 'error',
                'code': e.__class__.__name__,
                'message': str(e)
            }
        except Exception as e:
            return {
                'status': 'error',
                'code': 'PROCESSING_ERROR',
                'message': f"Failed to process face: {str(e)}"
            }
    
    def validate_quality(self, image_data):
     
        try:
            # Detect face
            face_image, _ = self.face_detector.detect_face(image_data)
            
            # Extract embedding
            embedding = self.face_embedder.get_embedding(face_image)
            embedding = embedding / np.linalg.norm(embedding)
            
            # Calculate blur score
            blur_score = self._calculate_blur_score(face_image)
            
            # Return successful response
            return {
                'status': 'success',
                'data': {
                    'embedding': embedding.tolist(),
                    'quality_metrics': {
                        'blur_score': blur_score
                    }
                }
            }
            
        except (NoFaceDetectedError, MultipleFacesError, FaceDetectionError) as e:
            return {
                'status': 'error',
                'code': e.__class__.__name__,
                'message': str(e)
            }
        except Exception as e:
            return {
                'status': 'error',
                'code': 'QUALITY_VALIDATION_ERROR',
                'message': f"Failed to validate image quality: {str(e)}"
            }
    
    def verify_face(self, image_data, class_id, nim):

        session = self.db.get_session()
        try:
            # Get student
            student = self._get_student_by_nim(session, nim)
            
            # Check if student has registered face
            if not student.face_registered:
                raise FaceNotRegisteredError(f"Student with NIM {nim} has not registered a face")
            
            # Get student's face data
            from app.models.database import FaceData, ClassSchedule
            
            face_data = session.query(FaceData).filter(
                FaceData.student_id == student.id,
                FaceData.is_active == True
            ).first()
            
            if not face_data:
                raise FaceNotRegisteredError(f"No active face data found for student with NIM {nim}")
            
            # Check if student is enrolled in the class
            class_schedule = session.get(ClassSchedule, class_id)
            if not class_schedule or class_schedule.classroom_id != student.classroom_id:
                return {
                    'status': 'error',
                    'message': f"Student with NIM {nim} is not enrolled in this class"
                }
            
            # Detect face in the verification image
            face_image, _ = self.face_detector.detect_face(image_data)
            
            # Generate face embedding
            embedding = self.face_embedder.get_embedding(face_image)
            
            # Get stored embedding
            stored_embedding = face_data.get_embedding_array()
            
            # Compare embeddings
            similarity = self._compare_embeddings(embedding, stored_embedding)
            
            # Check if similarity is above threshold
            if similarity < self.recognition_threshold:
                return {
                    'status': 'error',
                    'message': f"Face verification failed: similarity {similarity:.2f} below threshold {self.recognition_threshold:.2f}"
                }
            
            return {
                'status': 'success',
                'message': 'Face verified successfully',
                'student_id': student.id,
                'nim': student.nim,
                'similarity': similarity
            }
            
        except FaceRecognitionException as e:
            return {
                'status': 'error',
                'message': str(e)
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': f"Failed to verify face: {str(e)}"
            }
        finally:
            session.close()