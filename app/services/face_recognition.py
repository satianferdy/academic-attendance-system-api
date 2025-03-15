import os
import numpy as np
from datetime import datetime
import uuid
from PIL import Image
from app.core.exceptions import (
    FaceRecognitionException, FaceVerificationError, 
    StudentNotFoundError, FaceNotRegisteredError, DatabaseError
)

class FaceRecognitionServiceInterface:
    """Interface for face recognition services."""
    
    def register_face(self, image_data, nim):

        raise NotImplementedError("Subclasses must implement register_face")
    
    def verify_face(self, image_data, class_id, nim):

        raise NotImplementedError("Subclasses must implement verify_face")

class FaceRecognitionService(FaceRecognitionServiceInterface):
    """Implementation of face recognition service."""
    
    def __init__(self, face_detector, face_embedder, recognition_threshold, db):

        self.face_detector = face_detector
        self.face_embedder = face_embedder
        self.recognition_threshold = recognition_threshold
        self.db = db
        self._ensure_upload_dir()
    
    def _ensure_upload_dir(self):
        """Ensure that the upload directory exists."""
        os.makedirs('storage/faces', exist_ok=True)
    
    def _get_student_by_nim(self, session, nim):
        """
        Get a student by NIM.
        
        Args:
            session: Database session
            nim: Student identification number
            
        Returns:
            Student: Student object
            
        Raises:
            StudentNotFoundError: If student is not found
        """
        from app.models.database import Student
        
        student = session.query(Student).filter(Student.nim == nim).first()
        if not student:
            raise StudentNotFoundError(f"Student with NIM {nim} not found")
        return student
    
    def _save_image(self, image, student_id):
        """
        Save a face image to the filesystem.
        
        Args:
            image: PIL Image object
            student_id: Student ID
            
        Returns:
            str: Path to the saved image
        """
        # Generate a unique filename
        filename = f"{student_id}_{uuid.uuid4()}.jpg"
        filepath = os.path.join('storage/faces', filename)
        
        # Save the image
        image.save(filepath, 'JPEG')
        
        return filepath
    
    def _compare_embeddings(self, embedding1, embedding2):
        """
        Compare two face embeddings.
        
        Args:
            embedding1: First face embedding
            embedding2: Second face embedding
            
        Returns:
            float: Similarity score (higher = more similar)
        """
        # Calculate Euclidean distance
        distance = np.linalg.norm(embedding1 - embedding2)
        
        # Convert distance to similarity (1 - normalized distance)
        # Typical threshold for FaceNet: ~0.7-0.8 for similarity
        similarity = 1.0 - min(distance / 2.0, 1.0)
        
        return similarity
    
    def register_face(self, image_data, nim):

        session = self.db.get_session()
        try:
            # Get student
            student = self._get_student_by_nim(session, nim)
            
            # Detect face
            face_image, face_box = self.face_detector.detect_face(image_data)
            
            # Generate face embedding
            embedding = self.face_embedder.get_embedding(face_image)
            
            # Save face image
            image_path = self._save_image(face_image, student.id)
            
            # Check if student already has face data
            from app.models.database import FaceData
            
            face_data = session.query(FaceData).filter(FaceData.student_id == student.id).first()
            
            if face_data:
                # Update existing face data
                face_data.set_embedding_array(embedding)
                face_data.image_path = image_path
                face_data.is_active = True
            else:
                # Create new face data
                face_data = FaceData(
                    student_id=student.id,
                    image_path=image_path,
                    is_active=True
                )
                face_data.set_embedding_array(embedding)
                session.add(face_data)
            
            # Update student face registration status
            student.face_registered = True
            
            # Commit changes
            session.commit()
            
            return {
                'status': 'success',
                'message': 'Face registered successfully',
                'student_id': student.id,
                'nim': student.nim
            }
            
        except FaceRecognitionException as e:
            session.rollback()
            return {
                'status': 'error',
                'message': str(e)
            }
        except Exception as e:
            session.rollback()
            return {
                'status': 'error',
                'message': f"Failed to register face: {str(e)}"
            }
        finally:
            session.close()
    
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