import pytest
from unittest.mock import MagicMock
import numpy as np
from app.services.face_recognition import FaceRecognitionService
from app.core.exceptions import FaceNotRegisteredError, StudentNotFoundError

@pytest.fixture
def face_service(database, mock_face_detector, mock_face_embedder, test_config):
    return FaceRecognitionService(
        face_detector=mock_face_detector,
        face_embedder=mock_face_embedder,
        recognition_threshold=test_config.FACE_RECOGNITION_THRESHOLD,
        db=database
    )

def test_register_face(face_service, database):
    session = database.get_session()
    try:
        # Create test student
        from app.models.database import Student
        student = Student(nim='12345', classroom_id=1)
        session.add(student)
        session.commit()
        
        # Test registration
        result = face_service.register_face(b'dummy_image', '12345')
        assert result['status'] == 'success'
        
        # Check database
        student = session.query(Student).filter_by(nim='12345').first()
        assert student.face_registered is True
    finally:
        session.rollback()
        session.close()

def test_verify_unregistered_face(face_service, database):
    session = database.get_session()
    try:
        # Get or create test student with face_registered=False
        from app.models.database import Student
        student = session.query(Student).filter_by(nim='12345').first()
        if student:
            # Update existing student
            student.face_registered = False
        else:
            # Create new student
            student = Student(nim='12345', classroom_id=1, face_registered=False)
            session.add(student)
        session.commit()
        
        # Test verification - should return error status
        result = face_service.verify_face(b'dummy_image', 1, '12345')
        assert result['status'] == 'error'
        assert "not registered a face" in result['message']
    finally:
        session.rollback()
        session.close()

def test_face_similarity_calculation(face_service):
    emb1 = np.random.randn(128)
    emb2 = emb1 + np.random.normal(0, 0.1, 128)
    similarity = face_service._compare_embeddings(emb1, emb2)
    assert 0 <= similarity <= 1
    