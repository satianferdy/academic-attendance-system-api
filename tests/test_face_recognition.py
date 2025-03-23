import pytest
import numpy as np
from unittest.mock import MagicMock, patch
import io
from PIL import Image
from app.core.exceptions import (
    NoFaceDetectedError, MultipleFacesError, FaceNotRegisteredError, 
    StudentNotFoundError
)

@pytest.fixture
def face_recognition_service(mock_face_detector, mock_face_embedder, database):
    """Create a FaceRecognitionService instance for testing."""
    from app.services.face_recognition import FaceRecognitionService
    return FaceRecognitionService(
        face_detector=mock_face_detector,
        face_embedder=mock_face_embedder,
        recognition_threshold=0.7,
        db=database
    )

class TestFaceRecognitionService:
    """Test suite for the FaceRecognitionService."""
    
    def test_calculate_blur_score(self, face_recognition_service):
        """Test the _calculate_blur_score method."""
        # Create a test image
        image = Image.new('RGB', (160, 160))
        blur_score = face_recognition_service._calculate_blur_score(image)
        assert isinstance(blur_score, float)
    
    def test_face_to_base64(self, face_recognition_service):
        """Test the _face_to_base64 method."""
        image = Image.new('RGB', (160, 160))
        base64_str = face_recognition_service._face_to_base64(image)
        assert isinstance(base64_str, str)
        assert len(base64_str) > 0
    
    def test_compare_embeddings(self, face_recognition_service):
        """Test the _compare_embeddings method."""
        # Create two normalized embeddings
        embedding1 = np.random.randn(128)
        embedding1 = embedding1 / np.linalg.norm(embedding1)
        
        # Same embedding should have similarity 1.0
        similarity = face_recognition_service._compare_embeddings(embedding1, embedding1)
        assert similarity > 0.99
        
        # Different embeddings should have lower similarity
        embedding2 = np.random.randn(128)
        embedding2 = embedding2 / np.linalg.norm(embedding2)
        similarity = face_recognition_service._compare_embeddings(embedding1, embedding2)
        assert 0 <= similarity <= 1.0
    
    def test_process_face_success(self, face_recognition_service, sample_image):
        """Test the process_face method with successful detection."""
        # Setup mocks
        face_recognition_service.face_detector.detect_face.return_value = (
            Image.new('RGB', (160, 160)), [0, 0, 100, 100]
        )
        face_recognition_service.face_embedder.get_embedding.return_value = np.ones(128)
        face_recognition_service._calculate_blur_score = MagicMock(return_value=50.0)
        
        # Call the method
        result = face_recognition_service.process_face(sample_image.read(), '12345')
        
        # Assertions
        assert result['status'] == 'success'
        assert result['data']['nim'] == '12345'
        assert 'embedding' in result['data']
        assert 'face_image' in result['data']
        assert 'quality_metrics' in result['data']
        assert result['data']['quality_metrics']['blur_score'] == 50.0
    
    def test_process_face_no_face(self, face_recognition_service, sample_image):
        """Test the process_face method when no face is detected."""
        # Setup mock to raise NoFaceDetectedError
        face_recognition_service.face_detector.detect_face.side_effect = NoFaceDetectedError("No face detected")
        
        # Call the method
        result = face_recognition_service.process_face(sample_image.read(), '12345')
        
        # Assertions
        assert result['status'] == 'error'
        assert result['code'] == 'NoFaceDetectedError'
    
    def test_process_face_multiple_faces(self, face_recognition_service, sample_image):
        """Test the process_face method when multiple faces are detected."""
        # Setup mock to raise MultipleFacesError
        face_recognition_service.face_detector.detect_face.side_effect = MultipleFacesError("Multiple faces detected")
        
        # Call the method
        result = face_recognition_service.process_face(sample_image.read(), '12345')
        
        # Assertions
        assert result['status'] == 'error'
        assert result['code'] == 'MultipleFacesError'
    
    def test_process_face_low_quality(self, face_recognition_service, sample_image):
        """Test the process_face method with a low quality (blurry) image."""
        # Setup mocks
        face_recognition_service.face_detector.detect_face.return_value = (
            Image.new('RGB', (160, 160)), [0, 0, 100, 100]
        )
        face_recognition_service.face_embedder.get_embedding.return_value = np.ones(128)
        face_recognition_service._calculate_blur_score = MagicMock(return_value=15.0)  # Below threshold
        
        # Call the method
        result = face_recognition_service.process_face(sample_image.read(), '12345')
        
        # Assertions
        assert result['status'] == 'error'
        assert result['code'] == 'LOW_QUALITY_IMAGE'
        assert 'blur_score' in result
    
    def test_validate_quality_success(self, face_recognition_service, sample_image):
        """Test the validate_quality method."""
        # Setup mocks
        face_recognition_service.face_detector.detect_face.return_value = (
            Image.new('RGB', (160, 160)), [0, 0, 100, 100]
        )
        face_recognition_service.face_embedder.get_embedding.return_value = np.ones(128)
        face_recognition_service._calculate_blur_score = MagicMock(return_value=75.0)
        
        # Call the method
        result = face_recognition_service.validate_quality(sample_image.read())
        
        # Assertions
        assert result['status'] == 'success'
        assert 'data' in result
        assert 'quality_metrics' in result['data']
        assert result['data']['quality_metrics']['blur_score'] == 75.0
    
    def test_validate_quality_no_face(self, face_recognition_service, sample_image):
        """Test the validate_quality method when no face is detected."""
        # Setup mock to raise NoFaceDetectedError
        face_recognition_service.face_detector.detect_face.side_effect = NoFaceDetectedError("No face detected")
        
        # Call the method
        result = face_recognition_service.validate_quality(sample_image.read())
        
        # Assertions
        assert result['status'] == 'error'
        assert result['code'] == 'NoFaceDetectedError'
    
    def test_verify_face_success(self, face_recognition_service, sample_image, database):
        """Test the verify_face method with successful verification."""
        # Setup session mock
        session_mock = MagicMock()
        database.get_session = MagicMock(return_value=session_mock)
        
        # Mock the _get_student_by_nim method
        student_mock = MagicMock()
        student_mock.id = 1
        student_mock.nim = '12345'
        student_mock.face_registered = True
        student_mock.classroom_id = 1
        
        # Use patch to mock the query method chain
        with patch('app.models.database.Student', autospec=True) as mock_student:
            # Configure the session query chain for Student
            mock_query = MagicMock()
            mock_filter = MagicMock()
            mock_first = MagicMock(return_value=student_mock)
            mock_filter.first = mock_first
            mock_query.filter = MagicMock(return_value=mock_filter)
            session_mock.query = MagicMock(return_value=mock_query)
            
            # Configure the session query chain for FaceData
            with patch('app.models.database.FaceData', autospec=True) as mock_face_data:
                face_data_mock = MagicMock()
                face_data_mock.get_embedding_array = MagicMock(return_value=np.ones(128))
                
                mock_face_query = MagicMock()
                mock_face_filter = MagicMock()
                mock_face_first = MagicMock(return_value=face_data_mock)
                mock_face_filter.first = mock_face_first
                mock_face_query.filter = MagicMock(return_value=mock_face_filter)
                
                # Configure session.query behavior based on the argument
                def query_side_effect(model):
                    if model == mock_student:
                        return mock_query
                    elif model == mock_face_data:
                        return mock_face_query
                
                session_mock.query = MagicMock(side_effect=query_side_effect)
                
                # Configure ClassSchedule mock
                with patch('app.models.database.ClassSchedule', autospec=True) as mock_class:
                    class_mock = MagicMock()
                    class_mock.classroom_id = 1
                    session_mock.get = MagicMock(return_value=class_mock)
                    
                    # Setup face detection and embedding mocks
                    face_recognition_service.face_detector.detect_face.return_value = (
                        Image.new('RGB', (160, 160)), [0, 0, 100, 100]
                    )
                    face_recognition_service.face_embedder.get_embedding.return_value = np.ones(128)
                    face_recognition_service._compare_embeddings = MagicMock(return_value=0.9)  # High similarity
                    
                    # Call the method
                    result = face_recognition_service.verify_face(sample_image.read(), 1, '12345')
                    
                    # Assertions
                    assert result['status'] == 'success'
                    assert result['student_id'] == 1
                    assert result['nim'] == '12345'
                    assert result['similarity'] == 0.9
    
    def test_verify_face_student_not_found(self, face_recognition_service, sample_image, database):
        """Test the verify_face method when student is not found."""
        # Setup session mock
        session_mock = MagicMock()
        database.get_session = MagicMock(return_value=session_mock)
        
        # Setup query chain to return None (student not found)
        with patch('app.models.database.Student', autospec=True) as mock_student:
            mock_query = MagicMock()
            mock_filter = MagicMock()
            mock_filter.first = MagicMock(return_value=None)  # Student not found
            mock_query.filter = MagicMock(return_value=mock_filter)
            session_mock.query = MagicMock(return_value=mock_query)
            
            # Call the method
            result = face_recognition_service.verify_face(sample_image.read(), 1, '12345')
            
            # Assertions
            assert result['status'] == 'error'
            assert 'not found' in result['message'].lower()
    
    def test_verify_face_not_registered(self, face_recognition_service, sample_image, database):
        """Test the verify_face method when face is not registered."""
        # Setup session mock
        session_mock = MagicMock()
        database.get_session = MagicMock(return_value=session_mock)
        
        # Mock the student with face_registered = False
        student_mock = MagicMock()
        student_mock.id = 1
        student_mock.nim = '12345'
        student_mock.face_registered = False  # Not registered
        
        with patch('app.models.database.Student', autospec=True) as mock_student:
            # Configure the query chain
            mock_query = MagicMock()
            mock_filter = MagicMock()
            mock_filter.first = MagicMock(return_value=student_mock)
            mock_query.filter = MagicMock(return_value=mock_filter)
            session_mock.query = MagicMock(return_value=mock_query)
            
            # Call the method
            result = face_recognition_service.verify_face(sample_image.read(), 1, '12345')
            
            # Assertions
            assert result['status'] == 'error'
            assert 'not registered' in result['message'].lower()
    
    def test_verify_face_not_enrolled(self, face_recognition_service, sample_image, database):
        """Test the verify_face method when student is not enrolled in class."""
        # Setup session mock
        session_mock = MagicMock()
        database.get_session = MagicMock(return_value=session_mock)
        
        # Mock the student
        student_mock = MagicMock()
        student_mock.id = 1
        student_mock.nim = '12345'
        student_mock.face_registered = True
        student_mock.classroom_id = 1  # Student's classroom
        
        # Use patch to mock the query method chain
        with patch('app.models.database.Student', autospec=True) as mock_student:
            # Configure the session query chain for Student
            mock_query = MagicMock()
            mock_filter = MagicMock()
            mock_filter.first = MagicMock(return_value=student_mock)
            mock_query.filter = MagicMock(return_value=mock_filter)
            session_mock.query = MagicMock(return_value=mock_query)
            
            # Configure the session query chain for FaceData
            with patch('app.models.database.FaceData', autospec=True) as mock_face_data:
                face_data_mock = MagicMock()
                
                mock_face_query = MagicMock()
                mock_face_filter = MagicMock()
                mock_face_filter.first = MagicMock(return_value=face_data_mock)
                mock_face_query.filter = MagicMock(return_value=mock_face_filter)
                
                # Configure session.query behavior based on the argument
                def query_side_effect(model):
                    if model == mock_student:
                        return mock_query
                    elif model == mock_face_data:
                        return mock_face_query
                
                session_mock.query = MagicMock(side_effect=query_side_effect)
                
                # Configure ClassSchedule mock with different classroom_id
                with patch('app.models.database.ClassSchedule', autospec=True) as mock_class:
                    class_mock = MagicMock()
                    class_mock.classroom_id = 2  # Different classroom ID
                    session_mock.get = MagicMock(return_value=class_mock)
                    
                    # Call the method
                    result = face_recognition_service.verify_face(sample_image.read(), 1, '12345')
                    
                    # Assertions
                    assert result['status'] == 'error'
                    assert 'not enrolled' in result['message'].lower()
    
    def test_verify_face_low_similarity(self, face_recognition_service, sample_image, database):
        """Test the verify_face method with low similarity (failed verification)."""
        # Setup session mock
        session_mock = MagicMock()
        database.get_session = MagicMock(return_value=session_mock)
        
        # Mock the student
        student_mock = MagicMock()
        student_mock.id = 1
        student_mock.nim = '12345'
        student_mock.face_registered = True
        student_mock.classroom_id = 1
        
        # Use patch to mock the query method chain
        with patch('app.models.database.Student', autospec=True) as mock_student:
            # Configure the session query chain for Student
            mock_query = MagicMock()
            mock_filter = MagicMock()
            mock_filter.first = MagicMock(return_value=student_mock)
            mock_query.filter = MagicMock(return_value=mock_filter)
            session_mock.query = MagicMock(return_value=mock_query)
            
            # Configure the session query chain for FaceData
            with patch('app.models.database.FaceData', autospec=True) as mock_face_data:
                face_data_mock = MagicMock()
                face_data_mock.get_embedding_array = MagicMock(return_value=np.ones(128))
                
                mock_face_query = MagicMock()
                mock_face_filter = MagicMock()
                mock_face_filter.first = MagicMock(return_value=face_data_mock)
                mock_face_query.filter = MagicMock(return_value=mock_face_filter)
                
                # Configure session.query behavior based on the argument
                def query_side_effect(model):
                    if model == mock_student:
                        return mock_query
                    elif model == mock_face_data:
                        return mock_face_query
                
                session_mock.query = MagicMock(side_effect=query_side_effect)
                
                # Configure ClassSchedule mock
                with patch('app.models.database.ClassSchedule', autospec=True) as mock_class:
                    class_mock = MagicMock()
                    class_mock.classroom_id = 1
                    session_mock.get = MagicMock(return_value=class_mock)
                    
                    # Setup face detection and embedding mocks
                    face_recognition_service.face_detector.detect_face.return_value = (
                        Image.new('RGB', (160, 160)), [0, 0, 100, 100]
                    )
                    face_recognition_service.face_embedder.get_embedding.return_value = np.ones(128)
                    face_recognition_service._compare_embeddings = MagicMock(return_value=0.5)  # Low similarity
                    
                    # Call the method
                    result = face_recognition_service.verify_face(sample_image.read(), 1, '12345')
                    
                    # Assertions
                    assert result['status'] == 'error'
                    assert 'similarity' in result['message'].lower()
                    assert '0.5' in result['message']