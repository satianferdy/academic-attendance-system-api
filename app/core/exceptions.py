class FaceRecognitionException(Exception):
    """Base exception for face recognition errors."""
    pass

class FaceDetectionError(FaceRecognitionException):
    """Raised when face detection fails."""
    pass

class MultipleFacesError(FaceRecognitionException):
    """Raised when multiple faces are detected in an image."""
    pass

class NoFaceDetectedError(FaceRecognitionException):
    """Raised when no face is detected in an image."""
    pass

class FaceEmbeddingError(FaceRecognitionException):
    """Raised when face embedding extraction fails."""
    pass

class FaceVerificationError(FaceRecognitionException):
    """Raised when face verification fails."""
    pass

class StudentNotFoundError(FaceRecognitionException):
    """Raised when a student is not found in the database."""
    pass

class FaceNotRegisteredError(FaceRecognitionException):
    """Raised when a student has not registered their face."""
    pass

class DatabaseError(FaceRecognitionException):
    """Raised when a database operation fails."""
    pass

class UnauthorizedError(Exception):
    """Raised when authentication fails."""
    pass