from dependency_injector import containers, providers
from app.services.face_detection import MTCNNFaceDetector
from app.services.face_embedding import FaceNetEmbedding
from app.services.face_recognition import FaceRecognitionService
from app.models.database import Database
from app.config import Config

class Container(containers.DeclarativeContainer):
    """Dependency injection container."""
    
    config = providers.Singleton(Config)
    
    db = providers.Singleton(
        Database,
        db_uri=config.provided.SQLALCHEMY_DATABASE_URI
    )
    
    face_detector = providers.Singleton(
        MTCNNFaceDetector,
        min_confidence=config.provided.FACE_DETECTION_CONFIDENCE
    )
    
    face_embedder = providers.Singleton(
        FaceNetEmbedding,
        model_path=config.provided.FACE_RECOGNITION_MODEL_PATH,
        image_size=config.provided.FACE_IMAGE_SIZE
    )
    
    face_recognition_service = providers.Singleton(
        FaceRecognitionService,
        face_detector=face_detector,
        face_embedder=face_embedder,
        recognition_threshold=config.provided.FACE_RECOGNITION_THRESHOLD,
        db=db
    )