"""
API Routes DENGAN Dependency Injection - untuk perbandingan
Dependencies di-inject dari container, bisa menggunakan mock untuk testing
"""

from flask import Blueprint, request, jsonify
from app.core.security import validate_api_key
from app.api.validators import validate_verify_face_request
from dependency_injector.wiring import inject, Provide
from app.core.dependencies import Container
from app.services.interfaces import FaceRecognitionServiceInterface
import logging

# Blueprint dengan DI
api_di_blueprint = Blueprint('api_di', __name__)
logger = logging.getLogger(__name__)

@api_di_blueprint.route('/verify-face-di', methods=['POST'])
@validate_api_key
@inject
def verify_face_di(
    face_recognition_service: FaceRecognitionServiceInterface = Provide[Container.face_recognition_service]
):
    """
    Verify face endpoint DENGAN dependency injection
    Service sudah di-inject dari container (bisa real atau mock)
    """
    logger.info("=== Starting DI face verification ===")
    
    # Validate request
    is_valid, data_or_errors = validate_verify_face_request(request)
    
    if not is_valid:
        return jsonify({
            'status': 'error',
            'message': 'Invalid request', 
            'errors': data_or_errors
        }), 400
    
    # Extract data
    image = data_or_errors['image']
    class_id = data_or_errors['class_id']
    nim = data_or_errors['nim']
    
    # MENGGUNAKAN SERVICE YANG SUDAH DI-INJECT
    # Service sudah siap pakai (bisa real atau mock tergantung konfigurasi)
    logger.info("Using injected face recognition service...")
    result = face_recognition_service.verify_face(image.read(), class_id, nim)
    
    logger.info("=== DI face verification completed ===")
    
    # Return response
    status_code = 200 if result['status'] == 'success' else 400
    return jsonify(result), status_code

@api_di_blueprint.route('/health-di', methods=['GET'])
def health_check_di():
    """Health check for DI endpoint"""
    return jsonify({
        'status': 'success',
        'message': 'DI Face verification service is running',
        'type': 'DEPENDENCY_INJECTION'
    })