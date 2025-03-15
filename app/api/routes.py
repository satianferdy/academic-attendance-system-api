from flask import Blueprint, request, jsonify
from app.core.security import validate_api_key
from app.api.validators import validate_process_face_request, validate_verify_face_request, validate_quality_request
from dependency_injector.wiring import inject, Provide
from app.core.dependencies import Container
from app.services.face_recognition import FaceRecognitionService
import logging

api_blueprint = Blueprint('api', __name__)
logger = logging.getLogger(__name__)

@api_blueprint.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'success',
        'message': 'Face recognition service is running'
    })

@api_blueprint.route('/process-face', methods=['POST'])
@validate_api_key
@inject
def process_face(face_recognition_service: FaceRecognitionService = Provide[Container.face_recognition_service]):
    """
    Process a face image and return facial embedding and quality metrics.
    """
    # Validate request
    is_valid, data_or_errors = validate_process_face_request(request)
    
    if not is_valid:
        return jsonify({
            'status': 'error',
            'message': 'Invalid request',
            'errors': data_or_errors
        }), 400
    
    # Extract data
    image_file = data_or_errors['image']
    nim = data_or_errors['nim']
    
    # Process face
    result = face_recognition_service.process_face(image_file.read(), nim)
    
    # Return response
    status_code = 200 if result['status'] == 'success' else 400
    return jsonify(result), status_code

@api_blueprint.route('/verify-face', methods=['POST'])
@validate_api_key
@inject
def verify_face(face_recognition_service: FaceRecognitionService = Provide[Container.face_recognition_service]):
    """
    Verify a face for attendance.
    """
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
    
    # Verify face
    result = face_recognition_service.verify_face(image, class_id, nim)
    
    # Return response
    status_code = 200 if result['status'] == 'success' else 400
    return jsonify(result), status_code

@api_blueprint.route('/validate-quality', methods=['POST'])
@validate_api_key
@inject
def validate_quality(face_recognition_service: FaceRecognitionService = Provide[Container.face_recognition_service]):
    """
    Validate face image quality and return embedding.
    """
    # Validate request
    is_valid, data_or_errors = validate_quality_request(request)
    
    if not is_valid:
        return jsonify({
            'status': 'error',
            'message': 'Invalid request',
            'errors': data_or_errors
        }), 400
    
    # Extract data
    image_file = data_or_errors['image']
    
    # Validate quality
    result = face_recognition_service.validate_quality(image_file.read())
    
    # Return response
    status_code = 200 if result['status'] == 'success' else 400
    return jsonify(result), status_code