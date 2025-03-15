from flask import Blueprint, request, jsonify
from app.core.security import validate_api_key
from app.api.validators import validate_verify_face_request, validate_quality_request, validate_process_face_request
from dependency_injector.wiring import inject, Provide
from app.core.dependencies import Container
from app.core.exceptions import (
    FaceDetectionError,
    MultipleFacesError,
    NoFaceDetectedError,
    FaceRecognitionException,
    FaceEmbeddingError
)
from app.services.face_recognition import FaceRecognitionService
import cv2
import numpy as np
import base64
import io
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
    try:
        # 1. Validasi input
        if 'image' not in request.files:
            return jsonify({
                'status': 'error',
                'code': 'MISSING_IMAGE',
                'message': 'No image file provided'
            }), 400

        if 'nim' not in request.form:
            return jsonify({
                'status': 'error',
                'code': 'MISSING_NIM',
                'message': 'NIM parameter is required'
            }), 400

        image_file = request.files['image']
        nim = request.form['nim']

        # 2. Baca dan validasi gambar
        try:
            image_data = image_file.read()
            if len(image_data) == 0:
                raise ValueError("Empty image file")
        except Exception as e:
            logger.error(f"Image read error: {str(e)}")
            return jsonify({
                'status': 'error',
                'code': 'INVALID_IMAGE',
                'message': 'Failed to read image file'
            }), 400

        # 3. Deteksi wajah
        try:
            face_image, face_box = face_recognition_service.face_detector.detect_face(image_data)
        except NoFaceDetectedError:
            return jsonify({
                'status': 'error',
                'code': 'NO_FACE_DETECTED',
                'message': 'No face detected in the image'
            }), 400
        except MultipleFacesError:
            return jsonify({
                'status': 'error',
                'code': 'MULTIPLE_FACES',
                'message': 'Multiple faces detected in the image'
            }), 400
        except FaceDetectionError as e:
            logger.error(f"Face detection failed: {str(e)}")
            return jsonify({
                'status': 'error',
                'code': 'FACE_DETECTION_FAILED',
                'message': 'Face detection process failed'
            }), 500

        # 4. Ekstraksi embedding
        try:
            embedding = face_recognition_service.face_embedder.get_embedding(face_image)
            # Normalisasi embedding
            embedding = embedding / np.linalg.norm(embedding)
        except Exception as e:
            logger.error(f"Embedding extraction failed: {str(e)}")
            return jsonify({
                'status': 'error',
                'code': 'EMBEDDING_FAILED',
                'message': 'Failed to generate face embedding'
            }), 500

        # 5. Validasi kualitas gambar (opsional)
        try:
            # Hitung blur score menggunakan Laplacian variance
            gray = cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2GRAY)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

            if blur_score < 10:  # Threshold blur
                return jsonify({
                    'status': 'error',
                    'code': 'LOW_QUALITY_IMAGE',
                    'message': 'Image is too blurry',
                    'blur_score': blur_score
                }), 400
        except Exception as e:
            logger.warning(f"Blur detection failed: {str(e)}")

        # 6. Konversi gambar wajah ke base64
        try:
            img_byte_arr = io.BytesIO()
            face_image.save(img_byte_arr, format='JPEG', quality=90)
            base64_image = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
        except Exception as e:
            logger.error(f"Image conversion failed: {str(e)}")
            return jsonify({
                'status': 'error',
                'code': 'IMAGE_CONVERSION_FAILED',
                'message': 'Failed to process face image'
            }), 500

        # 7. Response sukses
        return jsonify({
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
                    'size': len(base64_image)  # Size dalam bytes
                },
                'quality_metrics': {
                    'blur_score': blur_score if 'blur_score' in locals() else None
                }
            }
        }), 200

    except Exception as e:
        logger.critical(f"Unexpected error: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'code': 'INTERNAL_SERVER_ERROR',
            'message': 'An unexpected error occurred'
        }), 500


@api_blueprint.route('/verify-face', methods=['POST'])  # <-- Hapus /api di sini
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
    try:
        # 1. Validasi input
        if 'image' not in request.files:
            return jsonify({
                'status': 'error',
                'code': 'MISSING_IMAGE',
                'message': 'No image file provided'
            }), 400

        image_file = request.files['image']

        # 2. Baca dan validasi gambar
        try:
            image_data = image_file.read()
            if len(image_data) == 0:
                raise ValueError("Empty image file")
        except Exception as e:
            logger.error(f"Image read error: {str(e)}")
            return jsonify({
                'status': 'error',
                'code': 'INVALID_IMAGE',
                'message': 'Failed to read image file'
            }), 400

        # 3. Deteksi wajah
        try:
            face_image, _ = face_recognition_service.face_detector.detect_face(image_data)
        except NoFaceDetectedError:
            return jsonify({
                'status': 'error',
                'code': 'NO_FACE_DETECTED',
                'message': 'No face detected in the image'
            }), 400
        except MultipleFacesError:
            return jsonify({
                'status': 'error',
                'code': 'MULTIPLE_FACES',
                'message': 'Multiple faces detected in the image'
            }), 400
        except FaceDetectionError as e:
            logger.error(f"Face detection failed: {str(e)}")
            return jsonify({
                'status': 'error',
                'code': 'FACE_DETECTION_FAILED',
                'message': 'Face detection process failed'
            }), 500

        # 4. Ekstraksi embedding
        try:
            embedding = face_recognition_service.face_embedder.get_embedding(face_image)
            # Normalisasi embedding dan konversi ke list
            embedding = (embedding / np.linalg.norm(embedding)).tolist()
        except Exception as e:
            logger.error(f"Embedding extraction failed: {str(e)}")
            return jsonify({
                'status': 'error',
                'code': 'EMBEDDING_FAILED',
                'message': 'Failed to generate face embedding'
            }), 500

        # 5. Validasi kualitas gambar (blur detection)
        try:
            gray = cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2GRAY)
            blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())  # Konversi ke float
        except Exception as e:
            logger.warning(f"Blur detection failed: {str(e)}")
            blur_score = None

        # 6. Response sukses
        return jsonify({
            'status': 'success',
            'data': {
                'embedding': embedding,  # Sudah dalam bentuk list
                'quality_metrics': {
                    'blur_score': blur_score
                }
            }
        }), 200

    except Exception as e:
        logger.critical(f"Unexpected error: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'code': 'INTERNAL_SERVER_ERROR',
            'message': 'An unexpected error occurred'
        }), 500