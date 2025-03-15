from functools import wraps
from flask import request, jsonify, current_app
from app.core.exceptions import UnauthorizedError

def validate_api_key(f):
    """Decorator to validate API key in request header."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        # log api key recieved
        current_app.logger.info(f"API Key: {api_key}")
        current_app.logger.info(f"Expected API Key: '{current_app.config['API_KEY']}'")
        if not api_key or api_key != current_app.config['API_KEY']:
            return jsonify({'status': 'error', 'message': 'Unauthorized access'}), 401
        return f(*args, **kwargs)
    return decorated_function

def get_api_key():
    """Get API key from request header."""
    api_key = request.headers.get('X-API-Key')
    if not api_key or api_key != current_app.config['API_KEY']:
        raise UnauthorizedError("Invalid API key")
    return api_key