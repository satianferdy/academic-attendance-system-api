from flask import Flask
from flask_cors import CORS
from .api.routes import api_blueprint
from .core.dependencies import Container

def create_app(testing=False):
    """
    Application factory function.
    
    Args:
        testing: Flag to indicate if the app is in testing mode
        
    Returns:
        Flask application instance
    """
    # Create Flask app
    app = Flask(__name__)
     
    
    # Load configuration
    app.config.from_object('app.config.Config')
    
    # Configure testing mode
    if testing:
        app.config['TESTING'] = True
        # You can override other config settings for testing here
    
    # Setup CORS
    CORS(app, resources={r"/api/*": {"origins": "*"}})
    
    # Setup container
    container = Container()
    app.container = container
    
    # Register blueprints
    app.register_blueprint(api_blueprint, url_prefix='/api')
    
    # Wire container to endpoints
    container.wire(modules=['.api.routes'])
    
    return app