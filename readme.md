# Academic Attendance System API v2

A Flask-based REST API for academic attendance management using face recognition technology. This system provides secure face detection, recognition, and verification capabilities for educational institutions.

## 🚀 Features

- **Face Detection**: MTCNN-based face detection with configurable confidence thresholds
- **Face Recognition**: FaceNet neural network for generating face embeddings
- **Face Verification**: Compare face embeddings for attendance verification
- **Quality Validation**: Automatic image quality assessment (blur detection)
- **Dependency Injection**: Clean architecture with testable components
- **RESTful API**: Well-structured endpoints with comprehensive validation
- **Database Integration**: SQLAlchemy ORM with MySQL support
- **Security**: API key authentication and input validation
- **Testing**: Comprehensive test suite with mocking capabilities
- **Performance Benchmarking**: Built-in performance comparison tools

## 🏗️ Architecture

The system follows clean architecture principles with dependency injection:

```
app/
├── api/              # REST API endpoints and validation
├── core/             # Core utilities and dependency injection
├── models/           # Database models and ORM
├── services/         # Business logic and AI services
└── utils/            # Helper utilities
```

### Key Components

- **Face Detection Service**: MTCNN implementation for face detection
- **Face Embedding Service**: FaceNet for generating face embeddings
- **Face Recognition Service**: Main service orchestrating the workflow
- **Database Models**: Student, attendance, and face data management

## 📋 Prerequisites

- Python 3.7+
- MySQL database
- Face recognition models (FaceNet)
- Required system libraries for OpenCV

## 🛠️ Installation

1. **Clone the repository**

```bash
git clone <repository-url>
cd academic-attendance-system-api-v2
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Set up environment variables**

```bash
cp .env.example .env
# Edit .env with your configuration
```

4. **Download AI models**

```bash
# Place facenet_keras.h5 in models/ directory
mkdir models
# Download FaceNet model to models/facenet_keras.h5
```

5. **Initialize database**

```bash
# Run database migrations
python -c "from app.models.database import Database, Base; db = Database('your_db_uri'); Base.metadata.create_all(db.engine)"
```

## 🚀 Running the Application

### Development

```bash
python run.py
```

### Production (Docker)

```bash
docker build -t attendance-api .
docker run -p 5000:5000 attendance-api
```

### Production (Gunicorn)

```bash
gunicorn -w 4 -b 0.0.0.0:5000 wsgi:app
```

## 📡 API Endpoints

### Health Check

```http
GET /api/health
```

### Process Face

```http
POST /api/process-face
Content-Type: multipart/form-data
X-API-Key: your-api-key

image: <image_file>
nim: <student_id>
```

### Verify Face

```http
POST /api/verify-face
Content-Type: multipart/form-data
X-API-Key: your-api-key

image: <image_file>
class_id: <class_schedule_id>
nim: <student_id>
```

### Validate Quality

```http
POST /api/validate-quality
Content-Type: multipart/form-data
X-API-Key: your-api-key

image: <image_file>
```

## 🔧 Configuration

Key configuration options in `.env`:

```env
# Database
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=password
DB_NAME=attendance_db

# Security
API_KEY=your-secure-api-key
SECRET_KEY=your-flask-secret-key

# AI Models
FACE_RECOGNITION_MODEL_PATH=models/facenet_keras.h5
FACE_DETECTION_CONFIDENCE=0.95
FACE_RECOGNITION_THRESHOLD=0.7

# Storage
UPLOAD_FOLDER=storage/faces
MAX_CONTENT_LENGTH=5242880
```

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/test_face_recognition.py
```

### Performance Benchmarking

Compare dependency injection vs non-DI implementations:

```bash
python benchmark_comparison.py
```

## 📊 Performance Metrics

The system includes built-in benchmarking tools that compare:

- **Setup Time**: Initialization overhead
- **Execution Time**: Request processing speed
- **Memory Usage**: Resource consumption
- **Code Complexity**: Lines of code and dependencies

## 🐳 Docker Deployment

The application includes a production-ready Dockerfile with:

- Optimized Python 3.7 base image
- System dependencies for OpenCV
- Health checks
- Multi-stage build support

## 🔒 Security Features

- API key authentication
- Input validation with Marshmallow
- File upload restrictions
- SQL injection prevention (SQLAlchemy ORM)
- CORS configuration

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Troubleshooting

### Common Issues

1. **Model Loading Errors**: Ensure FaceNet model is in the correct path
2. **Database Connection**: Verify database credentials and connectivity
3. **Dependency Issues**: Use the exact versions in requirements.txt
4. **Memory Issues**: Adjust batch sizes for large-scale deployments

### Support

For technical support or questions, please open an issue in the repository.

---

**Built with ❤️ for educational institutions**
