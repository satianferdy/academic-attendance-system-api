from marshmallow import Schema, fields, validate, ValidationError
from werkzeug.datastructures import FileStorage

class FileField(fields.Field):
    def _deserialize(self, value, attr, data, **kwargs):
        if not isinstance(value, FileStorage):
            raise ValidationError('Not a valid file upload')
        return value

# Schema untuk /process-face
class ProcessFaceSchema(Schema):
    image = FileField(required=True)
    nim = fields.String(required=True, validate=validate.Length(min=1))

def validate_process_face_request(request):
    schema = ProcessFaceSchema()
    try:
        data = {
            'image': request.files.get('image'),
            'nim': request.form.get('nim')
        }
        validated_data = schema.load(data)
        return True, validated_data
    except ValidationError as err:
        return False, err.messages

# Schema untuk /validate-quality
class ValidateQualitySchema(Schema):
    image = FileField(required=True)

def validate_quality_request(request):
    schema = ValidateQualitySchema()
    try:
        data = {'image': request.files.get('image')}
        validated_data = schema.load(data)
        return True, validated_data
    except ValidationError as err:
        return False, err.messages

# Schema untuk /verify-face (sudah ada)
class VerifyFaceSchema(Schema):
    image = FileField(required=True)
    class_id = fields.Integer(required=True, strict=False)
    nim = fields.String(required=True, validate=validate.Length(min=1))

def validate_verify_face_request(request):
    schema = VerifyFaceSchema()
    try:
        data = {
            'image': request.files.get('image'),
            'class_id': request.form.get('class_id'),
            'nim': request.form.get('nim')
        }
        validated_data = schema.load(data)
        return True, validated_data
    except ValidationError as err:
        return False, err.messages