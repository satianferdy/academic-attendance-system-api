import pytest
import json
import numpy as np  
from app.models.database import Student, FaceData, Database

def test_student_face_relationship(database):
    session = database.get_session()
    try:
        # Buat data student
        student = Student(nim='12345', classroom_id=1)
        session.add(student)
        session.commit()

        # Buat data face_data dengan face_embedding
        face_data = FaceData(
            student_id=student.id,
            face_embedding=json.dumps([1.0, 2.0, 3.0]),  # <-- Isi dengan data valid
            image_path='test.jpg'
        )
        session.add(face_data)
        session.commit()

        # Verifikasi relasi
        assert student.face_data == face_data
        assert face_data.student == student
    finally:
        session.rollback()
        session.close()

def test_face_embedding_conversion():
    face_data = FaceData()
    original = np.random.randn(128)
    face_data.set_embedding_array(original)
    restored = face_data.get_embedding_array()
    assert np.allclose(original, restored)