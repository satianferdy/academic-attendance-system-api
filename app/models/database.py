from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, ForeignKey, Text, DateTime, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker, scoped_session, declarative_base
import json
import numpy as np

Base = declarative_base()

class Student(Base):
    __tablename__ = 'students'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    classroom_id = Column(Integer, ForeignKey('classrooms.id'))
    nim = Column(String(20), unique=True, nullable=False)
    department = Column(String(100))
    faculty = Column(String(100))
    face_registered = Column(Boolean, default=False)
    
    # Relationships
    user = relationship("User", back_populates="student")
    classroom = relationship("Classroom", back_populates="students")
    face_data = relationship("FaceData", back_populates="student", uselist=False)
    attendances = relationship("Attendance", back_populates="student")

class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    password = Column(String(255), nullable=False)
    role = Column(String(20), nullable=False)
    
    # Relationships
    student = relationship("Student", back_populates="user", uselist=False)
    lecturer = relationship("Lecturer", back_populates="user", uselist=False)

class Lecturer(Base):
    __tablename__ = 'lecturers'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    nip = Column(String(20), unique=True, nullable=False)
    department = Column(String(100))
    faculty = Column(String(100))
    
    # Relationships
    user = relationship("User", back_populates="lecturer")
    schedules = relationship("ClassSchedule", back_populates="lecturer")

class Classroom(Base):
    __tablename__ = 'classrooms'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    department = Column(String(100))
    faculty = Column(String(100))
    
    # Relationships
    students = relationship("Student", back_populates="classroom")
    schedules = relationship("ClassSchedule", back_populates="classroom")

class FaceData(Base):
    __tablename__ = 'face_data'
    
    id = Column(Integer, primary_key=True)
    student_id = Column(Integer, ForeignKey('students.id'), unique=True, nullable=False)
    face_embedding = Column(Text, nullable=False)  # Storing as JSON string
    image_path = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    student = relationship("Student", back_populates="face_data")
    
    def get_embedding_array(self):
        """Convert stored JSON embedding to numpy array."""
        return np.array(json.loads(self.face_embedding))
    
    def set_embedding_array(self, embedding_array):
        """Convert numpy array to JSON for storage."""
        self.face_embedding = json.dumps(embedding_array.tolist())

class ClassSchedule(Base):
    __tablename__ = 'class_schedules'
    
    id = Column(Integer, primary_key=True)
    course_id = Column(Integer, ForeignKey('courses.id'))
    lecturer_id = Column(Integer, ForeignKey('lecturers.id'))
    classroom_id = Column(Integer, ForeignKey('classrooms.id'))
    room = Column(String(50))
    day = Column(String(20))
    semester = Column(String(20))
    academic_year = Column(String(20))
    
    # Relationships
    course = relationship("Course", back_populates="schedules")
    lecturer = relationship("Lecturer", back_populates="schedules")
    classroom = relationship("Classroom", back_populates="schedules")
    attendances = relationship("Attendance", back_populates="class_schedule")
    session_attendances = relationship("SessionAttendance", back_populates="class_schedule")
    
class Course(Base):
    __tablename__ = 'courses'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    code = Column(String(20), nullable=False)
    
    # Relationships
    schedules = relationship("ClassSchedule", back_populates="course")

class Attendance(Base):
    __tablename__ = 'attendances'
    
    id = Column(Integer, primary_key=True)
    class_schedule_id = Column(Integer, ForeignKey('class_schedules.id'))
    student_id = Column(Integer, ForeignKey('students.id'))
    date = Column(DateTime, nullable=False)
    status = Column(String(20), nullable=False)
    remarks = Column(Text)
    qr_token = Column(String(255))
    attendance_time = Column(DateTime)
    
    # Relationships
    class_schedule = relationship("ClassSchedule", back_populates="attendances")
    student = relationship("Student", back_populates="attendances")

class SessionAttendance(Base):
    __tablename__ = 'session_attendance'
    
    id = Column(Integer, primary_key=True)
    class_schedule_id = Column(Integer, ForeignKey('class_schedules.id'))
    session_date = Column(DateTime, nullable=False)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=False)
    qr_code = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    class_schedule = relationship("ClassSchedule", back_populates="session_attendances")

class Database:
    """Database connection manager"""
    
    def __init__(self, db_uri):
        self.engine = create_engine(db_uri)
        self.session_factory = sessionmaker(bind=self.engine)
        self.Session = scoped_session(self.session_factory)
    
    def get_session(self):
        return self.Session()
    
    def close_session(self):
        self.Session.remove()