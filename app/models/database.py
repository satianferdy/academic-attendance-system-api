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
    study_program_id = Column(Integer, ForeignKey('study_programs.id'))
    nim = Column(String(20), unique=True, nullable=False)
    face_registered = Column(Boolean, default=False)
    
    # Relationships
    user = relationship("User", back_populates="student")
    classroom = relationship("Classroom", back_populates="students")
    study_program = relationship("StudyProgram", back_populates="students")
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
    
    # Relationships
    user = relationship("User", back_populates="lecturer")
    schedules = relationship("ClassSchedule", back_populates="lecturer")

class StudyProgram(Base):
    __tablename__ = 'study_programs'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    code = Column(String(20), nullable=False)
    degree_level = Column(String(50))
    faculty = Column(String(100))
    description = Column(Text)
    
    # Relationships
    students = relationship("Student", back_populates="study_program")
    classrooms = relationship("Classroom", back_populates="study_program")
    courses = relationship("Course", back_populates="study_program")
    class_schedules = relationship("ClassSchedule", back_populates="study_program")

class Semester(Base):
    __tablename__ = 'semesters'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    academic_year = Column(String(20), nullable=False)
    term = Column(String(20), nullable=False)
    start_date = Column(DateTime)
    end_date = Column(DateTime)
    is_active = Column(Boolean, default=False)
    
    # Relationships
    classrooms = relationship("Classroom", back_populates="semester")
    class_schedules = relationship("ClassSchedule", back_populates="semester")

class Classroom(Base):
    __tablename__ = 'classrooms'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    study_program_id = Column(Integer, ForeignKey('study_programs.id'))
    capacity = Column(Integer)
    semester_id = Column(Integer, ForeignKey('semesters.id'))
    
    # Relationships
    study_program = relationship("StudyProgram", back_populates="classrooms")
    semester = relationship("Semester", back_populates="classrooms")
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
    semester_id = Column(Integer, ForeignKey('semesters.id'))
    study_program_id = Column(Integer, ForeignKey('study_programs.id'))
    room = Column(String(50))
    day = Column(String(20))
    semester = Column(String(20))  # Legacy field to be deprecated
    total_weeks = Column(Integer)
    meetings_per_week = Column(Integer)
    
    # Relationships
    course = relationship("Course", back_populates="schedules")
    lecturer = relationship("Lecturer", back_populates="schedules")
    classroom = relationship("Classroom", back_populates="schedules")
    semester = relationship("Semester", back_populates="class_schedules")
    study_program = relationship("StudyProgram", back_populates="class_schedules")
    attendances = relationship("Attendance", back_populates="class_schedule")
    session_attendances = relationship("SessionAttendance", back_populates="class_schedule")
    time_slots = relationship("ScheduleTimeSlot", back_populates="class_schedule")

class ScheduleTimeSlot(Base):
    __tablename__ = 'schedule_time_slots'
    
    id = Column(Integer, primary_key=True)
    class_schedule_id = Column(Integer, ForeignKey('class_schedules.id'))
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=False)
    
    # Relationships
    class_schedule = relationship("ClassSchedule", back_populates="time_slots")

class Course(Base):
    __tablename__ = 'courses'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    code = Column(String(20), nullable=False)
    study_program_id = Column(Integer, ForeignKey('study_programs.id'))
    credits = Column(Integer)
    description = Column(Text)
    
    # Relationships
    schedules = relationship("ClassSchedule", back_populates="course")
    study_program = relationship("StudyProgram", back_populates="courses")

class Attendance(Base):
    __tablename__ = 'attendances'
    
    id = Column(Integer, primary_key=True)
    class_schedule_id = Column(Integer, ForeignKey('class_schedules.id'))
    student_id = Column(Integer, ForeignKey('students.id'))
    date = Column(DateTime, nullable=False)
    status = Column(String(20), nullable=False)
    remarks = Column(Text)
    edit_notes = Column(Text)
    hours_present = Column(Float)
    hours_absent = Column(Float)
    hours_permitted = Column(Float)
    hours_sick = Column(Float)
    qr_token = Column(String(255))
    attendance_time = Column(DateTime)
    last_edited_by = Column(Integer, ForeignKey('users.id'))
    last_edited_at = Column(DateTime)
    
    # Relationships
    class_schedule = relationship("ClassSchedule", back_populates="attendances")
    student = relationship("Student", back_populates="attendances")
    editor = relationship("User", foreign_keys=[last_edited_by])

class SessionAttendance(Base):
    __tablename__ = 'session_attendance'
    
    id = Column(Integer, primary_key=True)
    class_schedule_id = Column(Integer, ForeignKey('class_schedules.id'))
    session_date = Column(DateTime, nullable=False)
    week = Column(Integer)
    meetings = Column(Integer)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=False)
    total_hours = Column(Float)
    tolerance_minutes = Column(Integer)
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