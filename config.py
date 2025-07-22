# Face Recognition with OpenCV, Python, and Deep Learning
# Project Structure and Main Components

"""
Project Directory Structure:
face_recognition_project/
│
├── requirements.txt
├── main.py
├── face_encoder.py
├── face_recognizer.py
├── utils.py
├── config.py
│
├── known_faces/
│   ├── person1/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── person2/
│       ├── image1.jpg
│       └── image2.jpg
│
├── encodings/
│   └── face_encodings.pkl
│
└── output/
    └── recognized_faces/
"""

# requirements.txt content:
"""
opencv-python==4.8.1.78
face-recognition==1.3.0
numpy==1.24.3
Pillow==10.0.1
imutils==0.5.4
dlib==19.24.2
cmake==3.27.7
"""

# config.py - Configuration settings
import os

class Config:
    # Directories
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    KNOWN_FACES_DIR = os.path.join(BASE_DIR, "known_faces")
    ENCODINGS_DIR = os.path.join(BASE_DIR, "encodings")
    OUTPUT_DIR = os.path.join(BASE_DIR, "output")
    
    # Files
    ENCODINGS_FILE = os.path.join(ENCODINGS_DIR, "face_encodings.pkl")
    
    # Face recognition parameters
    TOLERANCE = 0.6  # Lower = more strict
    MODEL = "hog"  # or "cnn" for better accuracy but slower
    
    # Video settings
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    
    # Create directories if they don't exist
    @classmethod
    def create_directories(cls):
        os.makedirs(cls.KNOWN_FACES_DIR, exist_ok=True)
        os.makedirs(cls.ENCODINGS_DIR, exist_ok=True)
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)