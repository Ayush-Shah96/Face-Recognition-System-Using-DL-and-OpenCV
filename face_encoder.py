# face_encoder.py - Face encoding generation and management

import os
import pickle
import face_recognition
import cv2
from PIL import Image
import numpy as np
from config import Config
from utils import get_image_paths, load_image_safely

class FaceEncoder:
    def __init__(self):
        self.known_encodings = []
        self.known_names = []
        Config.create_directories()
    
    def encode_faces_from_directory(self, directory_path=None):
        """
        Encode faces from the known_faces directory structure.
        Expected structure: known_faces/person_name/image.jpg
        """
        if directory_path is None:
            directory_path = Config.KNOWN_FACES_DIR
        
        print(f"[INFO] Processing faces from {directory_path}")
        
        # Reset encodings
        self.known_encodings = []
        self.known_names = []
        
        # Process each person's folder
        for person_name in os.listdir(directory_path):
            person_path = os.path.join(directory_path, person_name)
            
            if not os.path.isdir(person_path):
                continue
                
            print(f"[INFO] Processing {person_name}")
            
            # Get all images for this person
            image_paths = get_image_paths(person_path)
            
            for image_path in image_paths:
                # Load and process image
                image = load_image_safely(image_path)
                if image is None:
                    print(f"[WARNING] Could not load {image_path}")
                    continue
                
                # Convert BGR to RGB (OpenCV loads in BGR)
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Find face locations
                face_locations = face_recognition.face_locations(rgb_image, model=Config.MODEL)
                
                if len(face_locations) == 0:
                    print(f"[WARNING] No faces found in {image_path}")
                    continue
                
                # Generate face encodings
                face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
                
                # Store each encoding (in case multiple faces in one image)
                for encoding in face_encodings:
                    self.known_encodings.append(encoding)
                    self.known_names.append(person_name)
                    print(f"[INFO] Encoded face for {person_name}")
        
        print(f"[INFO] Total faces encoded: {len(self.known_encodings)}")
        return len(self.known_encodings)
    
    def save_encodings(self, filepath=None):
        """Save encodings to pickle file"""
        if filepath is None:
            filepath = Config.ENCODINGS_FILE
        
        data = {
            "encodings": self.known_encodings,
            "names": self.known_names
        }
        
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
        
        print(f"[INFO] Encodings saved to {filepath}")
    
    def load_encodings(self, filepath=None):
        """Load encodings from pickle file"""
        if filepath is None:
            filepath = Config.ENCODINGS_FILE
        
        if not os.path.exists(filepath):
            print(f"[WARNING] Encodings file not found: {filepath}")
            return False
        
        try:
            with open(filepath, "rb") as f:
                data = pickle.load(f)
            
            self.known_encodings = data["encodings"]
            self.known_names = data["names"]
            
            print(f"[INFO] Loaded {len(self.known_encodings)} face encodings")
            return True
        
        except Exception as e:
            print(f"[ERROR] Failed to load encodings: {str(e)}")
            return False
    
    def add_face_from_image(self, image_path, person_name):
        """Add a single face encoding from an image"""
        image = load_image_safely(image_path)
        if image is None:
            return False
        
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_image, model=Config.MODEL)
        
        if len(face_locations) == 0:
            print(f"[WARNING] No faces found in {image_path}")
            return False
        
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
        
        for encoding in face_encodings:
            self.known_encodings.append(encoding)
            self.known_names.append(person_name)
        
        print(f"[INFO] Added {len(face_encodings)} face(s) for {person_name}")
        return True
    
    def remove_person(self, person_name):
        """Remove all encodings for a specific person"""
        indices_to_remove = []
        
        for i, name in enumerate(self.known_names):
            if name == person_name:
                indices_to_remove.append(i)
        
        # Remove in reverse order to maintain indices
        for i in reversed(indices_to_remove):
            del self.known_encodings[i]
            del self.known_names[i]
        
        print(f"[INFO] Removed {len(indices_to_remove)} encodings for {person_name}")
        return len(indices_to_remove)
    
    def get_encoded_names(self):
        """Get list of unique names that have been encoded"""
        return list(set(self.known_names))

if __name__ == "__main__":
    # Example usage
    encoder = FaceEncoder()
    
    # Encode all faces from directory
    num_faces = encoder.encode_faces_from_directory()
    
    if num_faces > 0:
        # Save encodings
        encoder.save_encodings()
        print("Face encoding completed successfully!")
    else:
        print("No faces were encoded. Please check your known_faces directory structure.")