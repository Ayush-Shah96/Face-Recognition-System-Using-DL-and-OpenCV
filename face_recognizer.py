# face_recognizer.py - Real-time face recognition

import cv2
import face_recognition
import numpy as np
import os
from datetime import datetime
from config import Config
from face_encoder import FaceEncoder
from utils import draw_face_box, calculate_confidence

class FaceRecognizer:
    def __init__(self):
        self.encoder = FaceEncoder()
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Load encodings
        if not self.encoder.load_encodings():
            print("[WARNING] No pre-encoded faces found. Please run face encoding first.")
    
    def recognize_faces_in_image(self, image_path, save_result=True):
        """Recognize faces in a single image"""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"[ERROR] Could not load image: {image_path}")
            return None
        
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Find faces
        face_locations = face_recognition.face_locations(rgb_image, model=Config.MODEL)
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
        
        recognized_faces = []
        
        # Process each face
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Compare with known faces
            matches = face_recognition.compare_faces(
                self.encoder.known_encodings, 
                face_encoding, 
                tolerance=Config.TOLERANCE
            )
            
            name = "Unknown"
            confidence = 0
            
            if True in matches:
                # Calculate distances to find best match
                face_distances = face_recognition.face_distance(self.encoder.known_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                
                if matches[best_match_index]:
                    name = self.encoder.known_names[best_match_index]
                    confidence = calculate_confidence(face_distances[best_match_index])
            
            recognized_faces.append({
                'name': name,
                'confidence': confidence,
                'location': (left, top, right, bottom)
            })
            
            # Draw rectangle and label
            image = draw_face_box(image, (left, top, right, bottom), name, confidence)
        
        # Save result if requested
        if save_result and recognized_faces:
            output_path = os.path.join(Config.OUTPUT_DIR, f"recognized_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
            cv2.imwrite(output_path, image)
            print(f"[INFO] Result saved to {output_path}")
        
        return image, recognized_faces
    
    def recognize_faces_webcam(self):
        """Real-time face recognition using webcam"""
        print("[INFO] Starting webcam face recognition. Press 'q' to quit, 's' to save frame.")
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)
        
        # Process every nth frame for performance
        process_every_n_frames = 3
        frame_count = 0
        
        face_locations = []
        face_encodings = []
        face_names = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to capture frame")
                break
            
            # Resize frame for faster processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # Process faces every nth frame
            if frame_count % process_every_n_frames == 0:
                face_locations = face_recognition.face_locations(rgb_small_frame, model=Config.MODEL)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                
                face_names = []
                face_confidences = []
                
                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(
                        self.encoder.known_encodings, 
                        face_encoding, 
                        tolerance=Config.TOLERANCE
                    )
                    
                    name = "Unknown"
                    confidence = 0
                    
                    if True in matches:
                        face_distances = face_recognition.face_distance(self.encoder.known_encodings, face_encoding)
                        best_match_index = np.argmin(face_distances)
                        
                        if matches[best_match_index]:
                            name = self.encoder.known_names[best_match_index]
                            confidence = calculate_confidence(face_distances[best_match_index])
                    
                    face_names.append(name)
                    face_confidences.append(confidence)
            
            # Draw results on full frame
            for (top, right, bottom, left), name, confidence in zip(face_locations, face_names, face_confidences):
                # Scale back up face locations
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                
                frame = draw_face_box(frame, (left, top, right, bottom), name, confidence)
            
            # Display the frame
            cv2.imshow('Face Recognition', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                save_path = os.path.join(Config.OUTPUT_DIR, f"webcam_capture_{timestamp}.jpg")
                cv2.imwrite(save_path, frame)
                print(f"[INFO] Frame saved to {save_path}")
            
            frame_count += 1
        
        cap.release()
        cv2.destroyAllWindows()
    
    def recognize_faces_video(self, video_path, output_path=None):
        """Process video file for face recognition"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"[ERROR] Could not open video: {video_path}")
            return False
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup video writer if output path provided
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"[INFO] Processing video: {total_frames} frames at {fps} FPS")
        
        frame_count = 0
        process_every_n_frames = 3  # Process every 3rd frame for performance
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            processed_frame = frame.copy()
            
            # Process faces every nth frame
            if frame_count % process_every_n_frames == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_frame, model=Config.MODEL)
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                
                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    matches = face_recognition.compare_faces(
                        self.encoder.known_encodings, 
                        face_encoding, 
                        tolerance=Config.TOLERANCE
                    )
                    
                    name = "Unknown"
                    confidence = 0
                    
                    if True in matches:
                        face_distances = face_recognition.face_distance(self.encoder.known_encodings, face_encoding)
                        best_match_index = np.argmin(face_distances)
                        
                        if matches[best_match_index]:
                            name = self.encoder.known_names[best_match_index]
                            confidence = calculate_confidence(face_distances[best_match_index])
                    
                    processed_frame = draw_face_box(processed_frame, (left, top, right, bottom), name, confidence)
            
            # Write frame if output specified
            if output_path:
                out.write(processed_frame)
            
            # Progress indicator
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"[INFO] Progress: {progress:.1f}%")
            
            frame_count += 1
        
        cap.release()
        if output_path:
            out.release()
            print(f"[INFO] Processed video saved to {output_path}")
        
        return True
    
    def get_recognition_stats(self):
        """Get statistics about known faces"""
        if not self.encoder.known_encodings:
            return "No faces encoded"
        
        unique_names = set(self.encoder.known_names)
        stats = {
            "total_encodings": len(self.encoder.known_encodings),
            "unique_people": len(unique_names),
            "people": list(unique_names)
        }
        
        return stats

if __name__ == "__main__":
    recognizer = FaceRecognizer()
    
    # Print stats
    stats = recognizer.get_recognition_stats()
    print(f"Recognition Stats: {stats}")
    
    # Start webcam recognition
    recognizer.recognize_faces_webcam()