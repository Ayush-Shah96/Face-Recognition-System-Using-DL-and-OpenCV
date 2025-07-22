# main.py - Main application interface for Face Recognition System

import os
import sys
import argparse
from datetime import datetime
from config import Config
from face_encoder import FaceEncoder
from face_recognizer import FaceRecognizer
from utils import create_training_data_report, benchmark_recognition_speed

class FaceRecognitionApp:
    def __init__(self):
        self.encoder = FaceEncoder()
        self.recognizer = FaceRecognizer()
        Config.create_directories()
    
    def setup_training_data(self):
        """Interactive setup for training data"""
        print("\n=== Face Recognition Setup ===")
        print(f"Place your training images in: {Config.KNOWN_FACES_DIR}")
        print("Structure: known_faces/person_name/image1.jpg")
        print("Example:")
        print("  known_faces/")
        print("  ├── john_doe/")
        print("  │   ├── photo1.jpg")
        print("  │   └── photo2.jpg")
        print("  └── jane_smith/")
        print("      ├── image1.jpg")
        print("      └── image2.jpg")
        
        input("\nPress Enter when you have added your training images...")
        
        # Generate report
        report = create_training_data_report(Config.KNOWN_FACES_DIR)
        
        if report['total_people'] == 0:
            print("\n[WARNING] No training data found!")
            print("Please add images to the known_faces directory first.")
            return False
        
        print(f"\n[INFO] Found {report['total_people']} people with {report['total_images']} total images:")
        for person, details in report['people_details'].items():
            print(f"  - {person}: {details['image_count']} images")
        
        return True
    
    def train_model(self):
        """Train the face recognition model"""
        print("\n=== Training Face Recognition Model ===")
        
        # Encode faces
        num_faces = self.encoder.encode_faces_from_directory()
        
        if num_faces == 0:
            print("[ERROR] No faces were encoded. Please check your training data.")
            return False
        
        # Save encodings
        self.encoder.save_encodings()
        print(f"[SUCCESS] Model trained with {num_faces} face encodings!")
        
        return True
    
    def interactive_menu(self):
        """Interactive command-line menu"""
        while True:
            print("\n" + "="*50)
            print("FACE RECOGNITION SYSTEM")
            print("="*50)
            print("1. Setup Training Data")
            print("2. Train Model")
            print("3. Recognize Faces (Webcam)")
            print("4. Recognize Faces (Image)")
            print("5. Recognize Faces (Video)")
            print("6. Add New Person")
            print("7. Remove Person")
            print("8. View Statistics")
            print("9. Benchmark Performance")
            print("0. Exit")
            print("-"*50)
            
            try:
                choice = input("Select option: ").strip()
                
                if choice == '1':
                    self.setup_training_data()
                
                elif choice == '2':
                    self.train_model()
                
                elif choice == '3':
                    self.start_webcam_recognition()
                
                elif choice == '4':
                    self.recognize_image()
                
                elif choice == '5':
                    self.recognize_video()
                
                elif choice == '6':
                    self.add_new_person()
                
                elif choice == '7':
                    self.remove_person()
                
                elif choice == '8':
                    self.show_statistics()
                
                elif choice == '9':
                    self.run_benchmark()
                
                elif choice == '0':
                    print("Goodbye!")
                    break
                
                else:
                    print("[ERROR] Invalid option. Please try again.")
            
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"[ERROR] An error occurred: {str(e)}")
    
    def start_webcam_recognition(self):
        """Start webcam face recognition"""
        stats = self.recognizer.get_recognition_stats()
        if isinstance(stats, str):
            print(f"[ERROR] {stats}")
            print("Please train the model first.")
            return
        
        print(f"\n[INFO] Starting webcam recognition...")
        print(f"[INFO] Loaded {stats['total_encodings']} encodings for {stats['unique_people']} people")
        print("[INFO] Press 'q' to quit, 's' to save frame")
        
        self.recognizer.recognize_faces_webcam()
    
    def recognize_image(self):
        """Recognize faces in an image file"""
        image_path = input("\nEnter image path: ").strip()
        
        if not os.path.exists(image_path):
            print(f"[ERROR] Image not found: {image_path}")
            return
        
        print("[INFO] Processing image...")
        result = self.recognizer.recognize_faces_in_image(image_path)
        
        if result:
            image, faces = result
            print(f"[INFO] Found {len(faces)} faces:")
            
            for i, face in enumerate(faces, 1):
                name = face['name']
                confidence = face['confidence']
                print(f"  Face {i}: {name} (Confidence: {confidence:.1f}%)")
            
            print("[INFO] Result saved to output directory")
        else:
            print("[ERROR] Failed to process image")
    
    def recognize_video(self):
        """Recognize faces in a video file"""
        video_path = input("\nEnter video path: ").strip()
        
        if not os.path.exists(video_path):
            print(f"[ERROR] Video not found: {video_path}")
            return
        
        output_path = input("Enter output path (press Enter to skip saving): ").strip()
        if not output_path:
            output_path = None
        
        print("[INFO] Processing video...")
        success = self.recognizer.recognize_faces_video(video_path, output_path)
        
        if success:
            print("[INFO] Video processing completed!")
        else:
            print("[ERROR] Failed to process video")
    
    def add_new_person(self):
        """Add a new person to the recognition system"""
        person_name = input("\nEnter person's name: ").strip()
        if not person_name:
            print("[ERROR] Name cannot be empty")
            return
        
        image_path = input("Enter image path: ").strip()
        if not os.path.exists(image_path):
            print(f"[ERROR] Image not found: {image_path}")
            return
        
        # Load existing encodings
        self.encoder.load_encodings()
        
        # Add new face
        success = self.encoder.add_face_from_image(image_path, person_name)
        
        if success:
            # Save updated encodings
            self.encoder.save_encodings()
            print(f"[SUCCESS] Added {person_name} to the system!")
            
            # Reload in recognizer
            self.recognizer.encoder.load_encodings()
        else:
            print(f"[ERROR] Failed to add {person_name}")
    
    def remove_person(self):
        """Remove a person from the recognition system"""
        # Load encodings first
        self.encoder.load_encodings()
        
        encoded_names = self.encoder.get_encoded_names()
        if not encoded_names:
            print("[ERROR] No people in the system")
            return
        
        print("\nCurrent people in system:")
        for i, name in enumerate(encoded_names, 1):
            print(f"  {i}. {name}")
        
        try:
            choice = int(input("Select person to remove (number): "))
            if 1 <= choice <= len(encoded_names):
                person_name = encoded_names[choice - 1]
                removed_count = self.encoder.remove_person(person_name)
                
                if removed_count > 0:
                    # Save updated encodings
                    self.encoder.save_encodings()
                    print(f"[SUCCESS] Removed {person_name} ({removed_count} encodings)")
                    
                    # Reload in recognizer
                    self.recognizer.encoder.load_encodings()
                else:
                    print(f"[ERROR] Failed to remove {person_name}")
            else:
                print("[ERROR] Invalid selection")
        except ValueError:
            print("[ERROR] Please enter a valid number")
    
    def show_statistics(self):
        """Display system statistics"""
        print("\n" + "="*30)
        print("SYSTEM STATISTICS")
        print("="*30)
        
        stats = self.recognizer.get_recognition_stats()
        
        if isinstance(stats, str):
            print(stats)
        else:
            print(f"Total encodings: {stats['total_encodings']}")
            print(f"Unique people: {stats['unique_people']}")
            print("\nPeople in system:")
            for person in stats['people']:
                print(f"  - {person}")
        
        # Training data report
        report = create_training_data_report(Config.KNOWN_FACES_DIR)
        print(f"\nTraining data:")
        print(f"  People: {report['total_people']}")
        print(f"  Images: {report['total_images']}")
        
        # Configuration
        print(f"\nConfiguration:")
        print(f"  Model: {Config.MODEL}")
        print(f"  Tolerance: {Config.TOLERANCE}")
        print(f"  Known faces directory: {Config.KNOWN_FACES_DIR}")
        print(f"  Encodings file: {Config.ENCODINGS_FILE}")
    
    def run_benchmark(self):
        """Run performance benchmark"""
        print("\n[INFO] Running performance benchmark...")
        
        # Get test images from known_faces directory
        test_images = []
        for person_dir in os.listdir(Config.KNOWN_FACES_DIR):
            person_path = os.path.join(Config.KNOWN_FACES_DIR, person_dir)
            if os.path.isdir(person_path):
                for img_file in os.listdir(person_path):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        test_images.append(os.path.join(person_path, img_file))
        
        if not test_images:
            print("[ERROR] No test images found")
            return
        
        # Run benchmark (limit to first 10 images for speed)
        benchmark_results = benchmark_recognition_speed(
            self.recognizer, 
            test_images[:10], 
            num_runs=3
        )
        
        print(f"\nBenchmark Results:")
        print(f"  Avg time per image: {benchmark_results['avg_time_per_image']:.3f}s")
        print(f"  Avg faces per second: {benchmark_results['avg_faces_per_second']:.1f}")
        print(f"  Total processing time: {benchmark_results['total_time']:.2f}s")
        print(f"  Total faces processed: {benchmark_results['total_faces']}")

def main():
    parser = argparse.ArgumentParser(description='Face Recognition System')
    parser.add_argument('--mode', choices=['interactive', 'webcam', 'image', 'video', 'train'], 
                       default='interactive', help='Operation mode')
    parser.add_argument('--input', help='Input file path (for image/video mode)')
    parser.add_argument('--output', help='Output file path')
    parser.add_argument('--tolerance', type=float, default=0.6, help='Recognition tolerance')
    parser.add_argument('--model', choices=['hog', 'cnn'], default='hog', help='Face detection model')
    
    args = parser.parse_args()
    
    # Update config with command line arguments
    Config.TOLERANCE = args.tolerance
    Config.MODEL = args.model
    
    app = FaceRecognitionApp()
    
    try:
        if args.mode == 'interactive':
            app.interactive_menu()
        
        elif args.mode == 'train':
            print("Training face recognition model...")
            if app.setup_training_data():
                app.train_model()
        
        elif args.mode == 'webcam':
            app.start_webcam_recognition()
        
        elif args.mode == 'image':
            if not args.input:
                print("Error: --input required for image mode")
                sys.exit(1)
            app.recognize_image_from_path(args.input)
        
        elif args.mode == 'video':
            if not args.input:
                print("Error: --input required for video mode")
                sys.exit(1)
            app.recognize_video_from_path(args.input, args.output)
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()