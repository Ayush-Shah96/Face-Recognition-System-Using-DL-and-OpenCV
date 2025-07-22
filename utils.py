# utils.py - Utility functions for face recognition project

import os
import cv2
import numpy as np
from PIL import Image

def get_image_paths(directory):
    """Get all image file paths from a directory"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_paths = []
    
    for filename in os.listdir(directory):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            image_paths.append(os.path.join(directory, filename))
    
    return sorted(image_paths)

def load_image_safely(image_path):
    """Safely load an image with error handling"""
    try:
        image = cv2.imread(image_path)
        if image is None:
            # Try with PIL for different formats
            pil_image = Image.open(image_path)
            image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        return image
    except Exception as e:
        print(f"[ERROR] Failed to load image {image_path}: {str(e)}")
        return None

def resize_image(image, max_width=800, max_height=600):
    """Resize image while maintaining aspect ratio"""
    height, width = image.shape[:2]
    
    # Calculate scaling factor
    scale_w = max_width / width
    scale_h = max_height / height
    scale = min(scale_w, scale_h, 1.0)  # Don't upscale
    
    if scale < 1.0:
        new_width = int(width * scale)
        new_height = int(height * scale)
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    return image

def draw_face_box(image, face_location, name, confidence):
    """Draw bounding box and label on face"""
    left, top, right, bottom = face_location
    
    # Choose color based on recognition
    if name == "Unknown":
        color = (0, 0, 255)  # Red for unknown
    else:
        color = (0, 255, 0)  # Green for known
    
    # Draw rectangle
    cv2.rectangle(image, (left, top), (right, bottom), color, 2)
    
    # Draw label background
    label = f"{name} ({confidence:.1f}%)" if confidence > 0 else name
    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)[0]
    
    cv2.rectangle(
        image, 
        (left, bottom - label_size[1] - 10),
        (left + label_size[0], bottom),
        color, 
        cv2.FILLED
    )
    
    # Draw label text
    cv2.putText(
        image, 
        label, 
        (left + 6, bottom - 6), 
        cv2.FONT_HERSHEY_DUPLEX, 
        0.6, 
        (255, 255, 255), 
        1
    )
    
    return image

def calculate_confidence(distance, threshold=0.6):
    """Calculate confidence percentage from face distance"""
    if distance > threshold:
        return 0.0
    
    # Convert distance to confidence percentage
    confidence = (1.0 - distance) * 100
    return max(0.0, min(100.0, confidence))

def create_face_mosaic(images, grid_size=(3, 3)):
    """Create a mosaic of face images"""
    if not images:
        return None
    
    rows, cols = grid_size
    if len(images) < rows * cols:
        # Pad with blank images if needed
        blank = np.zeros_like(images[0])
        images.extend([blank] * (rows * cols - len(images)))
    
    # Resize all images to same size
    target_size = (150, 150)
    resized_images = [cv2.resize(img, target_size) for img in images[:rows * cols]]
    
    # Create mosaic
    mosaic_rows = []
    for row in range(rows):
        row_images = resized_images[row * cols:(row + 1) * cols]
        mosaic_rows.append(np.hstack(row_images))
    
    mosaic = np.vstack(mosaic_rows)
    return mosaic

def extract_face_from_image(image, face_location, padding=20):
    """Extract face region from image with padding"""
    height, width = image.shape[:2]
    left, top, right, bottom = face_location
    
    # Add padding
    left = max(0, left - padding)
    top = max(0, top - padding)
    right = min(width, right + padding)
    bottom = min(height, bottom + padding)
    
    return image[top:bottom, left:right]

def enhance_image(image):
    """Apply basic image enhancement"""
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Split channels
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l)
    
    # Merge channels and convert back to BGR
    enhanced_lab = cv2.merge([enhanced_l, a, b])
    enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    return enhanced_image

def validate_image_quality(image, min_size=(50, 50)):
    """Check if image meets quality requirements for face recognition"""
    if image is None:
        return False, "Image is None"
    
    height, width = image.shape[:2]
    
    if width < min_size[0] or height < min_size[1]:
        return False, f"Image too small: {width}x{height}, minimum: {min_size}"
    
    # Check if image is too dark
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    
    if mean_brightness < 30:
        return False, "Image too dark"
    
    # Check if image is too blurry
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < 100:
        return False, "Image too blurry"
    
    return True, "Image quality acceptable"

def log_recognition_event(name, confidence, timestamp, image_path=None):
    """Log recognition events to file"""
    log_entry = f"{timestamp} - Recognized: {name} (Confidence: {confidence:.2f}%)"
    if image_path:
        log_entry += f" - Image: {image_path}"
    
    # Append to log file
    with open("face_recognition.log", "a") as f:
        f.write(log_entry + "\n")

def create_training_data_report(known_faces_dir):
    """Generate a report of training data"""
    report = {
        'total_people': 0,
        'total_images': 0,
        'people_details': {}
    }
    
    if not os.path.exists(known_faces_dir):
        return report
    
    for person_name in os.listdir(known_faces_dir):
        person_path = os.path.join(known_faces_dir, person_name)
        
        if not os.path.isdir(person_path):
            continue
        
        image_paths = get_image_paths(person_path)
        report['people_details'][person_name] = {
            'image_count': len(image_paths),
            'image_paths': image_paths
        }
        
        report['total_people'] += 1
        report['total_images'] += len(image_paths)
    
    return report

def benchmark_recognition_speed(recognizer, test_images, num_runs=5):
    """Benchmark face recognition speed"""
    import time
    
    total_time = 0
    total_faces = 0
    
    for run in range(num_runs):
        run_start = time.time()
        
        for image_path in test_images:
            image = load_image_safely(image_path)
            if image is None:
                continue
            
            start_time = time.time()
            _, faces = recognizer.recognize_faces_in_image(image_path, save_result=False)
            end_time = time.time()
            
            total_time += (end_time - start_time)
            total_faces += len(faces) if faces else 0
        
        run_end = time.time()
        print(f"Run {run + 1}: {run_end - run_start:.2f}s")
    
    avg_time_per_image = total_time / (len(test_images) * num_runs)
    avg_faces_per_second = total_faces / total_time if total_time > 0 else 0
    
    return {
        'avg_time_per_image': avg_time_per_image,
        'avg_faces_per_second': avg_faces_per_second,
        'total_time': total_time,
        'total_faces': total_faces
    }