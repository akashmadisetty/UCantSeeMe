import sys
import os
from datetime import datetime
import logging
from threading import Timer
from PIL import Image, ImageFilter, ImageDraw
import pytesseract
import re
import spacy
import numpy as np
import cv2
from huggingface_hub import hf_hub_download
from doclayout_yolo import YOLOv10
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QWidget, QVBoxLayout
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QPixmap, QScreen, QImage

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FileCleanup:
    @staticmethod
    def schedule_file_deletion(filepath, delay=30):
        """Schedule a file for deletion after specified delay in seconds"""
        def delete_file():
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
                    logger.info(f"Deleted file: {filepath}")
            except Exception as e:
                logger.error(f"Error deleting file {filepath}: {e}")
        
        Timer(delay, delete_file).start()

        
class PIIDetector:
    def __init__(self, debug_mode=True):
        self.debug_mode = debug_mode
        self.model = self.load_yolo_model()
        self.setup_pii_patterns()
        #self.setup_code_patterns()
        self.nlp = spacy.load('en_core_web_sm')

    '''def setup_code_patterns(self):
        """Patterns to identify code snippets and documentation"""
        self.code_patterns = {
            "import_statement": r'^\s*(?:from|import)\s+[\w\s.,*]+',
            "function_def": r'^\s*def\s+\w+\s*\(',
            "variable_assignment": r'^\s*\w+\s*=\s*',
            "comment": r'^\s*#.*$',
            "method_call": r'\w+\.(predict|plot|imwrite|save)',
            "parameter_list": r'^\s*[a-z_]+\s*=\s*[\d."]+\s*(?:#|$)',
            "path_pattern": r'(?:path/to|/|\.\.)/[\w./]+',
            "code_keywords": r'\b(import|def|class|return|print|if|else|for|while|try|except)\b'
        }
        
        self.code_related_words = {
            'model', 'predict', 'image', 'cuda', 'cpu', 'size', 'conf', 'device',
            'path', 'imgsz', 'result', 'annotate', 'frame', 'detection', 'script',
            'sdk', 'yolo', 'demo', 'prediction', 'threshold'
        }'''

    def load_yolo_model(self):
        """Load the YOLO model from Hugging Face"""
        try:
            filepath = hf_hub_download(
                repo_id="juliozhao/DocLayout-YOLO-DocStructBench", 
                filename="doclayout_yolo_docstructbench_imgsz1024.pt"
            )
            return YOLOv10(filepath)
        except Exception as e:
            logger.error(f"Error loading YOLO model: {e}")
            raise

    def setup_pii_patterns(self):
        """Define patterns for PII detection"""
        self.patterns = {
            "phone": r"\b(?:\+?(\d{1,3}))?[-. (](\d{3})[-. )](\d{3})[-. ]*(\d{4})\b",
            "ssn": r"\b\d{3}[- ]?\d{2}[- ]?\d{4}\b",
            "credit_card": r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12}|3[0-9]{4}|[36][89][0-9]{13})\b",
            "date": r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
            "address": r"\b\d+\s+[A-Za-z\s,]+(?:Street|St|Avenue|Ave|Marg|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr)\b",
            "name": r"\b(?:Mr\.|Mrs\.|Ms\.|Dr\.)?\s[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b",
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "ip_address": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
            "license_plate": r"\b[A-Z]{1,2}\d{1,6}[A-Z]{0,2}\b",
            "bank_account": r"\b\d{9,18}\b",
            "passport_number": r"\b[A-Z]{2}\d{7,9}[A-Z]{1,2}\b",
        }

    def process_image(self, image_path, output_path="processed_document.jpg", debug_visualization=True):
        """Process the image and blur sensitive regions"""
        try:
            image = Image.open(image_path)
            regions = self.detect_regions(image_path)
            
            if debug_visualization:
                debug_image = image.copy()
                draw = ImageDraw.Draw(debug_image)
            
            total_regions = len(regions)
            blurred_regions = 0
            
            for idx, region in enumerate(regions):
                coords = region['coordinates']
                extracted_region = image.crop(coords)
                text = self.perform_ocr(extracted_region)
                
                if self.detect_sensitive_identifiers(text):
                    blurred_regions += 1
                    region_width = coords[2] - coords[0]
                    blur_radius = max(10, int(region_width * 0.1))
                    blurred_region = extracted_region.filter(ImageFilter.GaussianBlur(radius=blur_radius))
                    image.paste(blurred_region, coords)
                    
                    if debug_visualization:
                        draw.rectangle(coords, outline="red", width=2)
                        draw.text((coords[0], coords[1]-20), f"PII {idx}", fill="red")
            
            logger.info(f"Processed {total_regions} regions, blurred {blurred_regions} sensitive regions")
            image.save(output_path)
            
            if debug_visualization:
                debug_image.save("debug_visualization.jpg")
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return False

    def detect_regions(self, image_path, conf_threshold=0.3):
        """Detect regions in the document using YOLO"""
        try:
            det_res = self.model.predict(
                image_path,
                imgsz=1024,
                conf=conf_threshold,
                device="cpu"
            )
            
            regions = []
            for i in range(len(det_res[0])):
                box = det_res[0].boxes[i]
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                confidence = box.conf[0].item()
                class_id = int(box.cls[0].item())
                class_name = det_res[0].names[class_id]
                
                padding = 5
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = x2 + padding
                y2 = y2 + padding
                
                regions.append({
                    'coordinates': (x1, y1, x2, y2),
                    'class': class_name,
                    'confidence': confidence
                })
            
            return regions
            
        except Exception as e:
            logger.error(f"Error in region detection: {e}")
            return []

    def improve_image_for_ocr(self, image):
        """Enhance image quality for better OCR results"""
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        return Image.fromarray(enhanced)

    def perform_ocr(self, image):
        """Perform OCR with improved image preprocessing"""
        '''enhanced_image = self.improve_image_for_ocr(image)
        custom_config = r'--oem 3 --psm 6'''
        try:
            text = pytesseract.image_to_string(image)
            return text
        except Exception as e:
            logger.error(f"OCR Error: {e}")
            return ""
        
    def detect_sensitive_identifiers(self, text):
      if not text.strip():
        return False

      #if self.is_code_content(text):
      #  return False

      for pattern_name, pattern in self.patterns.items():
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
          matched_text = match.group()
          if matched_text:
            return True

      doc = self.nlp(text)
      sensitive_ner_types = {'PERSON', 'ORG', 'GPE', 'MONEY', 'DATE'}

      for ent in doc.ents:
        if ent.label_ in sensitive_ner_types:
          #if not self.is_code_content(ent.text):
          #  return True
          return True

      return False
    

    '''def is_code_content(self, text):
        """Check if the text appears to be code-related"""
        text_lower = text.lower()
        
        for pattern in self.code_patterns.values():
            if re.search(pattern, text, re.MULTILINE):
                return True
                
        words = set(text_lower.split())
        if len(words.intersection(self.code_related_words)) > 0:
            return True
            
        if any(char in text for char in '{}[]()='):
            return True
            
        return False'''

import sys
import os
from datetime import datetime
import logging
from PIL import Image, ImageFilter, ImageDraw
import pytesseract
import re
import spacy
import numpy as np
import cv2
from huggingface_hub import hf_hub_download
from doclayout_yolo import YOLOv10
from PyQt6.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel, 
                            QWidget, QVBoxLayout, QHBoxLayout, QSizePolicy)
from PyQt6.QtCore import QTimer, Qt, QSize
from PyQt6.QtGui import QPixmap, QScreen, QImage

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProcessedScreenWindow(QWidget):
    def __init__(self, processed_dir):
        super().__init__()
        self.setWindowTitle("Protected View")
        self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.FramelessWindowHint)
        print('in first one!!')
        # Store the processed directory path
        self.processed_dir = processed_dir
        
        # Create layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create screen label
        self.screen_label = QLabel()
        self.screen_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.screen_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.screen_label)
        
        # Take initial screenshot
        self.take_initial_screenshot()
        
        # Set up timer for checking processed images
        self.timer = QTimer()
        self.timer.timeout.connect(self.check_processed_images)
        self.timer.start(500)  # Check every 500ms
        
        self.last_processed_image = None
        self.current_pixmap = None
        self.showMaximized()
        
    def take_initial_screenshot(self):
        screen = QApplication.primaryScreen()
        screenshot = screen.grabWindow(0)
        self.update_display(screenshot)
        
    def update_display(self, pixmap):
        try:
            if isinstance(pixmap, QPixmap) and not pixmap.isNull():
                # Store the current pixmap
                self.current_pixmap = pixmap
                # Scale the pixmap to fit the window while maintaining aspect ratio
                scaled_pixmap = pixmap.scaled(
                    self.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                self.screen_label.setPixmap(scaled_pixmap)
        except Exception as e:
            logger.error(f"Error updating display: {e}")
            
    def check_processed_images(self):
        try:
            processed_files = [f for f in os.listdir(self.processed_dir) 
                             if f.startswith('processed_') and f.endswith('.png')]
            
            if processed_files:
                latest_image = max(processed_files, 
                                 key=lambda x: os.path.getctime(
                                     os.path.join(self.processed_dir, x)))
                latest_path = os.path.join(self.processed_dir, latest_image)
                
                # Only update if we have a new image
                if latest_path != self.last_processed_image:
                    pixmap = QPixmap(latest_path)
                    if not pixmap.isNull():
                        self.update_display(pixmap)
                        self.last_processed_image = latest_path
                        
        except Exception as e:
            logger.error(f"Error checking processed images: {e}")
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Re-scale the current pixmap when window is resized
        if self.current_pixmap:
            self.update_display(self.current_pixmap)
            
    def mousePressEvent(self, event):
        self.close()

class ScreenWindow(QWidget):
    def __init__(self, change_label, count_label, pii_label, captured_dir, processed_dir, pii_detector, processed_window=None):
        super().__init__()
        self.setWindowTitle("UCan'tSeeMe")
        self.change_label = change_label
        self.count_label = count_label
        self.pii_label = pii_label
        self.captured_dir = captured_dir
        self.processed_dir = processed_dir
        self.pii_detector = pii_detector
        self.processed_window = processed_window
        self.screenshot_count = 0
        
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        
        layout = QVBoxLayout(self)
        self.screen_label = QLabel()
        layout.addWidget(self.screen_label)
        
        self.previous_image = None
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.check_screen_change)
        self.timer.start(500)  # Check every 500ms
        
        self.showMaximized()
    
    def qimage_to_bytes(self, qimage):
        """Convert QImage to bytes for comparison"""
        buffer = QImage(qimage)
        if buffer.format() != QImage.Format.Format_RGB888:
            buffer = buffer.convertToFormat(QImage.Format.Format_RGB888)
        
        buffer = buffer.scaled(100, 100, Qt.AspectRatioMode.KeepAspectRatio)
        byte_array = bytes(buffer.bits().asarray(buffer.width() * buffer.height() * 3))
        return byte_array
    
    def save_and_process_screenshot(self, screenshot):
        """Save screenshot and process it for PII"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.screenshot_count += 1
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save original screenshot
            original_filename = f"screenshot_{timestamp}.png"
            original_filepath = os.path.join(self.captured_dir, original_filename)
            screenshot.save(original_filepath, "PNG")

            #to delete
            FileCleanup.schedule_file_deletion(original_filepath)
            
            # Process for PII
            processed_filename = f"processed_{timestamp}.png"
            processed_filepath = os.path.join(self.processed_dir, processed_filename)
            
            success = self.pii_detector.process_image(
                original_filepath,
                processed_filepath,
                debug_visualization=True
            )

            FileCleanup.schedule_file_deletion(processed_filepath)

            debug_filepath = "debug_visualization.jpg"
            if os.path.exists(debug_filepath):
                FileCleanup.schedule_file_deletion(debug_filepath)
            self.count_label.setText(f"Screenshots: {self.screenshot_count}")
            self.pii_label.setText(f"PII Status: {'Detected and blurred' if success else 'Not detected'}")
            
            # If processed window exists, trigger an immediate update
            if self.processed_window and success:
                self.processed_window.check_new_image()
                
            logger.info(f"Saved: {original_filepath}")
            if success:
                logger.info(f"Processed: {processed_filepath}")
                
        except Exception as e:
            logger.error(f"Error in save_and_process_screenshot: {e}")
    
    def check_screen_change(self):
        try:
            screen = QApplication.primaryScreen()
            screenshot = screen.grabWindow(0)
            current_image = screenshot.toImage()
            current_bytes = self.qimage_to_bytes(current_image)
            
            if self.previous_image is not None:
                has_changed = current_bytes != self.previous_image
                status = "Change Detected!" if has_changed else "No Change"
                self.change_label.setText(f"Status: {status}")
                
                if has_changed:
                    self.save_and_process_screenshot(screenshot)
            
            # Update display
            self.screen_label.setPixmap(screenshot.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            ))
            
            # Update previous image
            self.previous_image = current_bytes
            
        except Exception as e:
            logger.error(f"Error in check_screen_change: {e}")
            self.change_label.setText(f"Error: {str(e)}")
    
    def mousePressEvent(self, event):
        self.close()

class ProcessedScreenWindow(QWidget):
    def __init__(self, processed_dir):
        super().__init__()
        self.setWindowTitle("Protected View")
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        print('in second one!!')
        # Store the processed directory path
        self.processed_dir = processed_dir
        
        # Create layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create screen label
        self.screen_label = QLabel()
        layout.addWidget(self.screen_label)
        
        # Take initial screenshot
        self.take_initial_screenshot()
        
        # Set up timer for checking processed images
        self.timer = QTimer()
        self.timer.timeout.connect(self.check_processed_images)
        self.timer.start(500)  # Check every 500ms
        
        self.last_processed_image = None
        self.showMaximized()
        
    def take_initial_screenshot(self):
        screen = QApplication.primaryScreen()
        screenshot = screen.grabWindow(0)
        self.update_display(screenshot)
        
    def update_display(self, pixmap):
        scaled_pixmap = pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.screen_label.setPixmap(scaled_pixmap)
        
    def check_processed_images(self):
        try:
            processed_files = [f for f in os.listdir(self.processed_dir) 
                             if f.startswith('processed_') and f.endswith('.png')]
            
            if processed_files:
                latest_image = max(processed_files, 
                                 key=lambda x: os.path.getctime(
                                     os.path.join(self.processed_dir, x)))
                latest_path = os.path.join(self.processed_dir, latest_image)
                
                if latest_path != self.last_processed_image:
                    pixmap = QPixmap(latest_path)
                    if not pixmap.isNull():
                        self.update_display(pixmap)
                        self.last_processed_image = latest_path
                        
        except Exception as e:
            logger.error(f"Error checking processed images: {e}")
            
    def mousePressEvent(self, event):
        self.close()

class ScreenShareApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Screen Monitor with PII Detection")
        self.setGeometry(100, 100, 300, 200)
        
        # Initialize PII detector
        self.pii_detector = PIIDetector()
        
        # Create directories
        self.captured_dir = "captured"
        os.makedirs(self.captured_dir, exist_ok=True)
        self.processed_dir = "processed"
        os.makedirs(self.processed_dir, exist_ok=True)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Create buttons layout
        button_layout = QHBoxLayout()
        
        # Create start button
        self.start_button = QPushButton("Start Monitoring", self)
        self.start_button.clicked.connect(self.start_screen_sharing)
        button_layout.addWidget(self.start_button)
        
        # Create show processed view button
        self.show_processed_button = QPushButton("Show Protected View", self)
        self.show_processed_button.clicked.connect(self.show_processed_window)
        self.show_processed_button.setCheckable(True)  # Make button toggleable
        button_layout.addWidget(self.show_processed_button)
        
        layout.addLayout(button_layout)
        
        # Add status labels
        self.change_label = QLabel("Status: Not started")
        layout.addWidget(self.change_label)
        
        self.count_label = QLabel("Screenshots: 0")
        layout.addWidget(self.count_label)
        
        self.pii_label = QLabel("PII Status: Not detected")
        layout.addWidget(self.pii_label)
        
        # Initialize windows
        self.screen_window = None
        self.processed_screen_window = None
        
    def closeEvent(self, event):
        """Handle application closing"""
        for directory in [self.captured_dir, self.processed_dir]:
            try:
                for filename in os.listdir(directory):
                    filepath = os.path.join(directory, filename)
                    try:
                        if os.path.isfile(filepath):
                            os.remove(filepath)
                            logger.info(f"Cleaned up file: {filepath}")
                    except Exception as e:
                        logger.error(f"Error cleaning up file {filepath}: {e}")
            except Exception as e:
                logger.error(f"Error cleaning up directory {directory}: {e}")
                
        if self.processed_screen_window:
            self.processed_screen_window.close()
        if self.screen_window:
            self.screen_window.close()
        event.accept()
            
    def show_processed_window(self):
        """Toggle the processed window view"""
        if not self.processed_screen_window:
            # Create and show the window
            self.processed_screen_window = ProcessedScreenWindow(self.processed_dir)
            self.processed_screen_window.show()
            self.show_processed_button.setText("Hide Protected View")
            self.show_processed_button.setChecked(True)
            
            # Connect the window's close event
            self.processed_screen_window.destroyed.connect(self.on_processed_window_closed)
        else:
            self.processed_screen_window.close()
            self.processed_screen_window = None
            self.show_processed_button.setText("Show Protected View")
            self.show_processed_button.setChecked(False)
    
    def on_processed_window_closed(self):
        """Handle processed window being closed"""
        self.processed_screen_window = None
        self.show_processed_button.setText("Show Protected View")
        self.show_processed_button.setChecked(False)
            
    def start_screen_sharing(self):
        """Toggle screen sharing"""
        if self.screen_window is None:
            self.screen_window = ScreenWindow(
                self.change_label,
                self.count_label,
                self.pii_label,
                self.captured_dir,
                self.processed_dir,
                self.pii_detector,
                self.processed_screen_window
            )
            self.screen_window.show()
            self.start_button.setText("Stop Monitoring")
        else:
            self.screen_window.close()
            self.screen_window = None
            self.start_button.setText("Start Monitoring")

def main():
    try:
        app = QApplication(sys.argv)
        window = ScreenShareApp()
        window.show()
        sys.exit(app.exec())
    except Exception as e:
        logger.error(f"Application error: {e}")

if __name__ == '__main__':
    main()


