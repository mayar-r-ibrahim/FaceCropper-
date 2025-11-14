import sys
import os
import cv2
import numpy as np
import logging
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QFileDialog, QProgressBar, QMessageBox,
    QTreeView, QAbstractItemView, QListView, QSpinBox, QDoubleSpinBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QStandardItemModel, QStandardItem

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='face_detection.log',
    filemode='w'
)
logger = logging.getLogger(__name__)

def read_image_unicode(path):
    """Read image with proper Unicode path handling"""
    try:
        # Try normal reading first
        img = cv2.imread(path)
        if img is not None:
            return img
            
        # If normal reading fails, try alternative method
        stream = open(path, "rb")
        bytes = bytearray(stream.read())
        numpyarray = np.asarray(bytes, dtype=np.uint8)
        img = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)
        return img
    except Exception as e:
        logger.error(f"Error reading image: {str(e)}")
        return None

class FaceDetectorThread(QThread):
    progress_updated = pyqtSignal(int, str)
    status_updated = pyqtSignal(str)
    faces_detected = pyqtSignal(int, str)
    folder_completed = pyqtSignal(str)
    finished = pyqtSignal()
    
    def __init__(self, input_folders, prototxt_path, caffemodel_path, margin, confidence_threshold):
        super().__init__()
        self.input_folders = input_folders
        self.prototxt_path = prototxt_path
        self.caffemodel_path = caffemodel_path
        self.margin = margin
        self.confidence_threshold = confidence_threshold
        self.cancel_requested = False
        logger.info(
            f"Initialized FaceDetectorThread with {len(input_folders)} folders, "
            f"margin {self.margin}, and confidence threshold {self.confidence_threshold}"
        )
        
    def run(self):
        logger.info("Starting face detection process")
        
        # Initialize DNN face detector
        try:
            logger.info(f"Loading Caffe model from: {self.prototxt_path} and {self.caffemodel_path}")
            net = cv2.dnn.readNetFromCaffe(self.prototxt_path, self.caffemodel_path)
            logger.info("Successfully loaded Caffe model")
        except Exception as e:
            error_msg = f"Error loading model: {str(e)}"
            logger.error(error_msg)
            self.status_updated.emit(error_msg)
            return
            
        total_folders = len(self.input_folders)
        processed_folders = 0
        
        for folder_idx, input_folder in enumerate(self.input_folders):
            if self.cancel_requested:
                logger.info("Processing canceled by user")
                break
                
            # Create output folder if it doesn't exist
            output_folder = os.path.join(input_folder, "cropped_faces")
            os.makedirs(output_folder, exist_ok=True)
            logger.info(f"Processing folder {folder_idx+1}/{total_folders}: {input_folder}")
            logger.info(f"Output folder: {output_folder}")
            
            # Get list of image files
            image_files = []
            valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
            for f in os.listdir(input_folder):
                file_path = os.path.join(input_folder, f)
                if os.path.isfile(file_path) and f.lower().endswith(valid_extensions):
                    image_files.append(file_path)
            
            logger.info(f"Found {len(image_files)} images in folder")
            
            if not image_files:
                status_msg = f"No supported images in {os.path.basename(input_folder)}"
                logger.warning(status_msg)
                self.status_updated.emit(status_msg)
                self.progress_updated.emit(int(((folder_idx+1) / total_folders) * 100), 
                                           f"Skipped: {os.path.basename(input_folder)} (no images)")
                processed_folders += 1
                continue
                
            total_faces = 0
            processed_count = 0
            
            for i, img_path in enumerate(image_files):
                if self.cancel_requested:
                    logger.info("Processing canceled by user")
                    break
                    
                current_status = f"Folder {folder_idx+1}/{total_folders}: Processing {os.path.basename(img_path)}"
                logger.info(current_status)
                self.status_updated.emit(current_status)
                
                # Calculate overall progress
                folder_progress = (folder_idx / total_folders) * 100
                file_progress = (i / len(image_files)) * (100 / total_folders)
                overall_progress = int(folder_progress + file_progress)
                self.progress_updated.emit(overall_progress, 
                                           f"{os.path.basename(input_folder)}: {i+1}/{len(image_files)}")
                
                try:
                    # Read image with Unicode support
                    logger.debug(f"Reading image: {img_path}")
                    img = read_image_unicode(img_path)
                    if img is None:
                        logger.warning(f"Failed to read image: {img_path}")
                        continue
                        
                    # Convert 4-channel images to 3-channel BGR (fix for RGBA images)
                    if len(img.shape) == 3 and img.shape[2] == 4:
                        logger.debug(f"Converting 4-channel image to 3-channel: {img_path}")
                        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                        
                    (h, w) = img.shape[:2]
                    logger.debug(f"Image dimensions: {w}x{h}, Channels: {img.shape[2] if len(img.shape) == 3 else 1}")
                    
                    # Prepare input image for DNN
                    logger.debug("Preparing blob for DNN")
                    blob = cv2.dnn.blobFromImage(
                        cv2.resize(img, (300, 300)), 
                        1.0, 
                        (300, 300), 
                        (104.0, 177.0, 123.0)  # Mean subtraction values
                    )
                    
                    # Perform face detection
                    logger.debug("Running face detection")
                    net.setInput(blob)
                    detections = net.forward()
                    logger.debug(f"Detections shape: {detections.shape}")
                    
                    # Process detections
                    faces = []
                    for j in range(0, detections.shape[2]):
                        confidence = detections[0, 0, j, 2]
                        logger.debug(f"Detection {j}: confidence = {confidence:.2f}")
                        
                        # Filter out weak detections
                        if confidence > self.confidence_threshold:  # Confidence threshold
                            box = detections[0, 0, j, 3:7] * np.array([w, h, w, h])
                            (x1, y1, x2, y2) = box.astype("int")
                            logger.debug(f"Raw box coordinates: {x1}, {y1}, {x2}, {y2}")
                            
                            # Ensure coordinates are within image bounds
                            x1 = max(0, x1)
                            y1 = max(0, y1)
                            x2 = min(w, x2)
                            y2 = min(h, y2)
                            
                            # Only add if valid face area
                            if x2 > x1 and y2 > y1:
                                faces.append((x1, y1, x2, y2))
                                logger.debug(f"Valid face area: {x1}, {y1}, {x2}, {y2}")
                            else:
                                logger.debug("Invalid face area - skipped")
                    
                    logger.info(f"Found {len(faces)} faces in {os.path.basename(img_path)}")
                    
                    # Crop and save each face
                    for face_idx, (x1, y1, x2, y2) in enumerate(faces):
                        # Add specified margin
                        margin = self.margin
                        x1 = max(0, x1 - margin)
                        y1 = max(0, y1 - margin)
                        x2 = min(w, x2 + margin)
                        y2 = min(h, y2 + margin)
                        
                        # Crop face with margin
                        face_img = img[y1:y2, x1:x2]
                        
                        # Save cropped face
                        filename = os.path.basename(img_path)
                        name, ext = os.path.splitext(filename)
                        output_path = os.path.join(
                            output_folder, 
                            f"{name}_face_{face_idx+1}{ext}"
                        )
                        
                        # Use imwrite with Unicode support
                        success = cv2.imencode(ext, face_img)[1].tofile(output_path)
                        if success:
                            logger.debug(f"Saved face to: {output_path}")
                        else:
                            logger.error(f"Failed to save face to: {output_path}")
                        
                        total_faces += 1
                    
                    processed_count += 1
                except Exception as e:
                    error_msg = f"Error processing {img_path}: {str(e)}"
                    logger.error(error_msg, exc_info=True)
                    self.status_updated.emit(error_msg)
            
            # Emit folder completion signal
            folder_result = f"Processed {processed_count} images, found {total_faces} faces in {os.path.basename(input_folder)}"
            self.faces_detected.emit(total_faces, os.path.basename(input_folder))
            self.folder_completed.emit(folder_result)
            processed_folders += 1
        
        final_status = (
            f"Completed {processed_folders}/{total_folders} folders" + 
            (" (canceled)" if self.cancel_requested else "")
        )
        logger.info(final_status)
        self.progress_updated.emit(100, "Completed")
        self.status_updated.emit(final_status)
        self.finished.emit()
    
    def cancel(self):
        logger.info("Cancel requested")
        self.cancel_requested = True

class FolderSelectionDialog(QFileDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setOption(QFileDialog.DontUseNativeDialog, True)
        self.setFileMode(QFileDialog.Directory)
        self.setOption(QFileDialog.ShowDirsOnly, True)
        
        # Enable multi-selection
        tree_view = self.findChild(QTreeView)
        if tree_view:
            tree_view.setSelectionMode(QAbstractItemView.ExtendedSelection)
            tree_view.setSelectionBehavior(QAbstractItemView.SelectRows)
        
        list_view = self.findChild(QListView)
        if list_view:
            list_view.setSelectionMode(QAbstractItemView.ExtendedSelection)
            list_view.setSelectionBehavior(QAbstractItemView.SelectRows)

class FaceCropperApp(QMainWindow):
    def __init__(self):
        super().__init__()
        logger.info("Initializing FaceCropperApp")
        
        # Model paths - update these if needed
        self.model_dir = r"D:\programs\facedetect"
        self.prototxt_path = os.path.join(self.model_dir, "deploy.prototxt")
        self.caffemodel_path = os.path.join(self.model_dir, "res10_300x300_ssd_iter_140000.caffemodel")
        logger.info(f"Model paths - prototxt: {self.prototxt_path}, caffemodel: {self.caffemodel_path}")
        
        # Verify model files exist
        if not os.path.exists(self.prototxt_path):
            logger.error(f"Prototxt file not found at: {self.prototxt_path}")
        if not os.path.exists(self.caffemodel_path):
            logger.error(f"Caffemodel file not found at: {self.caffemodel_path}")
        
        self.worker = None
        self.selected_folders = []
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("Face Detection and Cropping Tool (Multi-Folder)")
        self.setGeometry(100, 100, 800, 600)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Folder selection
        folder_layout = QHBoxLayout()
        self.folder_label = QLabel("No folders selected")
        folder_layout.addWidget(self.folder_label)
        
        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.clicked.connect(self.select_folders)
        folder_layout.addWidget(self.browse_btn)
        
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.clear_selection)
        self.clear_btn.setEnabled(False)
        folder_layout.addWidget(self.clear_btn)
        
        layout.addLayout(folder_layout)
        
        # Selected folders list
        self.folder_list = QTreeView()
        self.folder_list.setHeaderHidden(True)
        self.folder_list.setRootIsDecorated(False)
        self.folder_model = QStandardItemModel()
        self.folder_list.setModel(self.folder_model)
        layout.addWidget(QLabel("Selected Folders:"))
        layout.addWidget(self.folder_list)
        
        # Margin setting
        margin_layout = QHBoxLayout()
        margin_layout.addWidget(QLabel("Face Margin (pixels):"))
        self.margin_spinbox = QSpinBox()
        self.margin_spinbox.setRange(0, 500)
        self.margin_spinbox.setValue(50)
        self.margin_spinbox.setToolTip("The number of pixels to add as a margin around the cropped face.")
        margin_layout.addWidget(self.margin_spinbox)
        
        # Confidence threshold setting
        confidence_layout = QHBoxLayout()
        confidence_layout.addWidget(QLabel("Confidence Threshold:"))
        self.confidence_spinbox = QDoubleSpinBox()
        self.confidence_spinbox.setRange(0.0, 1.0)
        self.confidence_spinbox.setSingleStep(0.05)
        self.confidence_spinbox.setValue(0.5)
        self.confidence_spinbox.setToolTip("The minimum confidence level to detect a face (0.0 to 1.0).")
        confidence_layout.addWidget(self.confidence_spinbox)
        confidence_layout.addStretch()
        layout.addLayout(margin_layout)
        layout.addLayout(confidence_layout)
        
        # Status label
        self.status_label = QLabel("Ready to start")
        layout.addWidget(self.status_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)
        
        # Progress details
        self.progress_detail = QLabel("")
        layout.addWidget(self.progress_detail)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("Start Processing")
        self.start_btn.clicked.connect(self.start_processing)
        self.start_btn.setEnabled(False)
        btn_layout.addWidget(self.start_btn)
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.cancel_processing)
        self.cancel_btn.setEnabled(False)
        btn_layout.addWidget(self.cancel_btn)
        
        layout.addLayout(btn_layout)
        
        # Results info
        self.results_label = QLabel("")
        layout.addWidget(self.results_label)
        
        # Folder results
        self.folder_results = QLabel("")
        self.folder_results.setWordWrap(True)
        layout.addWidget(self.folder_results)
        
        # Set styles
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QLabel {
                font-size: 12px;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px 16px;
                font-size: 14px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
            QProgressBar {
                height: 20px;
                border-radius: 10px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 10px;
            }
            QTreeView {
                border: 1px solid #ccc;
                border-radius: 4px;
                background-color: white;
                min-height: 100px;
            }
        """)
        
    def select_folders(self):
        logger.info("Browse folders button clicked")
        dialog = FolderSelectionDialog(self)
        dialog.setWindowTitle("Select Image Folders")
        if dialog.exec_() == QFileDialog.Accepted:
            folders = dialog.selectedFiles()
            if folders:
                logger.info(f"Selected {len(folders)} folders")
                self.selected_folders = folders
                self.update_folder_list()
                self.start_btn.setEnabled(True)
                self.clear_btn.setEnabled(True)
                self.status_label.setText(f"{len(folders)} folders selected. Ready to process.")
    
    def clear_selection(self):
        logger.info("Clearing folder selection")
        self.selected_folders = []
        self.folder_model.clear()
        self.folder_label.setText("No folders selected")
        self.start_btn.setEnabled(False)
        self.clear_btn.setEnabled(False)
        self.status_label.setText("Ready to start")
    
    def update_folder_list(self):
        self.folder_model.clear()
        for folder in self.selected_folders:
            item = QStandardItem(folder)
            self.folder_model.appendRow(item)
        self.folder_label.setText(f"{len(self.selected_folders)} folders selected")
    
    def start_processing(self):
        logger.info("Start processing button clicked")
        
        if not self.selected_folders:
            warning_msg = "Please select folders first"
            logger.warning(warning_msg)
            QMessageBox.warning(self, "Warning", warning_msg)
            return
            
        # Check if model files exist
        if not os.path.exists(self.prototxt_path):
            error_msg = f"Prototxt file not found at: {self.prototxt_path}"
            logger.error(error_msg)
            QMessageBox.critical(self, "Error", error_msg)
            return
            
        if not os.path.exists(self.caffemodel_path):
            error_msg = f"Caffemodel file not found at: {self.caffemodel_path}"
            logger.error(error_msg)
            QMessageBox.critical(self, "Error", error_msg)
            return
            
        # Disable UI elements during processing
        self.browse_btn.setEnabled(False)
        self.clear_btn.setEnabled(False)
        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.status_label.setText("Processing...")
        self.progress_bar.setValue(0)
        self.progress_detail.setText("")
        self.results_label.setText("")
        self.folder_results.setText("")
        
        # Get margin and confidence values
        margin_value = self.margin_spinbox.value()
        confidence_value = self.confidence_spinbox.value()
        
        # Create and start worker thread
        logger.info(f"Creating FaceDetectorThread with margin: {margin_value} and confidence: {confidence_value}")
        self.worker = FaceDetectorThread(
            self.selected_folders,
            self.prototxt_path,
            self.caffemodel_path,
            margin_value,
            confidence_value
        )
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.status_updated.connect(self.update_status)
        self.worker.faces_detected.connect(self.update_folder_results)
        self.worker.folder_completed.connect(self.update_folder_completion)
        self.worker.finished.connect(self.processing_finished)
        self.worker.start()
    
    def cancel_processing(self):
        logger.info("Cancel processing button clicked")
        if self.worker and self.worker.isRunning():
            logger.info("Requesting thread cancellation")
            self.worker.cancel()
            self.cancel_btn.setEnabled(False)
            self.status_label.setText("Canceling...")
    
    def update_progress(self, value, detail):
        logger.debug(f"Progress updated: {value}% - {detail}")
        self.progress_bar.setValue(value)
        self.progress_detail.setText(detail)
    
    def update_status(self, message):
        logger.info(f"Status update: {message}")
        self.status_label.setText(message)
    
    def update_folder_results(self, face_count, folder_name):
        result_msg = f"Found {face_count} faces in {folder_name}"
        logger.info(result_msg)
        current_text = self.folder_results.text()
        self.folder_results.setText(current_text + "\n" + result_msg)
    
    def update_folder_completion(self, message):
        logger.info(f"Folder completed: {message}")
        current_text = self.results_label.text()
        self.results_label.setText(current_text + "\n" + message)
    
    def processing_finished(self):
        logger.info("Processing finished")
        # Re-enable UI elements
        self.browse_btn.setEnabled(True)
        self.clear_btn.setEnabled(True)
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        
        # Clean up worker
        self.worker = None

if __name__ == "__main__":
    logger.info("Starting application")
    app = QApplication(sys.argv)
    window = FaceCropperApp()
    window.show()
    logger.info("Application running")
    sys.exit(app.exec_())
