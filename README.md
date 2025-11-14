# FaceCropper 

A professional desktop application for batch face detection and cropping from images across multiple folders using deep learning.

## Features

- **Multi-Folder Processing**: Process multiple image folders simultaneously
- **Deep Learning Detection**: Uses OpenCV's DNN module with Caffe model for accurate face detection
- **Customizable Parameters**: Adjustable margin and confidence threshold
- **Real-time Progress Tracking**: Live progress updates and status monitoring
- **Unicode Path Support**: Handles international characters in file paths
- **Batch Processing**: Automatically processes all supported images in selected folders
- **User-Friendly Interface**: Clean, intuitive PyQt5-based GUI

## Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff)
- WebP (.webp)

## Requirements

### Python Packages
- Python 3.6+
- OpenCV (cv2)
- PyQt5
- NumPy

### Model Files
The application requires the following Caffe model files:
- `deploy.prototxt`
- `res10_300x300_ssd_iter_140000.caffemodel`

By default, the application looks for these files in `D:\programs\facedetect\`. You can modify the `model_dir` path in the code to point to your model files location.

## Installation

1. Clone or download this repository
2. Install required packages:
   ```bash
   pip install opencv-python pyqt5 numpy
   ```
3. Download the Caffe model files:
   - [deploy.prototxt](https://github.com/opencv/opencv/blob/master/samples/dnn/face_detector/deploy.prototxt)
   - [res10_300x300_ssd_iter_140000.caffemodel](https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel)

4. Update the `model_dir` path in the code if your model files are in a different location

## Usage

1. **Launch the Application**:
   ```bash
   python face_cropper_pro.py
   ```

2. **Select Folders**:
   - Click "Browse..." to select one or multiple folders containing images
   - Use Ctrl+Click to select multiple folders
   - Click "Clear" to remove all selected folders

3. **Configure Settings**:
   - **Face Margin**: Number of pixels to add around detected faces (default: 50)
   - **Confidence Threshold**: Minimum confidence level for face detection (0.0-1.0, default: 0.5)

4. **Start Processing**:
   - Click "Start Processing" to begin face detection and cropping
   - Monitor progress through the progress bar and status messages
   - Use "Cancel" to stop processing at any time

5. **View Results**:
   - Processed faces are saved in "cropped_faces" subfolders within each input folder
   - Results are named as: `original_filename_face_1.jpg`, `original_filename_face_2.jpg`, etc.

## Output Structure

```
Input Folder/
├── image1.jpg
├── image2.png
└── cropped_faces/
    ├── image1_face_1.jpg
    ├── image1_face_2.jpg
    ├── image2_face_1.jpg
    └── ...
```

## Configuration

### Model Path
Update the `model_dir` variable in the `__init__` method of `FaceCropperApp` class:

```python
self.model_dir = r"your/model/directory/path"
```

### Logging
The application creates a log file `face_detection.log` with detailed processing information for debugging.

## Technical Details

- **Face Detection**: Uses OpenCV's DNN module with ResNet-10 SSD model
- **Image Processing**: Handles various color formats (RGB, RGBA, grayscale)
- **Threading**: Separate worker thread for processing to keep UI responsive
- **Error Handling**: Comprehensive error handling and logging
- **Memory Management**: Efficient image processing with proper resource cleanup

## Troubleshooting

### Common Issues

1. **Model files not found**:
   - Ensure the model files are in the correct directory
   - Update the `model_dir` path in the code

2. **No faces detected**:
   - Lower the confidence threshold
   - Ensure images contain clear, front-facing faces
   - Check that model files are properly loaded

3. **Application crashes**:
   - Check the log file for detailed error messages
   - Ensure all required packages are installed
   - Verify image file integrity

### Log File
Check `face_detection.log` for detailed processing information and error messages.

## License

This project is provided for educational and personal use. Please ensure you comply with the licenses of the underlying libraries and model files.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## Acknowledgments

- Uses OpenCV's deep learning module for face detection
- Caffe model provided by the OpenCV community
- Built with PyQt5 for the user interface
