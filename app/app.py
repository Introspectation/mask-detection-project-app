from PyQt5 import QtWidgets as qtw
from PyQt5 import QtGui as qtg
from PyQt5 import QtCore as qtc
import sys
import numpy as np
import cv2
from deeplearning import face_mask_prediction
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import time

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)
        
    def plot_histogram(self, image):
        self.axes.clear()  # Clear any previous plot
        if image is not None:
            # Calculate histogram for each color channel
            colors = ('b', 'g', 'r')
            for i, color in enumerate(colors):
                hist = cv2.calcHist([image], [i], None, [256], [0, 256])
                self.axes.plot(hist, color=color)
            self.axes.set_title('Intensity Histogram')
            self.axes.set_xlabel('Pixel Value')
            self.axes.set_ylabel('Frequency')
        self.draw()  # Redraw the canvas

class VideoCapture(qtc.QThread):
    # Signal to update the pixmap on the GUI
    change_pixmap_signal = qtc.pyqtSignal(np.ndarray)
    update_histogram_signal = qtc.pyqtSignal(np.ndarray) # New signal for histogram updates
    
    def __init__(self):
        super().__init__()
        self.run_flag = True # Flag to control the camera thread
        
        
        
    def run(self):
        #video capture kısmında gene arkadaki işlemler gibi değiştirmeyi unutma!
        # Connect to the video source (e.g., a phone camera via DroidCam)
        droidcam_url = "http://192.168.1.101:4747/video"
        cap = cv2.VideoCapture(droidcam_url)
        last_histogram_time = time.time()
        
        while self.run_flag:
            ret, frame = cap.read()
            if ret:
                processed_frame = face_mask_prediction(frame)
                self.change_pixmap_signal.emit(processed_frame)

                # Update histogram every 5 seconds
                if time.time() - last_histogram_time > 5:
                    self.update_histogram_signal.emit(processed_frame)
                    last_histogram_time = time.time()

        cap.release()# Release the video capture
        
    def stop(self):
        # Stop the video capture loop
        self.run_flag = False
        self.wait() # Wait for the thread to finish

class mainWindow(qtw.QWidget):
    def __init__(self):
        super().__init__()
        self.camera_active = False  # Keep track of camera status
        self.capture = None  # To hold the video capture thread
        self.initUI()
        
    def initUI(self):
        self.setWindowIcon(qtg.QIcon('../images/icon.png')) #window icon
        self.setWindowTitle("Face Mask Recognition Software")
        self.setFixedSize(600, 600)
        
        headerLabel = qtw.QLabel('<h2>Face Mask Recognition Application</h2>')
        
        #Adding widgets for camera
        self.cameraButton = qtw.QPushButton('Open Camera', clicked=self.cameraButtonClick, checkable=True)
        self.cameraButton.setStyleSheet("""
            QPushButton {
                background-color: Blue;
                color: white;
                font-size: 16px;
                border: 2px solid #8f8f91;
                border-radius: 6px;
                padding: 6px;
            }
            QPushButton:pressed {
                background-color: navy;
            }
        """)
        
        #Adding widgets for Upload Image
        self.uploadButton = qtw.QPushButton('Upload Image', clicked=self.uploadImage)
        self.uploadButton.setStyleSheet("""
            QPushButton {
                background-color: Green;
                color: white;
                font-size: 16px;
                border: 2px solid #8f8f91;
                border-radius: 6px;
                padding: 6px;
            }
            QPushButton:pressed {
                background-color: darkGreen;
            }
        """)
        
        # Display area for video or images
        self.screen = qtw.QLabel()
        self.img = qtg.QPixmap(600, 480)
        self.img.fill(qtg.QColor('darkGray'))
        self.screen.setPixmap(self.img)
        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        
        # Organize widgets in a vertical layout
        layout = qtw.QVBoxLayout()
        layout.addWidget(headerLabel)
        layout.addWidget(self.cameraButton)
        layout.addWidget(self.uploadButton)
        layout.addWidget(self.screen)
        layout.addWidget(self.canvas)
        
        self.setLayout(layout)
        self.show()
        
        
    def updateHistogram(self, frame):
        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.canvas.plot_histogram(rgb_img)
        
        
    def cameraButtonClick(self):
        # Handle camera button click
        print('camera button clicked')
        if self.cameraButton.isChecked():
            self.cameraButton.setText('Closed Camera')
            if not self.camera_active:
            # open the camera
                 self.capture = VideoCapture()
                 self.capture.change_pixmap_signal.connect(self.updateImage)
                 self.capture.update_histogram_signal.connect(self.updateHistogram)  # Connect histogram update signal
                 self.capture.start()
                 self.camera_active = True
        else:
            self.cameraButton.setText('Open Camera')
            self.capture.stop()
            
    def uploadImage(self):
        # Stop camera if active when trying to upload an image
        if self.camera_active:
            self.stopCamera()
            self.cameraButton.setChecked(False)
            self.cameraButton.setText('Open Camera')
            
        fname, _ = qtw.QFileDialog.getOpenFileName(self, 'OpenFile', qtc.QDir.currentPath(), "Image Files (*.jpg *.jpeg *.png)")
        if fname:
            img = cv2.imread(fname)
            if img is not None:
                processed_image = face_mask_prediction(img)
                self.updateImage(processed_image)
                self.updateHistogram(processed_image)
            else:
                print("failed to load image.")
    def stopCamera(self):
        # Stop the video capture
        if self.capture:
            self.capture.stop()
            # Disconnect both signals to avoid any dangling connections
            self.capture.change_pixmap_signal.disconnect(self.updateImage)
            self.capture.update_histogram_signal.disconnect(self.updateHistogram)
            self.camera_active = False
            self.capture = None

    
    @qtc.pyqtSlot(np.ndarray)
    def updateImage(self, image_array):
        rgb_img = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_img.shape
        bytes_per_line = ch * w
        convertedImage = qtg.QImage(rgb_img.data, w, h, bytes_per_line, qtg.QImage.Format_RGB888)
        scaledImage = convertedImage.scaled(600, 400, qtc.Qt.KeepAspectRatio)
        qt_img = qtg.QPixmap.fromImage(scaledImage)
        self.screen.setPixmap(qt_img)
        
    def updateHistogram(self, frame):
        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.canvas.plot_histogram(rgb_img)


        
if __name__ == "__main__":
    app = qtw.QApplication(sys.argv)
    mw = mainWindow()
    sys.exit(app.exec())