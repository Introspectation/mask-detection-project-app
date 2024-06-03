import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QFileDialog
)
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
import cv2
import numpy as np
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._run_flag = True

    def run(self):
        cap = cv2.VideoCapture(0)
        while self._run_flag:
            ret, frame = cap.read()
            if ret:
                self.change_pixmap_signal.emit(frame)
        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Mask Detection App")
        self.setGeometry(100, 100, 1280, 720)

        # Real-time detection part
        self.real_time_label = QLabel(self)
        self.real_time_label.setFixedSize(640, 480)
        self.real_time_label.setStyleSheet("background-color: black;")

        self.start_button = QPushButton("Start", self)
        self.start_button.clicked.connect(self.start_real_time_detection)

        self.pause_button = QPushButton("Pause", self)
        self.pause_button.clicked.connect(self.pause_real_time_detection)

        self.exit_button_rt = QPushButton("Exit", self)
        self.exit_button_rt.clicked.connect(self.close)

        real_time_button_layout = QHBoxLayout()
        real_time_button_layout.setSpacing(10)  # Set spacing between buttons
        real_time_button_layout.addWidget(self.start_button)
        real_time_button_layout.addWidget(self.pause_button)
        real_time_button_layout.addWidget(self.exit_button_rt)

        real_time_layout = QVBoxLayout()
        real_time_layout.addWidget(self.real_time_label)
        real_time_layout.addLayout(real_time_button_layout)

        # Image upload detection part
        self.upload_label = QLabel(self)
        self.upload_label.setFixedSize(640, 480)
        self.upload_label.setStyleSheet("background-color: black;")

        self.upload_button = QPushButton("Upload Image", self)
        self.upload_button.clicked.connect(self.upload_image)

        self.exit_button_upload = QPushButton("Exit", self)
        self.exit_button_upload.clicked.connect(self.close)

        upload_button_layout = QHBoxLayout()
        upload_button_layout.setSpacing(10)  # Set spacing between buttons
        upload_button_layout.addWidget(self.upload_button)
        upload_button_layout.addWidget(self.exit_button_upload)

        upload_layout = QVBoxLayout()
        upload_layout.addWidget(self.upload_label)
        upload_layout.addLayout(upload_button_layout)

        # Combine layouts
        main_layout = QHBoxLayout()
        main_layout.setSpacing(20)  # Set spacing between real-time and upload sections
        main_layout.addLayout(real_time_layout)
        main_layout.addLayout(upload_layout)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)

    def start_real_time_detection(self):
        self.thread.start()

    def pause_real_time_detection(self):
        self.thread.stop()

    def update_image(self, frame):
        qt_image = self.convert_cv_qt(frame)
        self.real_time_label.setPixmap(qt_image)

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(convert_to_Qt_format)

    def upload_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Image Files (*.png *.jpg *.bmp)", options=options)
        if file_name:
            image = cv2.imread(file_name)
            qt_image = self.convert_cv_qt(image)
            self.upload_label.setPixmap(qt_image)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
