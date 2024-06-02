from PyQt5 import QtWidgets as qtw
from PyQt5 import QtGui as qtg
from PyQt5 import QtCore as qtc
import sys
import numpy as np
import cv2


class VideoCapture(qtc.QThread):
    change_pixmap_signal = qtc.pyqtSignal(np.ndarray)
    
    def __init__(self):
        super().__init__()
        self.run_flag = True
        
        
    def run(self):
        #video capture kısmında gene arkadaki işlemler gibi değiştirmeyi unutma!
        cap = cv2.VideoCapture(0)
        
        while self.run_flag:
            ret, frame = cap.read() #BGR format
            
            if ret == True:
                self.change_pixmap_signal.emit(frame)
                
                
        cap.release()
        
    def stop(self):
        self.run_flag = False
        self.wait()

class mainWindow(qtw.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowIcon(qtg.QIcon('../images/icon.png')) #window icon
        self.setWindowTitle("Face Mask Recognition Software")
        self.setFixedSize(600, 600)
        
        headerLabel = qtw.QLabel('<h2>Face Mask Recognition Application</h2>')
        
        #Adding widgets for camera
        self.cameraButton = qtw.QPushButton('Open Camera', self)
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
        self.uploadButton = qtw.QPushButton('Upload Image', self)
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
        
        # screen
        self.screen = qtw.QLabel()
        self.img = qtg.QPixmap(600, 480)
        self.img.fill(qtg.QColor('darkGray'))
        self.screen.setPixmap(self.img)
        
        #Layout
        layout = qtw.QVBoxLayout()
        layout.addWidget(headerLabel)
        layout.addWidget(self.cameraButton)
        layout.addWidget(self.uploadButton)
        layout.addWidget(self.screen)
        
        self.setLayout(layout)
        self.show()
        
        
if __name__ == "__main__":
    app = qtw.QApplication(sys.argv)
    mw = mainWindow()
    sys.exit(app.exec())