import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QStackedWidget, QWidget, QVBoxLayout, QLabel, QHBoxLayout, QPushButton, QFileDialog, QProgressBar
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QFont, QPixmap, QPainter, QIcon, QImage,QFontDatabase
import cv2
from keras.models import load_model
import numpy as np
import time
import tensorflow as tf
import os

current_file_dir=os.path.dirname(__file__)

print(current_file_dir)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Face Rater')
        self.setGeometry(100, 100, 1000,600)
        self.backgroundPixmap = QPixmap("images/mainwindow.jpeg")
        self.initUI()
        self.show_face=False
        self.class_names=["Attractive","Average","Ugly"]
        self.model=load_model("facerater.h5")
        self.label = QLabel(self)

        self.attrlbl=QLabel(self)
        self.avglbl=QLabel(self)
        self.uglbl=QLabel(self)

        self.attrprog=QProgressBar(self)
        self.avgprog=QProgressBar(self)
        self.ugprog=QProgressBar(self)

    def initUI(self):
        self.camerabutton=QPushButton(self)
        self.camerabutton.setIcon(QIcon("images/camera2.png"))
        self.camerabutton.setIconSize(QSize(100,100))
        self.camerabutton.setGeometry(420,0,100,70)
        self.camerabutton.clicked.connect(self.openFileDialog)
        self.camerabutton.show()


        self.image_frame=QLabel(self)
        self.image_frame.setGeometry(650,30,300,300)
        self.image_frame.show()
        self.descfont=QFontDatabase.addApplicationFont("/Users/adityachakraborty/Desktop/Programming:CS3/Machine Learning/Face Rater/fonts/AquireLight-YzE0o.otf")
        self.descfont = QFontDatabase.applicationFontFamilies(self.descfont)
        descfont = QFont(self.descfont[0], 20)
        face_detected=QLabel(self,text="Detected faces:")
        face_detected.setFont(descfont)
        face_detected.setGeometry(650,430,200,20)
        face_detected.show()
        self.label=QLabel(self, text="No Face detected")
        self.label.setGeometry(650,430,300,200)
        self.label.hide()
    def rate_face(self, frame, x, y, x2, y2):
        # Check if the region of interest is valid
        if x2 > x and y2 > y:
            display_image = cv2.resize(frame[y:y2, x:x2], (300, 300))
            self.imgshow = display_image
            self.image = QImage(self.imgshow.data, self.imgshow.shape[1], self.imgshow.shape[0], QImage.Format_RGB888).rgbSwapped()
            self.image_frame.setPixmap(QPixmap.fromImage(self.image))
            
            # Preprocess the image for prediction
            predict_image = cv2.resize(frame[y:y2, x:x2], (200, 200)) / 255
            predict_image = np.reshape(predict_image, (1, 200, 200, 3))
            
            # Perform prediction
            @tf.autograph.experimental.do_not_convert()
            def predict():
                prediction = self.model.predict(predict_image, verbose=False)
                return prediction
            prediction=predict()

            # Find the corresponding class name
            key=self.class_names[np.argmax(prediction, axis=1)[0]]
            fonttoUse = QFont(self.descfont[0], 72)
            self.label.setFont(fonttoUse)
            self.label.setGeometry(100, 100, 500, 72)
            self.label.setText(key)
            self.label.show()

            self.attrprog.setGeometry(50, 280, 400, 20)
            self.avgprog.setGeometry(50, 380, 400, 20)
            self.ugprog.setGeometry(50, 480, 400, 20)

            self.attrlbl.setGeometry(50,230,400,30)
            self.avglbl.setGeometry(50,330,400,30)
            self.uglbl.setGeometry(50,430,400,30)

            self.attrlbl.setText("Attractive")
            self.avglbl.setText("Average")
            self.uglbl.setText("Ugly")

            self.attrlbl.setFont(QFont(None, 20))
            self.avglbl.setFont(QFont(None, 20))
            self.uglbl.setFont(QFont(None, 20))

            self.ugprog.show()
            self.avgprog.show()
            self.attrprog.show()

            atval=int(prediction[0][0]*100)
            avgval=int(prediction[0][1]*100)
            ugval=int(prediction[0][2]*100)

            self.attrprog.setValue(atval)
            self.avgprog.setValue(avgval)
            self.ugprog.setValue(ugval)

            for i in range(atval):
                self.attrprog.setValue(i+1)
            for i in range(avgval):
                self.avgprog.setValue(i+1)
            for i in range(ugval):
                self.ugprog.setValue(i+1)
            

        else:
            # If the region of interest is invalid, hide the label
            self.label.setVisible(False)


    def openFileDialog(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open file", "", "Images (*.png *.jpg *.jpeg)")

        # Load the image and check if it's loaded correctly.
        image = cv2.imread(filename)
        
        # Specify the paths to the model's weights and configuration
        modelFile = "model/res10_300x300_ssd_iter_140000_fp16.caffemodel"
        configFile = "model/deploy.prototxt"

        # Load the Caffe model
        net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

        # Prepare the image for the neural network
        h, w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                    (300, 300), (104.0, 177.0, 123.0))

        # Use the neural network to detect faces
        net.setInput(blob)
        detections = net.forward()

        # Assume we take the first detected face for simplicity
        # In a real application, you'd probably want to handle multiple faces
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < 0.5:
                continue

            # Compute the (x, y)-coordinates of the bounding box for the face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            self.rate_face(image, startX, startY, endX, endY)
            break  # Assuming we're only interested in the first detected face for this example



        # When everything is done, release the capture 
    def paintEvent(self, event):
        painter = QPainter(self)
        # Scale pixmap to fit widget size while maintaining aspect ratio
        scaledPixmap = self.backgroundPixmap.scaled(self.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
        # Calculate top-left position to center the pixmap
        x = (self.width() - scaledPixmap.width()) // 2
        y = (self.height() - scaledPixmap.height()) // 2
        painter.drawPixmap(x, y, scaledPixmap)
        painter.drawRect(600,0,10,600)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    startWindow = MainWindow()
    startWindow.show()
    sys.exit(app.exec_())