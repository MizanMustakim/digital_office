import torch
import cv2
import numpy as np
import time
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QTextEdit
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer

# Load the YOLOv7 model
model = torch.hub.load('./yolov7', 'custom', './yolov7/yolov7-tiny.pt', source='local')

# Function to detect humans in each frame using YOLOv7
def detect_humans(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    start_time = time.time()
    
    results = model(img)
    
    end_time = time.time()
    detections = results.xyxy[0].cpu().numpy()

    humans = []
    for detection in detections:
        if int(detection[5]) == 0:  # Person class
            humans.append(detection)
    
    return humans, f"Inference time {end_time - start_time:.2f} seconds"

def determine_posture(bbox, desk_y_level):
    x1, y1, x2, y2 = bbox[:4]
    height = y2 - y1
    
    if height < 400:
        return "sitting"
    else:
        return "standing"
    
    # if y2 > desk_y_level:
    #     return "sitting"
    # elif y1 < desk_y_level:
    #     return "standing"
    # else:
    #     return "transition"

def detect_desk_leaving(humans, desk_y_level):
    leaving = False
    for human in humans:
        x1, y1, x2, y2 = human[:4]
        ax = (y1+y2)//2
        if ax < desk_y_level:
            leaving = True
    return leaving


class HumanDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Set up the main window
        self.setWindowTitle("Human Movement Detection")
        self.setGeometry(100, 100, 800, 600)

        # Create a label to display video frames
        self.video_label = QLabel(self)
        self.video_label.resize(640, 480)

        # Create a text box to display posture and inference information
        self.info_box = QTextEdit(self)
        self.info_box.setReadOnly(True)
        
        # Create a button to stop/start the video
        self.start_button = QPushButton("Stop Detection", self)
        self.start_button.clicked.connect(self.start_detection)
        
        # Set up layout
        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.info_box)
        layout.addWidget(self.start_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Video capture
        self.cap = cv2.VideoCapture(0)
        self.desk_y_level = 300  # Example y-coordinate where the desk is located
        self.prev_posture = {}

        # Timer to update the video frames
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        # Start the timer immediately
        self.timer.start(20)  # Update frame every 20 ms

    def start_detection(self):
        if self.timer.isActive():
            self.timer.stop()
            self.cap.release()
            self.start_button.setText("Start Detection")
        else:
            self.cap = cv2.VideoCapture(0)
            self.timer.start(20)
            self.start_button.setText("Stop Detection")

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        # Detect humans in the frame
        humans, inference = detect_humans(frame)

        # Analyze human movements and postures
        for i, human in enumerate(humans):
            x1, y1, x2, y2 = human[:4]
            current_posture = determine_posture(human, self.desk_y_level)
            
            # Draw bounding boxes around detected humans
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"{current_posture}", (int(x1)+30, int(y1)+30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            if i in self.prev_posture:
                if self.prev_posture[i] == "sitting" and current_posture == "standing":
                    cv2.putText(frame, "Standing Up Detected", (150, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    # cv2.putText(frame, "Standing Up Detected", (int(x1), int(y1) - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                elif self.prev_posture[i] == "standing" and current_posture == "sitting":
                    cv2.putText(frame, "Sitting Down Detected", (150, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            self.prev_posture[i] = current_posture

        if detect_desk_leaving(humans, self.desk_y_level):
            cv2.putText(frame, "Leaving Desk Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        # Convert frame to QImage to display in QLabel
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_image))

        # Update info box with inference time
        self.info_box.setText(inference)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HumanDetectionApp()
    window.show()
    sys.exit(app.exec_())