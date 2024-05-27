import time
import cv2
import numpy as np
from yolo5_onnx_cv import YOLOv5_ONNX_CV

# ONNX 모델 로드
model_path = 'yolov5.onnx'
yolo_model = YOLOv5_ONNX_CV(model_path)

# 클래스 레이블
classes = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
           "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
           "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
           "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
           "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
           "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
           "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
           "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
           "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
           "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
           "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
           "toothbrush"]

# 비디오 스트림 설정
cap = cv2.VideoCapture(0)  # 0은 기본 카메라
print("Starting video stream...")

# 이전 감지 시간 초기화
prev_time = time.time()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        current_time = time.time()
        height, width, _ = frame.shape

        # 5초마다 객체 탐지
        if current_time - prev_time >= 5:
            detections = yolo_model.detect(frame)

            person_count = 0
            for detection in detections:
                class_id = int(detection[5])
                confidence = detection[4]
                if confidence > 0.5 and classes[class_id] == "person":
                    person_count += 1
                    box = detection[:4].astype("int")
                    (startX, startY, endX, endY) = box
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    label = f"{classes[class_id]}: {confidence:.2f}"
                    cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            print(f"Detected {person_count} people.")
            prev_time = current_time

        # Display the output frame
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Stream interrupted.")
finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Video stream stopped.")
