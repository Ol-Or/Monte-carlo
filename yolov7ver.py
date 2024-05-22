import torch
import torch.onnx

# 모델 로드
model = torch.load('yolov7.pt', map_location=torch.device('cpu'))
model.eval()

# 더미 입력 생성
dummy_input = torch.randn(1, 3, 640, 640)

# ONNX로 내보내기
torch.onnx.export(model, dummy_input, "yolov7.onnx", verbose=True, input_names=['input'], output_names=['output'], opset_version=11)

import cv2
import numpy as np
import time
import subprocess

# ONNX 모델 로드
net = cv2.dnn.readNetFromONNX('yolov7.onnx')
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# libcamera를 이용해 사진 촬영 및 저장
def capture_image(filename):
    command = f"libcamera-still -o {filename}"
    subprocess.run(command.split(), check=True)

# 이미지 캡처
image_path = "/tmp/image.jpg"

try:
    while True:
        # 이미지 촬영
        capture_image(image_path)
        
        # 저장된 이미지 로드
        img = cv2.imread(image_path)
        if img is None:
            print("Image not loaded properly. Skipping iteration.")
            continue
        
        # 객체 탐지를 위한 전처리
        blob = cv2.dnn.blobFromImage(img, 1/255, (640, 640), [0, 0, 0], 1, crop=False)
        net.setInput(blob)
        
        # 추론
        outs = net.forward()

        # 결과 처리
        class_ids = []
        confidences = []
        boxes = []
        for detection in outs:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        # 결과 출력
        for i in indexes:
            i = i[0]
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = np.random.uniform(0, 255, 3)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 5초 대기
        time.sleep(5)

except KeyboardInterrupt:
    print("Program terminated.")
except Exception as e:
    print(f"An error occurred: {e}")
