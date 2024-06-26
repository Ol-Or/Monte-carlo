import cv2
import numpy as np
import time
import subprocess
import os

# YOLO 모델 로드
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))
font = cv2.FONT_HERSHEY_PLAIN

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
        
        # 객체 탐지
        height, width, channels = img.shape
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        # 정보를 화면에 표시
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and class_id == 0:  # '0'은 사람 클래스
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

        # 사람 수 및 좌표값 계산
        person_count = 0
        person_coordinates = {}
        for i in range(len(boxes)):
            if i in indexes:
                person_count += 1
                x, y, w, h = boxes[i]
                center_x = x + w // 2
                center_y = y + h // 2
                person_coordinates[f'person{person_count}'] = (center_x, center_y)
        
        # 결과 출력
        print(f"Number of People: {person_count}")
        for person, coordinates in person_coordinates.items():
            print(f"{person} : {coordinates}")

        # 이미지를 해제하여 메모리 누수 방지
        del img

        # 5초 대기
        time.sleep(5)

except KeyboardInterrupt:
    print("Program terminated.")
except Exception as e:
    print(f"An error occurred: {e}")
