import cv2
import numpy as np
import time
import subprocess  # subprocess 모듈 임포트

# YOLO 모델 로드
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
classes = []
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

try:
    while True:
        # libcamera-jpeg 명령어 실행
        subprocess.run(['libcamera-jpeg', '-o', 'cam.jpg'])
        img = cv2.imread('cam.jpg')  # 이미지 파일 읽기

        if img is None:
            print("Failed to capture image.")
            continue

        # 객체 탐지
        height, width, channels = img.shape
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        # 검출 결과 처리
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
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

        # 사람 수 및 좌표
        person_count = 0
        person_coordinates = {}
        for i in indexes.flatten():
            if class_ids[i] == 0:
                person_count += 1
                x, y, w, h = boxes[i]
                center_x = x + w // 2
                center_y = y + h // 2
                person_coordinates[f'person{person_count}'] = (center_x, center_y)

        # 결과 출력
        print(f"Number of People: {person_count}")
        for person, coordinates in person_coordinates.items():
            print(f"{person} : {coordinates}")

        # 일정 시간 대기
        time.sleep(5)

except KeyboardInterrupt:
    print("Program terminated.")