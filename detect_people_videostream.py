import cv2
import numpy as np
import subprocess
import time

# YOLO 모델 로드
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# libcamera-vid를 사용하 여비디오 스트림 생성 및 파이프
command = "libcamera-vid -t 0 --inline --codec yuv420 -o -"
process = subprocess.Popen(command.split(), stdout=subprocess.PIPE, bufsize=10**8)

# 비디오 스트림 읽기 (파이프)
cap = cv2.VideoCapture(process.stdout.fileno(), cv2.CAP_ANY)

last_time_printed = time.time()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 객체 탐지
        height, width, channels = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        # 정보 처리
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

        # 결과 출력
        person_count = 0
        person_coordinates = {}
        for i in range(len(boxes)):
            if i in indexes:
                person_count += 1
                x, y, w, h = boxes[i]
                center_x = x + w // 2
                center_y = y + h // 2
                person_coordinates[f'person{person_count}'] = (center_x, center_y)

        # 5초마다 결과 출력
        if time.time() - last_time_printed >= 5:
            print(f"Number of People: {person_count}")
            for person, coordinates in person_coordinates.items():
                print(f"{person} : {coordinates}")
            last_time_printed = time.time()

except KeyboardInterrupt:
    print("Program terminated.")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    cap.release()
    process.terminate()
