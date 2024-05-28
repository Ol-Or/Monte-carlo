import cv2
import numpy as np
import time
from picamera2 import Picamera2

# YOLO 모델 로드
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))
font = cv2.FONT_HERSHEY_PLAIN

# PiCamera2 초기화 및 설정
picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (640, 480)})
picam2.configure(config)
picam2.start()

frame_interval = 10  # 감지 간격을 위한 프레임 간격 설정
frame_count = 0

try:
    while True:
        # 프레임 캡처
        frame = picam2.capture_array()
        frame_count += 1

        # 객체 탐지를 위한 프레임 선택
        if frame_count % frame_interval == 0:
            # 객체 탐지 수행
            height, width, channels = frame.shape
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers)

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

            # 감지된 객체에 박스 그리기
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    color = colors[class_ids[i]]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, label, (x, y - 10), font, 1, color, 2)

        # 비디오 스트리밍 출력
        cv2.imshow("Image", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Program terminated.")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    picam2.stop()
    cv2.destroyAllWindows()
