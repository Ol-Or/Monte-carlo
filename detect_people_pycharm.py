import cv2
import numpy as np
import time

# YOLO 모델 로드
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()

# 네트워크에서 연결되지 않은 레이어의 인덱스 확인 및 출력 레이어 설정
output_layers = [layer for layer in layer_names if 'yolo' in layer]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

#네트워크에서 연결되지 않은 레이어의 인덱스 확인
unconnected_out_layers = net.getUnconnectedOutLayers()
print("Unconnected out layers:", unconnected_out_layers)

# 레이어 이름 확인
print("Layer names:", layer_names)
font = cv2.FONT_HERSHEY_PLAIN
'''
# 비디오 캡처 초기화
cap = cv2.VideoCapture(0)  # '0'은 첫 번째 카메라 장치
starting_time = time.time()
frame_id = 0
'''
# 이미지 가져오기
img = cv2.imread("IMG_2811.JPG")
img = cv2.resize(img, None, fx=0.4, fy=0.4)
height, width, channels = img.shape

# 객체 탐지
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
        if confidence > 0.5:
            # 객체 감지
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            # 좌표
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
# 인덱스 출력 추가
print("Indexes:", indexes)

# 사람 수 카운트
person_count = sum(1 for i in indexes if class_ids[i] == 0)

# 사람 수 카운트 및 중심 좌표 출력
person_count = 0
person_coordinates = {}
for i in indexes:
    # 인덱스 처리 방식을 확인한 후 적절한 접근 방식을 사용
    if isinstance(i, np.ndarray):
        i = i[0]  # np.ndarray라면 첫 번째 요소를 사용
    elif isinstance(i, int):
        pass  # i가 이미 정수형이라면 그대로 사용

    if class_ids[i] == 0:  # 'person' 클래스 확인
        person_count += 1
        x, y, w, h = boxes[i]
        center_x = x + w // 2
        center_y = y + h // 2
        label = str(classes[class_ids[i]])
        color = colors[i]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y - 10), font, 1, color, 2)
        person_coordinates[f'person{person_count}'] = (center_x, center_y)

# 각 사람의 중심 좌표 출력
for person, coordinates in person_coordinates.items():
    print(f"{person} : {coordinates}")

print(f"Number of People: {person_count}")

# 이미지에 사람 수 및 바운딩 박스 표시
for i in indexes:
    i = i[0] if isinstance(i, list) else i
    x, y, w, h = boxes[i]
    label = str(classes[class_ids[i]])
    color = colors[i]
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    cv2.putText(img, label, (x, y - 10), font, 1, color, 2)
cv2.putText(img, f"Number of People: {person_count}", (10, 50), font, 2, (0, 255, 0), 3)

cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()