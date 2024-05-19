
#!/usr/bin/env python3

# pip install smbus2

import cv2
import numpy as np
import time
import subprocess
import os
import smbus2 as smbus

BUS = smbus.SMBus(1)

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

def write_word(addr, data):
    global BLEN
    temp = data
    if BLEN == 1:
        temp |= 0x08
    else:
        temp &= 0xF7
    BUS.write_byte(addr, temp)

def send_command(comm):
    # Send bit7-4 firstly
    buf = comm & 0xF0
    buf |= 0x04               # RS = 0, RW = 0, EN = 1
    write_word(LCD_ADDR, buf)
    time.sleep(0.002)
    buf &= 0xFB               # Make EN = 0
    write_word(LCD_ADDR, buf)

    # Send bit3-0 secondly
    buf = (comm & 0x0F) << 4
    buf |= 0x04               # RS = 0, RW = 0, EN = 1
    write_word(LCD_ADDR, buf)
    time.sleep(0.002)
    buf &= 0xFB               # Make EN = 0
    write_word(LCD_ADDR, buf)

def send_data(data):
    # Send bit7-4 firstly
    buf = data & 0xF0
    buf |= 0x05               # RS = 1, RW = 0, EN = 1
    write_word(LCD_ADDR, buf)
    time.sleep(0.002)
    buf &= 0xFB               # Make EN = 0
    write_word(LCD_ADDR, buf)

    # Send bit3-0 secondly
    buf = (data & 0x0F) << 4
    buf |= 0x05               # RS = 1, RW = 0, EN = 1
    write_word(LCD_ADDR, buf)
    time.sleep(0.002)
    buf &= 0xFB               # Make EN = 0
    write_word(LCD_ADDR, buf)

def init(addr, bl):
    global LCD_ADDR
    global BLEN
    LCD_ADDR = addr
    BLEN = bl
    try:
        send_command(0x33) # Must initialize to 8-line mode at first
        time.sleep(0.005)
        send_command(0x32) # Then initialize to 4-line mode
        time.sleep(0.005)
        send_command(0x28) # 2 Lines & 5*7 dots
        time.sleep(0.005)
        send_command(0x0C) # Enable display without cursor
        time.sleep(0.005)
        send_command(0x01) # Clear Screen
        BUS.write_byte(LCD_ADDR, 0x08)
    except:
        return False
    else:
        return True

def clear():
    send_command(0x01) # Clear Screen

def openlight():  # Enable the backlight
    BUS.write_byte(0x27, 0x08)

def closelight():  # Disable the backlight
    BUS.write_byte(0x27, 0x00)

def write(x, y, text):
    if x < 0:
        x = 0
    if x > 15:
        x = 15
    if y < 0:
        y = 0
    if y > 1:
        y = 1

    # Move cursor
    addr = 0x80 + 0x40 * y + x
    send_command(addr)

    for chr in text:
        send_data(ord(chr))

# 이미지 캡처
image_path = "/tmp/image.jpg"

try:
    init(0x27, 1)
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
       
        # LCD에 결과 표시
        if person_count == 2:
            write(1, 0, 'Temperature is')
            write(3, 1, '24.0')
        else:
            write(1, 0, 'Temperature is')
            write(3, 1, '26.0')

        # 5초 대기
        time.sleep(5)
        clear()  # 다음 반복 전에 LCD를 지웁니다.

except KeyboardInterrupt:
    clear()
    closelight()
    print("LCD turned off and program terminated.")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    BUS.close()  # Ensure the bus is closed properly