#카메라 연결된 라즈베리파이

#client.py
import socket

HOST = '192.168.0.10'
# Enter IP or Hostname of your server
PORT = 65535
# Pick an open Port (1000+ recommended), must match the server port
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST,PORT))

# 예시를 위한 person_count와 person_coordinates 정의
person_count = 2
person_coordinates = {'Person1': 'Location1', 'Person2': 'Location2'}

#Lets loop awaiting for your input
while True:
    for i in range(person_count):     # for person, coordinates in person_coordinates.items():
        command = f"{person} : {coordinates}"
        s.send(command.encode('utf-8'))
    command="a" # 끝내려고 
    reply = s.recv(1024).decode('utf-8')
    if reply == 'Terminate':
        break
    print(reply)
    
    time.sleep(60) # 몇 초마다 보내줄지