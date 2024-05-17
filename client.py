#카메라 연결된 라즈베리파이

#client.py
import socket

HOST = '192.168.0.10'
# Enter IP or Hostname of your server
PORT = 65535
# Pick an open Port (1000+ recommended), must match the server port
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST,PORT))

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
    else:
        # for loop completed without breaking
        continue
    # for loop was broken
break
    
time.sleep(60) # 몇 초마다 보내줄지