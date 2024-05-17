#모터 연결된 라즈베리파이
import socket

# server.py
HOST = '192.168.0.10'
PORT = 65535
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print('Socket created')

# 에러 관리
try:
    s.bind((HOST, PORT))
except OSError as e:
    print(f'Bind failed: {e}')

s.listen(5)
print('Socket awaiting messages')
conn, addr = s.accept()
print('Connected')

# 메시지 수신 대기
while True:
    data = conn.recv(1024)
    if not data:
        break  # 클라이언트가 연결을 끊었을 경우
    decodedata = data.decode('utf-8')
    print('I sent a message back in response to: ' + decodedata)

    # 메시지 처리
    if decodedata == "a":
        conn.send(b'Terminating')
        break
    else:
        reply = 'Message received'
        conn.send(reply.encode('utf-8'))

conn.close()  # 연결 종료