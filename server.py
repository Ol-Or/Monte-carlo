#모터 연결된 라즈베리파이
import socket

#server.py
HOST = '192.168.0.10'
# Server IP or Hostname
PORT = 65535
# Pick an open Port (1000+ recommended), must match the client sport
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print ('Socket created')

#managing error exception
try:
	s.bind((HOST, PORT))
except socket.error:
	print ('Bind failed ')

s.listen(5)
print ('Socket awaiting messages')
(conn, addr) = s.accept()
print ('Connected')

# awaiting for message
while True:
	data = conn.recv(1024)
	decodedata = data.decode('utf-8')
	print('I sent a message back in response to: ' + decodedata)
	reply = ''
	rp = reply.encode('utf-8')

	# process your message
	if decodedata == 'Hello':
		reply = 'Hi, back!'
	elif decodedata == 'This is important':
		reply = 'OK, I have done the important thing you have asked me!'
	#and so on and on until...
	elif decodedata == 'quit':
		conn.send('Terminating')
		break
	else:
		reply = 'Unknown command'

	# Sending reply
	conn.send(reply.encode('utf-8'))
conn.close()
# Close connections
