import socket

print(123)

sock = socket.socket()

sock.bind(('', 9090))

sock.listen(1)
conn, addr = sock.accept()

print('connected ', addr)

while True:

    conn.send((input() + '\n').encode())
    print('sent')
