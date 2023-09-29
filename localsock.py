import os
import socket
import struct
import time

class Socket:
    def __init__(self):
        socketPath = '/tmp/codellama_evaluator.sock'
        while not os.path.exists(socketPath):
            time.sleep(0.1)
        self.client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.client.connect(socketPath)

    def sendTokens(self, toks):
        data = struct.pack(f"{1+len(toks)}i", len(toks), *toks)
        self.client.sendall(data)

    def recvLogits(self):
        responseTokens = []
        disconnected = False
        while True:
            chunks = []
            chunkSize = 0
            while not disconnected and chunkSize < 4:
                moreBytes = self.client.recv(4 - chunkSize)
                if not moreBytes:
                    disconnected = True
                    break
                chunkSize += len(moreBytes)
                chunks.append(moreBytes)
            if disconnected:
                break
            responseSizeBuffer = b''.join(chunks)
            responseSize = struct.unpack('i', responseSizeBuffer)[0]

            if responseSize == -1:
                break #end of response reached
            chunks = []
            chunkSize = 0
            while not disconnected and chunkSize < 4 * responseSize:
                moreBytes = self.client.recv(4*responseSize - chunkSize)
                if not moreBytes:
                    disconnected = True
                    break
                chunkSize += len(moreBytes)
                chunks.append(moreBytes)
            responseBuffer = b''.join(chunks)
            response = struct.unpack(f"{responseSize}f", responseBuffer)
            responseTokens.extend(response)
        return (responseTokens, disconnected)

    def close(self):
        self.client.close()

