//
// Created by Grady Schofield on 9/22/23.
//

#include<iostream>
#include<filesystem>

#include<Socket.h>
#include<Common.h>

using namespace std;
namespace fs = std::filesystem;


Socket::Socket()
    : clientSocket(-1), serverSocket(-1)
{
    struct sockaddr_un serverAddress;

    serverSocket = socket(AF_UNIX, SOCK_STREAM, 0);
    if (serverSocket == -1) {
        fout << "Failed to create socket." << endl;
    }
    memset(&serverAddress, 0, sizeof(serverAddress));
    serverAddress.sun_family = AF_UNIX;
    strncpy(serverAddress.sun_path, socketPath.c_str(), sizeof(serverAddress.sun_path) - 1);
    if (fs::exists(socketPath) && unlink(socketPath.c_str())) {
        fout << "Error removing socket path " << strerror(errno) << endl;
    }

    if (bind(serverSocket, (struct sockaddr *)&serverAddress, sizeof(serverAddress)) == -1) {
        fout << "Bind error." << endl;
    }

    if (listen(serverSocket, 1) == -1) {
        fout << "Listen error." << endl;
    }

    struct sockaddr_un clientAddress;
    socklen_t clientAddressSize;
    clientAddressSize = sizeof(clientAddress);
    clientSocket = accept(serverSocket, (struct sockaddr *)&clientAddress, &clientAddressSize);
    if (clientSocket == -1) {
        fout << "Accept error." << endl;
    }
}

vector<int> Socket::getIntArray(int numTokens) {
    vector<char> buffer(4 * numTokens);
    int numRead = 0;
    int len = 4 * numTokens;
    while(numRead < len) {
        int num = recv(clientSocket, &buffer[numRead], len - numRead, 0);
        if(num <= 0) throw runtime_error("Client disconnected");
        numRead += num;
    }
    vector<int> ret(numTokens);
    for(int i = 0; i < numTokens; ++i) {
        ret[i] = ((int*)buffer.data())[i];
    }
    fout << "Recieved tokens: ";
    for(int i : ret) {
        fout << i << " ";
    }
    fout << endl;
    return ret;
}

int Socket::getInt() {
    return getIntArray(1)[0];
}

void Socket::sendInt(int i) {
    int numSent = 0;
    while(numSent < 4) {
        int num = send(clientSocket, &((int8_t*)&i)[numSent], 4 - numSent, 0);
        if(num <= -1) throw runtime_error("Client disconnected");
        numSent += num;
    }
}

void Socket::sendFloatArray(vector<float> tokens) {
    /*
    fout << "Sending floats ";
    for(float t : tokens) {
        fout << t << " ";
    }
    fout << endl;
     */
    sendInt(tokens.size());
    vector<char> buffer(4 * tokens.size());
    memcpy(buffer.data(), tokens.data(), 4*tokens.size());
    int numSent = 0;
    while(numSent < 4 * (int)tokens.size()) {
        int num = send(clientSocket, &buffer[numSent], 4*tokens.size() - numSent, 0);
        if(num <= 0) throw runtime_error("Client disconnected");
        numSent += num;
    }
}

Socket::~Socket() {
    if(clientSocket != -1) close(clientSocket);
    if(serverSocket != -1) close(serverSocket);
    if(unlink(socketPath.c_str()) == -1) {
        fout << "Could not unlink socket file: " << strerror(errno) << endl;
    }
}