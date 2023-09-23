//
// Created by Grady Schofield on 9/22/23.
//

#ifndef STREAMING_LLAMA_SOCKET_H
#define STREAMING_LLAMA_SOCKET_H

#include<sys/socket.h>
#include<sys/un.h>
#include<unistd.h>

#include<string>
#include<vector>

using namespace std;

class Socket {
    string const socketPath = "/tmp/codellama_evaluator.sock";
    int clientSocket;
    int serverSocket;
public:
    Socket();
    void sendInt(int i);
    void sendFloatArray(vector<float> tokens);
    int getInt();
    vector<int> getIntArray(int numTokens);
    ~Socket();
};


#endif //STREAMING_LLAMA_SOCKET_H
