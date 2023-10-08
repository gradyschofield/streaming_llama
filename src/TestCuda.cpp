#include<iostream>

#include<Cuda.h>
#include<Exception.h>

using namespace std;

int main(int argc, char ** argv) {
    Cuda * cuda = getCuda();
    cout << cuda << "\n";
    freeCuda();
    return 0;
}
