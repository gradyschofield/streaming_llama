#include<iostream>
#include<list>
#include<memory>
#include<thread>
#include<vector>

#include<LlamaModel.h>
#include<Socket.h>
#include<Timer.h>

/*
 * TODO Cuda benchmark gemmEx
 * TODO full Checker verification in the transformer block
 * TODO Cuda scratch buffers
 * TODO Cuda matrixMultiply
 * TODO Cuda kernels layer norm, toary embedding, swilu
 * TODO handle different number of kv heads for largest model
 */

using namespace std;
using namespace Common;

#define MANUAL_TESTING 0

int main(int argc, char ** argv) {
    //logger = ofstream("log");
    int maxSequenceLength = 100; // 19
    int cacheSize = 500; // 0
#if MANUAL_TESTING==1
    string filename1 = "llama_model_7.bin";
    string filename2 = "llama_model_7_bf16.bin";
    float bfloat16Tolerance = 2 * pow(2,-7);
    shared_ptr<Checker> checker = make_shared<Checker>(bfloat16Tolerance);
    shared_ptr<LLamaModelInterface> model1 = createLlamaModel<Cpu>(filename1, maxSequenceLength, cacheSize, checker);
    shared_ptr<LLamaModelInterface> model2 = createLlamaModel<Cpu>(filename2, maxSequenceLength, cacheSize, checker);
    auto runEvaluate = [](shared_ptr<LLamaModelInterface> model, vector<float> & ret) {
        ret = model->evaluate({1,518,25580,29962,6028,366,2436,263,22172,3186,1824,297,315,1817,29973,29961,29914,25580,29962});
    };
    vector<float> logits1, logits2;
    vector<thread> threads;
    threads.emplace_back(runEvaluate, model1, std::ref(logits1));
    threads.emplace_back(runEvaluate, model2, std::ref(logits2));
    for(thread & t : threads) t.join();
    for(int i = 0; i < min(100, (int)logits1.size()); ++i) {
        cout << i << ": " << logits1[i] << " " << logits2[i] << "\n";
    }
#else
    Socket socket;
    string filename = "release/llama_model_7_bf16.bin";
    //string filename = "release/llama_model_notran_emb.bin";
    shared_ptr<LLamaModelInterface> model = createLlamaModel<Cpu>(filename, maxSequenceLength, cacheSize);
    while (true) {
        int numTokens = 0;
        vector<int> tokens;
        try {
            numTokens = socket.getInt();
            tokens = socket.getIntArray(numTokens);
        } catch(exception & e) {
            cout << "Client disconnected" << endl;
            break;
        }
        Timer timer;
        vector<float> logits = model->evaluate(tokens);
        cout << timer.elapsed()/tokens.size() << " sec per token.  sending back " << logits.size() << " logits" << endl;
        timer.start();
        socket.sendFloatArray(logits);
        socket.sendInt(-1);
    }
#endif
    return 0;
}
