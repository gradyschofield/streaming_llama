#include<iostream>
#include<list>
#include<memory>
#include<thread>
#include<vector>

#include<LlamaModel.h>
#include<Socket.h>
#include<Timer.h>

/*
 * TOOD measure distribution of outputs for quantization purposes
 * TODO write model files for quantized weights
 *
 * TODO handle different number of kv heads for largest model
 *
 * TODO metal shaders for bfloat16
 *      read matrices into memory
 *
 */

using namespace std;
using namespace Common;

#define MANUAL_TESTING 0

int main(int argc, char ** argv) {
    //logger = ofstream("log");
    int maxSequenceLength = 100; // 19
    int cacheSize = 5000; // 0
    bool unmapWeights = false;
#if MANUAL_TESTING==1
    string filename1 = "llama_model_7.bin";
    string filename2 = "llama_model_7_bf16.bin";
    float bfloat16Tolerance = 2 * pow(2,-7);
    shared_ptr<Checker> checker = make_shared<Checker>(bfloat16Tolerance);
    shared_ptr<LLamaModelInterface> model1 = createLlamaModel(filename1, maxSequenceLength, cacheSize, unmapWeights, checker);
    shared_ptr<LLamaModelInterface> model2 = createLlamaModel(filename2, maxSequenceLength, cacheSize, unmapWeights, checker);
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
    //string filename = "release/llama_model_13.bin";
    //string filename = "release/llama_model_notran_emb.bin";
    unique_ptr<LLamaModelInterface> model = createLlamaModel(filename, maxSequenceLength, cacheSize, unmapWeights);
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
