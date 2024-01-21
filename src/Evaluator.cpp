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
    int maxSequenceLength = 1000; // 19
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
        //ret = model->evaluate({1,518,25580,29962,6028,366,2436,263,22172,3186,1824,297,315,1817,29973,29961,29914,25580,29962});
        ret = model->evaluate({1, 518, 25580, 29962, 6028, 366, 2436, 263, 315, 1817, 740, 773, 4943, 12241, 393, 10174, 931, 20238, 297, 278, 21971, 28585, 2400, 322, 3639, 263, 5101, 393, 3743, 278, 1369, 322, 5040, 931, 310, 278, 3464, 408, 263, 28167, 14334, 29973, 29871, 1670, 526, 2211, 4072, 310, 21971, 29889, 13, 13, 1252, 9422, 310, 278, 937, 3402, 526, 376, 26626, 29871, 29906, 29900, 29896, 29900, 304, 1085, 29871, 29906, 29900, 29896, 29945, 613, 376, 27501, 29871, 29906, 29900, 29900, 29941, 304, 29639, 29871, 29906, 29900, 29900, 29955, 613, 322, 376, 29943, 774, 29871, 29896, 29929, 29947, 29955, 304, 4756, 29871, 29896, 29929, 29929, 29929, 1642, 29871, 306, 884, 864, 363, 278, 4098, 5993, 304, 367, 13136, 29889, 29871, 1105, 376, 29906, 29900, 29900, 29941, 304, 29639, 29871, 29906, 29900, 29900, 29955, 29908, 470, 376, 27501, 29871, 29906, 29900, 29900, 29945, 304, 29871, 29906, 29900, 29896, 29941, 29908, 338, 884, 2854, 29889, 13, 13, 1252, 9422, 310, 278, 1473, 3402, 526, 376, 29896, 29900, 2440, 613, 376, 29906, 2440, 613, 376, 29906, 7378, 613, 376, 29896, 1629, 613, 376, 29896, 4098, 613, 376, 29896, 2462, 613, 322, 376, 29945, 3841, 1642, 29871, 512, 445, 1206, 29892, 278, 1095, 931, 338, 278, 1857, 2462, 322, 278, 1347, 11524, 278, 920, 2215, 1250, 304, 748, 304, 1284, 278, 1369, 931, 29889, 13, 13, 12881, 635, 29892, 6455, 310, 278, 1833, 3402, 526, 376, 29941, 2440, 304, 29871, 29906, 2440, 613, 376, 29896, 1629, 304, 29871, 29945, 3841, 613, 376, 29896, 29900, 2440, 304, 29871, 29896, 1629, 613, 376, 29896, 4098, 304, 29871, 29896, 29945, 3841, 1642, 29871, 512, 445, 1206, 29892, 278, 1369, 322, 1095, 931, 526, 15712, 773, 278, 1857, 2462, 408, 263, 3407, 1298, 322, 3063, 1250, 1328, 29892, 2788, 304, 278, 1473, 3402, 29892, 541, 1716, 278, 1369, 322, 1095, 931, 526, 6790, 29892, 2788, 304, 278, 937, 3402, 29889, 13, 29961, 29914, 25580, 29962});
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
    //string filename = "release/llama_model_7_bf16.bin";
    string filename = "release/llama_model_13.bin";
    //string filename = "release/llama_model_notran_emb.bin";
    unique_ptr<LLamaModelInterface> model = createLlamaModel(filename, maxSequenceLength, cacheSize, unmapWeights);
    while (true) {
        int numTokens = 0;
        vector<int> tokens;
        try {
            numTokens = socket.getInt();
            tokens = socket.getIntArray(numTokens);
        } catch(exception & e) {
            fout << "Client disconnected" << endl;
            break;
        }
        Timer timer;
        vector<float> logits = model->evaluate(tokens);
        fout << timer.elapsed()/tokens.size() << " sec per token.  sending back " << logits.size() << " logits" << endl;
        timer.start();
        socket.sendFloatArray(logits);
        socket.sendInt(-1);
    }
#endif
    return 0;
}
