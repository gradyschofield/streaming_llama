#include<cuda_bf16.h>
extern "C" __global__ void layerNormFp32(float *weights, float *src, int numRows, int leadingDimension, int seqlen, float normEps) {
    for (int j = 0; j < seqlen; ++j) {
        float accum = 0;
        float *ptr = &src[j * leadingDimension];
        for (int i = 0; i < numRows; ++i) {
            accum += (float) ptr[i] * (float) ptr[i];
        }
        float norm = 1.0 / sqrt(accum / numRows + normEps);
        for (int i = 0; i < numRows; ++i) {
            ptr[i] *= (float) weights[i] * norm;
        }
    }
}

extern "C" __global__ void layerNormBf16(__nv_bfloat16 *weights,
                                         __nv_bfloat16 *src,
                                         int numRows,
                                         int leadingDimension,
                                         int seqlen,
                                         float normEps) {
    for (int j = 0; j < seqlen; ++j) {
        float accum = 0;
        __nv_bflaot16 *ptr = &src[j * leadingDimension];
        for (int i = 0; i < numRows; ++i) {
            accum += (float) ptr[i] * (float) ptr[i];
        }
        float norm = 1.0 / sqrt(accum / numRows + normEps);
        for (int i = 0; i < numRows; ++i) {
            ptr[i] *= (float) weights[i] * norm;
        }
    }
}
