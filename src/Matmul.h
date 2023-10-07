//
// Created by Grady Schofield on 10/1/23.
//

#ifndef STREAMING_LLAMA_MATMUL_H
#define STREAMING_LLAMA_MATMUL_H

#ifdef __APPLE__
#include<Accelerate/Accelerate.h>
#else
#include<mkl.h>
#endif

#include<Common.h>

using namespace Common;

#ifdef __APPLE__
template<typename T, Processor P>
void multiplyMatrices(const enum CBLAS_ORDER ORDER,
                      const enum CBLAS_TRANSPOSE TRANSA,
                      const enum CBLAS_TRANSPOSE TRANSB, const int M, const int N,
                      const int K, const T ALPHA, const T * A, const int LDA,
                      const T * B, const int LDB, const T BETA, T * C,
                      const int LDC);
#else
template<typename T, Processor P>
void multiplyMatrices(const enum CBLAS_LAYOUT ORDER,
                      const enum CBLAS_TRANSPOSE TRANSA,
                      const enum CBLAS_TRANSPOSE TRANSB, const int M, const int N,
                      const int K, const T ALPHA, const T * A, const int LDA,
                      const T * B, const int LDB, const T BETA, T * C,
                      const int LDC);
#endif
#endif //STREAMING_LLAMA_MATMUL_H
