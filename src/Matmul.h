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
#include<Metal.hpp>

using namespace Common;

template<typename T>
void multiplyMatrices(const enum CBLAS_ORDER ORDER,
                      const enum CBLAS_TRANSPOSE TRANSA,
                      const enum CBLAS_TRANSPOSE TRANSB, const int M, const int N,
                      const int K, const T ALPHA, const T * A, const int LDA,
                      const T * B, const int LDB, const T BETA, T * C,
                      const int LDC);

void matvec(MTL::Buffer * mat, MTL::Buffer * in, MTL::Buffer * out, long numRows, long numCols, long outputOffsetElements = 0);

template<typename T>
void multiplyMatrices(const enum CBLAS_ORDER ORDER,
                      const enum CBLAS_TRANSPOSE TRANSA,
                      const enum CBLAS_TRANSPOSE TRANSB, const int M, const int N,
                      const int K, const T ALPHA, MTL::Buffer * A, const int LDA,
                      MTL::Buffer * B, const int LDB, const T BETA, MTL::Buffer * C,
                      const int LDC) {
    if (N > 1 || TRANSA == CblasTrans || LDA != M || LDB != K || LDC != M) {
        multiplyMatrices(ORDER, TRANSA, TRANSB, M, N, K, ALPHA, (T const *) A->contents(), LDA,
                         (T const *) B->contents(), LDB, BETA, (T *) C->contents(), LDC);
    } else {
        matvec(A, B, C, M, K);
        //metal matvec
    }
}

template<typename T>
void multiplyMatrices(const enum CBLAS_ORDER ORDER,
                      const enum CBLAS_TRANSPOSE TRANSA,
                      const enum CBLAS_TRANSPOSE TRANSB, const int M, const int N,
                      const int K, const T ALPHA, MTL::Buffer * A, const int LDA,
                      MTL::Buffer * B, const int LDB, const T BETA, MTL::Buffer * C,
                      const long outputOffsetElements,
                      const int LDC) {
    if (N > 1 || TRANSA == CblasTrans || LDA != M || LDB != K || LDC != M) {
        multiplyMatrices(ORDER, TRANSA, TRANSB, M, N, K, ALPHA, (T const *) A->contents(), LDA,
                         (T const *) B->contents(), LDB, BETA, ((T *)C->contents()) + outputOffsetElements, LDC);
    } else {
        matvec(A, B, C, M, K, outputOffsetElements);
        //metal matvec
    }
}

template<typename T>
void multiheadMatvec(Scratch<T> & mat,
                     MTL::Buffer * in,
                     MTL::Buffer * out,
                     long headDimension,
                     long numHeads,
                     long currentToken,
                     long seqlen,
                     long leadingDimension) {
    if (seqlen > 1) {
        Scratch<T> * qkOut = transformerBlockScratch->getQKout();
        T * qkOutPtr = qkOut->getPtr();
        int qkOutLeadingDim = qkOut->getLeadingDimension();
        for(int head = 0; head < numHeads; ++head) {
            int M = currentToken + seqlen;
            int N = seqlen;
            int K = headDimension;
            int inputHeadOffset = head * headDimension;
            int outputHeadOffset = head * (currentToken + seqlen);
            timings.start("Key/Query matrix product");
            multiplyMatrices<T>(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                M, N, K,
                                1.0,
                                &wkOutPtr[head * headDimension * wkOutLeadingDim],
                                wkOutLeadingDim,
                                &wqOutPtr[inputHeadOffset],
                                queryWeights->getLeadingDimension(),
                                0.0,
                                &qkOutPtr[outputHeadOffset],
                                qkOutLeadingDim);
            timings.finish("Key/Query matrix product");
            if (checker) {
                checker->submitResult(createDataAccessor(&qkOutPtr[outputHeadOffset],
                                                         {(currentToken + seqlen),
                                                          seqlen},
                                                         qkOutLeadingDim));
            }
        }
    }
    long numRows = currentToken + seqlen;
}
void reclaimMatvecBuffers();

#endif //STREAMING_LLAMA_MATMUL_H
