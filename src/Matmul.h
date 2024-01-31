//
// Created by Grady Schofield on 10/1/23.
//

#ifndef STREAMING_LLAMA_MATMUL_H
#define STREAMING_LLAMA_MATMUL_H

#include<Accelerate/Accelerate.h>

#include<Common.h>
#include<Metal.hpp>
#include<Scratch.h>

using namespace Common;

template<typename T>
void multiplyMatrices(const enum CBLAS_ORDER ORDER,
                      const enum CBLAS_TRANSPOSE TRANSA,
                      const enum CBLAS_TRANSPOSE TRANSB, const int M, const int N,
                      const int K, const T ALPHA, const T * A, const int LDA,
                      const T * B, const int LDB, const T BETA, T * C,
                      const int LDC);

void matvec(MTL::Buffer * mat, MTL::Buffer * in, MTL::Buffer * out, long numRows, long numCols, long outputOffsetElements = 0);
void multiheadMatvecMetal(MTL::Buffer * mat, MTL::Buffer * in, MTL::Buffer * out,
                          long headDimension, long numHeads, long numRows, long leadingDimension);

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
void multiheadMatvec(Scratch<T> * wkOut,
                     Scratch<T> * wqOut,
                     Scratch<T> * qkOut,
                     long headDimension,
                     long numHeads,
                     long currentToken,
                     long seqlen) {
    /*
     * Compute K^T * Q for each head of the attention mechanism
     * We are stepping through horizontal bands of each of K, Q and the output matrix.
     * We are asking for a transpose on a horizontal band of K, not K itself.
     * Imagine the output matrix as numHeads vertically stacked blocks of (cacheSize + seqlen) x seqlen
     */
    if (seqlen > 1) {
    //if (true) {
        T* wkOutPtr = wkOut->getPtr();
        int wkOutLeadingDim = wkOut->getLeadingDimension();
        T* wqOutPtr = wqOut->getPtr();
        int wqOutLeadingDim = wqOut->getLeadingDimension();
        T * qkOutPtr = qkOut->getPtr();
        int qkOutLeadingDim = qkOut->getLeadingDimension();
        for(int head = 0; head < numHeads; ++head) {
            int M = currentToken + seqlen;
            int N = seqlen;
            int K = headDimension;
            int inputHeadOffset = head * headDimension;
            int outputHeadOffset = head * (currentToken + seqlen);
            multiplyMatrices<T>(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                M, N, K,
                                1.0,
                                &wkOutPtr[head * headDimension * wkOutLeadingDim],
                                wkOutLeadingDim,
                                &wqOutPtr[inputHeadOffset],
                                wqOutLeadingDim,
                                0.0,
                                &qkOutPtr[outputHeadOffset],
                                qkOutLeadingDim);
        }
    } else {
        long numRows = currentToken + seqlen;
        long leadingDimension = wkOut->getLeadingDimension();
        multiheadMatvecMetal(wkOut->getMetalBuffer(),
                             wqOut->getMetalBuffer(),
                             qkOut->getMetalBuffer(),
                             headDimension,
                             numHeads,
                             numRows,
                             leadingDimension);
    }
}
void reclaimMatvecBuffers();

#endif //STREAMING_LLAMA_MATMUL_H
