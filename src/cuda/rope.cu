#include<cuda_bf16.h>
__device__ void rope(__nv_bfloat16 * a, int num) {
    for(int i = 0; i < num; ++i) {
        a[i] = 0.0;
    }
}
