#include <iostream>

int gpu_check() {
    int* count = new int;
    cudaError_t x = cudaGetDeviceCount(count);
    if (x==cudaSuccess) return 0;
    else if (x==cudaErrorNoDevice) return -1;
    else if (x==cudaErrorInsufficientDriver) return -2;
    return 1;
}
