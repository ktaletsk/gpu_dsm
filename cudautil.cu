#include <iostream>
#if defined(_MSC_VER)
    #include <windows.h>
#endif
#include <stdio.h>
#include <cuda_gl_interop.h>

#include "cudautil.h"

    void checkCUDA(int dev)
    {
    //checks for presence of GPU
    //selects device number dev to use
    //just copy of "CUDA programming guide" code
	int deviceCount;
	CUDA_SAFE_CALL_NOSYNC(cudaGetDeviceCount(&deviceCount));
	if (deviceCount == 0) {
		fprintf(stderr, "no CUDA device found!\n");
		exit(1);
	}


// 	int dev = deviceCount - 1;
	printf("device Count %d \n", deviceCount);
	if (dev>=deviceCount) printf("device %d not present\n", dev);
	cudaDeviceProp deviceProp;
	CUDA_SAFE_CALL_NOSYNC(cudaGetDeviceProperties(&deviceProp, dev));
	if (deviceProp.major < 1) {
		fprintf(stderr, "device %d does not support CUDA!\n", dev);
		exit(1);
	}

	printf("Using device %d: %s\n", dev, deviceProp.name );
	CUDA_SAFE_CALL(cudaSetDevice(dev));
	CUDA_SAFE_CALL(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
    }


