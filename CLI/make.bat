nvcc -c main.cu -gencode arch=compute_50,code=sm_50 -gencode arch=compute_50,code=sm_50 -o main.obj
nvcc -c ../common/cudautil.cu -gencode arch=compute_50,code=sm_50 -o cudautil.obj
nvcc -c ../common/gpu_random.cu -gencode arch=compute_50,code=sm_50 -gencode arch=compute_50,code=sm_50 -o gpu_random.obj
nvcc -c ../common/stress.cpp -gencode arch=compute_50,code=sm_50 -gencode arch=compute_50,code=sm_50 -o stress.obj
nvcc -c ../common/chain.cu -gencode arch=compute_50,code=sm_50 -gencode arch=compute_50,code=sm_50 -o chain.obj
nvcc -c ../common/ensemble.cu -gencode arch=compute_50,code=sm_50 -gencode arch=compute_50,code=sm_50 -o ensemble.obj
nvcc -c ../common/correlator.cu -gencode arch=compute_50,code=sm_50 -o correlator.obj
nvcc -o gpu_DSM  main.obj cudautil.obj gpu_random.obj stress.obj chain.obj ensemble.obj correlator.obj
