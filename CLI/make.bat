nvcc -c main.cu -gencode arch=compute_50,code=sm_50 -gencode arch=compute_50,code=sm_50 --maxrregcount=24 -o main.obj
nvcc -c ../common/cudautil.cu -gencode arch=compute_50,code=sm_50 --maxrregcount=24 -o cudautil.obj
nvcc -c ../common/gpu_random.cu -gencode arch=compute_50,code=sm_50 -gencode arch=compute_50,code=sm_50 --maxrregcount=24 -o gpu_random.obj
nvcc -c ../common/stress.cpp -gencode arch=compute_50,code=sm_50 -gencode arch=compute_50,code=sm_50 --maxrregcount=24 -o stress.obj
nvcc -c ../common/chain.cu -gencode arch=compute_50,code=sm_50 -gencode arch=compute_50,code=sm_50 --maxrregcount=24 -o chain.obj
nvcc -c ../common/ensemble.cu -gencode arch=compute_50,code=sm_50 -gencode arch=compute_50,code=sm_50 --maxrregcount=24 -o ensemble.obj
nvcc -c ../common/correlator.cu -gencode arch=compute_50,code=sm_50 --maxrregcount=24 -o correlator.obj
nvcc -c ../common/gamma.cpp  -gencode arch=compute_35,code=sm_35 -gencode arch=compute_35,code=compute_35  -o gamma.obj
nvcc -o gpu_DSM  main.obj cudautil.obj gpu_random.obj stress.obj chain.obj ensemble.obj correlator.obj gamma.obj
