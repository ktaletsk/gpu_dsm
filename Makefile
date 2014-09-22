OBJS =  cudautil.o gpu_random.o stress.o chain.o ensemble.o 
CC = nvcc
FLAGS =  -arch=sm_35 -O3 
#by explicitly specifing compute achitecture you can generate smaller executable and gain around 5% performance 
# FLAGS =  -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_32,code=sm_32 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_50,code=compute_50
#no architecture 10 support, sorry

DEBUGFLAGS =
DEPS = gpu_random.h cudautil.h cuda_call.h stress.h chain.h ensemble.h ensemble_kernel.cu ensemble_call_block.cu textures_surfaces.h pcd_tau.h detailed_balance.h job_ID.h

%.o: %.cpp $(DEPS)
	$(CC) -c $<  $(FLAGS) $(DEBUGFLAGS) -o $@
%.o: %.cu $(DEPS)
	$(CC) -c $<  $(FLAGS) $(DEBUGFLAGS) -o $@
	
gpu_DSM:  main.o $(OBJS) 
	$(CC) -o gpu_DSM  main.o $(OBJS)  $(LIBS)
clean:
	rm *.o *~

