OBJS =  cudautil.o gpu_random.o stress.o chain.o ensemble.o correlator.o gamma.o
CC = nvcc
#FLAGS =  -gencode arch=compute_35,code=sm_35 
#by explicitly specifing compute achitecture you can generate smaller executable and gain around 5% performance 
FLAGS = -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35
#no architecture 10 support, sorry

#For debug you need to generate CPU (-g) and GPU (-G) debug information
#For ability to watch state of variables you need to generate additional information (-keep)
DEPS = gpu_random.h cudautil.h cuda_call.h stress.h chain.h ensemble.h ensemble_kernel.cu ensemble_call_block.cu textures_surfaces.h pcd_tau.h detailed_balance.h job_ID.h correlator.h gamma.h

all: gpu_DSM

debug: DEBUGFLAGS = -g -G -keep

gpu_DSM:  main.o $(OBJS) 
	$(CC) main.o $(OBJS) $(LIBS) -o gpu_DSM
%.o: %.cpp $(DEPS)
	$(CC) -c $<  $(FLAGS) $(DEBUGFLAGS) -o $@
%.o: %.cu $(DEPS)
	$(CC) -c $<  $(FLAGS) $(DEBUGFLAGS) -o $@
clean:
	rm -f *.o *~ *.cubin *.ptx *.ii *.i *module_id *.hash *fatbin* *cudafe*