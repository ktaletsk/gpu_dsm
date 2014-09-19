OBJS =  cudautil.o gpu_random.o chain.o ensemble.o binomial.o orientation_tensor.o
CC = nvcc
#GTX560
# FLAGS =  -arch=sm_21 -O3 
#GTX680
FLAGS =  -arch=sm_35 -O3 
DEBUGFLAGS =
DEPS = gpu_random.h cudautil.h cuda_call.h chain.h ensemble.h ensemble_kernel.cu ensemble_call_block.cu textures_surfaces.h pcd_tau.h detailed_balance.h job_ID.h

#gpu_dynamic_linear_chain.o : gpu_dynamic_linear_chain.cu $(DEPS) gpu_abstract_chain.cugpu_Gauss_free_energy.cu 
#	$(CC) -c $<  $(FLAGS) $(DEBUGFLAGS) -o $@
%.o: %.cpp $(DEPS)
	$(CC) -c $<  $(FLAGS) $(DEBUGFLAGS) -o $@
%.o: %.cu $(DEPS)
	$(CC) -c $<  $(FLAGS) $(DEBUGFLAGS) -o $@
	
gpu_DSM: main_boost.o $(OBJS) 
	$(CC) -o gpu_DSM  main_boost.o $(OBJS)  $(LIBS)
gpu_DSM_noboost: main.o $(OBJS) 
	$(CC) -o gpu_DSM_noboost  main.o $(OBJS)  $(LIBS)
clean:
	rm *.o *~

