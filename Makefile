# Copyright 2015 Marat Andreev, Konstantin Taletskiy, Maria Katzarova
# 
# This file is part of gpu_dsm.
# 
# gpu_dsm is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# at your option) any later version.
# 
# gpu_dsm is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with gpu_dsm.  If not, see <http://www.gnu.org/licenses/>.

OBJS =  cudautil.o gpu_random.o stress.o chain.o ensemble.o correlator.o gamma.o
CC = nvcc

#Gencode arguments
SMS = 20 30 35 37 50 52

ifeq ($(GENCODE_FLAGS),)
# Generate SASS code for each SM architecture listed in $(SMS)
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

# Generate PTX code from the highest SM architecture in $(SMS) to guarantee forward-compatibility
HIGHEST_SM := $(lastword $(sort $(SMS)))
ifneq ($(HIGHEST_SM),)
GENCODE_FLAGS += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)
endif
endif

DEPS = gpu_random.h cudautil.h cuda_call.h stress.h chain.h ensemble.h ensemble_kernel.cu ensemble_call_block.cu textures_surfaces.h pcd_tau.h detailed_balance.h job_ID.h correlator.h gamma.h

#For debug you need to generate CPU (-g) and GPU (-G) debug information
#For ability to watch state of variables you need to generate additional information (-keep)
debug: DEBUGFLAGS = -g -G -keep
debug: gpu_DSM

all: gpu_DSM

gpu_DSM:  main.o $(OBJS) 
	$(CC) main.o $(OBJS) $(LIBS) -o gpu_DSM
%.o: %.cpp $(DEPS)
	$(CC) -c $<  $(GENCODE_FLAGS) $(DEBUGFLAGS) -o $@
%.o: %.cu $(DEPS)
	$(CC) -c $<  $(GENCODE_FLAGS) $(DEBUGFLAGS) -o $@
clean:
	rm -f *.o *~ *.cubin *.ptx *.ii *.i *module_id *.hash *fatbin* *cudafe*