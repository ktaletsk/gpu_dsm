// Copyright 2015 Marat Andreev, Konstantin Taletskiy, Maria Katzarova
// 
// This file is part of gpu_dsm.
// 
// gpu_dsm is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// at your option) any later version.
// 
// gpu_dsm is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with gpu_dsm.  If not, see <http://www.gnu.org/licenses/>.

#include <iostream>
#include <stdlib.h>
#include "cudautil.h"
#include "cuda_call.h"
#include "textures_surfaces.h"
#include "correlator.h"

#define tpb_corr_kernel 32

using namespace std;

correlator::correlator(int n, int s) {//Initialize correlator
	nc = n; //size of the ensemble
	numcorrelators = s;

	//Dimensions of the memory
	int width = correlator_size;
	int height = numcorrelators;
	int depth = nc;
	cudaExtent extent_f4 = make_cudaExtent(width * sizeof(float4), height, depth);
	cudaExtent extent_f = make_cudaExtent(width * sizeof(float), height, depth);

	//Allocate device 3D memory [width X height X depth]
	CUDA_SAFE_CALL(cudaMalloc3D(&(gpu_corr.d_shift), extent_f4));
	CUDA_SAFE_CALL(cudaMalloc3D(&(gpu_corr.d_correlation), extent_f4));
	CUDA_SAFE_CALL(cudaMalloc3D(&(gpu_corr.d_ncorrelation), extent_f));

	//Allocate device 2D memory [height X depth]
	CUDA_SAFE_CALL(cudaMallocPitch((float4**)&(gpu_corr.d_accumulator), &(gpu_corr.d_accumulator_pitch), height * sizeof(float4), depth));
	CUDA_SAFE_CALL(cudaMallocPitch((int**)&(gpu_corr.d_naccumulator), &(gpu_corr.d_naccumulator_pitch), height * sizeof(int), depth));
	CUDA_SAFE_CALL(cudaMallocPitch((int**)&(gpu_corr.d_insertindex), &(gpu_corr.d_insertindex_pitch), height * sizeof(int), depth));

	//Allocate device 1D memory [depth]
	CUDA_SAFE_CALL(cudaMalloc((int**)&(gpu_corr.d_kmax), nc * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((float4**)&(gpu_corr.d_accval), nc * sizeof(float4)));

	//Allocate device variables:
	CUDA_SAFE_CALL(cudaMalloc((int**)&(gpu_corr.d_nc), sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((int**)&(gpu_corr.d_numcorrelators), sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((int**)&(gpu_corr.d_dmin), sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((int**)&(gpu_corr.d_correlator_size), sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((int**)&(gpu_corr.d_correlator_aver_size), sizeof(int)));

	//Initialize 3D memory
	CUDA_SAFE_CALL(cudaMemset3D((gpu_corr.d_shift), 0, extent_f4));
	CUDA_SAFE_CALL(cudaMemset3D((gpu_corr.d_correlation), 0, extent_f4));
	CUDA_SAFE_CALL(cudaMemset3D((gpu_corr.d_ncorrelation), 0, extent_f));

	//Initialize 2D memory
	CUDA_SAFE_CALL(cudaMemset2D((void *)(gpu_corr.d_accumulator), (gpu_corr.d_accumulator_pitch), 0, height * sizeof(float4), depth)); //not sure if initialization with int works
	CUDA_SAFE_CALL(cudaMemset2D((void *)(gpu_corr.d_naccumulator), (gpu_corr.d_naccumulator_pitch), 0, height * sizeof(int), depth));
	CUDA_SAFE_CALL(cudaMemset2D((void *)(gpu_corr.d_insertindex), (gpu_corr.d_insertindex_pitch), 0, height * sizeof(int), depth));

	//Initialize 1D memory
	CUDA_SAFE_CALL(cudaMemset((gpu_corr.d_kmax), 0, depth*sizeof(int)));
	CUDA_SAFE_CALL(cudaMemset((gpu_corr.d_accval), 0, depth*sizeof(float4)));

	//Initialize variables
	CUDA_SAFE_CALL(cudaMemcpy((gpu_corr.d_nc), &nc, sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy((gpu_corr.d_numcorrelators), &numcorrelators, sizeof(int), cudaMemcpyHostToDevice));
	int temp = correlator_size/correlator_res;
	CUDA_SAFE_CALL(cudaMemcpy((gpu_corr.d_dmin), &temp, sizeof(int), cudaMemcpyHostToDevice));
	temp = correlator_size;
	CUDA_SAFE_CALL(cudaMemcpy((gpu_corr.d_correlator_size), &temp, sizeof(int), cudaMemcpyHostToDevice));
	temp = correlator_res;
	CUDA_SAFE_CALL(cudaMemcpy((gpu_corr.d_correlator_aver_size), &temp, sizeof(int), cudaMemcpyHostToDevice));

	//Initialize stress array
	cudaChannelFormatDesc channelDesc4 = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
	CUDA_SAFE_CALL(cudaMallocArray(&(gpu_corr.d_correlator), &channelDesc4, sizeof(float4)*stressarray_count, n, cudaArraySurfaceLoadStore));

	CUT_CHECK_ERROR("kernel execution failed");
}

correlator::~correlator() {
	cudaFree(gpu_corr.d_shift.ptr);
	cudaFree(gpu_corr.d_correlation.ptr);
	cudaFree(gpu_corr.d_ncorrelation.ptr);
	cudaFree(gpu_corr.d_accumulator);
	cudaFree(gpu_corr.d_naccumulator);
	cudaFree(gpu_corr.d_insertindex);
	cudaFree(gpu_corr.d_kmax);
	cudaFree(gpu_corr.d_accval);
	cudaFree(gpu_corr.d_correlator);
}

__global__ void corr_function_calc_kernel(cudaPitchedPtr d_correlation, cudaPitchedPtr d_ncorrelation, float* d_lag, float* d_corr, int* d_nc, int* d_numcorrelators, int *d_dmin, int *d_correlator_size, int *d_correlator_aver_size) {
	//Calculate kernel index
	int k = blockIdx.y * blockDim.y + threadIdx.y; //correlator level
	int j = blockIdx.x * blockDim.x + threadIdx.x; //lag

	//Check if kernel index is outside boundaries
	if ((k >=*d_numcorrelators) || (j >=*d_correlator_size))
		return;

	if (k>0 && j < *d_dmin)
		return;

	float lag = (float)j * powf((float)(*d_correlator_aver_size), k); // time lag

	float stress = 0.0f; //local variable to accumulate correlation from all chains

	char* correlation_ptr = (char *)(d_correlation.ptr);
	size_t correlation_pitch = d_correlation.pitch;
	size_t correlation_slicePitch = correlation_pitch * *d_numcorrelators;

	char* ncorrelation_ptr = (char *)(d_ncorrelation.ptr);
	size_t ncorrelation_pitch = d_ncorrelation.pitch;
	size_t ncorrelation_slicePitch = ncorrelation_pitch * *d_numcorrelators;

	for (int n = 0; n < *(d_nc); ++n) { //iterate chains
		char* correlation_slice = correlation_ptr + n * correlation_slicePitch;
		float4* correlation = (float4*) (correlation_slice + k * correlation_pitch);
		float4 element = correlation[j];

		char* ncorrelation_slice = ncorrelation_ptr + n * ncorrelation_slicePitch;
		float* ncorrelation = (float*) (ncorrelation_slice + k * ncorrelation_pitch);
		float weight = ncorrelation[j];

		if (weight > 0)
			stress +=__fdividef((element.x + element.y + element.z), weight);
	}

	//Write results to output array
	d_lag[k*(*d_correlator_size)+j]=lag;
	d_corr[k*(*d_correlator_size)+j]=stress;
}

void correlator::calc(int *t, float *x, int correlator_type){

	//allocate and initialize d_corr (flatten 2D, correlation results)
	CUDA_SAFE_CALL(cudaMalloc((float**)&d_lag, correlator_size * numcorrelators * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc((float**)&d_corr,correlator_size * numcorrelators * sizeof(float)));

	CUDA_SAFE_CALL(cudaMemset(d_lag, 0, correlator_size * numcorrelators * sizeof(float)));
	CUDA_SAFE_CALL(cudaMemset(d_corr, 0, correlator_size * numcorrelators * sizeof(float)));

	//2D grid [k X p]
	dim3 dimBlock(tpb_corr_kernel, tpb_corr_kernel);
	dim3 dimGrid((correlator_size + dimBlock.x - 1) / dimBlock.x,(numcorrelators + dimBlock.y - 1) / dimBlock.y);

	corr_function_calc_kernel<<<dimGrid, dimBlock>>>(gpu_corr.d_correlation, gpu_corr.d_ncorrelation, d_lag, d_corr, gpu_corr.d_nc, gpu_corr.d_numcorrelators, gpu_corr.d_dmin, gpu_corr.d_correlator_size, gpu_corr.d_correlator_aver_size);

	float *lag_buffer = new float[correlator_size * numcorrelators];
	float *corr_buffer = new float[correlator_size * numcorrelators];

	//memcpy d_corr, d_lag back to host
	cudaMemcpy(lag_buffer, d_lag, correlator_size * numcorrelators * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(corr_buffer,d_corr,correlator_size * numcorrelators * sizeof(float), cudaMemcpyDeviceToHost);

	//output to t and x
	int im = 0;
	for (unsigned int i=0; i<correlator_size; ++i) {
		t[im] = (int)(lag_buffer[i]);
		if (correlator_type==0)	x[im] = corr_buffer[i]/3.0f;
		if (correlator_type==1)	x[im] = corr_buffer[i];
		++im;
	}
	for (int k=1; k<numcorrelators; ++k) {
			for (int i=correlator_size/correlator_res; i<correlator_size; ++i) {
				t[im] = (int)(lag_buffer[k * correlator_size + i]);
				if (correlator_type==0)	x[im] = corr_buffer[k* correlator_size + i]/3.0f;
				if (correlator_type==1)	x[im] = corr_buffer[k* correlator_size + i];
				++im;
			}
	}
	npcorr = im;
	delete[] lag_buffer;
	delete[] corr_buffer;
}
