// Copyright 2014 Marat Andreev
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

using namespace std;

//correlator constant
__constant__ int d_correlator_res;
__constant__ int d_corr_function_length;
__constant__ int d_correlator_counter;
__constant__ int d_corr_nc;

void init_correlator() {

	//correlator constants
	int corr_temp = max_corr_function_length;
	CUDA_SAFE_CALL(
			cudaMemcpyToSymbol(d_corr_function_length, &(corr_temp),
					sizeof(int)));
}

c_correlator::c_correlator(int na) {
	nc = na;
	counter = 0;
	cudaChannelFormatDesc channelDesc4 = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
	cudaChannelFormatDesc channelDesc1 = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	cudaMallocArray(&d_correlator, &channelDesc4, correlator_size, nc, cudaArraySurfaceLoadStore);
	cudaMallocArray(&(d_corr_function), &channelDesc1, max_corr_function_length, nc, cudaArraySurfaceLoadStore);
	CUT_CHECK_ERROR("kernel execution failed");
}

c_correlator::~c_correlator() {
	cudaFreeArray(d_correlator);
	cudaFreeArray(d_corr_function);
}

__global__ __launch_bounds__(ran_tpd) void corr_function_calc_kernel(int *d_time_ticks) {
	int i = blockIdx.x * blockDim.x + threadIdx.x; //i- entanglement index
	if (i >= d_corr_nc)
		return;
	for (int j = 0; j < d_corr_function_length; j++) {
//   	for (int j=0;j<1;j++){

		int ct = d_time_ticks[j];
		float4 data1, data2;
		float cf = 0.0f;
		for (int k = 0; k < d_correlator_counter - ct; k++) {
// 	      for (int k=0;k<1;k++){
			data1 = tex2D(t_correlator, k, i);
			data2 = tex2D(t_correlator, k + ct, i);
			cf += data1.x * data2.x + data1.y * data2.y + data1.z * data2.z;
//  		  cf+=data2.x;

		}
		cf = __fdividef(cf, 3.0f * (d_correlator_counter - ct));
		surf2Dwrite(cf, s_corr_function, 4 * j, i);
	}
}

void c_correlator::calc(int *t, float *x, int np) {

	cudaChannelFormatDesc channelDesc1 = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	cudaChannelFormatDesc channelDesc4 = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);

	cudaBindTextureToArray(t_correlator, d_correlator, channelDesc4);

	cudaBindSurfaceToArray(s_corr_function, d_corr_function);

	int corr_temp = np;
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_corr_function_length, &(corr_temp), sizeof(int)));
	corr_temp = nc;
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_corr_nc, &(corr_temp), sizeof(int)));
	corr_temp = counter;
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_correlator_counter, &(corr_temp),sizeof(int)));
	//correlation calculation
// 	int *ti=new int[np];
// 	for(int j=0;j<np;j++) {ti[j]=int (t[j]);}

	int *d_ti;
	cudaMalloc(&d_ti, sizeof(int) * max_corr_function_length);
	cudaMemcpy(d_ti, t, sizeof(int) * np, cudaMemcpyHostToDevice);

	corr_function_calc_kernel<<<(nc + ran_tpd - 1) / ran_tpd, ran_tpd>>>(d_ti);
	CUT_CHECK_ERROR("corr_function_calc_kernel execution failed");
// 	delete []ti;
	cudaFree(d_ti);
	cudaUnbindTexture(t_correlator);

	//average

	float *x_buf = new float[nc * max_corr_function_length];
	cudaMemcpy2DFromArray(x_buf, sizeof(float) * max_corr_function_length,
			d_corr_function, 0, 0, sizeof(float) * max_corr_function_length, nc,
			cudaMemcpyDeviceToHost);

	int error = 0;

	float *sum_x = new float[max_corr_function_length];
	for (int j = 0; j < np; j++) {
		sum_x[j] = 0.0;
		for (int i = 0; i < nc; i++) {
			sum_x[j] += x_buf[i * max_corr_function_length + j];
			if (x_buf[i * max_corr_function_length + j] != x_buf[i * max_corr_function_length + j])
				error ++;
		}
		x[j] = sum_x[j] / nc;
	}
	delete[] x_buf;
	delete[] sum_x;
	cout << "\n" << error << " NaNs\n";
}
