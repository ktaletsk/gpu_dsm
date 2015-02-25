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

//  #include <cuda.h>
#include "gpu_random.h"
#include "random.h"
#include "pcd_tau.h"
#include "ensemble.h"
#include "gamma.cu"
#include <fstream>

extern p_cd *pcd;
extern bool PD_flag;
// #include <iostream>


#if defined(_MSC_VER)
#define uint unsigned int
#endif

#include "cudautil.h"
#include "cuda_call.h"      
#include "textures_surfaces.h"

//CD constants
__constant__ float d_At, d_Ct, d_Dt, d_Adt, d_Bdt, d_Cdt, d_Ddt;
__constant__ float d_g, d_alpha, d_tau_0, d_tau_max, d_tau_d, d_tau_d_inv;
__constant__ bool d_PD_flag;

void gpu_ran_init (){
	cout << "preparing GPU random number generator parameters..\n";

	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_g, &(pcd->g), sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_alpha, &(pcd->alpha), sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_tau_0, &(pcd->tau_0), sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_tau_max, &(pcd->tau_max), sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_tau_d, &(pcd->tau_d), sizeof(float)));
	float cdtemp = 1.0f / pcd->tau_d;
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_tau_d_inv, &(cdtemp), sizeof(float)));

	cdtemp = 1.0f / pcd->At;
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_At, &cdtemp, sizeof(float)));
	cdtemp = powf(pcd->tau_0, pcd->alpha);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_Dt, &cdtemp, sizeof(float)));
	cdtemp = -1.0f / pcd->alpha;
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_Ct, &cdtemp, sizeof(float)));
	cdtemp = pcd->normdt / pcd->Adt;
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_Adt, &cdtemp, sizeof(float)));
	cdtemp = pcd->Bdt / pcd->normdt;
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_Bdt, &cdtemp, sizeof(float)));
	cdtemp = -1.0f / (pcd->alpha - 1.0f);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_Cdt, &cdtemp, sizeof(float)));
	cdtemp = powf(pcd->tau_0, pcd->alpha - 1.0f);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_Ddt, &(cdtemp), sizeof(float)));

	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_PD_flag, &PD_flag, sizeof(int)));

	if(PD_flag){
		double a, b, mp, Mk;
		ifstream in2;
		in2.open("polydisp.dat");
		in2 >> a;
		in2 >> b;
		in2 >> mp;
		in2 >> Mk;
		gamma(a,b);
		cudaArray* d_gamma_table;
		cudaChannelFormatDesc channelDesc1 = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
		cudaMallocArray(&(d_gamma_table), &channelDesc1, uniformrandom_count, table_size, cudaArraySurfaceLoadStore);
		//TODO cudaMallocArray(&d_sum_W, &channelDesc1, z_max, rsz, cudaArraySurfaceLoadStore);
		//
	}
	cout << "device random number generator parameters done\n";
	}

//
__global__ __launch_bounds__(ran_tpd) void fill_surface_rand (gpu_Ran *state,int n,int count ){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	float tmp;
	if (i<n){
		curandState localState = state[i];
		for (int j=0; j<count;j++){
			tmp=curand_uniform (&localState);
			surf2Dwrite(tmp,rand_buffer,4*j,i);
		}
		state[i] = localState;
	}
}

//
__global__ __launch_bounds__(ran_tpd) void array_seed (gpu_Ran *gr,int sz,int seed_offset){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	if (i<sz) curand_init(seed_offset, i, 0, &gr[i]);
}

//
void gr_array_seed (gpu_Ran *gr,int sz, int seed_offset){      
	array_seed<<<(sz+ ran_tpd-1)/ ran_tpd, ran_tpd>>>(gr,sz, seed_offset);
	CUT_CHECK_ERROR("kernel execution failed");
 	cudaDeviceSynchronize();
}

//
void gr_fill_surface_uniformrand(gpu_Ran *gr,int sz,int count , cudaArray*  d_uniformrand){
	cudaBindSurfaceToArray(rand_buffer, d_uniformrand);
	CUT_CHECK_ERROR("kernel execution failed");
	fill_surface_rand<<<(sz+ ran_tpd-1)/ ran_tpd, ran_tpd>>>(gr,sz,count);
	CUT_CHECK_ERROR("kernel execution failed");
	cudaDeviceSynchronize();
}

//lifetime generation from uniform random number p
__device__ __forceinline__ float d_tau_CD_f_d_t(float p, float d_Adt, float d_Bdt, float d_Cdt, float d_Ddt, float d_tau_d_inv) {
	return p < d_Bdt ? __powf(p * d_Adt + d_Ddt, d_Cdt) : d_tau_d_inv;
}
__device__ __forceinline__ float d_tau_CD_f_t(float p, float d_At, float d_Ct, float d_Dt, float d_tau_d_inv, float d_g) {
	return p < 1.0f - d_g ? __powf(p * d_At + d_Dt, d_Ct) : d_tau_d_inv;
}

//
__global__ __launch_bounds__(ran_tpd) void fill_surface_taucd_gauss_rand (gpu_Ran *state, int n, int count, bool SDCD_toggle){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	float4 tmp;
	float g=0.0f;
	float2 g2;
	if (i<n){
		curandState localState = state[i];
		for (int j=0; j<count;j++){
			//Pcd generation for new entanglements

			//Pick a uniform distributed random number
			tmp.x=curand_uniform (&localState);

			if (d_PD_flag){
				tmp.y=curand_uniform (&localState); //For picking random molecular weight
				//TODO tmp.y=READ_TEXTURE[t_gamma_table[curand_uniform (&localState)/step]
				//TODO Calculate Pcd parameters (How?)
				if (SDCD_toggle == true)
					tmp.w = d_tau_CD_f_t(tmp.x, d_At, d_Ct, d_Dt, d_tau_d_inv, d_g);
				else
					tmp.w = d_tau_CD_f_d_t(tmp.x, d_Adt, d_Bdt, d_Cdt, d_Ddt, d_tau_d_inv);
				//TODO Substitute with calculated Pcd values
			}
			else{
				if (SDCD_toggle == true)
					tmp.w = d_tau_CD_f_t(tmp.x, d_At, d_Ct, d_Dt, d_tau_d_inv, d_g);
				else
					tmp.w = d_tau_CD_f_d_t(tmp.x, d_Adt, d_Bdt, d_Cdt, d_Ddt, d_tau_d_inv);
			}

			//Q vector generation for new entanglements
			if (g==0.0f){
				g2=curand_normal2(&localState);
				tmp.x=g2.x;
				tmp.y=g2.y;
				g2=curand_normal2(&localState);
				tmp.z=g2.x;
				g=g2.y;
			}else{
				tmp.x=g;
				g2=curand_normal2(&localState);
				tmp.y=g2.x;
				tmp.z=g2.y;
				g=0.0f;
			}
			surf2Dwrite(tmp,rand_buffer,16*j,i);
	    }
	    state[i] = localState;
	}
}

//
__global__ __launch_bounds__(ran_tpd) void refill_surface_taucd_gauss_rand (gpu_Ran *state, int n, int *count, bool SDCD_toggle){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	float4 tmp;
	float g=0.0f;
	float2 g2;
	if (i<n){
		int cnt=count[i];
		curandState localState = state[i];
	    for (int j=0; j<cnt;j++){
	    	tmp.x=curand_uniform (&localState);
			if (SDCD_toggle == true)
				tmp.w=d_tau_CD_f_t(tmp.x, d_At, d_Ct, d_Dt, d_tau_d_inv, d_g);
			else
				tmp.w=d_tau_CD_f_d_t(tmp.x, d_Adt, d_Bdt, d_Cdt, d_Ddt, d_tau_d_inv);
			if (g==0.0f){
				g2=curand_normal2(&localState);
				tmp.x=g2.x;
				tmp.y=g2.y;
				g2=curand_normal2(&localState);
				tmp.z=g2.x;
				g=g2.y;
			}else{
				tmp.x=g;
				g2=curand_normal2(&localState);
				tmp.y=g2.x;
				tmp.z=g2.y;
				g=0.0f;
			}
			surf2Dwrite(tmp,rand_buffer,16*j,i);
		}
		state[i] = localState;
	}
}

//
void gr_fill_surface_taucd_gauss_rand(gpu_Ran *gr, int sz, int count, bool SDCD_toggle, cudaArray* d_taucd_gauss_rand){
	cudaBindSurfaceToArray(rand_buffer, d_taucd_gauss_rand);
    fill_surface_taucd_gauss_rand<<<(sz+ ran_tpd-1)/ ran_tpd, ran_tpd>>>(gr,sz,count,SDCD_toggle);
	CUT_CHECK_ERROR("kernel execution failed");
 	cudaDeviceSynchronize();
}

//
void gr_refill_surface_taucd_gauss_rand(gpu_Ran *gr, int sz, int *count, bool SDCD_toggle, cudaArray* d_taucd_gauss_rand){
	cudaBindSurfaceToArray(rand_buffer, d_taucd_gauss_rand);
	refill_surface_taucd_gauss_rand<<<(sz+ ran_tpd-1)/ ran_tpd, ran_tpd>>>(gr,sz,count,SDCD_toggle);
	CUT_CHECK_ERROR("kernel execution failed");
 	cudaDeviceSynchronize();
}
