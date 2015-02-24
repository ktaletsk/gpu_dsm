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

extern p_cd *pcd;
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
__device__ __forceinline__ float d_tau_CD_f_d_t(float p) {
	return p < d_Bdt ? __powf(p * d_Adt + d_Ddt, d_Cdt) : d_tau_d_inv;
}
__device__ __forceinline__ float d_tau_CD_f_t(float p) {
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
			//Pick a uniform distributed random number
			tmp.x=curand_uniform (&localState);
			//TODO Pick random molecular weight using texture t_gamma_table and RNG and save to tmp.w
			if (g==0.0f){
				if (SDCD_toggle == true)
					tmp.w=d_tau_CD_f_t(tmp.x);
				else
					tmp.w=d_tau_CD_f_d_t(tmp.x);
				g2=curand_normal2(&localState);
				tmp.x=g2.x;
				tmp.y=g2.y;
				g2=curand_normal2(&localState);
				tmp.z=g2.x;
				g=g2.y;
			}else{
				if (SDCD_toggle == true)
					tmp.w=d_tau_CD_f_t(tmp.x);
				else
					tmp.w=d_tau_CD_f_d_t(tmp.x);
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
			if (g==0.0f){
				if (SDCD_toggle == true)
					tmp.w=d_tau_CD_f_t(tmp.x);
				else
					tmp.w=d_tau_CD_f_d_t(tmp.x);
				g2=curand_normal2(&localState);
				tmp.x=g2.x;
				tmp.y=g2.y;
				g2=curand_normal2(&localState);
				tmp.z=g2.x;
				g=g2.y;
			}else{
				if (SDCD_toggle == true)
					tmp.w=d_tau_CD_f_t(tmp.x);
				else
					tmp.w=d_tau_CD_f_d_t(tmp.x);
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
