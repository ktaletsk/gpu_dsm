//  #include <cuda.h>
#include "gpu_random.h"
#include "random.h"
// #include <iostream>



#if defined(_MSC_VER)
#define uint unsigned int
#endif

 #include "cudautil.h"
 #include "cuda_call.h"      
#include "textures_surfaces.h"



    void gpu_ran_init (){//dummy
    }
      
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

     __global__ __launch_bounds__(ran_tpd) void fill_surface_taucd_gauss_rand (gpu_Ran *state,int n,int count ){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	float4 tmp;
	float g=0.0f;
	float2 g2;
	if (i<n){
	    curandState localState = state[i];
	    for (int j=0; j<count;j++){
		if (g==0.0f){
		    tmp.w=curand_uniform (&localState);
		    g2=curand_normal2(&localState);
		    tmp.x=g2.x;
		    tmp.y=g2.y;
		    g2=curand_normal2(&localState);
		    tmp.z=g2.x;
		    g=g2.y;
		}else{
		    tmp.w=curand_uniform (&localState);
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

    __global__ __launch_bounds__(ran_tpd) void refill_surface_taucd_gauss_rand (gpu_Ran *state,int n,int *count ){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	float4 tmp;
	float g=0.0f;
	float2 g2;
	if (i<n){
	    int cnt=count[i]; 
	    curandState localState = state[i];
	    for (int j=0; j<cnt;j++){
		if (g==0.0f){
		    tmp.w=curand_uniform (&localState);
		    g2=curand_normal2(&localState);
		    tmp.x=g2.x;
		    tmp.y=g2.y;
		    g2=curand_normal2(&localState);
		    tmp.z=g2.x;
		    g=g2.y;
		}else{
		    tmp.w=curand_uniform (&localState);
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

    __global__ __launch_bounds__(ran_tpd) void array_seed (gpu_Ran *gr,int sz,int seed_offset){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	if (i<sz) curand_init(seed_offset, i, 0, &gr[i]);
    }
    
    void gr_array_seed (gpu_Ran *gr,int sz, int seed_offset){
      
	array_seed<<<(sz+ ran_tpd-1)/ ran_tpd, ran_tpd>>>(gr,sz, seed_offset);
	CUT_CHECK_ERROR("kernel execution failed");
 	cudaDeviceSynchronize();

    }

    void gr_fill_surface_uniformrand(gpu_Ran *gr,int sz,int count , cudaArray*  d_uniformrand){
	cudaBindSurfaceToArray(rand_buffer, d_uniformrand);
	CUT_CHECK_ERROR("kernel execution failed");
        fill_surface_rand<<<(sz+ ran_tpd-1)/ ran_tpd, ran_tpd>>>(gr,sz,count);
	CUT_CHECK_ERROR("kernel execution failed");
 	cudaDeviceSynchronize();

    }

    void gr_fill_surface_taucd_gauss_rand(gpu_Ran *gr,int sz,int count,cudaArray*  d_taucd_gauss_rand ){

	cudaBindSurfaceToArray(rand_buffer, d_taucd_gauss_rand);
        fill_surface_taucd_gauss_rand<<<(sz+ ran_tpd-1)/ ran_tpd, ran_tpd>>>(gr,sz,count);
	CUT_CHECK_ERROR("kernel execution failed");
 	cudaDeviceSynchronize();

    }
    
    
     void gr_refill_surface_taucd_gauss_rand(gpu_Ran *gr,int sz,int *count,cudaArray*  d_taucd_gauss_rand ){

	cudaBindSurfaceToArray(rand_buffer, d_taucd_gauss_rand);
        refill_surface_taucd_gauss_rand<<<(sz+ ran_tpd-1)/ ran_tpd, ran_tpd>>>(gr,sz,count);
	CUT_CHECK_ERROR("kernel execution failed");
 	cudaDeviceSynchronize();

    }    
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
