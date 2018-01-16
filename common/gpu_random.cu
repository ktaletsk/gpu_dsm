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

//  #include <cuda.h>
#include "gpu_random.h"
#include "random.h"
#include "pcd_tau.h"
#include "ensemble.h"
#include <fstream>

extern float step, step_d;
extern float a,b,mp,Mk;
extern int table_size, table_size_d;
extern float* GEX_table;
extern float* GEXd_table;
extern float gamma_table_cutoff;

extern p_cd *pcd;
extern bool PD_flag;

#if defined(_MSC_VER)
#define uint unsigned int
#endif

#include "cudautil.h"
#include "cuda_call.h"
#include "textures_surfaces.h"

//CD constants
__constant__ float d_At, d_Ct, d_Dt, d_Adt, d_Bdt, d_Cdt, d_Ddt;
__constant__ float d_A1, d_B1, d_A2, d_B2, d_normdt;
__constant__ float d_g, d_alpha_1, d_alpha_2, d_tau_0, d_tau_1, d_tau_2, d_tau_d_inv;
__constant__ bool d_PD_flag;

//Polydispersity constatnts
__constant__ float d_step, d_step_d;
__constant__ float d_Mk, d_mp;
__constant__ float d_Be;
__constant__ float d_gamma_table_cutoff;

cudaArray* d_gamma_table;
cudaArray* d_gamma_table_d;
texture<float, cudaTextureType1D, cudaReadModeElementType> t_gamma_table;
texture<float, cudaTextureType1D, cudaReadModeElementType> t_gamma_table_d;

__constant__ int d_pcd_table_size_cr, d_pcd_table_size_eq;
cudaArray* d_pcd_table_cr;
cudaArray* d_pcd_table_eq;
texture<float, cudaTextureType1D, cudaReadModeElementType> t_pcd_table_cr;
texture<float, cudaTextureType1D, cudaReadModeElementType> t_pcd_table_eq;

void gpu_ran_init (p_cd* pcd) {
	cout << "preparing GPU random number generator parameters..\n";

    if(PD_flag){
        cudaChannelFormatDesc channelDesc1 = cudaCreateChannelDesc<float>();
        CUDA_SAFE_CALL(cudaMallocArray(&d_gamma_table, &channelDesc1, table_size));
        CUDA_SAFE_CALL(cudaMallocArray(&d_gamma_table_d, &channelDesc1, table_size_d));
        CUDA_SAFE_CALL(cudaMemcpyToArray(d_gamma_table, 0, 0, GEX_table, table_size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpyToArray(d_gamma_table_d, 0, 0, GEXd_table, table_size_d * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaBindTextureToArray(t_gamma_table, d_gamma_table, channelDesc1));
        CUDA_SAFE_CALL(cudaBindTextureToArray(t_gamma_table_d, d_gamma_table_d, channelDesc1));
        CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_step, &step, sizeof(float)));
        CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_step_d, &step_d, sizeof(float)));
        CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_mp, &mp, sizeof(float)));
        CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_Mk, &Mk, sizeof(float)));
        CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_Be, &Be, sizeof(float)));
        CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_gamma_table_cutoff, &gamma_table_cutoff, sizeof(float)));
    }
    else {
        //CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_g, &(pcd->g), sizeof(float)));
        //CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_alpha_1, &(pcd->alpha_1), sizeof(float)));
        //CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_alpha_2, &(pcd->alpha_2), sizeof(float)));
        //CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_tau_0, &(pcd->tau_0), sizeof(float)));
        //CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_tau_1, &(pcd->tau_1), sizeof(float)));
        //CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_tau_2, &(pcd->tau_2), sizeof(float)));
        //float cdtemp = 1.0f / pcd->tau_d;
        //CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_tau_d_inv, &(cdtemp), sizeof(float)));

        //CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_A1, &(pcd->A1), sizeof(float)));
        //CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_B1, &(pcd->B1), sizeof(float)));
        //CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_A2, &(pcd->A2), sizeof(float)));
        //CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_B2, &(pcd->B2), sizeof(float)));
        //CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_Bdt, &(pcd->Bdt), sizeof(float)));
        //CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_normdt, &(pcd->normdt), sizeof(float)));

        ////cdtemp = 1.0f * (pcd->c1) / (pcd->At);
        ////CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_At, &cdtemp, sizeof(float)));
        ////cdtemp = powf(pcd->tau_0, pcd->alpha);
        ////CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_Dt, &cdtemp, sizeof(float)));
        ////cdtemp = -1.0f / pcd->alpha;
        ////CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_Ct, &cdtemp, sizeof(float)));
        ////cdtemp = pcd->normdt * (pcd->c1) / (pcd->Adt);
        ////CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_Adt, &cdtemp, sizeof(float)));
        ////cdtemp = pcd->Bdt;

        ////cdtemp = -1.0f / (pcd->alpha - 1.0f);
        ////CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_Cdt, &cdtemp, sizeof(float)));
        ////cdtemp = powf(pcd->tau_0, pcd->alpha - 1.0f);
        ////CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_Ddt, &(cdtemp), sizeof(float)));

        //Find the smallest g_i
        float min_gi = 1;
        for (int i=0; i < pcd->nmodes; i++){
            float gi = pcd->g[i];
            if (gi < min_gi)
                min_gi = gi;
        }
        int pcd_table_size_cr = int(100/min_gi);
        if (pcd_table_size_cr>10000)    pcd_table_size_cr=10000;
        float* pcd_table_cr = new float[pcd_table_size_cr];
        int j = 0;
        float sum = pcd->g[j];
        for (int i=0; i < pcd_table_size_cr; i++){
            if (float(i)/float(pcd_table_size_cr) > sum){
                j++;
                sum += pcd->g[j];
            }
            pcd_table_cr[i] = 1.0/pcd->tau[j];
        }

        min_gi = 1;
        for (int i=0; i < pcd->nmodes; i++){
            float gi = pcd->tau[i]*pcd->g[i]/pcd->ptau_sum;
            if (gi < min_gi)
                min_gi = gi;
        }

        int pcd_table_size_eq = int(100/min_gi);
        if (pcd_table_size_eq>10000)    pcd_table_size_eq=10000;
        cout << "\nEquilibrium creation table size\t" << pcd_table_size_eq;
        float* pcd_table_eq = new float[pcd_table_size_eq];
        j = 0;
        sum = pcd->tau[j]*pcd->g[j]/pcd->ptau_sum;
        for (int i=0; i < pcd_table_size_eq; i++){
            if (float(i)/float(pcd_table_size_eq) > sum){
                j++;
                sum += pcd->tau[j]*pcd->g[j]/pcd->ptau_sum;
            }
            pcd_table_eq[i] = 1.0/pcd->tau[j];
        }
        //copy p^CD tables to GPU and bind textures
        CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_pcd_table_size_cr, &pcd_table_size_cr, sizeof(int)));
        cudaChannelFormatDesc channelDesc1 = cudaCreateChannelDesc<float>();
        CUDA_SAFE_CALL(cudaMallocArray(&d_pcd_table_cr, &channelDesc1, pcd_table_size_cr));
        CUDA_SAFE_CALL(cudaMemcpyToArray(d_pcd_table_cr, 0, 0, pcd_table_cr, pcd_table_size_cr * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaBindTextureToArray(t_pcd_table_cr, d_pcd_table_cr, channelDesc1));

        CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_pcd_table_size_eq, &pcd_table_size_eq, sizeof(int)));
        CUDA_SAFE_CALL(cudaMallocArray(&d_pcd_table_eq, &channelDesc1, pcd_table_size_eq));
        CUDA_SAFE_CALL(cudaMemcpyToArray(d_pcd_table_eq, 0, 0, pcd_table_eq, pcd_table_size_eq * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaBindTextureToArray(t_pcd_table_eq, d_pcd_table_eq, channelDesc1));

    }

    CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_PD_flag, &PD_flag, sizeof(bool)));
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
void gr_fill_surface_uniformrand(gpu_Ran *gr,int sz,int count, cudaArray* d_uniformrand, cudaStream_t stream_calc){
    cudaBindSurfaceToArray(rand_buffer, d_uniformrand);
    CUT_CHECK_ERROR("kernel execution failed");
    fill_surface_rand<<<(sz+ ran_tpd-1)/ ran_tpd, ran_tpd,0,stream_calc>>>(gr,sz,count);
    CUT_CHECK_ERROR("kernel execution failed");
    cudaDeviceSynchronize();
}

//lifetime generation from uniform random number p
__device__ __forceinline__ float d_tau_CD_f_d_t_linear(float p, float d_Adt, float d_Bdt, float d_Cdt, float d_Ddt, float d_tau_d_inv) {
	return p < d_Bdt ? __powf(p * d_Adt + d_Ddt, d_Cdt) : d_tau_d_inv;
}
__device__ __forceinline__ float d_tau_CD_f_t_linear(float p, float d_At, float d_Ct, float d_Dt, float d_tau_d_inv, float d_g) {
	return p < 1.0f - d_g ? __powf(p * d_At + d_Dt, d_Ct) : d_tau_d_inv;
}

__device__ float d_tau_CD_f_d_t_star(float p, float A1, float B1, float A2, float B2, float alpha_1, float alpha_2, float tau_0, float tau_1, float tau_d_inv, float g, float normdt, float Bdt) {
    p = p * normdt;
    if (p < Bdt)
        return p < (1.0f - g)*A2 / (A1 + B1) ? __powf(p * (alpha_1 - 1) * (A1 + B1) / (1 - g) + __powf(tau_0, alpha_1 - 1), -1.0f / (alpha_1 - 1)) : __powf((alpha_2 - 1) / __powf(tau_1, alpha_1 - alpha_2) * (p * (A1 + B1) / (1 - g) - A2) + __powf(tau_1, alpha_2 - 1), -1.0f / (alpha_2 - 1));
    else
        return tau_d_inv;
}
__device__ float d_tau_CD_f_t_star(float p, float A1, float B1, float A2, float B2, float alpha_1, float alpha_2, float tau_0, float tau_1, float tau_d_inv, float g) {
	if (p < (1.0f - g)) {
		return p < (1.0f - g)*A1 / (A1 + B1) ? __powf(p * alpha_1 * (A1 + B1) / (1 - g) + __powf(tau_0, alpha_1), -1.0f / alpha_1) : __powf(alpha_2 / __powf(tau_1, alpha_1 - alpha_2) * (p * (A1 + B1) / (1 - g) - A1) + __powf(tau_1, alpha_2), -1.0f / alpha_2);
	}
	else {
		return tau_d_inv;
	}
}

__global__ __launch_bounds__(ran_tpd) void fill_surface_taucd_gauss_rand (gpu_Ran *state, int n, int count, bool SDCD_toggle){
    float d_p_At, d_p_Ct, d_p_Dt, d_p_g, d_p_Adt, d_p_Bdt, d_p_Cdt, d_p_Ddt, d_p_tau_d_inv; //Dynamic fdt parameters for given Nk in polydisperse solution
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    float4 tmp;
    float h=0.0f;
    float2 g2;
    if (i<n){
        curandState localState = state[i];
        for (int j=0; j<count;j++){
            //Pcd generation for new entanglements
            tmp.x=curand_uniform (&localState); //Pick a uniform distributed random number

            if (SDCD_toggle == true)
                tmp.w = tex1D(t_pcd_table_eq, int(tmp.x * d_pcd_table_size_eq));
            else
                tmp.w = tex1D(t_pcd_table_cr, int(tmp.x * d_pcd_table_size_cr));

            //Q vector generation for new entanglements
            if (h==0.0f){
                g2=curand_normal2(&localState);
                tmp.x=g2.x;
                tmp.y=g2.y;
                g2=curand_normal2(&localState);
                tmp.z=g2.x;
                h=g2.y;
            }else{
                tmp.x=h;
                g2=curand_normal2(&localState);
                tmp.y=g2.x;
                tmp.z=g2.y;
                h=0.0f;
            }
            surf2Dwrite(tmp,rand_buffer,16*j,i);
        }
        state[i] = localState;
    }
}

//
__global__ __launch_bounds__(ran_tpd) void refill_surface_taucd_gauss_rand (gpu_Ran *state, int n, int *count, bool SDCD_toggle){
	float d_p_At, d_p_Ct, d_p_Dt, d_p_g, d_p_Adt, d_p_Bdt, d_p_Cdt, d_p_Ddt, d_p_tau_d_inv; //Dynamic fdt parameters for given Nk in polydisperse solution
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
                tmp.w = tex1D(t_pcd_table_eq, int(tmp.x * d_pcd_table_size_eq));
            else
                tmp.w = tex1D(t_pcd_table_cr, int(tmp.x * d_pcd_table_size_cr));

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
void gr_fill_surface_taucd_gauss_rand(gpu_Ran *gr, int sz, int count, bool SDCD_toggle, cudaArray* d_taucd_gauss_rand, cudaStream_t stream_calc){
	cudaBindSurfaceToArray(rand_buffer, d_taucd_gauss_rand);
    fill_surface_taucd_gauss_rand<<<(sz+ ran_tpd-1)/ ran_tpd, ran_tpd,0,stream_calc>>>(gr,sz,count,SDCD_toggle);
	CUT_CHECK_ERROR("kernel execution failed");
 	cudaDeviceSynchronize();
}

//
void gr_refill_surface_taucd_gauss_rand(gpu_Ran *gr, int sz, int *count, bool SDCD_toggle, cudaArray* d_taucd_gauss_rand, cudaStream_t stream_calc){
	cudaBindSurfaceToArray(rand_buffer, d_taucd_gauss_rand);
	refill_surface_taucd_gauss_rand<<<(sz+ ran_tpd-1)/ ran_tpd, ran_tpd,0,stream_calc>>>(gr,sz,count,SDCD_toggle);
	CUT_CHECK_ERROR("kernel execution failed");
 	cudaDeviceSynchronize();
}

//TO REIMPLEMENT LATER:
//__device__ void p_cd_(float Be, float Nk, float *d_p_At, float *d_p_Ct, float *d_p_Dt, float *d_p_g, float *d_p_Adt, float *d_p_Bdt, float *d_p_Cdt, float *d_p_Ddt, float *d_p_tau_d_inv) {
//	//Generates \tau_CD lifetimes
//	//uses analytical approximation to P_cd parameters
//	float c1, cm1, rcm1c1, At, Adt, Bdt, normdt, g, alpha, tau_0, tau_max, tau_d;
//	double z = (Nk + Be) / (Be + 1.0);
//	g = 0.667f;
//	if (Be != 1.0f) {
//		//Analytical approximation to P_cd parameters for FSM
//		//Unpublished Pilyugina E. (2012)
//
//		alpha = (0.053f * logf(Be) + 0.31f) * powf(z, -0.012f * logf(Be) - 0.024f);
//		tau_0 = 0.285f * powf(Be + 2.0f, 0.515f);
//		tau_max = ((Nk<2.0f) ? tau_0 : 0.025f * powf(Be + 2.0f, 2.6f) * powf(z, 2.83f));
//		tau_d = ((Nk<2.0f) ? tau_0 : 0.036f * powf(Be + 2.0f, 3.07f) * powf(z - 1.0f, 3.02f));
//	} else {
//		//Analytical approximation to P_cd parameters CFSM
//		//Andreev, M., Feng, H., Yang, L., and Schieber, J. D.,J. Rheol. 58, 723 (2014).
//		//DOI:10.1122/1.4869252
//
//		alpha = 0.267096f - 0.375571f * expf(-0.0838237f * Nk);
//		tau_0 = 0.460277f + 0.298913f * expf(-0.0705314f * Nk);
//		tau_max = ((Nk<4.0f) ? tau_0 : 0.0156137f * powf(Nk, 3.18849f));
//		tau_d = ((Nk<4.0f) ? tau_0 : 0.0740131f * powf(Nk, 3.18363f));
//	}
//	//Workaround for short chains
//	//Previous fits of parameters {alpha,tau_0,tau_max,tau_d) did not include short chains <Z> < 3
//	//And expression for power law part of p^CD stop working there
//	//So we set tau_max=tau_0 if Nk<4 and apply analytically found limit of (tau_max^(alpha-1)-tau_0^(alpha-1))/(tau_max^alpha-tau_0^alpha) -> alpha/(alpha-1)
//	//And the code doesn't break here
//
//	//init vars
//	c1 = powf(tau_max, alpha) - powf(tau_0, alpha);
//	cm1 = powf(tau_max, alpha - 1.0f) - powf(tau_0, alpha - 1.0f);
//	rcm1c1 = ((c1==0.0f) ? (alpha - 1.0f)/alpha/tau_0 : (cm1/c1));//ratio beween c1 and c-1
//	At = (1.0f - g);
//	Adt = At * alpha / (alpha - 1.0f);
//	Bdt = Adt * rcm1c1;
//	normdt = Bdt + g / tau_d;
//
//	*d_p_g=g;
//	*d_p_tau_d_inv = __fdividef(1.0f, tau_d);
//	*d_p_At = __fdividef(c1, At);
//	*d_p_Dt = powf(tau_0, alpha);
//	*d_p_Ct = __fdividef(-1.0f, alpha);
//	*d_p_Adt = __fdividef(normdt*c1, Adt);
//	*d_p_Bdt = __fdividef(Bdt, normdt);
//	*d_p_Cdt = __fdividef(-1.0f, alpha - 1.0f);
//	*d_p_Ddt = powf(tau_0, alpha - 1.0f);
//}
