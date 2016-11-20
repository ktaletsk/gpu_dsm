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

// Short intro
//
// CUDA devices possess enormous computation capabilities,
// however memory access (especially writing) is relatively slow.
// Unfortunately DSM flow simulation require to update significant part
// of chain conformations variables every time step, which normally bottlenecks the performance.
// First time conformation updated when flow deformation of strand orientation vectors is applied, second time when jump process is applied.
// If the jump process is SD shift, only two neighboring N_i must be updated,
// but in case entanglement creation/destruction major portion of chain conformation
// arrays must be moved. On GPU it is a very expensive operation,
// almost as expensive as updating {Q_i} during deformation.
// Thus we combined two conformation updates into one.
// It is done through "delayed dynamics". This means that jump process is not applied
// immediately, but information about it stored  in temporary variables until deformation applied. 
// Next time step shifting of arrays applied simultaneously together with flow deformation.

#ifndef _ENSEMBLE_KERNEL_
#define _ENSEMBLE_KERNEL_

#if defined(_MSC_VER)
#define uint unsigned int
#endif

#define tpb_chain_kernel 256
#define tpb_strent_kernel 32

#define NUM_BANKS 16
#define LOG_NUM_BANKS 4
#ifdef ZERO_BANK_CONFLICTS
#define CONFLICT_FREE_OFFSET(n) ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))
#else
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)
#endif

#include "textures_surfaces.h"
#include "chain.h"

//d means device variables
__constant__ float d_universal_time;
__constant__ float dBe;
__constant__ int dnk;
__constant__ int dnk_arms[100];
__constant__ int d_z_max; //limits actual array size. might come useful for large beta values and for polydisperse systems
__constant__ int d_z_max_arms[100];//limits for each arm
__constant__ int d_narms;
__constant__ int dn_cha_per_call; //number of chains in this call. cannot be bigger than chains_per_call
__constant__ float d_kappa_xx, d_kappa_xy, d_kappa_xz, d_kappa_yx, d_kappa_yy,d_kappa_yz, d_kappa_zx, d_kappa_zy, d_kappa_zz;

//CD constants
__constant__ int d_CD_flag;
__constant__ float d_CD_create_prefact;
__constant__ int d_correlator_res;

// delayed dynamics --- how does it work:
// There are entanglement parallel portion of the code and chain parallel portion.
// The entanglement parallel part applies flow deformation and calculates jump process probabilities.
// The chain parallel part picks one of the jump processes, generates a new orientation vector and a tau_CD if needed.
// It applies only some simpliest chain conformation changes(SD shifting).
// The Information about complex chain conformation changes(entanglement creation/destruction) is stored in temp arrays d_offset, d_new_strent,d_new_tau_CD.
// Complex changes are applied next time step by entanglement parallel part.

//float4 math operators
inline __device__ void operator+=(float4 &a, float4 b) {
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
	a.w += b.w;
}
inline __host__ __device__ float4 operator/(float4 a, float b) {
	return make_float4(a.x / b, a.y / b, a.z / b, a.w / b);
}
inline __host__ __device__ float4 operator*(float4 a, float4 b) {
	return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}
inline __host__ __device__ float4 operator*(float4 a, float b) {
	return make_float4(a.x * b, a.y * b, a.z * b, a.w * b);
}
inline __host__ __device__ float4 operator+(float4 a, float4 b) {
	return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
inline __host__ __device__ float4 operator-(float4 a, float4 b) {
	return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

//offset in 2 component vector {shifting starting index, shifting direction}
//offset stores both components in the one int variable
//index in first 3 bytes, direction in last byte
__device__ __forceinline__ int offset_code(int offset_index, int offset_dir) {
	return (offset_dir + 1) | (offset_index << 8);
}

// returns i or i+/- 1 from offset
__device__ __forceinline__ int make_offset(int i, int offset) {
	//offset&0xffff00)>>8 offset_index
	//offset&0xff-1; offset_dir
	return i >= ((offset & 0xffff00) >> 8) ? i + ((offset & 0xff) - 1) : i;
}

//returns components of offset
__device__ __forceinline__ int offset_index(int offset) {
	return ((offset & 0xffff00) >> 8);
}

__device__ __forceinline__ int offset_dir(int offset) {
	return (offset & 0xff) - 1;
}

//returns true if d_new_strent should be inserted at index i
__device__ __forceinline__ bool fetch_new_strent(int i, int offset) {
	return (i == offset_index(offset)) && (offset_dir(offset) == -1);
}

//deformation
__device__   __forceinline__ float4 kappa(const float4 QN, const float dt) {//Qx is different for consitency with old version
	return make_float4(
			QN.x + dt * d_kappa_xx * QN.x + dt * d_kappa_xy * QN.y + dt * d_kappa_xz * QN.z,
			QN.y + dt * d_kappa_yx * QN.x + dt * d_kappa_yy * QN.y + dt * d_kappa_yz * QN.z,
			QN.z + dt * d_kappa_zx * QN.x + dt * d_kappa_zy * QN.y + dt * d_kappa_zz * QN.z, QN.w);
}

//The entanglement parallel part of the code
//2D kernel: i- entanglement index j - chain index
template<int type> __global__ __launch_bounds__(tpb_strent_kernel*tpb_strent_kernel) void strent_kernel(
		scalar_chains* chain_heads, float *tdt, int *d_offset, float4 *d_new_strent, float *d_new_tau_CD)
{
	//Calculate kernel index
	int i = blockIdx.x * blockDim.x + threadIdx.x;//strent index
	int j = blockIdx.y * blockDim.y + threadIdx.y;//chain index

	//Check if kernel index is outside boundaries
	if ((j >= dn_cha_per_call) || (i >= d_z_max))
		return;

	int arm=0;
	int run_sum=0;
	for (int k=0; i>=run_sum; k++){
		run_sum+=d_z_max_arms[k];
		arm = k;
	}
	int ii = i-run_sum+d_z_max_arms[arm];
	int jj = j*d_narms+arm;

	int tz = chain_heads[j].Z[arm]; //Current chain size
	if (ii >= tz) //Check if entaglement index is over chain size
		return;

	//When new entaglements are created we need to shift index +1(destruction, skip one strent), 0(nothing happens) or -1(new strent created before)
	int oft = d_offset[jj]; //Offset for current chain

	//fetch
	float4 QN;
	if (fetch_new_strent(i, oft)){
		QN = d_new_strent[j]; //second check if strent created last time step should go here
//		printf("\nNew strent\t%i\t%i\t%i\t%i\t%i\t%f\t%f\t%f\t%f\t%i",i,arm,ii,d_z_max_arms[arm],tz,QN.x,QN.y,QN.z,QN.w,oft);
	}
	else{
		QN = tex2D(t_a_QN, make_offset(i, oft), j); // all access to strents is done through two operations: first texture fetch
//		printf("\nRead\t%i\t%i\t%i\t%i\t%i\t%f\t%f\t%f\t%f\t%i\t%i\t%i\t%i",i,arm,ii,d_z_max_arms[arm],tz,QN.x,QN.y,QN.z,QN.w,oft,offset_index(oft),offset_dir(oft),make_offset(i, oft));
	}

	float tcd;
//	if (d_CD_flag) { //If constraint dynamics is enabled
		if (fetch_new_strent(i, oft))
			tcd = d_new_tau_CD[j];
		else
			tcd = tex2D(t_a_tCD, make_offset(i, oft), j);
//	} else
//		tcd = 0;

	float dt;
	if (type==1){//transform
		dt = tdt[j];
		QN = kappa(QN, dt);
	}
	//printf("\n%i\t%i\t%i\t%i\t%i\t%f\t%f\t%f\t%f\t%i",i,arm,ii,d_z_max_arms[arm],tz,QN.x,QN.y,QN.z,QN.w,oft);
	//fetch next strent
	if ((ii > 0) && (ii < tz - 1)) {
		float4 wsh = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
		float4 QN2; //Q for next strent
		if (fetch_new_strent(i + 1, oft))
			QN2 = d_new_strent[j];
		else
			QN2 = tex2D(t_a_QN, make_offset(i + 1, oft), j);

		if (type==1){//transform
			QN2 = kappa(QN2, dt);
		}
		//w_shift probability calc

		float Q = QN.x * QN.x + QN.y * QN.y + QN.z * QN.z;
		float Q2 = QN2.x * QN2.x + QN2.y * QN2.y + QN2.z * QN2.z;

		if (QN2.w > 1.0f) { //N=1 mean that shift is not possible, also ot will lead to dividing on zero error
			//float prefact=__powf( __fdividef(QN.w*QN2.w,(QN.w+1)*(QN2.w-1)),0.75f);
//			printf("\n%i\t%i\t%i\tQN2.w %f",i,arm,ii,QN2.w);
			float sig1 = __fdividef(0.75f, QN.w * (QN.w + 1)); //fdivedf - fast divide float
			float sig2 = __fdividef(0.75f, QN2.w * (QN2.w - 1));
			float prefact1 = (Q == 0.0f) ? 1.0f : __fdividef(QN.w, (QN.w + 1));
			float prefact2 = (Q2 == 0.0f) ? 1.0f : __fdividef(QN2.w, (QN2.w - 1));
			float f1 = (ii == 0) ? 2.0f * QN.w + 0.5f : QN.w;
			float f2 = (ii == tz-2) ? 2.0f * QN2.w - 0.5f : QN2.w;
			float friction = __fdividef(2.0f, f1 + f2);
			wsh.x = friction * __powf(prefact1 * prefact2, 0.75f)* __expf(Q * sig1 - Q2 * sig2);
		}
		if (QN.w > 1.0f) {//N=1 mean that shift is not possible, also ot will lead to dividing on zero error
//			printf("\n%i\t%i\t%i\tQN.w %f",i,arm,ii,QN.w);
			float sig1 = __fdividef(0.75f, QN.w * (QN.w - 1.0f));
			float sig2 = __fdividef(0.75f, QN2.w * (QN2.w + 1.0f));
			float prefact1 = (Q == 0.0f) ? 1.0f : __fdividef(QN.w, (QN.w - 1.0f));
			float prefact2 = (Q2 == 0.0f) ? 1.0f : __fdividef(QN2.w, (QN2.w + 1.0f));
			float f1 = (ii == 0) ? 2.0f * QN.w - 0.5f : QN.w;
			float f2 = (ii == tz-2) ? 2.0f * QN2.w + 0.5f : QN2.w;
			float friction = __fdividef(2.0f, f1 + f2);
			wsh.y = friction * __powf(prefact1 * prefact2, 0.75f) * __expf(-Q * sig1 + Q2 * sig2);
		}
//			printf("\nStrent %i\t%i\t%i\t%i\t%i\t%i\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f",j,i,arm,ii,d_z_max_arms[arm],tz,QN.x,QN.y,QN.z,QN.w,wsh.x + wsh.y,wsh.x,wsh.y,QN2.x,QN2.y,QN2.z,QN2.w);
//			printf("\n%i\t%i\t%i\t%f\t%f\t%f\t%f",arm,ii,tz,QN.x,QN.y,QN.z,QN.w);
//		surf2Dwrite(wsh.x + wsh.y /*+ d_CD_flag* (tcd + d_CD_create_prefact * (QN.w - 1.0f))*/,s_sum_W, 4 * i, j);
//		printf("\nwsh %i %i: %f\t%f\t%f\t%f",arm,ii,wsh.x,wsh.y,wsh.z,wsh.w);
		surf2Dwrite(wsh, s_sum_W, sizeof(float4)*i, j);
		//probability of Kuhn step shitt + probability of entanglement destruction by CD
		// + probability of entanglement creation by CD
	}

//	if (j==1)	printf("\n%i\t%i\t%i\t%i\t%i\t%f\t%f",j,i,arm,ii,tz,QN.w,d_universal_time + chain_heads[j].time);

	//write updated chain conformation
	surf2Dwrite(QN, s_b_QN, 16 * i, j);
	surf2Dwrite(tcd, s_b_tCD, 4 * i, j);
}

__global__ void boundary1_kernel(
		scalar_chains* chain_heads, int *d_offset, float4 *d_new_strent)
{
	//calculate probabilities at the ends of chain
//	int i = blockIdx.x * blockDim.x + threadIdx.x;//chain index
	int ii = blockIdx.x * blockDim.x + threadIdx.x;//arm index in ensemble
	int i = int(ii/d_narms);
	int arm = ii - i*d_narms;

	if (i >= dn_cha_per_call)
		return;
	float4 QNtail; //last strent
	float4 QNtailp;
	float4 new_strent = d_new_strent[i];
	float sumW = 0;

	int run_sum=0;

	for (int u=0; u<arm; u++){
		run_sum += d_z_max_arms[arm];
	}

	int tz = chain_heads[i].Z[arm];
	uint oft = d_offset[ii];

	float4 probs_z = make_float4(0.0f,0.0f,0.0f,0.0f);

	if (fetch_new_strent(tz - 1 + run_sum, oft))
		QNtail = new_strent;
	else
		QNtail = tex2D(t_a_QN, make_offset((tz - 1)+ run_sum, oft), i);

	if (tz == 1) {
		probs_z.y = __fdividef(1.0f, (dBe * dnk_arms[arm]));//Creation at the end by SD
	} else {
		if (QNtail.w == 1.0f) {//destruction by SD at the end
			if (fetch_new_strent(tz - 2 + run_sum, oft))
				QNtailp = new_strent;
			else
				QNtailp = tex2D(t_a_QN, make_offset(tz - 2 + run_sum, oft), i);
			float f1 = (tz == 2) ? QNtailp.w + 0.25f : 0.5f * QNtailp.w;
			probs_z.x = __fdividef(1.0f, f1 + 0.75f);
		} else {//creation by SD at the end
			probs_z.y = __fdividef(2.0f, dBe * (QNtail.w + 0.5f));
		}
	}
	sumW += probs_z.x + probs_z.y;
	surf2Dwrite(probs_z, s_sum_W, sizeof(float4)*(tz - 1+run_sum), i);
}

template<int narms> __global__ void boundary2_kernel(
		scalar_chains* chain_heads, int *d_offset, float4 *d_new_strent)
{
	//calculate probabilities at the ends of chain
//	int i = blockIdx.x * blockDim.x + threadIdx.x;//chain index
	int ii = blockIdx.x * blockDim.x + threadIdx.x;//arm index in ensemble
	int i = int(ii/narms);
	int arm = ii - i*narms;

	if (i >= dn_cha_per_call)
		return;
//	clock_t t1 = clock();
	float4 QNhead_arms[narms]; // first strent
	float4 new_strent = d_new_strent[i];
	float sumW = 0;
	int tz;
	int run_sum=0;
	for (int u=0; u<narms; u++){
		tz = chain_heads[i].Z[u];
		uint oft = d_offset[i*narms+u];

		float4 probs_z = make_float4(0.0f,0.0f,0.0f,0.0f);

		if (fetch_new_strent(0 + run_sum, oft))
			QNhead_arms[u] = new_strent;
		else
			QNhead_arms[u] = tex2D(t_a_QN, make_offset(0 + run_sum, oft), i);
		run_sum += d_z_max_arms[u];
	}
//	clock_t t2 = clock();
	//Shifts near branching point
	float4 QN2; //Q for next strent
	run_sum=0;
	float upsum1 = 0.0f;
	float upsum2 = 0.0f;
	float downsum1 = 0.0f;
	float downsum2 = 0.0f;
	float temp;

	run_sum=0;
	for (int u=0; u<arm; u++){
		run_sum += d_z_max_arms[u];
	}
//	for (int arm=0; arm<narms; arm++){
	tz = chain_heads[i].Z[arm];
	if (tz>1){
		uint oft = d_offset[i*narms+arm];
		if (fetch_new_strent(1 + run_sum, oft))
			QN2 = new_strent;
		else
			QN2 = tex2D(t_a_QN, make_offset(1 + run_sum, oft), i);

		float Q2 = QN2.x * QN2.x + QN2.y * QN2.y + QN2.z * QN2.z;
		float4 probs_1 = make_float4(0.0f,0.0f,0.0f,0.0f);
		//backward shift
		if (QN2.w > 1.0f) { //N=1 mean that shift is not possible
			upsum1 = 0.0f;
			upsum2 = 0.0f;
			downsum1 = 0.0f;
			downsum2 = 0.0f;
			for (int q1=0; q1 < narms; q1++){
				if (chain_heads[i].Z[q1]>1){
					for (int q2=q1+1; q2 < narms; q2++){
						if (chain_heads[i].Z[q2]>1){
							temp = (QNhead_arms[q2].x - QNhead_arms[q1].x)*(QNhead_arms[q2].x - QNhead_arms[q1].x)+(QNhead_arms[q2].y - QNhead_arms[q1].y)*(QNhead_arms[q2].y - QNhead_arms[q1].y)+(QNhead_arms[q2].z - QNhead_arms[q1].z)*(QNhead_arms[q2].z - QNhead_arms[q1].z);
							upsum1 += __fdividef(0.75 * temp,QNhead_arms[q2].w * QNhead_arms[q1].w);
							upsum2 += __fdividef(0.75 * temp,(QNhead_arms[q2].w + (int)(q2==arm)) * (QNhead_arms[q1].w + (int)(q1==arm)));
						}
					}
					downsum1 += __fdividef(1.0f, QNhead_arms[q1].w);
					downsum2 += __fdividef(1.0f, QNhead_arms[q1].w + (int)(q1==arm));
				}
			}

			float sig2 = __fdividef(0.75f, QN2.w * (QN2.w - 1));
			float prefact1 = __fdividef(QNhead_arms[arm].w, (QNhead_arms[arm].w + 1));
			float prefact2 = (Q2 == 0.0f) ? 1.0f : __fdividef(QN2.w, (QN2.w - 1));
			float f1 = QNhead_arms[arm].w;
			float f2 = (tz == 2) ? 2.0f * QN2.w - 0.5f : QN2.w;
			float friction = __fdividef(2.0f, f1 + f2);
			probs_1.x = friction * __powf(__fdividef(prefact1 * prefact2 * downsum1,downsum2), 0.75f)* __expf(__fdividef(upsum1,downsum1) - __fdividef(upsum2,downsum2) - Q2 * sig2);
		}

		//forward shift
		if (QNhead_arms[arm].w > 1.0f) { //N=1 mean that shift is not possible
			upsum1 = 0.0f;
			upsum2 = 0.0f;
			downsum1 = 0.0f;
			downsum2 = 0.0f;
			for (int q1=0; q1 < narms; q1++){
				if (chain_heads[i].Z[q1]>1){
					for (int q2=q1+1; q2 < narms; q2++){
						if (chain_heads[i].Z[q2]>1){
							temp = (QNhead_arms[q2].x - QNhead_arms[q1].x)*(QNhead_arms[q2].x - QNhead_arms[q1].x)+(QNhead_arms[q2].y - QNhead_arms[q1].y)*(QNhead_arms[q2].y - QNhead_arms[q1].y)+(QNhead_arms[q2].z - QNhead_arms[q1].z)*(QNhead_arms[q2].z - QNhead_arms[q1].z);
							upsum1 += __fdividef(0.75 * temp,QNhead_arms[q2].w * QNhead_arms[q1].w);
							upsum2 += __fdividef(0.75 * temp,(QNhead_arms[q2].w - (int)(q2==arm)) * (QNhead_arms[q1].w - (int)(q1==arm)));
						}
					}
					downsum1 += __fdividef(1.0f, QNhead_arms[q1].w);
					downsum2 += __fdividef(1.0f, QNhead_arms[q1].w - (int)(q1==arm));
				}
			}

			float sig2 = __fdividef(0.75f, QN2.w * (QN2.w + 1));
			float prefact1 = __fdividef(QNhead_arms[arm].w, (QNhead_arms[arm].w - 1));
			float prefact2 = (Q2 == 0.0f) ? 1.0f : __fdividef(QN2.w, (QN2.w + 1));
			float f1 = QNhead_arms[arm].w;
			float f2 = (tz == 2) ? 2.0f * QN2.w - 0.5f : QN2.w;
			float friction = __fdividef(2.0f, f1 + f2);
			probs_1.y = friction * __powf(__fdividef(prefact1 * prefact2 * downsum1,downsum2), 0.75f)* __expf(__fdividef(upsum1,downsum1) - __fdividef(upsum2,downsum2) + Q2 * sig2);
		}
		//sumW += probs_1.x + probs_1.y;
		surf2Dwrite(probs_1, s_sum_W, sizeof(float4)*run_sum, i);
	}
//		run_sum += d_z_max_arms[arm];
//	}
//	clock_t t3 = clock();
//	printf("\n%i\t%i",(int)(t2-t1),(int)(t3-t2));
}

__global__ void scan_kernel(scalar_chains* chain_heads, int *rand_used, int* found_index,  int* found_shift) {
	extern __shared__ double s[];
	//Calculate kernel index
	int i = blockIdx.x * blockDim.x + threadIdx.x;//strent index
	int j = blockIdx.y * blockDim.y + threadIdx.y;//chain index
	//Check if kernel index is outside boundaries
	if ((j >= dn_cha_per_call) || (i >= d_z_max))
		return;

	int arm, run_sum, ii, tz;
	float4 temp;

	arm=0;
	run_sum=0;

	for (arm=0; i>=run_sum+d_z_max_arms[arm]; arm++){
		run_sum+=d_z_max_arms[arm];
	}
//	clock_t t1 = clock();
	tz = chain_heads[j].Z[arm]; //Current chain size
	temp = make_float4(0.0f,0.0f,0.0f,0.0f);
	ii = i-run_sum;

	if ((ii >= 0) && (ii < tz)){
		surf2Dread(&temp, s_sum_W, sizeof(float4)*i, j);
	}
//	clock_t t2 = clock();
	//parallel scan in s (naive)
	int pout = 0, pin = 1;
	s[pout*d_z_max+i] = (double)temp.x + (double)temp.y;
//	clock_t t3 = clock();
	__syncthreads();
	for (int offset = 1; offset < d_z_max; offset *= 2) {
		pout = 1 - pout; // swap double buffer indices
		pin = 1 - pout;

		if (i >= offset)
		    s[pout*d_z_max + i] = s[pin*d_z_max + i] + s[pin*d_z_max + i - offset];
		else
			s[pout*d_z_max + i] = s[pin*d_z_max + i];
		__syncthreads();
	}
	surf2Dwrite((float)s[pout*d_z_max + i], s_sum_W_sorted, sizeof(float)*i, j);
//	clock_t t4 = clock();
	//search
	float ran = tex2D(t_uniformrand, rand_used[j], j);
	double x = s[pout*d_z_max+d_z_max-1]*(double)ran;
	double left = (i==0)? 0.0f : s[pout*d_z_max + i - 1];
	double right = s[pout*d_z_max + i];
	bool xFound = (left < x) && (x <= right);
	if (xFound){
		found_index[j]=i;
		if (temp.y == 0.0f)
			found_shift[j]=1;
		else{
			if (x-left > temp.x)
				found_shift[j]=0;
			else
				found_shift[j]=1;
		}
	}
//	clock_t t5 = clock();
//	printf("\n%i\t%i\t%i\t%i",(int)(t2-t1),(int)(t3-t2),(int)(t4-t3),(int)(t5-t4));
}

//Add new value w to k-th level of correlator corr for chain i
__device__ void corr_add(corr_device gpu_corr, float4 w, int k, int i, int type) {
	int s = *(gpu_corr.d_numcorrelators);
	//s is the last correlator level
	if (k == s)
		return;

	int dm = *(gpu_corr.d_dmin);
	int p = *(gpu_corr.d_correlator_size);
	int corr_aver_s = *(gpu_corr.d_correlator_aver_size);

	//extract 3D array pointers and pitches
	char* shift_ptr = (char *) (gpu_corr.d_shift.ptr);
	size_t shift_pitch = gpu_corr.d_shift.pitch;
	size_t shift_slicePitch = shift_pitch * s;
	char* shift_slice = shift_ptr + i * shift_slicePitch;
	float4* shift = (float4*) (shift_slice + k * shift_pitch);

	char* correlation_ptr = (char *) (gpu_corr.d_correlation.ptr);
	size_t correlation_pitch = gpu_corr.d_correlation.pitch;
	size_t correlation_slicePitch = correlation_pitch * s;
	char* correlation_slice = correlation_ptr + i * correlation_slicePitch;
	float4* correlation = (float4*) (correlation_slice + k * correlation_pitch);

	char* ncorrelation_ptr = (char *) (gpu_corr.d_ncorrelation.ptr);
	size_t ncorrelation_pitch = gpu_corr.d_ncorrelation.pitch;
	size_t ncorrelation_slicePitch = ncorrelation_pitch * s;
	char* ncorrelation_slice = ncorrelation_ptr + i * ncorrelation_slicePitch;
	float* ncorrelation = (float*) (ncorrelation_slice + k * ncorrelation_pitch);

	//Extract 2D array pointers and pitches
	float4* accumulator = (float4*) ((char*) gpu_corr.d_accumulator + i * gpu_corr.d_accumulator_pitch);
	int* naccumulator = (int*) ((char*) gpu_corr.d_naccumulator + i * gpu_corr.d_naccumulator_pitch);
	int* insertindex = (int*) ((char*) gpu_corr.d_insertindex + i * gpu_corr.d_insertindex_pitch);

	//update maximum attained correlator level
	if (k > gpu_corr.d_kmax[i])
		gpu_corr.d_kmax[i] = k;

	//Write new value to the shift array
	shift[insertindex[k]] = w;

	//Update average value
	if (k == 0){
		gpu_corr.d_accval[i] += w;
	}

	//Add to accumulator and send to the next level, if needed
	if (type==0 || (type==1 && naccumulator[k]==0)){
		accumulator[k] += w;
	}
	naccumulator[k]++;
	if (naccumulator[k] == corr_aver_s) {
		//Calling next correlator
		if (type==0)	corr_add(gpu_corr, (accumulator[k]) / float(corr_aver_s), k + 1, i, type);
		if (type==1)	corr_add(gpu_corr, (accumulator[k]), k + 1, i, type);
		accumulator[k] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
		naccumulator[k] = 0;
	}

	//Update correlation results
	int ind1 = insertindex[k];
	float4 temp_shift_1 = shift[ind1];//cache frequently used value in register
	float4 temp_shift_2;
	if (k == 0) {
		int ind2 = ind1;
		for (int j = 0; j < p; ++j) {
			temp_shift_2 = shift[ind2];
			if (temp_shift_2.x != 0.0f || temp_shift_2.y != 0.0f || temp_shift_2.z != 0.0f) {
				if (type==0)	correlation[j] += temp_shift_1 * temp_shift_2;
				if (type==1)	correlation[j] += (temp_shift_1 - temp_shift_2)*(temp_shift_1 - temp_shift_2);
				ncorrelation[j] += 1.0f;
			}
			--ind2;
			if (ind2 < 0)
				ind2 += p;
		}
	} else {
		int ind2 = ind1 - dm;
		for (int j = dm; j < p; ++j) {
			if (ind2 < 0)
				ind2 += p;
			temp_shift_2 = shift[ind2];
			if (temp_shift_2.x != 0.0f || temp_shift_2.y != 0.0f || temp_shift_2.z != 0.0f) {
				if (type==0)	correlation[j] += temp_shift_1 * temp_shift_2;
				if (type==1)	correlation[j] += (temp_shift_1 - temp_shift_2)*(temp_shift_1 - temp_shift_2);
				ncorrelation[j] += 1.0f;
			}
			--ind2;
		}
	}
	insertindex[k]++;
	if (insertindex[k] == p)
		insertindex[k] = 0;
}

__global__ void update_correlator(corr_device gpu_corr, int n, int type){
	int i = blockIdx.x * blockDim.x + threadIdx.x; //Chain index
	if (i >= dn_cha_per_call)
		return;
	float4 stress;
	for (int j=0; j<n; j++){
		stress = tex2D(t_corr, i, j);
		if (stress.w != -1.0f){
			corr_add(gpu_corr, stress, 0, i, type); //add new value to the correlator
		}
	}
}

__global__ __launch_bounds__(tpb_chain_kernel) void flow_stress(corr_device gpu_corr, int n, float4* stress_average, int nc){
	int i = blockIdx.x * blockDim.x + threadIdx.x; //Chain index
	if (i >= dn_cha_per_call)
		return;
	float4 stress;
	for (int j=0; j<n; j++){
		stress = tex2D(t_corr, i, j);
		if (stress.w != -1.0f){
			stress_average[(int)stress.w * nc + i].x = stress.x;
			stress_average[(int)stress.w * nc + i].y = stress.y;
			stress_average[(int)stress.w * nc + i].z = stress.z;
			//atomicAdd(&(stress_average[(int)stress.w].x),stress.x);
			//atomicAdd(&(stress_average[(int)stress.w].y),stress.y);
			//atomicAdd(&(stress_average[(int)stress.w].z),stress.z);
		}
	}
}

template<int type> __global__ __launch_bounds__(tpb_chain_kernel) void chain_kernel(
		scalar_chains* chain_heads, float *tdt, float *reach_flag, 
		float next_sync_time, int *d_offset, float4 *d_new_strent, 
		float *d_new_tau_CD, int *d_write_time, int correlator_type,
		int *rand_used, int *tau_CD_used_CD, int *tau_CD_used_SD, int stress_index,
		int* found_index, int* found_shift)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;//chain index

	if (i >= dn_cha_per_call)
		return;

	float4 sum_stress = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	surf2Dwrite(sum_stress, s_corr, sizeof(float4) * i, stress_index); //Write stress value to the stack

	surf1Dwrite(0.0f,s_ft,i*sizeof(float));
	if (reach_flag[i]!=0) {
		return;
	}

	if (((chain_heads[i].time >= next_sync_time) && (d_universal_time + next_sync_time <= d_write_time[i] * d_correlator_res)) || (chain_heads[i].stall_flag != 0)) {
		reach_flag[i] = 1;
		chain_heads[i].time-=next_sync_time;
		tdt[i] = 0.0f;
		for (int u=0; u<d_narms; u++){
			d_offset[i*d_narms+u] = offset_code(0xffff, +1);
		}
		return;
	}

	float olddt;
	if (type == 1) olddt = tdt[i];
	float4 new_strent = d_new_strent[i];

	//check for correlator
	if (d_universal_time + chain_heads[i].time > d_write_time[i] * d_correlator_res) { //TODO add d_correlator_time to gpu_chain_heads
		if (correlator_type == 0) {//stress calc
			int run_sum_ = 0;
			for (int arm_ = 0; arm_ < d_narms; arm_++) {
				int tz_ = chain_heads[i].Z[arm_];
				for (int j = 0; j < tz_; j++) {
					float4 QN1;
					if (fetch_new_strent(j + run_sum_, d_offset[i*d_narms + arm_]))
						QN1 = new_strent;
					else
						QN1 = tex2D(t_a_QN, make_offset(j + run_sum_, d_offset[i*d_narms + arm_]), i);
					
					sum_stress.x -= __fdividef(3.0f * QN1.x * QN1.y, QN1.w);
					sum_stress.y -= __fdividef(3.0f * QN1.y * QN1.z, QN1.w);
					sum_stress.z -= __fdividef(3.0f * QN1.x * QN1.z, QN1.w);
				}
				run_sum_ += d_z_max_arms[arm_];
			}
			sum_stress.w = 1.0f;
			//printf("\n%f\t%f\t%f", sum_stress.x, sum_stress.y, sum_stress.z);
			surf2Dwrite(sum_stress, s_corr, sizeof(float4) * i, stress_index); //Write stress value to the stack
		}

		//Update counter
		d_write_time[i]++;
	}
	//check again to stop if necessary
	if (d_universal_time + chain_heads[i].time > d_write_time[i] * d_correlator_res) {
		for (int u=0; u<d_narms; u++){
			d_offset[i*d_narms+u] = offset_code(0xffff, +1);
		}
		return;
	}

	float sumW; // sum of probabilities
	surf2Dread(&sumW, s_sum_W_sorted, sizeof(float)*(d_z_max-1), i);
//	printf("\n%f",sumW);
	//decide the timestep
	tdt[i] = __fdividef(1.0f, sumW);

	// error handling
	if (tdt[i] == 0.0f)
		chain_heads[i].stall_flag = 1;
	if (isnan(tdt[i]))
		chain_heads[i].stall_flag = 2;
	if (isinf(tdt[i]))
		chain_heads[i].stall_flag = 3;
	//update time
	chain_heads[i].time += tdt[i];
	//start picking the jump process
	rand_used[i]++;

	int j = found_index[i];
	int k = found_shift[i];
	int arm=0;
	int run_sum=0;
	for (arm=0; j>=run_sum+d_z_max_arms[arm]; arm++){
		run_sum+=d_z_max_arms[arm];
	}
	int jj = j-run_sum;
	int ii = i*d_narms+arm;

	//setup local variables
	int tz = chain_heads[i].Z[arm];
	uint oft = d_offset[ii];

	bool shfl = (jj < tz-1) && (tz>1);

	//for (int u=0; u<d_narms; u++){
	//	if (u!=arm)
	//		d_offset[i*d_narms+u] = offset_code(0xffff, +1);
	//}

	if (shfl) {//Shuffling
		float4 QN1 = tex2D(t_a_QN, make_offset(j, oft), i);
		if (fetch_new_strent(j, oft))
			QN1 = new_strent;
		float4 QN2 = tex2D(t_a_QN, make_offset(j + 1, oft), i);
		if (fetch_new_strent(j + 1, oft))
			QN2 = new_strent;
		if (type == 1){
			QN1 = kappa(QN1, olddt);
			QN2 = kappa(QN2, olddt);
		}

		if (k==1) {
			QN1.w = QN1.w + 1;
			QN2.w = QN2.w - 1;
			//printf("\nShuffling left, arm %i",arm);
		} else {
			QN1.w = QN1.w - 1;
			QN2.w = QN2.w + 1;
			//printf("\nShuffling right, arm %i", arm);
		}

		if (jj == 0) {//shuffling invoving branch-point
			float sumNinv = 0.0f;
			float4 temp_;
			int run_sum_ = 0;
			for (int arm_ = 0; arm_<d_narms; arm_++) {
				int tz_ = chain_heads[i].Z[arm_];
				//printf("\n%i\t%i", arm_,tz_);
				if ((tz_>1) && (arm_ != arm)) {//entangled arms
					if (fetch_new_strent(0 + run_sum_, d_offset[i*d_narms + arm_]))
						temp_ = new_strent;
					else
						temp_ = tex2D(t_a_QN, make_offset(0 + run_sum_, d_offset[i*d_narms + arm_]), i);
					//printf("\n%i\t%f\t%f\t%f\t%f", arm_, temp_.x, temp_.y, temp_.z, temp_.w);
					sumNinv += 1 / temp_.w;
					//printf("\n%i\t%f", arm_, 1 / temp_.w);
				}
				run_sum_ += d_z_max_arms[arm_];
			}
			sumNinv += 1 / QN1.w;
			float4 deltaQ = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
			if (k == 1) {//shift left
				deltaQ = QN1 / (-QN1.w*(QN1.w - 1)*sumNinv);
			}
			else {
				deltaQ = QN1 / (QN1.w*(QN1.w + 1)*sumNinv);
			}
			deltaQ.w = 0.0f;
			//printf("\nNear branch %i. Sum of inverses %f. Shifting origin at %f\t%f\t%f", arm, sumNinv, deltaQ.x, deltaQ.y, deltaQ.z);
			run_sum_ = 0;
			for (int arm_ = 0; arm_ < d_narms; arm_++) {
				int tz_ = chain_heads[i].Z[arm_];
				if ((tz_ > 1) && (arm_ != arm)) {//entangled arms
					if (fetch_new_strent(0 + run_sum_, d_offset[i*d_narms + arm_]))
						temp_ = new_strent;
					else
						temp_ = tex2D(t_a_QN, make_offset(0 + run_sum_, d_offset[i*d_narms + arm_]), i);
					//printf("\n%f\t%f\t%f", temp_.x, temp_.y, temp_.z);
					
					temp_ = temp_ - deltaQ;
					
					surf2Dwrite(temp_, s_b_QN, 16 * run_sum_, i);
				}
				run_sum_ += d_z_max_arms[arm_];
			}
			QN1 = QN1 - deltaQ;
		}
		surf2Dwrite(QN1, s_b_QN, 16 * j, i);
		surf2Dwrite(QN2, s_b_QN, 16 * (j + 1), i);
		d_offset[ii] = offset_code(0xffff, +1);
	} else {//Process at the end of chain
		if (k==0) {//  Creation by sliding dynamics
			if (tz == d_z_max_arms[arm])
				return;	// possible detail balance issue
			//printf("\nCreation by SD");
			float4 temp = tex2D(t_taucd_gauss_rand_SD, tau_CD_used_SD[i], i);
			tau_CD_used_SD[i]++;
			chain_heads[i].Z[arm]++;
			d_new_tau_CD[i] = d_universal_time + chain_heads[i].time;

			float4 QN_tail;
			if (fetch_new_strent(tz - 1 + run_sum, oft))
				QN_tail = new_strent;
			else
				QN_tail = tex2D(t_a_QN, make_offset((tz - 1)+ run_sum, oft), i);

			temp.w = QN_tail.w - 1.0f;
			float sigma = __fsqrt_rn(__fdividef(temp.w, 3.0f));
			temp.x *= sigma;
			temp.y *= sigma;
			temp.z *= sigma;
			if (tz==1){
				float4 temp_;
				int run_sum_ = 0;
				bool unent=true; //if all arms are unentangled -> set temp as (0,0,0)
				float sumNinv = 0.0f;
				for (int arm_=0; arm_<d_narms; arm_++){
					int tz_ = chain_heads[i].Z[arm_];
					if ((tz_>1) && (arm_!=arm)){//entangled arms
						unent=false;

						if (fetch_new_strent(0 + run_sum_, d_offset[i*d_narms + arm_]))
							temp_ = new_strent;
						else
							temp_ = tex2D(t_a_QN, make_offset(0 + run_sum_, d_offset[i*d_narms + arm_]), i);

						sumNinv += 1/temp_.w;
					}
					run_sum_ += d_z_max_arms[arm_];
				}

				float4 deltaQ = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
				deltaQ = temp / (-temp.w*sumNinv);
				deltaQ.w = 0.0f;


				run_sum_ = 0;
				for (int arm_ = 0; arm_ < d_narms; arm_++) {
					int tz_ = chain_heads[i].Z[arm_];
					if ((tz_ > 1) && (arm_ != arm)) {//entangled arms
						if (fetch_new_strent(0 + run_sum_, d_offset[i*d_narms + arm_]))
							temp_ = new_strent;
						else
							temp_ = tex2D(t_a_QN, make_offset(0 + run_sum_, d_offset[i*d_narms + arm_]), i);

						temp_ = temp_ + deltaQ;
						//printf("\n writing... %f\t%f\t%f\t%f\t",temp_.x,temp_.y,temp_.z,temp_.w);
						surf2Dwrite(temp_, s_b_QN, 16 * run_sum_, i);
					}
					run_sum_ += d_z_max_arms[arm_];
				}

				if(unent){
					temp.x = 0.0f;
					temp.y = 0.0f;
					temp.z = 0.0f;
				}
				//printf("\nCreating entanglement on chain %i, on unentangled arm %i, with N=%f",i,arm,temp.w);
			}
			surf2Dwrite(make_float4(0.0f, 0.0f, 0.0f, 1.0f), s_b_QN, 16 * (tz - 1 + run_sum), i);
			d_offset[ii] = offset_code(tz - 1 + run_sum, -1);
			d_new_strent[i] = temp;
		} else {
		// Destruction by sliding dynamics
			//printf("\nDestruction by SD");
			chain_heads[i].Z[arm]--;

			float4 QNtailp;
			if (fetch_new_strent(tz - 2 + run_sum, oft))
				QNtailp = new_strent;
			else
				QNtailp = tex2D(t_a_QN, make_offset(tz - 2 + run_sum, oft), i);

			surf2Dwrite(make_float4(0.0f, 0.0f, 0.0f, QNtailp.w + 1.0f), s_b_QN, 16 * (tz - 2 + run_sum), i);
			d_offset[ii] = offset_code(tz + run_sum, +1);
			if(chain_heads[i].Z[arm]==1){
				float4 temp_;
				int run_sum_ = 0;
				bool unent = true; //if all arms are unentangled -> set temp as (0,0,0)
				float sumNinv = 0.0f;
				for (int arm_ = 0; arm_<d_narms; arm_++) {
					int tz_ = chain_heads[i].Z[arm_];
					if ((tz_>1) && (arm_ != arm)) {//entangled arms
						unent = false;

						if (fetch_new_strent(0 + run_sum_, d_offset[i*d_narms + arm_]))
							temp_ = new_strent;
						else
							temp_ = tex2D(t_a_QN, make_offset(0 + run_sum_, d_offset[i*d_narms + arm_]), i);

						sumNinv += 1 / temp_.w;
					}
					run_sum_ += d_z_max_arms[arm_];
				}

				float4 deltaQ = QNtailp / (QNtailp.w*sumNinv);
				deltaQ.w = 0.0f;

				run_sum_ = 0;
				for (int arm_ = 0; arm_ < d_narms; arm_++) {
					int tz_ = chain_heads[i].Z[arm_];
					if ((tz_ > 1) && (arm_ != arm)) {//entangled arms
						if (fetch_new_strent(0 + run_sum_, d_offset[i*d_narms + arm_]))
							temp_ = new_strent;
						else
							temp_ = tex2D(t_a_QN, make_offset(0 + run_sum_, d_offset[i*d_narms + arm_]), i);

						temp_ = temp_ + deltaQ;

						surf2Dwrite(temp_, s_b_QN, 16 * run_sum_, i);
					}
					run_sum_ += d_z_max_arms[arm_];
				}

				//printf("\nDestroying entanglement on chain %i. Arm %i is now unentangled  with with N=%f",i,arm,QNtailp.w + 1.0f);
				//printf("\n deltaQ=%f\t%f\t%f",deltaQ.x, deltaQ.y, deltaQ.z);
			}
			float cr_time;
			if (fetch_new_strent(tz - 2 + run_sum, oft)){
				cr_time = d_new_tau_CD[i];
			}
			else{
				cr_time = tex2D(t_a_tCD, make_offset(tz - 2 + run_sum, oft), i);
			}
			if (cr_time!=0){
				surf1Dwrite(log10f(d_universal_time + chain_heads[i].time - cr_time)+10,s_ft,i*sizeof(float));
			}
//			printf("\nDestruction by SD %f\t%f",d_universal_time + chain_heads[i].time - cr_time,cr_time);
		}
	}
	for (int u = 0; u<d_narms; u++) {
		if (u != arm)
			d_offset[i*d_narms + u] = offset_code(0xffff, +1);
	}
	return;
}

__global__ __launch_bounds__(tpb_chain_kernel) //stress calculation
void stress_calc(scalar_chains* chain_heads, float *tdt, int *d_offset,
		float4 *d_new_strent, float4* QN, int size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= dn_cha_per_call)
		return;
	int tz = chain_heads[i].Z[1];
	uint oft = d_offset[i];
	float olddt = tdt[i];
	float4 new_strent = d_new_strent[i];

	float4 sum_stress = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	float4 sum_stress2 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	for (int j = 0; j < tz; j++) {
		float4 QN1 = tex2D(t_a_QN, make_offset(j, oft), i);
		if (fetch_new_strent(j, oft))
			QN1 = new_strent;
		QN1 = kappa(QN1, olddt);
		sum_stress.x -= __fdividef(3.0f * QN1.x * QN1.x, QN1.w);
		sum_stress.y -= __fdividef(3.0f * QN1.y * QN1.y, QN1.w);
		sum_stress.z -= __fdividef(3.0f * QN1.z * QN1.z, QN1.w);
		sum_stress.w -= __fdividef(3.0f * QN1.x * QN1.y, QN1.w);
		sum_stress2.x -= __fdividef(3.0f * QN1.y * QN1.z, QN1.w);
		sum_stress2.y -= __fdividef(3.0f * QN1.x * QN1.z, QN1.w);
		sum_stress2.z += __fsqrt_rn(QN1.x * QN1.x + QN1.y * QN1.y + QN1.z * QN1.z);
	}
	sum_stress2.w = float(tz);
	surf1Dwrite(sum_stress, s_stress, 32 * i);
	surf1Dwrite(sum_stress2, s_stress, 32 * i + 16);

}

#endif
