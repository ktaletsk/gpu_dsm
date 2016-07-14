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

#include "textures_surfaces.h"
#include "chain.h"

//d means device variables
__constant__ float d_universal_time;
__constant__ float dBe;
__constant__ int dnk;
__constant__ int d_z_max; //limits actual array size. might come useful for large beta values and for polydisperse systems
__constant__ int dn_cha_per_call; //number of chains in this call. cannot be bigger than chains_per_call
__constant__ float d_kappa_xx, d_kappa_xy, d_kappa_xz, d_kappa_yx, d_kappa_yy,d_kappa_yz, d_kappa_zx, d_kappa_zy, d_kappa_zz;

//CD constants
__constant__ int d_CD_flag;
__constant__ float d_CD_create_prefact;
__constant__ int d_correlator_res;
//Next variables actually declared in ensemble_block.h
//    float *d_dt; // time step size from prevous time step. used for appling deformation
//    float *reach_flag;// flag that chain evolution reached required time
//copied to host each times step

// delayed dynamics --- how does it work:
// There are entanglement parallel portion of the code and chain parallel portion.
// The entanglement parallel part applies flow deformation and calculates jump process probabilities.
// The chain parallel part picks one of the jump processes, generates a new orientation vector and a tau_CD if needed.
// It applies only some simpliest chain conformation changes(SD shifting).
// The Information about complex chain conformation changes(entanglement creation/destruction) is stored in temp arrays d_offset, d_new_strent,d_new_tau_CD.
// Complex changes are applied next time step by entanglement parallel part.

//    int *d_offset;//coded shifting parameters
//    float4 *d_new_strent;//new strent which shoud be inserted in the middle of the chain//TODO two new_strents will allow do all the updates at once
//    float *d_new_tau_CD;//new life time

//float4 math operators
inline __device__ void operator+=(float4 &a, float4 b) {
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
	a.w += b.w;
}
inline __device__ float4 operator/(float4 a, float b) {
	return make_float4(a.x / b, a.y / b, a.z / b, a.w / b);
}
inline __device__ float4 operator*(float4 a, float4 b) {
	return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}
inline __device__ float4 operator+(float4 a, float4 b) {
	return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
inline __device__ float4 operator-(float4 a, float4 b) {
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
		scalar_chains* gpu_chain_heads, float *tdt, int *d_offset, float4 *d_new_strent, 
		float *d_new_tau_CD)
{
	//Calculate kernel index
	int i = blockIdx.x * blockDim.x + threadIdx.x;//strent index
	int j = blockIdx.y * blockDim.y + threadIdx.y;//chain index

	//Check if kernel index is outside boundaries
	if ((j >= dn_cha_per_call) || (i >= d_z_max))
		return;

	int tz = gpu_chain_heads[j].Z; //Current chain size
	if (i >= tz) //Check if entaglement index is over chain size
		return;

	//When new entaglements are created we need to shift index +1(destruction, skip one strent), 0(nothing happens) or -1(new strent created before)
	int oft = d_offset[j]; //Offset for current chain

	//fetch
	float4 QN;
	if (fetch_new_strent(i, oft))
		QN = d_new_strent[j]; //second check if strent created last time step should go here
	else
		QN = tex2D(t_a_QN, make_offset(i, oft), j); // all access to strents is done through two operations: first texture fetch
	float tcd;
	if (d_CD_flag) { //If constraint dynamics is enabled
		if (fetch_new_strent(i, oft))
			tcd = d_new_tau_CD[j];
		else
			tcd = tex2D(t_a_tCD, make_offset(i, oft), j);
	} else
		tcd = 0;


	float dt;
	if (type==1){//transform
		dt = tdt[j];
		QN = kappa(QN, dt);
	}

	//fetch next strent
	if (i < tz - 1) {
		float2 wsh = make_float2(0.0f, 0.0f);
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

			float sig1 = __fdividef(0.75f, QN.w * (QN.w + 1)); //fdivedf - fast divide float
			float sig2 = __fdividef(0.75f, QN2.w * (QN2.w - 1));
			float prefact1 = (Q == 0.0f) ? 1.0f : __fdividef(QN.w, (QN.w + 1));
			float prefact2 =
					(Q2 == 0.0f) ? 1.0f : __fdividef(QN2.w, (QN2.w - 1));
			float f1 = (Q == 0.0f) ? 2.0f * QN.w + 0.5f : QN.w;
			float f2 = (Q2 == 0.0f) ? 2.0f * QN2.w - 0.5f : QN2.w;
			float friction = __fdividef(2.0f, f1 + f2);
			wsh.x = friction * __powf(prefact1 * prefact2, 0.75f)* __expf(Q * sig1 - Q2 * sig2);
		}
		if (QN.w > 1.0f) {//N=1 mean that shift is not possible, also ot will lead to dividing on zero error

			float sig1 = __fdividef(0.75f, QN.w * (QN.w - 1.0f));
			float sig2 = __fdividef(0.75f, QN2.w * (QN2.w + 1.0f));
			float prefact1 = (Q == 0.0f) ? 1.0f : __fdividef(QN.w, (QN.w - 1.0f));
			float prefact2 = (Q2 == 0.0f) ? 1.0f : __fdividef(QN2.w, (QN2.w + 1.0f));
			float f1 = (Q == 0.0f) ? 2.0f * QN.w - 0.5f : QN.w;
			float f2 = (Q2 == 0.0f) ? 2.0f * QN2.w + 0.5f : QN2.w;
			float friction = __fdividef(2.0f, f1 + f2);
			wsh.y = friction * __powf(prefact1 * prefact2, 0.75f) * __expf(-Q * sig1 + Q2 * sig2);
		}
		surf2Dwrite(wsh.x + wsh.y+ d_CD_flag* (tcd + d_CD_create_prefact * (QN.w - 1.0f)),s_sum_W, 4 * i, j);
		//probability of Kuhn step shitt + probability of entanglement destruction by CD
		// + probability of entanglement creation by CD
	}
	//write updated chain conformation
	surf2Dwrite(QN, s_b_QN, 16 * i, j);
	surf2Dwrite(tcd, s_b_tCD, 4 * i, j);
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

__global__ __launch_bounds__(tpb_chain_kernel) void update_correlator(corr_device gpu_corr, int n, int type){
	int i = blockIdx.x * blockDim.x + threadIdx.x; //Chain index
	if (i >= dn_cha_per_call)
		return;
	float4 stress;
	for (int j=0; j<n; j++){
		stress = tex2D(t_corr, i, j);
		if (stress.w == 1.0f){
			corr_add(gpu_corr, stress, 0, i, type); //add new value to the correlator
		}
	}
}

template<int type> __global__ __launch_bounds__(tpb_chain_kernel) void chain_kernel(
		scalar_chains* gpu_chain_heads, float *tdt, float *reach_flag, 
		float next_sync_time, int *d_offset, float4 *d_new_strent, 
		float *d_new_tau_CD, int *d_correlator_time, int correlator_type,
		int *rand_used, int *tau_CD_used_CD, int *tau_CD_used_SD, int stress_index)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= dn_cha_per_call)
		return;

	//setup local variables
	int tz = gpu_chain_heads[i].Z;
	uint oft = d_offset[i];
	d_offset[i] = offset_code(0xffff, +1);

	float4 sum_stress;
	if (type == 0){
		sum_stress = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
		surf2Dwrite(sum_stress, s_corr, sizeof(float4) * i, stress_index); //Write stress value to the stack
	}	

	if (reach_flag[i]!=0) {
		return;
	}

	bool stop_cond;
	if (type==0){
		stop_cond = (gpu_chain_heads[i].time >= next_sync_time) && (d_universal_time + next_sync_time <= d_correlator_time[i] * d_correlator_res);
	}
	else if (type==1){
		stop_cond = (gpu_chain_heads[i].time >= next_sync_time);
	}
	if (stop_cond || (gpu_chain_heads[i].stall_flag != 0)) {
		reach_flag[i] = 1;
		gpu_chain_heads[i].time-=next_sync_time;
		tdt[i] = 0.0f;
		return;
	}
	
	float olddt;
	if (type == 1) olddt = tdt[i];
	float4 new_strent = d_new_strent[i];
	float new_tCD = d_new_tau_CD[i];

	float4 R1;
	if (correlator_type==0){
		R1 = tex1D(t_a_R1,i);
		surf1Dwrite(R1,s_b_R1,i*sizeof(float4));
		
		//check for correlator
		if (d_universal_time + gpu_chain_heads[i].time > d_correlator_time[i] * d_correlator_res) { //TODO add d_correlator_time to gpu_chain_heads
			if(correlator_type==0){//stress calc
				for (int j = 0; j < tz; j++) {
					float4 QN1;
					if (fetch_new_strent(j, oft))
						QN1 = new_strent;
					else
						QN1 = tex2D(t_a_QN, make_offset(j, oft), i);
					sum_stress.x -= __fdividef(3.0f * QN1.x * QN1.y, QN1.w);
					sum_stress.y -= __fdividef(3.0f * QN1.y * QN1.z, QN1.w);
					sum_stress.z -= __fdividef(3.0f * QN1.x * QN1.z, QN1.w);
				}
				sum_stress.w = 1.0f;
				surf2Dwrite(sum_stress, s_corr, sizeof(float4) * i, stress_index); //Write stress value to the stack
			}

			if (correlator_type==1){//center of mass calc
				float4 temp_sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
				float4 center_mass = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
				float4 prev_q = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
				for (int j = 0; j < tz; j++) {
					float4 QN1;
					if (fetch_new_strent(j, oft))
						QN1 = new_strent;
					else
						QN1 = tex2D(t_a_QN, make_offset(j, oft), i);

					float4 term = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
					temp_sum	+=	prev_q;
					term		+=	temp_sum;
					term.x 		+=	__fdividef(QN1.x, 2);
					term.y		+=	__fdividef(QN1.y, 2);
					term.z		+=	__fdividef(QN1.z, 2);
					center_mass.x += term.x * QN1.w / dnk;
					center_mass.y += term.y * QN1.w / dnk;
					center_mass.z += term.z * QN1.w / dnk;

					prev_q = QN1;
				}
				center_mass+=R1;
				center_mass.w = 1.0f;
				surf2Dwrite(center_mass, s_corr, sizeof(float4) * i, stress_index); //Write center of mass to the stack
			}

			//Update counter
			d_correlator_time[i]++;
			if (d_universal_time + gpu_chain_heads[i].time > d_correlator_time[i] * d_correlator_res) {
				return;
				//do nothing until next step
			}
		}
	}

	// sum W_SD_shifts
	float sum_wshpm = 0.0f;
	float tsumw;
	for (int j = 0; j < tz - 1; j++) {
		surf2Dread(&tsumw, s_sum_W, 4 * j, i);
		sum_wshpm += tsumw;
	}

	// W_SD_c/d calc
	float W_SD_c_1 = 0.0f, W_SD_d_1 = 0.0f;
	float W_SD_c_z = 0.0f, W_SD_d_z = 0.0f;

	//declare vars to reuse later
	float4 QNheadn, QNtailp;

	// first strent
	float4 QNhead;
	if (fetch_new_strent(0, oft))
		QNhead = new_strent;
	else
		QNhead = tex2D(t_a_QN, make_offset(0, oft), i);

	//last strent
	float4 QNtail;
	if (fetch_new_strent(tz - 1, oft))
		QNtail = new_strent;
	else
		QNtail = tex2D(t_a_QN, make_offset(tz - 1, oft), i);

	float W_CD_c_z = (double)d_CD_flag * (double)d_CD_create_prefact * (QNtail.w - 1.0f); //Create CD on the last strand

	if (tz == 1) {
		W_SD_c_1 = __fdividef(1.0f, (dBe * dnk));
		W_SD_c_z = W_SD_c_1;
	} else {
		if (QNhead.w == 1.0f) {
			//destruction
			if (fetch_new_strent(1, oft))
				QNheadn = new_strent;
			else
				QNheadn = tex2D(t_a_QN, make_offset(1, oft), i);
			float f2 = (tz == 2) ? QNheadn.w + 0.25f : 0.5f * QNheadn.w;
			W_SD_d_1 = __fdividef(1.0f, 0.75f + f2);
		} else {
			//creation
			W_SD_c_1 = __fdividef(2.0f, dBe * (QNhead.w + 0.5f));
		}

		if (QNtail.w == 1.0f) {
			//destruction
			if (fetch_new_strent(tz - 2, oft))
				QNtailp = new_strent;
			else
				QNtailp = tex2D(t_a_QN, make_offset(tz - 2, oft), i);
			float f1 = (tz == 2) ? QNtailp.w + 0.25f : 0.5f * QNtailp.w;
			W_SD_d_z = __fdividef(1.0f, f1 + 0.75f);
		} else {
			//creation
			W_SD_c_z = __fdividef(2.0f, dBe * (QNtail.w + 0.5f));
		}
	}
	//sum all the probabilities
	float sumW = sum_wshpm + W_SD_c_1 + W_SD_c_z + W_SD_d_1 + W_SD_d_z + W_CD_c_z;
	tdt[i] = __fdividef(1.0f, sumW);
	// error handling
	if (tdt[i] == 0.0f)
		gpu_chain_heads[i].stall_flag = 1;
	if (isnan(tdt[i]))
		gpu_chain_heads[i].stall_flag = 2;
	if (isinf(tdt[i]))
		gpu_chain_heads[i].stall_flag = 3;
	//update time
	gpu_chain_heads[i].time += tdt[i];
	//start picking the jump process
	float pr = (sumW) * tex2D(t_uniformrand, rand_used[i], i);
	rand_used[i]++;
	int j = 0;
	float tpr = 0.0f;
	if (tz != 1)
		surf2Dread(&tpr, s_sum_W, 4 * j, i);
	// picking where(which strent) jump process will happen
	// excluding SD creation destruction
	// perhaps one of the most time consuming parts of the code
	while ((pr >= tpr) && (j < tz - 2)) {
		pr -= tpr;
		j++;
		surf2Dread(&tpr, s_sum_W, 4 * j, i);
	}

	if (pr < tpr) {
		// ok we picked some strent j
		// now we need to decide which(SD shift or CDd CDc) jump process will happen
		// TODO check if the order have an effect on performance

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

		// 1. CDd (destruction by constraint dynamics)
		float wcdd;
		if (d_CD_flag) {
			wcdd = tex2D(t_a_tCD, make_offset(j, oft), i);    //////////////////
			if (fetch_new_strent(j, oft))
				wcdd = new_tCD;
		} else
			wcdd = 0;
		if (pr < wcdd) {
			float4 temp = make_float4(QN1.x + QN2.x, QN1.y + QN2.y, QN1.z + QN2.z, QN1.w + QN2.w);
			if ((j == tz - 2) || (j == 0)) {
				temp = make_float4(0.0f, 0.0f, 0.0f, QN1.w + QN2.w);
			}
			surf2Dwrite(temp, s_b_QN, 16 * (j + 1), i);
			d_offset[i] = offset_code(j, +1);
			gpu_chain_heads[i].Z--;

			if (type==0){
				if(correlator_type==1 && j == 0){
					//update position of the first entanglement in the fixed frame
					float4 delta_R1 = make_float4(0.0f,0.0f,0.0f,0.0f);
					delta_R1.x = QN2.x;
					delta_R1.y = QN2.y;
					delta_R1.z = QN2.z;
					R1+=delta_R1;
					surf1Dwrite(R1,s_b_R1,i*sizeof(float4));
				}
			}

			return;
		} else {
			pr -= wcdd;
		}

		// 2. SD shift
		// SD shift probabilitiess are not saved from entanglement parallel part
		// so we need to recalculate them
		float2 twsh = make_float2(0.0f, 0.0f);
		float Q = QN1.x * QN1.x + QN1.y * QN1.y + QN1.z * QN1.z;
		float Q2 = QN2.x * QN2.x + QN2.y * QN2.y + QN2.z * QN2.z;

		if (QN2.w > 1.0f) {	//N=1 mean that shift is not possible, also ot will lead to dividing on zero error
			//float prefact=__powf( __fdividef(QN1.w*QN2.w,(QN1.w+1)*(QN2.w-1)),0.75f);
			//TODO replace powf with sqrt(x*x*x)

			float sig1 = __fdividef(0.75f, QN1.w * (QN1.w + 1));
			float sig2 = __fdividef(0.75f, QN2.w * (QN2.w - 1));
			float prefact1 = (Q == 0.0f) ? 1.0f : __fdividef(QN1.w, (QN1.w + 1));
			float prefact2 = (Q2 == 0.0f) ? 1.0f : __fdividef(QN2.w, (QN2.w - 1));
			float f1 = (Q == 0.0f) ? 2.0f * QN1.w + 0.5f : QN1.w;
			float f2 = (Q2 == 0.0f) ? 2.0f * QN2.w - 0.5f : QN2.w;
			float friction = __fdividef(2.0f, f1 + f2);
			twsh.x = friction * __powf(prefact1 * prefact2, 0.75f) * __expf(Q * sig1 - Q2 * sig2);
		}
		if (QN1.w > 1.0f) {	//N=1 mean that shift is not possible, also ot will lead to dividing on zero error

			float sig1 = __fdividef(0.75f, QN1.w * (QN1.w - 1.0f));
			float sig2 = __fdividef(0.75f, QN2.w * (QN2.w + 1.0f));
			float prefact1 = (Q == 0.0f) ? 1.0f : __fdividef(QN1.w, (QN1.w - 1.0f));
			float prefact2 = (Q2 == 0.0f) ? 1.0f : __fdividef(QN2.w, (QN2.w + 1.0f));
			float f1 = (Q == 0.0f) ? 2.0f * QN1.w - 0.5f : QN1.w;
			float f2 = (Q2 == 0.0f) ? 2.0f * QN2.w + 0.5f : QN2.w;
			float friction = __fdividef(2.0f, f1 + f2);
			twsh.y = friction * __powf(prefact1 * prefact2, 0.75f) * __expf(-Q * sig1 + Q2 * sig2);
		}

		if (pr < twsh.x + twsh.y) {

			if (pr < twsh.x) {
				QN1.w = QN1.w + 1;
				QN2.w = QN2.w - 1;
			} else {
				QN1.w = QN1.w - 1;
				QN2.w = QN2.w + 1;
			}
			surf2Dwrite(QN1, s_b_QN, 16 * j, i);
			surf2Dwrite(QN2, s_b_QN, 16 * (j + 1), i);
			return;
		} else {
			pr -= twsh.x + twsh.y;
		}

		// 3. CDc (creation by constraint dynamics in the middle)

		float wcdc = d_CD_create_prefact * (QN1.w - 1.0f);
		if (pr < wcdc) {
			if (tz == d_z_max)
				return;		// possible detail balance issue
			float4 temp = tex2D(t_taucd_gauss_rand_CD, tau_CD_used_CD[i], i);
			tau_CD_used_CD[i]++;
			gpu_chain_heads[i].Z++;
			d_new_tau_CD[i] = temp.w;//__fdividef(1.0f,d_tau_d);
			float newn = floorf(0.5f + __fdividef(pr * (QN1.w - 2.0f), wcdc)) + 1.0f;
			if (j == 0) {
				temp.w = QN1.w - newn;
				float sigma = __fsqrt_rn(__fdividef(temp.w, 3.0f));
				temp.x *= sigma;
				temp.y *= sigma;
				temp.z *= sigma;
				surf2Dwrite(temp, s_b_QN, 16 * 0, i);
				d_offset[i] = offset_code(0, -1);
				d_new_strent[i] = make_float4(0.0f, 0.0f, 0.0f, newn);
				if (type == 0){
					if (correlator_type==1){
					//update position of the first entanglement in the fixed frame
					float4 delta_R1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
					delta_R1.x = -temp.x;
					delta_R1.y = -temp.y;
					delta_R1.z = -temp.z;
					R1 += delta_R1;
					surf1Dwrite(R1,s_b_R1,i*sizeof(float4));
					}
				}
				return;
			}

			temp.w = newn;
			float sigma = __fsqrt_rn(__fdividef(newn * (QN1.w - newn), 3.0f * QN1.w));
			float ration = __fdividef(newn, QN1.w);
			temp.x *= sigma;
			temp.y *= sigma;
			temp.z *= sigma;
			temp.x += QN1.x * ration;
			temp.y += QN1.y * ration;
			temp.z += QN1.z * ration;
			surf2Dwrite(make_float4(QN1.x - temp.x, QN1.y - temp.y, QN1.z - temp.z, QN1.w - newn), s_b_QN, 16 * j, i);
			d_offset[i] = offset_code(j, -1);
			d_new_strent[i] = temp;
			return;
		} else {
			pr -= wcdc;
		}
	} else {
		pr -= tpr;
	}

	//None of the processes in the middle of the chain happened
	//Now check processes on the left end

	// 4. w_CD_c_z (creation by constraint dynamics on the left end)
	if (pr < W_CD_c_z) {
		if (tz == d_z_max) return;	// possible detail balance issue
		float4 temp = tex2D(t_taucd_gauss_rand_CD, tau_CD_used_CD[i], i);
		tau_CD_used_CD[i]++;
		gpu_chain_heads[i].Z++;
		d_new_tau_CD[i] = temp.w;	//__fdividef(1.0f,d_tau_d);

		float newn = floorf(0.5f + __fdividef(pr * (QNtail.w - 2.0f), W_CD_c_z)) + 1.0f;

		temp.w = newn;
		float sigma = (tz == 1) ? 0.0f : __fsqrt_rn(__fdividef(temp.w, 3.0f));
		temp.x *= sigma;
		temp.y *= sigma;
		temp.z *= sigma;
		surf2Dwrite(make_float4(0.0f, 0.0f, 0.0f, QNtail.w - newn), s_b_QN, 16 * (tz - 1), i);
		d_offset[i] = offset_code(tz - 1, -1);
		d_new_strent[i] = temp;
		return;
	} else {
		pr -= W_CD_c_z;
	}

	// 5. w_SD_c (creation by sliding dynamics)

	if (pr < W_SD_c_1 + W_SD_c_z) {
		if (tz == d_z_max)
			return;	// possible detail balance issue
		float4 temp = tex2D(t_taucd_gauss_rand_SD, tau_CD_used_SD[i], i);
		tau_CD_used_SD[i]++;
		gpu_chain_heads[i].Z++;
		d_new_tau_CD[i] = temp.w;

		if (pr < W_SD_c_1) {//creation on the left end
			temp.w = QNhead.w - 1.0f;
			float sigma = (tz == 1) ? 0.0f : __fsqrt_rn(__fdividef(temp.w, 3.0f));
			temp.x *= sigma;
			temp.y *= sigma;
			temp.z *= sigma;
			surf2Dwrite(temp, s_b_QN, 16 * 0, i);//TODO maybe deformation should be applied here
			if (type==0){
				if (correlator_type==1){
					//update position of the first entanglement in the fixed frame
					float4 delta_R1 = make_float4(0.0f,0.0f,0.0f,0.0f);
					delta_R1.x = - temp.x;
					delta_R1.y = - temp.y;
					delta_R1.z = - temp.z;
					R1+=delta_R1;
					surf1Dwrite(R1,s_b_R1,i*sizeof(float4));
				}
			}
			d_offset[i] = offset_code(0, -1);
			d_new_strent[i] = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
		} else {
			temp.w = QNtail.w - 1.0f;
			float sigma = (tz == 1) ? 0.0f : __fsqrt_rn(__fdividef(temp.w, 3.0f));
			temp.x *= sigma;
			temp.y *= sigma;
			temp.z *= sigma;
			surf2Dwrite(make_float4(0.0f, 0.0f, 0.0f, 1.0f), s_b_QN, 16 * (tz - 1), i);//TODO maybe deformation should be applied here
			d_offset[i] = offset_code(tz - 1, -1);
			d_new_strent[i] = temp;
		}

		return;
	} else {
		pr -= W_SD_c_1 + W_SD_c_z;

	}

	// 6. Destruction by sliding dynamics
	if (pr < W_SD_d_1 + W_SD_d_z) {	//to delete entanglement
	// update cell and neigbours
	//clear W_sd
	//form a list of free cell
		gpu_chain_heads[i].Z--;

		if (pr < W_SD_d_1) {//Destruction of the 1st entanglement
			surf2Dwrite(make_float4(0.0f, 0.0f, 0.0f, QNheadn.w + 1.0f), s_b_QN, 16 * 1, i);
			d_offset[i] = offset_code(0, +1);

			if(type==0){
				if (correlator_type==1){
					//update position of the first entanglement in the fixed frame
					float4 delta_R1 = tex2D(t_a_QN, make_offset(1, oft), i);
					if (fetch_new_strent(1, oft))
						delta_R1 = new_strent;
					R1.w=0.0f;
					R1+=delta_R1;
					surf1Dwrite(R1,s_b_R1,i*sizeof(float4));
				}
			}
		} else {
			surf2Dwrite(make_float4(0.0f, 0.0f, 0.0f, QNtailp.w + 1.0f), s_b_QN, 16 * (tz - 2), i);
			d_offset[i] = offset_code(tz, +1);
		}
		return;
	} else
		pr -= W_SD_d_1 + W_SD_d_z;
}

__global__ __launch_bounds__(tpb_chain_kernel) //stress calculation
void stress_calc(scalar_chains* gpu_chain_heads, float *tdt, int *d_offset,
		float4 *d_new_strent) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= dn_cha_per_call)
		return;
	int tz = gpu_chain_heads[i].Z;
	uint oft = d_offset[i];
	float olddt = tdt[i];
	float4 new_strent = d_new_strent[i];

	float4 sum_stress = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	float4 sum_stress2 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	float ree_x = 0.0, ree_y = 0.0, ree_z = 0.0;
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
		ree_x += QN1.x;
		ree_y += QN1.y;
		ree_z += QN1.z;
	}
	sum_stress2.w = float(tz); //__fsqrt_rn(ree_x*ree_x+ree_y*ree_y+ree_z*ree_z);
	surf1Dwrite(sum_stress, s_stress, 32 * i);
	surf1Dwrite(sum_stress2, s_stress, 32 * i + 16);

}

#endif
