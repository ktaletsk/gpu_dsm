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

//Equilibrium versions of kernels
//flow deformation turn off
//EQ_chain_CD_kernel fills s_correlator with of diagonal stress component

//      correlator constant
__constant__ int d_correlator_res;

//entanglement parallel part of the code
//2D kernel: i- entanglement index j - chain index
__global__ __launch_bounds__(tpb_strent_kernel*tpb_strent_kernel) void EQ_strent_kernel(chain_head* gpu_chain_heads, int *d_offset, float4 *d_new_strent,float *d_new_tau_CD) {
	//Calculate kernel index
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

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
	if (dCD_flag) {//If constraint dynamics is enabled
		if (fetch_new_strent(i, oft))
			tcd = d_new_tau_CD[j];
		else
			tcd = tex2D(t_a_tCD, make_offset(i, oft), j);
	} else
		tcd = 0;

	//write
	surf2Dwrite(QN, s_b_QN, 16 * i, j);
	surf2Dwrite(tcd, s_b_tCD, 4 * i, j);

	//fetch next strent (if strent is not the last)
	if (i < tz - 1) {
		float2 wsh = make_float2(0.0f, 0.0f); //Variable for shifting probability
		float4 QN2; //Q for next strent
		if (fetch_new_strent(i + 1, oft))
			QN2 = d_new_strent[j];
		else
			QN2 = tex2D(t_a_QN, make_offset(i + 1, oft), j);

		//w_shift probability calc
		float Q  = QN.x  * QN.x  + QN.y  * QN.y  + QN.z  * QN.z;
		float Q2 = QN2.x * QN2.x + QN2.y * QN2.y + QN2.z * QN2.z;

		//Calculating probability of 1 Kuhn step to jump from i strand to (i+1) strand
		if (QN2.w > 1.0f) { //N=1 mean that shift is not possible, also it will lead to dividing by zero
			//float prefact=__powf( __fdividef(QN.w*QN2.w,(QN.w+1)*(QN2.w-1)),0.75f);

			float sig1 = __fdividef(0.75f, QN.w * (QN.w + 1));
			float sig2 = __fdividef(0.75f, QN2.w * (QN2.w - 1));
			float prefact1 = (Q == 0.0f) ? 1.0f : __fdividef(QN.w, (QN.w + 1));
			float prefact2 = (Q2 == 0.0f) ? 1.0f : __fdividef(QN2.w, (QN2.w - 1));
			float f1 = (Q == 0.0f) ? 2.0f * QN.w + 0.5f : QN.w;
			float f2 = (Q2 == 0.0f) ? 2.0f * QN2.w - 0.5f : QN2.w;
			float friction = __fdividef(2.0f, f1 + f2);
			wsh.x = friction * __powf(prefact1 * prefact2, 0.75f)* __expf(Q * sig1 - Q2 * sig2);
		}
		if (QN.w > 1.0f) {//N=1 mean that shift is not possible, also it will lead to dividing by zero

			float sig1 = __fdividef(0.75f, QN.w * (QN.w - 1.0f));
			float sig2 = __fdividef(0.75f, QN2.w * (QN2.w + 1.0f));
			float prefact1 = (Q == 0.0f) ? 1.0f : __fdividef(QN.w, (QN.w - 1.0f));
			float prefact2 = (Q2 == 0.0f) ? 1.0f : __fdividef(QN2.w, (QN2.w + 1.0f));
			float f1 = (Q == 0.0f) ? 2.0f * QN.w - 0.5f : QN.w;
			float f2 = (Q2 == 0.0f) ? 2.0f * QN2.w + 0.5f : QN2.w;
			float friction = __fdividef(2.0f, f1 + f2);
			wsh.y = friction * __powf(prefact1 * prefact2, 0.75f)* __expf(-Q * sig1 + Q2 * sig2);
		}
// 	    surf2Dwrite(wsh.x,s_W_SD_pm,8*i,j);//TODO funny bug i have no idea but doesn't work other way
// 	    surf2Dwrite(wsh.y,s_W_SD_pm,8*i+4,j);//seems to work with float4 below
		surf2Dwrite(wsh.x + wsh.y + dCD_flag * (tcd + d_CD_create_prefact * (QN.w - 1.0f)), s_sum_W, 4 * i, j);
	}
}

__global__ __launch_bounds__(tpb_chain_kernel) void EQ_chain_kernel(chain_head* gpu_chain_heads, float *tdt, float *reach_flag, float next_sync_time, int *d_offset, float4 *d_new_strent, float *d_new_tau_CD, int *d_correlator_time, int *rand_used, int *tau_CD_used) {
	int i = blockIdx.x * blockDim.x + threadIdx.x; //Chain index

	if (i >= dn_cha_per_call)
		return;

	//setup local variables
	int tz = gpu_chain_heads[i].Z;
	uint oft = d_offset[i];
	d_offset[i] = offset_code(0xffff, +1);

	if (reach_flag[i]!=0) {
		return;
	}
	if (((gpu_chain_heads[i].time >= next_sync_time)&&
	     (d_universal_time+next_sync_time<=d_correlator_time[i] * d_correlator_res)) ||  (gpu_chain_heads[i].stall_flag != 0))
	   {
		reach_flag[i] = 1;
		gpu_chain_heads[i].time-=next_sync_time;
		tdt[i] = 0.0f;
		return;
	}
	float4 new_strent = d_new_strent[i];
	float new_tCD = d_new_tau_CD[i];

	//check for correlator
	if (d_universal_time+gpu_chain_heads[i].time > d_correlator_time[i] * d_correlator_res) {//TODO add d_correlator_time to gpu_chain_heads
		float4 sum_stress = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
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
		surf2Dwrite(sum_stress, s_correlator, 16 * d_correlator_time[i], i);
		d_correlator_time[i]++;
		if (d_universal_time+gpu_chain_heads[i].time > d_correlator_time[i] * d_correlator_res){
		  return;
		  //do nothing until next step
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

	float W_CD_c_z = dCD_flag * d_CD_create_prefact * (QNtail.w - 1.0f); //Create CD on the last strand

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
			W_SD_d_z = __fdividef(1.0f, 0.75f + f1);
		} else {
			//creation
			W_SD_c_z = __fdividef(2.0f, dBe * (QNtail.w + 0.5f));
		}
	}

	float sumW = sum_wshpm + W_SD_c_1 + W_SD_c_z + W_SD_d_1 + W_SD_d_z + W_CD_c_z;
	tdt[i] = __fdividef(1.0f, sumW);
	if (tdt[i] == 0.0f)
		gpu_chain_heads[i].stall_flag = 1;
	if (isnan(tdt[i]))
		gpu_chain_heads[i].stall_flag = 2;
	if (isinf(tdt[i]))
		gpu_chain_heads[i].stall_flag = 3;
	gpu_chain_heads[i].time += tdt[i];
// 	surf2Dread(&tdt[i],rand_buffer,4*0,i);

	float pr = (sumW) * tex2D(t_uniformrand, rand_used[i], i);
	rand_used[i]++;		//TODO just use step count constant instead of rand used
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

// 	  for (int j=0;j<tz-1;j++)
	if (pr < tpr) {
		// ok we pick some  strent j
		// now we need to decide which(SD shift or CDd CDc) jump process will happen
		// TODO check if order will have an effect on performance

		float4 QN1 = tex2D(t_a_QN, make_offset(j, oft), i);
		if (fetch_new_strent(j, oft))
			QN1 = new_strent;
		float4 QN2 = tex2D(t_a_QN, make_offset(j + 1, oft), i);
		if (fetch_new_strent(j + 1, oft))
			QN2 = new_strent;

		// 1. CDd (destruction by constraint dynamics)

		float wcdd;
		if (dCD_flag) {
			wcdd = tex2D(t_a_tCD, make_offset(j, oft), i); //Read CD
			if (fetch_new_strent(j, oft))
				wcdd = new_tCD;
		} else
			wcdd = 0;
		if (pr < wcdd) {

			float4 temp = make_float4(QN1.x + QN2.x, QN1.y + QN2.y,
					QN1.z + QN2.z, QN1.w + QN2.w);
			if ((j == tz - 2) || (j == 0)) {
				temp = make_float4(0.0f, 0.0f, 0.0f, QN1.w + QN2.w);
			}
			surf2Dwrite(temp, s_b_QN, 16 * (j + 1), i);
			d_offset[i] = offset_code(j, +1);
			gpu_chain_heads[i].Z--;

			return;
		} else {
			pr -= wcdd;
		}

		// 2. SD shift

		// SD shift probs are not saved from entanglement parallel part
		// so we need to recalculate it
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
		float wcdc = dCD_flag * d_CD_create_prefact * (QN1.w - 1.0f); //
		if (pr < wcdc) {
			if (tz == d_z_max)
				return;		// possible detail balance issue
			float4 temp = tex2D(t_taucd_gauss_rand, tau_CD_used[i], i);
			tau_CD_used[i]++;
			gpu_chain_heads[i].Z++;
			d_new_tau_CD[i] = d_tau_CD_f_d_t(temp.w);//__fdividef(1.0f,d_tau_d);
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
			surf2Dwrite(
					make_float4(QN1.x - temp.x, QN1.y - temp.y, QN1.z - temp.z,
							QN1.w - newn), s_b_QN, 16 * j, i);
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
		if (tz == d_z_max)
			return;	// possible detail balance issue

		float4 temp = tex2D(t_taucd_gauss_rand, tau_CD_used[i], i);
		tau_CD_used[i]++;
		gpu_chain_heads[i].Z++;
		d_new_tau_CD[i] = d_tau_CD_f_d_t(temp.w);	//__fdividef(1.0f,d_tau_d);

		float newn = 1.0f + floorf(0.5f + __fdividef(pr * (QNtail.w - 2.0f), W_CD_c_z));
//		floorf(__fdividef(pr*(QNtail.w-1.0f),W_CD_c_z))+1.0f;
// 	    gpu_chain_heads[i].dummy=1.0f+__fdiv_rn(pr*(QNtail.w-2.0f),W_CD_c_z);

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
		float4 temp = tex2D(t_taucd_gauss_rand, tau_CD_used[i], i);
		tau_CD_used[i]++;
		gpu_chain_heads[i].Z++;
//		d_new_tau_CD[i]=__fdividef(1.0f,d_tau_d);
		d_new_tau_CD[i] = d_tau_CD_f_t(temp.w);

		if (pr < W_SD_c_1) {
			temp.w = QNhead.w - 1.0f;
			float sigma = (tz == 1) ? 0.0f : __fsqrt_rn(__fdividef(temp.w, 3.0f));
			temp.x *= sigma;
			temp.y *= sigma;
			temp.z *= sigma;
			surf2Dwrite(temp, s_b_QN, 16 * 0, i);//TODO maybe deformation should be applied here
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

	// 6. Destruction by constraint dynamics
	if (pr < W_SD_d_1 + W_SD_d_z) {	//to delete entanglement
	// update cell and neigbours
	//clear W_sd
	//
	//form a list of free cell
		gpu_chain_heads[i].Z--;
		if (pr < W_SD_d_1) {
			surf2Dwrite(make_float4(0.0f, 0.0f, 0.0f, QNheadn.w + 1.0f), s_b_QN,
					16 * 1, i);
			d_offset[i] = offset_code(0, +1);
		} else {
			surf2Dwrite(make_float4(0.0f, 0.0f, 0.0f, QNtailp.w + 1.0f), s_b_QN,
					16 * (tz - 2), i);
			d_offset[i] = offset_code(tz, +1);

		}
		return;

	} else {
		pr -= W_SD_d_1 + W_SD_d_z;
	}

}

