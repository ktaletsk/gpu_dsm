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
#include "chain.h"
#include "gpu_random.h"
#include "ensemble_kernel.cu"
#include "ensemble_call_block.h"
#include "correlator.h"
#include <vector>
#define max_sync_interval 1E5
#define uniformrandom_count 250// size of the random arrays

//variable arrays, that are common for all the blocks
gpu_Ran *d_random_gens; // device random number generators
gpu_Ran *d_random_gens2; //first is used to pick jump process, second is used for creation of new entanglements
//temporary arrays fpr random numbers sequences
float* d_uniformrand; // uniform random number supply //used to pick jump process
float4* d_taucd_gauss_rand_CD; // 1x uniform + 3x normal distributed random number supply// used for creating entanglements by SD
float4* d_taucd_gauss_rand_SD; // used for creating entanglements by SD
int steps_count = 0;    //time step count
int *d_tau_CD_used_SD;
int *d_tau_CD_used_CD;
int *d_rand_used;
std::vector<float> pcd_data;
float4* d_a_QN; //device arrays for vector part of chain conformations
float* d_a_tCD; // these arrays used by time evolution kernels
float4* d_b_QN;
float* d_b_tCD;
float4* d_corr_a;
float4* d_corr_b;

float* d_sum_W; // sum of probabilities for each entanglement
float4* d_stress; // stress calculation temp array

// There are two arrays with the vector part of the chain conformations  on device.
// And there is only one array with the scalar part of the chain conformations
// Every timestep the vector part is copied from one array to another.
// The coping is done in entanglement parallel portion of the code
// This allows to use textures/surfaces and speeds up memory access
// The scalar part(chain headers) is updated in the chain parallel portion of the code
// Chain headers occupy less memory,and there are no specific memory access technics for them.

void init_call_block(ensemble_call_block *cb, int nc, sstrentp chains,
		chain_head* chain_heads, int s) {
	//allocates arrays, copies chain conformations to device
	//ensemble_call_block *cb pointer for call block structure, just ref parameter
	//int nc  is a number of chains in this ensemble_call_block.
	//sstrentp chains, chain_head* chain_heads pointers to array with the chain conformations

	//first take care about nc
	cb->nc = nc;
	cb->chains = chains;
	cb->chain_heads = chain_heads;
	// setup universal time
	cb->block_time = universal_time;

	cudaMalloc(&(cb->d_QN), z_max*cb->nc*sizeof(float4));
	cudaMalloc(&(cb->d_tCD),z_max*cb->nc*sizeof(float));
	cudaMalloc(&(cb->d_R1), sizeof(float4) * cb->nc);
	//blank dynamics probabilities
	float *buffer = new float[z_max * cb->nc];
	memset(buffer, 0, sizeof(float) * z_max * cb->nc);
	cudaMemcpy(d_sum_W, buffer, z_max * sizeof(float) * cb->nc, cudaMemcpyHostToDevice);
	delete[] buffer;
	//copy initial conformations to device
	cudaMemcpy(cb->d_QN, cb->chains.QN, z_max * sizeof(float4) * cb->nc, cudaMemcpyHostToDevice);
	cudaMemcpy(cb->d_tCD,cb->chains.tau_CD, z_max * sizeof(float)*cb->nc, cudaMemcpyHostToDevice);
	cudaMemcpy(cb->d_R1, cb->chains.R1, sizeof(float4)  * cb->nc, cudaMemcpyHostToDevice);
	cudaMalloc((void**) &cb->gpu_chain_heads, sizeof(chain_head) * cb->nc);
	cudaMemcpy(cb->gpu_chain_heads, cb->chain_heads, sizeof(chain_head) * cb->nc, cudaMemcpyHostToDevice);

	// allocating device arrays
	cudaMalloc(&(cb->d_dt), sizeof(float) * cb->nc);

	cudaMemset(cb->d_dt, 0, sizeof(float) * cb->nc);
	cudaMalloc(&(cb->reach_flag), sizeof(float) * cb->nc);
	cudaMalloc(&(cb->d_offset), sizeof(int) * cb->nc);
	cudaMalloc(&(cb->d_new_strent), sizeof(float) * 4 * cb->nc);
	cudaMalloc(&(cb->d_new_tau_CD), sizeof(float) * cb->nc);

	cudaMemset((cb->d_offset), 0xff, sizeof(float) * cb->nc);

	cb->corr = new correlator(cb->nc, s);//Allocate memory for c_correlator structure in cb
	cudaMalloc((void**) &cb->d_correlator_time, sizeof(int) * cb->nc);//Allocated memory on device for correlator time for every chain in block
	cudaMemset(cb->d_correlator_time, 0, sizeof(int) * cb->nc);	//Initialize d_correlator_time with zeros
	CUT_CHECK_ERROR("kernel execution failed");	//Debug feature, check error code
	//initialize cb->corr arrays with 0s
}

int time_step_call_block(double reach_time, ensemble_call_block *cb,
		bool* run_flag) {
	//bind textures_surfaces perform time evolution unbind textures_surfaces
	//ensemble_call_block *cb pointer for call block structure, just ref parameter
	//sstrentp chains, chain_head* chain_heads needed for debug//TODO not implemented

	cudaChannelFormatDesc channelDesc1 = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat); //to read float4
	cudaChannelFormatDesc channelDesc4 = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat); //to read float

	//loop preparing
	dim3 dimBlock(tpb_strent_kernel, tpb_strent_kernel);
	dim3 dimGrid((z_max + dimBlock.x - 1) / dimBlock.x,
			(cb->nc + dimBlock.y - 1) / dimBlock.y);

	activate_block(cb);

	float time_step_interval = reach_time - cb->block_time;
	int number_of_syncs = int(
			floor((time_step_interval - 0.5) / max_sync_interval)) + 1;

	float *rtbuffer = new float[cb->nc];

	//Loop begins

	for (int i_sync = 0; i_sync < number_of_syncs; i_sync++) {
		float sync_interval = max_sync_interval;
		if ((i_sync + 1) == number_of_syncs)
			sync_interval = time_step_interval - i_sync * max_sync_interval;

		bool reach_flag_all = false;
		cudaMemset(cb->reach_flag, 0, sizeof(float) * cb->nc);

		//update universal_time on device
		float tf = cb->block_time + i_sync * max_sync_interval;
		CUDA_SAFE_CALL(
				cudaMemcpyToSymbol(d_universal_time, &tf, sizeof(float)));

		while (!reach_flag_all) {
			//odd time steps
			if (!(steps_count & 0x00000001)) {

				strent_kernel<<<dimGrid, dimBlock>>>(d_a_QN,d_a_tCD,d_b_QN,d_b_tCD,d_sum_W,cb->gpu_chain_heads, cb->d_dt, cb->d_offset, cb->d_new_strent, cb->d_new_tau_CD);
				CUT_CHECK_ERROR("kernel execution failed");
				chain_kernel<<<
						(cb->nc + tpb_chain_kernel - 1) / tpb_chain_kernel,
						tpb_chain_kernel>>>(d_a_QN,d_a_tCD,d_b_QN,d_sum_W,cb->gpu_chain_heads, cb->d_dt,
						cb->reach_flag, sync_interval, cb->d_offset,
						cb->d_new_strent, cb->d_new_tau_CD, d_rand_used,
						d_tau_CD_used_CD, d_tau_CD_used_SD,d_uniformrand,d_taucd_gauss_rand_CD,d_taucd_gauss_rand_SD
                        );
				CUT_CHECK_ERROR("kernel execution failed");
			} else {

				strent_kernel<<<dimGrid, dimBlock>>>(d_b_QN,d_b_tCD,d_a_QN,d_a_tCD,d_sum_W,cb->gpu_chain_heads, cb->d_dt, cb->d_offset, cb->d_new_strent, cb->d_new_tau_CD);
				CUT_CHECK_ERROR("kernel execution failed");

				chain_kernel<<<
						(cb->nc + tpb_chain_kernel - 1) / tpb_chain_kernel,
						tpb_chain_kernel>>>(d_b_QN,d_b_tCD,d_a_QN,d_sum_W,cb->gpu_chain_heads, cb->d_dt,
						cb->reach_flag, sync_interval, cb->d_offset,
						cb->d_new_strent, cb->d_new_tau_CD, d_rand_used,
						d_tau_CD_used_CD, d_tau_CD_used_SD,d_uniformrand,d_taucd_gauss_rand_CD,d_taucd_gauss_rand_SD
                        );
				CUT_CHECK_ERROR("kernel execution failed");
			}

			steps_count++;
			// check for rand refill
			if (steps_count % uniformrandom_count == 0) {
				if (dbug)
					cout << "steps_count " << steps_count
							<< ". random_textures_refill()\n";
				random_textures_refill(cb->nc, 0);
				steps_count = 0;
			}

			// check for reached time
			cudaMemcpy(rtbuffer, cb->reach_flag, sizeof(float) * cb->nc,
					cudaMemcpyDeviceToHost);
			float sumrt = 0;
			for (int i = 0; i < cb->nc; i++) {
				sumrt += rtbuffer[i];
			}
			reach_flag_all = (sumrt == cb->nc);

			if (dbug) {
				float *dtbuffer = new float[cb->nc];
				cudaMemcpy(dtbuffer, cb->d_dt, sizeof(float) * cb->nc,
						cudaMemcpyDeviceToHost);

				for (int i = 0; i < 1 + 0 * cb->nc; i++) {	//just one chain
					cout << "dt " << dtbuffer[i] << '\n';
					cout << "\n";
				}
				delete[] dtbuffer;
			}
			if (*run_flag == false)
				return -1;
		}
	}	//loop end
	cb->block_time = reach_time;
	delete[] rtbuffer;
	deactivate_block(cb);
	return 0;
}

int EQ_time_step_call_block(double reach_time, ensemble_call_block *cb, int correlator_type, bool* run_flag, int *progress_bar) {
	//bind textures_surfaces perform time evolution unbinds textures_surfaces
	//ensemble_call_block *cb - pointer to call block structure

	//Declare and create stream for parallel correlator update
	cudaStream_t stream_calc;
	cudaStream_t stream_update;
	cudaStreamCreate(&stream_calc);
	cudaStreamCreate(&stream_update);

	cudaChannelFormatDesc channelDesc1 = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	cudaChannelFormatDesc channelDesc4 = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);

	//Loop preparing
	dim3 dimBlock(tpb_strent_kernel, tpb_strent_kernel);
	dim3 dimGrid((z_max + dimBlock.x - 1) / dimBlock.x, (cb->nc + dimBlock.y - 1) / dimBlock.y);

	steps_count = 0;
	activate_block(cb);

	float time_step_interval = reach_time - cb->block_time;
	int number_of_syncs = int(floor((time_step_interval - 0.5) / max_sync_interval)) + 1;

	float *rtbuffer;
	cudaMallocHost(&rtbuffer, sizeof(float)*cb->nc);

	int *tbuffer;
	cudaMallocHost(&tbuffer, sizeof(int)*cb->nc);

	bool texture_flag = true;
	//Loop begins
	for (int i_sync = 0; i_sync < number_of_syncs; i_sync++) {
		float sync_interval = max_sync_interval;
		if ((i_sync + 1) == number_of_syncs)
			sync_interval = time_step_interval - i_sync * max_sync_interval;

		bool reach_flag_all = false;
		cudaMemset(cb->reach_flag, 0, sizeof(float) * cb->nc);

		//update universal_time on device
		float tf = cb->block_time + i_sync * max_sync_interval;
		CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_universal_time, &tf, sizeof(float)));

		while (!reach_flag_all) {
			if (!(steps_count & 0x00000001)) {	//For odd number of steps

                float4 *d_corr;
				if (texture_flag == true){
// 					cudaBindSurfaceToArray(s_corr, d_corr_b);
                    d_corr=d_corr_b;
				} else {
// 					cudaBindSurfaceToArray(s_corr, d_corr_a);
                    d_corr=d_corr_a;
				}

				EQ_strent_kernel<<<dimGrid, dimBlock,0,stream_calc>>>(d_a_QN,d_a_tCD,d_b_QN,d_b_tCD,d_sum_W,cb->gpu_chain_heads, cb->d_offset, cb->d_new_strent, cb->d_new_tau_CD);
				CUT_CHECK_ERROR("kernel execution failed");

				EQ_chain_kernel<<<(cb->nc + tpb_chain_kernel - 1) / tpb_chain_kernel, tpb_chain_kernel,0,stream_calc>>>(d_a_QN,d_a_tCD,d_b_QN,d_sum_W,d_corr,cb->gpu_chain_heads, cb->d_dt, cb->reach_flag, sync_interval, cb->d_offset, cb->d_new_strent, cb->d_new_tau_CD, cb->d_correlator_time, correlator_type, d_rand_used, d_tau_CD_used_CD, d_tau_CD_used_SD,d_uniformrand,d_taucd_gauss_rand_CD,d_taucd_gauss_rand_SD, steps_count % stressarray_count,cb->d_R1);
				CUT_CHECK_ERROR("kernel execution failed");


			} else { //For even number of steps
                float4 *d_corr;              
				if (texture_flag == true){
// 					cudaBindSurfaceToArray(s_corr, d_corr_b);
                    d_corr=d_corr_b;
                    
				} else {
// 					cudaBindSurfaceToArray(s_corr, d_corr_a);
                    d_corr=d_corr_a;
                    
				}

				EQ_strent_kernel<<<dimGrid, dimBlock,0,stream_calc>>>(d_b_QN,d_b_tCD,d_a_QN,d_a_tCD,d_sum_W,cb->gpu_chain_heads,cb->d_offset, cb->d_new_strent, cb->d_new_tau_CD);
				CUT_CHECK_ERROR("kernel execution failed");

				EQ_chain_kernel<<<(cb->nc + tpb_chain_kernel - 1) / tpb_chain_kernel, tpb_chain_kernel,0,stream_calc>>>(d_b_QN,d_b_tCD,d_a_QN,d_sum_W,d_corr,cb->gpu_chain_heads, cb->d_dt, cb->reach_flag, sync_interval, cb->d_offset, cb->d_new_strent, cb->d_new_tau_CD, cb->d_correlator_time, correlator_type, d_rand_used, d_tau_CD_used_CD, d_tau_CD_used_SD,d_uniformrand,d_taucd_gauss_rand_CD,d_taucd_gauss_rand_SD, steps_count % stressarray_count,cb->d_R1);
				CUT_CHECK_ERROR("kernel execution failed");

			}
			steps_count++;

			// update progress bar
			if (steps_count % 50 == 0) {
				cudaStreamSynchronize(stream_calc);
				cudaMemcpyAsync(tbuffer, cb->d_correlator_time, sizeof(int) * cb->nc, cudaMemcpyDeviceToHost, stream_calc);
				cudaStreamSynchronize(stream_calc);
				int sumt = 0;
				for (int i = 0; i < cb->nc; i++)
					sumt += tbuffer[i];
				*progress_bar = (int)(100.0f * sumt / (cb->nc) / reach_time);
				cout << "\r" << *progress_bar << "%\t ";
			}

			// check for rand refill
			if (steps_count % uniformrandom_count == 0) {
				random_textures_refill(cb->nc, stream_calc);
				steps_count = 0;
			}

			if (steps_count % stressarray_count == 0) {
				cudaStreamSynchronize(stream_calc);
				cudaStreamSynchronize(stream_update);
                float4 *d_corr;              
				if (texture_flag==true){
// 					cudaBindTextureToArray(t_corr, d_corr_b, channelDesc4);
                    d_corr=d_corr_b;
					texture_flag = false;                   
				} else {
// 					cudaBindTextureToArray(t_corr, d_corr_a, channelDesc4);
                    d_corr=d_corr_a;
					texture_flag = true;
				}
				update_correlator<<<(cb->nc + tpb_chain_kernel - 1) / tpb_chain_kernel, tpb_chain_kernel,0,stream_update>>>(d_corr,(cb->corr)->gpu_corr, stressarray_count, correlator_type);
			}

			// check for reached time
			cudaStreamSynchronize(stream_calc);
			cudaMemcpyAsync(rtbuffer, cb->reach_flag, sizeof(float) * cb->nc, cudaMemcpyDeviceToHost, stream_calc);
			cudaStreamSynchronize(stream_calc);
			float sumrt = 0;
			for (int i = 0; i < cb->nc; i++)
				sumrt += rtbuffer[i];
			reach_flag_all = (sumrt == cb->nc);

			// stop, if run_flag is changed from outside
			if (*run_flag == false)
				return -1;
		}
	} //loop ends

	if (steps_count % stressarray_count != 0) {
		cudaStreamSynchronize(stream_calc);
		cudaStreamSynchronize(stream_update);
        float4 *d_corr;                      
		if (texture_flag==true){
// 			cudaBindTextureToArray(t_corr, d_corr_b, channelDesc4);
            d_corr=d_corr_b;            
			texture_flag = false;
		} else {
// 			cudaBindTextureToArray(t_corr, d_corr_a, channelDesc4);
            d_corr=d_corr_b;            
			texture_flag = true;
		}
		update_correlator<<<(cb->nc + tpb_chain_kernel - 1) / tpb_chain_kernel, tpb_chain_kernel,0,stream_update>>>(d_corr,(cb->corr)->gpu_corr, steps_count, correlator_type);
	}

	cb->block_time = reach_time;
	cudaHostUnregister(rtbuffer);
	cudaFreeHost(rtbuffer);
	deactivate_block(cb);
	cudaStreamDestroy(stream_update);
	cudaStreamDestroy(stream_calc);
	return 0;
}

// utility functions
//h means host(cpu) declarations
//host copies of gpu inline access functions
//purpose-- to recreate latest chain conformations from gpu memory (to account for delayed dynamics)
int hmake_offset(int i, int offset) {
	//offset&0xffff00)>>8 offset_index
	//offset&0xff-1; offset_dir
	return i >= ((offset & 0xffff00) >> 8) ? i + ((offset & 0xff) - 1) : i;
}
int hoffset_index(int offset) {
	return ((offset & 0xffff00) >> 8);
}

int hoffset_dir(int offset) {
	return (offset & 0xff) - 1;
}

bool fetch_hnew_strent(int i, int offset) {
	return (i == hoffset_index(offset)) && (hoffset_dir(offset) == -1);
}

void get_chain_to_device_call_block(ensemble_call_block *cb) {
	//copy conformations to device
	cudaMemcpy(cb->d_QN, cb->chains.QN, z_max * sizeof(float4)*cb->nc, cudaMemcpyHostToDevice);
	cudaMemcpy(cb->d_tCD, cb->chains.tau_CD, z_max * sizeof(float)*cb->nc, cudaMemcpyHostToDevice);
	cudaMemcpy(cb->d_R1, cb->chains.R1, sizeof(float4)*cb->nc, cudaMemcpyHostToDevice);
	cudaMemcpy(cb->gpu_chain_heads, cb->chain_heads, sizeof(chain_head) * cb->nc, cudaMemcpyHostToDevice);

	//blank delayed dynamics arrays
	cudaMemset(cb->d_dt, 0, sizeof(float) * cb->nc);
	cudaMemset((cb->d_offset), 0xff, sizeof(float) * cb->nc);

}

void get_chain_from_device_call_block(ensemble_call_block *cb) { //copy chains back
	//NOTE accounts for delayed dynamics

	cudaMemcpy(cb->chain_heads, cb->gpu_chain_heads, sizeof(chain_head) * cb->nc, cudaMemcpyDeviceToHost);

	cudaMemcpy(cb->chains.QN, cb->d_QN,  z_max * sizeof(float4) * cb->nc, cudaMemcpyDeviceToHost);
	cudaMemcpy(cb->chains.tau_CD, cb->d_tCD, z_max * sizeof(float)*cb->nc, cudaMemcpyDeviceToHost);
	cudaMemcpy(cb->chains.R1, cb->d_R1, sizeof(float) * 4 * cb->nc, cudaMemcpyDeviceToHost);

	//delayed dynamics
	int *h_offset = new int[cb->nc];
	float4 *h_new_strent = new float4[cb->nc];
	float *h_new_tau_CD = new float[cb->nc];
	cudaMemcpy(h_offset, cb->d_offset, sizeof(int) * cb->nc, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_new_strent, cb->d_new_strent, sizeof(float4) * cb->nc, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_new_tau_CD, cb->d_new_tau_CD, sizeof(float) * cb->nc, cudaMemcpyDeviceToHost);
	for (int i = 0; i < cb->nc; i++) {
		if (hoffset_dir(h_offset[i]) == -1) {
			for (int j = z_max - 1; j > 0; j--) {
				chains.QN[i * z_max + j] = chains.QN[i * z_max + hmake_offset(j, h_offset[i])];
				chains.tau_CD[i * z_max + j] = chains.tau_CD[i * z_max + hmake_offset(j, h_offset[i])];
			}
			chains.QN[i * z_max + hoffset_index(h_offset[i])] = h_new_strent[i];
			chains.tau_CD[i * z_max + hoffset_index(h_offset[i])] = h_new_tau_CD[i];
		} else {
			for (int j = 0; j < z_max - 2; j++) {
				chains.QN[i * z_max + j] = chains.QN[i * z_max + hmake_offset(j, h_offset[i])];
				chains.tau_CD[i * z_max + j] = chains.tau_CD[i * z_max + hmake_offset(j, h_offset[i])];
			}
		}
	}

	delete[] h_offset;
	delete[] h_new_strent;
	delete[] h_new_tau_CD;
}

stress_plus calc_stress_call_block(ensemble_call_block *cb,
		int *r_chain_count) {

	cudaChannelFormatDesc channelDesc4 = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(dn_cha_per_call, &cb->nc, sizeof(int)));
	stress_calc<<<(cb->nc + tpb_chain_kernel - 1) / tpb_chain_kernel, tpb_chain_kernel>>>(cb->d_QN,d_stress,cb->gpu_chain_heads, cb->d_dt, cb->d_offset, cb->d_new_strent);
	CUT_CHECK_ERROR("kernel execution failed");

	float4 *stress_buf = new float4[cb->nc * 2];
	cudaMemcpy(stress_buf, d_stress, cb->nc * sizeof(float4) * 2, cudaMemcpyDeviceToHost);
	float4 sum_stress = make_float4(0.0f, 0.0f, 0.0f, 0.0f); //stress: xx,yy,zz,xy
	float4 sum_stress2 = make_float4(0.0f, 0.0f, 0.0f, 0.0f); //stress: yz,xz; Lpp, Ree
	chain_head* tchain_heads;
	tchain_heads = new chain_head[cb->nc];

	cudaMemcpy(tchain_heads, cb->gpu_chain_heads, sizeof(chain_head) * cb->nc, cudaMemcpyDeviceToHost);
	int chain_count = cb->nc;
	for (int j = 0; j < cb->nc; j++) {
		if (tchain_heads[j].stall_flag == 0) {
			if (!isnan(stress_buf[j * 2].x)) {
				sum_stress.x += stress_buf[j * 2].x;
				sum_stress.y += stress_buf[j * 2].y;
				sum_stress.z += stress_buf[j * 2].z;
				sum_stress.w += stress_buf[j * 2].w;
				sum_stress2.x += stress_buf[j * 2 + 1].x;
				sum_stress2.y += stress_buf[j * 2 + 1].y;
				sum_stress2.z += stress_buf[j * 2 + 1].z;
				sum_stress2.w += stress_buf[j * 2 + 1].w;
//				cout<<"stress chain "<<j<<'\t'<<sum_stress.x<<'\t'<<sum_stress.y<<'\t'<<sum_stress.z<<'\t'<<sum_stress.w<<'\n';
			} else {
				chain_count--;
				cout << "chain stall " << j << '\n';  //TODO output gloval index
			}
		} else {
			chain_count--;
			cout << "chain stall " << j << '\n';    //TODO output gloval index
		}
	}
	stress_plus rs;
	rs.xx = sum_stress.x / chain_count;
	rs.yy = sum_stress.y / chain_count;
	rs.zz = sum_stress.z / chain_count;
	rs.xy = sum_stress.w / chain_count;
	rs.yz = sum_stress2.x / chain_count;
	rs.zx = sum_stress2.y / chain_count;
	rs.Lpp = sum_stress2.z / chain_count;
	rs.Z = sum_stress2.w / chain_count;
	delete[] stress_buf;
	delete[] tchain_heads;
	*r_chain_count = chain_count;
	return rs;
}

void free_block(ensemble_call_block *cb) {    //free memory
//	delete cb->nc;
//	delete cb->block_time;


	cudaFree(cb->gpu_chain_heads);
	cudaFree(cb->d_QN);
	cudaFree(cb->d_tCD);
	cudaFree(cb->d_R1);

	cudaFree(cb->d_dt);
	cudaFree(cb->reach_flag);
	cudaFree(cb->d_offset);
	cudaFree(cb->d_new_strent);
	cudaFree(cb->d_new_tau_CD);

	if (cb->corr != NULL) {
		cudaFree(cb->d_correlator_time);
		delete cb->corr;
	}
}

void activate_block(ensemble_call_block *cb) {
	//prepares block for performing time evolution
	//i.e. copies chain conformations to working memory
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(dn_cha_per_call, &cb->nc, sizeof(int)));

	float tf = cb->block_time;
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_universal_time, &tf, sizeof(float)));

	if (!(steps_count & 0x00000001)) {
		cudaMemcpy(d_a_QN, cb->d_QN,  z_max * sizeof(float4) * cb->nc, cudaMemcpyDeviceToDevice);
		cudaMemcpy(d_a_tCD,  cb->d_tCD, z_max * sizeof(float) * cb->nc, cudaMemcpyDeviceToDevice);
	} else {
		cudaMemcpy(d_b_QN, cb->d_QN,  z_max * sizeof(float4)* cb->nc, cudaMemcpyDeviceToDevice);
		cudaMemcpy(d_b_tCD, cb->d_tCD, z_max * sizeof(float) * cb->nc, cudaMemcpyDeviceToDevice);
	}

	//correlator binding
	cudaBindSurfaceToArray(s_correlator, cb->corr->gpu_corr.d_correlator);
}

void deactivate_block(ensemble_call_block *cb) {
	//copies chain conformations to storing memory

	if (!(steps_count & 0x00000001)) {
		cudaMemcpy(cb->d_QN, d_a_QN, z_max * sizeof(float4)*cb->nc, cudaMemcpyDeviceToDevice);
		cudaMemcpy(cb->d_tCD,d_a_tCD, z_max * sizeof(float) * cb->nc, cudaMemcpyDeviceToDevice);
	} else {
		cudaMemcpy(cb->d_QN, d_b_QN,  z_max * sizeof(float4) * cb->nc, cudaMemcpyDeviceToDevice);
		cudaMemcpy(cb->d_tCD,d_b_tCD, z_max * sizeof(float) * cb->nc, cudaMemcpyDeviceToDevice);
	}
}

