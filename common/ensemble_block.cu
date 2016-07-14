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
#include "textures_surfaces.h"
#include "chain.h"
#include "gpu_random.h"
#include "ensemble_kernel.cu"
#include "ensemble_block.h"
#include "correlator.h"
#include <vector>
#define max_sync_interval 1E5
//variable arrays, that are common for all the blocks
gpu_Ran *d_random_gens; // device random number generators
gpu_Ran *d_random_gens2; //first is used to pick jump process, second is used for creation of new entanglements
//temporary arrays fpr random numbers sequences
cudaArray* d_uniformrand; // uniform random number supply //used to pick jump process
cudaArray* d_taucd_gauss_rand_CD; // 1x uniform + 3x normal distributed random number supply// used for creating entanglements by SD
cudaArray* d_taucd_gauss_rand_SD; // used for creating entanglements by SD
int steps_count = 0;    //time step count
int *d_tau_CD_used_SD;
int *d_tau_CD_used_CD;
int *d_rand_used;
//std::vector<float> pcd_data;
cudaArray* d_a_QN; //device arrays for vector part of chain conformations
cudaArray* d_a_tCD; // these arrays used by time evolution kernels
cudaArray* d_b_QN;
cudaArray* d_b_tCD;
cudaArray* d_a_R1;
cudaArray* d_b_R1;
cudaArray* d_corr_a;
cudaArray* d_corr_b;

cudaArray* d_sum_W; // sum of probabilities for each entanglement
cudaArray* d_stress; // stress calculation temp array

// There are two arrays with the vector part of the chain conformations  on device.
// And there is only one array with the scalar part of the chain conformations
// Every timestep the vector part is copied from one array to another.
// The coping is done in entanglement parallel portion of the code
// This allows to use textures/surfaces and speeds up memory access
// The scalar part(chain headers) is updated in the chain parallel portion of the code
// Chain headers occupy less memory,and there are no specific memory access technics for them.

void ensemble_block::init(int nc_, vector_chains chains_, scalar_chains* chain_heads_, int s){
	//allocates arrays, copies chain conformations to device
	//ensemble_call_block *cb pointer for call block structure, just ref parameter
	//int nc  is a number of chains in this ensemble_call_block.
	//sstrentp chains, chain_head* chain_heads pointers to array with the chain conformations

	//first take care about nc
	nc = nc_;
	chains = chains_;
	chain_heads = chain_heads_;
	// setup universal time
	block_time = universal_time;

	cudaChannelFormatDesc channelDesc4 = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
	cudaChannelFormatDesc channelDesc2 = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindFloat);
	cudaChannelFormatDesc channelDesc1 = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

	cudaMallocArray(&d_QN, &channelDesc4, z_max, nc, cudaArraySurfaceLoadStore);
	cudaMallocArray(&d_tCD, &channelDesc1, z_max, nc,cudaArraySurfaceLoadStore);
	cudaMallocArray(&d_R1, &channelDesc4, nc, 1, cudaArraySurfaceLoadStore);

	//blank dynamics probabalities
	float *buffer = new float[z_max * nc];
	memset(buffer, 0, sizeof(float) * z_max * nc);
	cudaMemcpy2DToArray(d_sum_W, 0, 0, buffer, z_max * sizeof(float), z_max * sizeof(float), nc, cudaMemcpyHostToDevice);
	delete[] buffer;
	//copy initial conformations to device
//	cudaMemcpy2DToArray(d_QN, 0, 0, chains.QN, z_max * sizeof(float) * 4, z_max * sizeof(float) * 4, nc, cudaMemcpyHostToDevice);
//	cudaMemcpy2DToArray(d_tCD, 0, 0, chains.tau_CD, z_max * sizeof(float), z_max * sizeof(float), nc, cudaMemcpyHostToDevice);
//	cudaMemcpyToArray(d_R1, 0, 0, chains.R1, sizeof(float) * 4 * nc, cudaMemcpyHostToDevice);
	cudaMalloc((void**) &gpu_chain_heads, sizeof(scalar_chains) * nc);
	cudaMemcpy(gpu_chain_heads, chain_heads, sizeof(scalar_chains) * nc, cudaMemcpyHostToDevice);

	// allocating device arrays
	cudaMalloc(&d_dt, sizeof(float) * nc);

	cudaMemset(d_dt, 0, sizeof(float) * nc);
	cudaMalloc(&reach_flag, sizeof(float) * nc);
	cudaMalloc(&d_offset, sizeof(int) * nc);
	cudaMalloc(&d_new_strent, sizeof(float) * 4 * nc);
	cudaMalloc(&d_new_tau_CD, sizeof(float) * nc);

	cudaMemset((this->d_offset), 0xff, sizeof(float) * nc);

	this->corr = new correlator(nc, s);//Allocate memory for c_correlator structure in cb
	cudaMalloc((void**) &d_correlator_time, sizeof(int) * nc);//Allocated memory on device for correlator time for every chain in block
	cudaMemset(d_correlator_time, 0, sizeof(int) * nc);	//Initialize d_correlator_time with zeros
	CUT_CHECK_ERROR("kernel execution failed");	//Debug feature, check error code

	//initialize corr arrays with 0s
}

int ensemble_block::flow_time_step(double reach_time, bool* run_flag) {
	//bind textures_surfaces perform time evolution unbind textures_surfaces
	//ensemble_call_block *cb pointer for call block structure, just ref parameter
	//sstrentp chains, chain_head* chain_heads needed for debug//TODO not implemented

	cudaChannelFormatDesc channelDesc1 = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat); //to read float4
	cudaChannelFormatDesc channelDesc4 = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat); //to read float

	//loop preparing
	dim3 dimBlock(tpb_strent_kernel, tpb_strent_kernel);
	dim3 dimGrid((z_max + dimBlock.x - 1) / dimBlock.x, (nc + dimBlock.y - 1) / dimBlock.y);

	activate_block();

	float time_step_interval = reach_time - block_time;
	int number_of_syncs = int(floor((time_step_interval - 0.5) / max_sync_interval)) + 1;

	float *rtbuffer = new float[nc];

	//Loop begins
	for (int i_sync = 0; i_sync < number_of_syncs; i_sync++) {
		float sync_interval = max_sync_interval;
		if ((i_sync + 1) == number_of_syncs)
			sync_interval = time_step_interval - i_sync * max_sync_interval;

		bool reach_flag_all = false;
		cudaMemset(reach_flag, 0, sizeof(float) * nc);

		//update universal_time on device
		float tf = block_time + i_sync * max_sync_interval;
		CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_universal_time, &tf, sizeof(float)));

		while (!reach_flag_all) {
			//odd time steps
			if (!(steps_count & 0x00000001)) {

				cudaBindTextureToArray(t_a_QN, d_a_QN, channelDesc4);
				cudaBindSurfaceToArray(s_b_QN, d_b_QN);
				cudaBindTextureToArray(t_a_tCD, d_a_tCD, channelDesc1);
				cudaBindSurfaceToArray(s_b_tCD, d_b_tCD);


				strent_kernel<1><<<dimGrid, dimBlock>>>(gpu_chain_heads, d_dt, d_offset, d_new_strent, d_new_tau_CD);
				CUT_CHECK_ERROR("kernel execution failed");
				chain_kernel<1><<<(nc + tpb_chain_kernel - 1) / tpb_chain_kernel, tpb_chain_kernel>>>(gpu_chain_heads, d_dt, reach_flag, sync_interval, d_offset, d_new_strent, d_new_tau_CD, NULL, 0, d_rand_used, d_tau_CD_used_CD, d_tau_CD_used_SD, 0);
				CUT_CHECK_ERROR("kernel execution failed");
				cudaUnbindTexture(t_a_QN);
				cudaUnbindTexture(t_a_tCD);
			} else {
				cudaBindTextureToArray(t_a_QN, d_b_QN, channelDesc4);
				cudaBindSurfaceToArray(s_b_QN, d_a_QN);
				cudaBindTextureToArray(t_a_tCD, d_b_tCD, channelDesc1);
				cudaBindSurfaceToArray(s_b_tCD, d_a_tCD);

				strent_kernel<1><<<dimGrid, dimBlock>>>(gpu_chain_heads, d_dt, d_offset, d_new_strent, d_new_tau_CD);
				CUT_CHECK_ERROR("kernel execution failed");

				chain_kernel<1><<<(nc + tpb_chain_kernel - 1) / tpb_chain_kernel, tpb_chain_kernel>>>(gpu_chain_heads, d_dt, reach_flag, sync_interval, d_offset, d_new_strent, d_new_tau_CD, NULL, 0, d_rand_used, d_tau_CD_used_CD, d_tau_CD_used_SD, 0);
				CUT_CHECK_ERROR("kernel execution failed");
				cudaUnbindTexture(t_a_QN);
				cudaUnbindTexture(t_a_tCD);
			}

			steps_count++;
			// check for rand refill
			if (steps_count % uniformrandom_count == 0) {
				if (dbug)
					cout << "steps_count " << steps_count
							<< ". random_textures_refill()\n";
				random_textures_refill(nc, 0);
				steps_count = 0;
			}

			// check for reached time
			cudaMemcpy(rtbuffer, reach_flag, sizeof(float) * nc,
					cudaMemcpyDeviceToHost);
			float sumrt = 0;
			for (int i = 0; i < nc; i++) {
				sumrt += rtbuffer[i];
			}
			reach_flag_all = (sumrt == nc);

			if (dbug) {
				float *dtbuffer = new float[nc];
				cudaMemcpy(dtbuffer, d_dt, sizeof(float) * nc,
						cudaMemcpyDeviceToHost);

				for (int i = 0; i < 1 + 0 * nc; i++) {	//just one chain
					cout << "dt " << dtbuffer[i] << '\n';
					cout << "\n";
				}
				delete[] dtbuffer;
			}
			if (*run_flag == false)
				return -1;
		}
	}	//loop end
	block_time = reach_time;
	delete[] rtbuffer;
	deactivate_block();
	return 0;
}

int ensemble_block::equilibrium_time_step(double reach_time, int correlator_type, bool* run_flag, int *progress_bar) {
	//bind textures_surfaces perform time evolution unbinds textures_surfaces
	//ensemble_block *cb - pointer to call block class

	//Declare and create stream for parallel correlator update
	cudaStream_t stream_calc;
	cudaStream_t stream_update;
	cudaStreamCreate(&stream_calc);
	cudaStreamCreate(&stream_update);

	cudaChannelFormatDesc channelDesc1 = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	cudaChannelFormatDesc channelDesc4 = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);

	//Loop preparing
	dim3 dimBlock(tpb_strent_kernel, tpb_strent_kernel);
	dim3 dimGrid((z_max + dimBlock.x - 1) / dimBlock.x, (nc + dimBlock.y - 1) / dimBlock.y);

	steps_count = 0;
	activate_block();

	float time_step_interval = reach_time - block_time;
	int number_of_syncs = int(floor((time_step_interval - 0.5) / max_sync_interval)) + 1;

	float *rtbuffer;
	cudaMallocHost(&rtbuffer, sizeof(float)*nc);

	int *tbuffer;
	cudaMallocHost(&tbuffer, sizeof(int)*nc);

	bool texture_flag = true;
	//Loop begins
	for (int i_sync = 0; i_sync < number_of_syncs; i_sync++) {
		float sync_interval = max_sync_interval;
		if ((i_sync + 1) == number_of_syncs)
			sync_interval = time_step_interval - i_sync * max_sync_interval;

		bool reach_flag_all = false;
		cudaMemset(reach_flag, 0, sizeof(float) * nc);

		//update universal_time on device
		float tf = block_time + i_sync * max_sync_interval;
		CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_universal_time, &tf, sizeof(float)));

		while (!reach_flag_all) {
			if (!(steps_count & 0x00000001)) {	//For odd number of steps

				cudaBindTextureToArray(t_a_QN, d_a_QN, channelDesc4);
				cudaBindSurfaceToArray(s_b_QN, d_b_QN);
				cudaBindTextureToArray(t_a_tCD, d_a_tCD, channelDesc1);
				cudaBindSurfaceToArray(s_b_tCD, d_b_tCD);
				cudaBindTextureToArray(t_a_R1, d_a_R1, channelDesc4);
				cudaBindSurfaceToArray(s_b_R1, d_b_R1);
				if (texture_flag == true){
					cudaBindSurfaceToArray(s_corr, d_corr_b);
				} else {
					cudaBindSurfaceToArray(s_corr, d_corr_a);
				}

				strent_kernel<0><<<dimGrid, dimBlock,0,stream_calc>>>(gpu_chain_heads, NULL, d_offset, d_new_strent, d_new_tau_CD);
				CUT_CHECK_ERROR("kernel execution failed");

				chain_kernel<0><<<(nc + tpb_chain_kernel - 1) / tpb_chain_kernel, tpb_chain_kernel,0,stream_calc>>>(gpu_chain_heads, d_dt, reach_flag, sync_interval, d_offset, d_new_strent, d_new_tau_CD, d_correlator_time, correlator_type, d_rand_used, d_tau_CD_used_CD, d_tau_CD_used_SD, steps_count % stressarray_count);
				CUT_CHECK_ERROR("kernel execution failed");

				cudaUnbindTexture(t_a_QN);
				cudaUnbindTexture(t_a_tCD);

			} else { //For even number of steps
				cudaBindTextureToArray(t_a_QN, d_b_QN, channelDesc4);
				cudaBindSurfaceToArray(s_b_QN, d_a_QN);
				cudaBindTextureToArray(t_a_tCD, d_b_tCD, channelDesc1);
				cudaBindSurfaceToArray(s_b_tCD, d_a_tCD);
				cudaBindTextureToArray(t_a_R1, d_b_R1, channelDesc4);
				cudaBindSurfaceToArray(s_b_R1, d_a_R1);
				if (texture_flag == true){
					cudaBindSurfaceToArray(s_corr, d_corr_b);
				} else {
					cudaBindSurfaceToArray(s_corr, d_corr_a);
				}

				strent_kernel<0><<<dimGrid, dimBlock,0,stream_calc>>>(gpu_chain_heads, NULL, d_offset, d_new_strent, d_new_tau_CD);
				CUT_CHECK_ERROR("kernel execution failed");

				chain_kernel<0><<<(nc + tpb_chain_kernel - 1) / tpb_chain_kernel, tpb_chain_kernel,0,stream_calc>>>(gpu_chain_heads, d_dt, reach_flag, sync_interval, d_offset, d_new_strent, d_new_tau_CD, d_correlator_time, correlator_type, d_rand_used, d_tau_CD_used_CD, d_tau_CD_used_SD, steps_count % stressarray_count);
				CUT_CHECK_ERROR("kernel execution failed");

				cudaUnbindTexture(t_a_QN);
				cudaUnbindTexture(t_a_tCD);
			}
			steps_count++;

			// update progress bar
			if (steps_count % 50 == 0) {
				cudaStreamSynchronize(stream_calc);
				cudaMemcpyAsync(tbuffer, d_correlator_time, sizeof(int) * nc, cudaMemcpyDeviceToHost, stream_calc);
				cudaStreamSynchronize(stream_calc);
				int sumt = 0;
				for (int i = 0; i < nc; i++)
					sumt += tbuffer[i];
				*progress_bar = (int)(100.0f * sumt / (nc) / reach_time);
				cout << "\r" << *progress_bar << "%\t ";
			}

			// check for rand refill
			if (steps_count % uniformrandom_count == 0) {
				random_textures_refill(nc, stream_calc);
				steps_count = 0;
			}

			if (steps_count % stressarray_count == 0) {
				cudaStreamSynchronize(stream_calc);
				cudaStreamSynchronize(stream_update);
				cudaUnbindTexture(t_corr);
				if (texture_flag==true){
					cudaBindTextureToArray(t_corr, d_corr_b, channelDesc4);
					texture_flag = false;
				} else {
					cudaBindTextureToArray(t_corr, d_corr_a, channelDesc4);
					texture_flag = true;
				}
				update_correlator<<<(nc + tpb_chain_kernel - 1) / tpb_chain_kernel, tpb_chain_kernel,0,stream_update>>>((corr)->gpu_corr, stressarray_count, correlator_type);
			}

			// check for reached time
			cudaStreamSynchronize(stream_calc);
			cudaMemcpyAsync(rtbuffer, reach_flag, sizeof(float) * nc, cudaMemcpyDeviceToHost, stream_calc);
			cudaStreamSynchronize(stream_calc);
			float sumrt = 0;
			for (int i = 0; i < nc; i++)
				sumrt += rtbuffer[i];
			reach_flag_all = (sumrt == nc);

			// stop, if run_flag is changed from outside
			if (*run_flag == false)
				return -1;
		}
	} //loop ends

	if (steps_count % stressarray_count != 0) {
		cudaStreamSynchronize(stream_calc);
		cudaStreamSynchronize(stream_update);
		cudaUnbindTexture(t_corr);
		if (texture_flag==true){
			cudaBindTextureToArray(t_corr, d_corr_b, channelDesc4);
			texture_flag = false;
		} else {
			cudaBindTextureToArray(t_corr, d_corr_a, channelDesc4);
			texture_flag = true;
		}
		update_correlator<<<(nc + tpb_chain_kernel - 1) / tpb_chain_kernel, tpb_chain_kernel,0,stream_update>>>((corr)->gpu_corr, steps_count, correlator_type);
	}

	block_time = reach_time;
	cudaHostUnregister(rtbuffer);
	cudaFreeHost(rtbuffer);
	deactivate_block();
	cudaStreamDestroy(stream_update);
	cudaStreamDestroy(stream_calc);
	return 0;
}

int ensemble_block::equilibrium_calc(double length, int correlator_type, bool* run_flag, int *progress_bar, int np, float* t, float* x){
	transfer_to_device();
	cudaMemset(d_correlator_time, 0, sizeof(int) * nc);
	if(equilibrium_time_step(length, correlator_type, run_flag, progress_bar)==-1) return -1;
	int *tint = new int[np];
	float *x_buf = new float[np];
	corr->calc(tint, x_buf, correlator_type);
	transfer_from_device();
	for (int j = 0; j < corr->npcorr; j++) {
		t[j] = tint[j];
		x[j] += x_buf[j] / N_cha;
	}
	delete[] x_buf;
	delete[] tint;
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

void ensemble_block::transfer_to_device(){
	//copy conformations to device
	cudaMemcpy2DToArray(d_QN, 0, 0, chains.QN, z_max * sizeof(float) * 4, z_max * sizeof(float) * 4, nc, cudaMemcpyHostToDevice);
	cudaMemcpy2DToArray(d_tCD, 0, 0, chains.tau_CD, z_max * sizeof(float), z_max * sizeof(float), nc, cudaMemcpyHostToDevice);
	cudaMemcpyToArray(d_R1, 0, 0, chains.R1, sizeof(float) * 4 * nc, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_chain_heads, chain_heads, sizeof(scalar_chains) * nc, cudaMemcpyHostToDevice);

	//blank delayed dynamics arrays
	cudaMemset(d_dt, 0, sizeof(float) * nc);
	cudaMemset(d_offset, 0xff, sizeof(float) * nc);

}

void ensemble_block::transfer_from_device() { //copy chains back
	//NOTE accounts for delayed dynamics

	cudaMemcpy(chain_heads, gpu_chain_heads, sizeof(scalar_chains) * nc, cudaMemcpyDeviceToHost);

	cudaMemcpy2DFromArray(chains.QN, sizeof(float) * z_max * 4, d_QN, 0, 0, z_max * sizeof(float) * 4, nc, cudaMemcpyDeviceToHost);
	cudaMemcpy2DFromArray(chains.tau_CD, sizeof(float) * z_max, d_tCD,0, 0, z_max * sizeof(float), nc, cudaMemcpyDeviceToHost);
	cudaMemcpyFromArray(chains.R1, d_R1, 0, 0, sizeof(float) * 4 * nc, cudaMemcpyDeviceToHost);

	//delayed dynamics
	int *h_offset = new int[nc];
	float4 *h_new_strent = new float4[nc];
	float *h_new_tau_CD = new float[nc];
	cudaMemcpy(h_offset, d_offset, sizeof(int) * nc, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_new_strent, d_new_strent, sizeof(float4) * nc, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_new_tau_CD, d_new_tau_CD, sizeof(float) * nc, cudaMemcpyDeviceToHost);
	for (int i = 0; i < nc; i++) {
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

stress_plus ensemble_block::calc_stress(int *r_chain_count) {

	cudaChannelFormatDesc channelDesc4 = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(dn_cha_per_call, &nc, sizeof(int)));
	cudaBindTextureToArray(t_a_QN, d_QN, channelDesc4);
	stress_calc<<<(nc + tpb_chain_kernel - 1) / tpb_chain_kernel, tpb_chain_kernel>>>(gpu_chain_heads, d_dt, d_offset, d_new_strent);
	CUT_CHECK_ERROR("kernel execution failed");
	cudaUnbindTexture(t_a_QN);

	float4 *stress_buf = new float4[nc * 2];
	cudaMemcpyFromArray(stress_buf, d_stress, 0, 0, nc * sizeof(float4) * 2, cudaMemcpyDeviceToHost);
	float4 sum_stress = make_float4(0.0f, 0.0f, 0.0f, 0.0f); //stress: xx,yy,zz,xy
	float4 sum_stress2 = make_float4(0.0f, 0.0f, 0.0f, 0.0f); //stress: yz,xz; Lpp, Ree
	scalar_chains* tchain_heads;
	tchain_heads = new scalar_chains[nc];

	cudaMemcpy(tchain_heads, gpu_chain_heads, sizeof(scalar_chains) * nc, cudaMemcpyDeviceToHost);
	int chain_count = nc;
	for (int j = 0; j < nc; j++) {
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

void ensemble_block::free_block() {    //free memory
	cudaFree(chains.QN);
	cudaFree(chains.tau_CD);
	cudaFree(chains.R1);

	delete[] chain_heads;

	cudaFree(gpu_chain_heads);
	cudaFreeArray(d_QN);
	cudaFreeArray(d_tCD);
	cudaFreeArray(d_R1);

	cudaFree(d_dt);
	cudaFree(reach_flag);
	cudaFree(d_offset);
	cudaFree(d_new_strent);
	cudaFree(d_new_tau_CD);

	if (corr != NULL) {
		cudaFree(d_correlator_time);
		delete corr;
	}
}

void ensemble_block::activate_block() {
	//prepares block for performing time evolution
	//i.e. copies chain conformations to working memory
	cudaChannelFormatDesc channelDesc1 = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	cudaChannelFormatDesc channelDesc4 = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(dn_cha_per_call, &nc, sizeof(int)));

	float tf = block_time;
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_universal_time, &tf, sizeof(float)));

	if (!(steps_count & 0x00000001)) {
		cudaMemcpy2DToArray(d_a_QN, 0, 0, chains.QN, z_max * sizeof(float) * 4, z_max * sizeof(float) * 4, nc, cudaMemcpyDeviceToDevice);
		cudaMemcpy2DToArray(d_a_tCD, 0, 0, chains.tau_CD, z_max * sizeof(float), z_max * sizeof(float), nc, cudaMemcpyDeviceToDevice);
		cudaMemcpyToArray(d_a_R1, 0, 0, chains.R1, sizeof(float) * 4 * nc, cudaMemcpyDeviceToDevice);
	} else {
		cudaMemcpy2DToArray(d_b_QN, 0, 0, chains.QN, z_max * sizeof(float) * 4, z_max * sizeof(float) * 4, nc, cudaMemcpyDeviceToDevice);
		cudaMemcpy2DToArray(d_b_tCD, 0, 0, chains.tau_CD, z_max * sizeof(float), z_max * sizeof(float), nc, cudaMemcpyDeviceToDevice);
		cudaMemcpyToArray(d_b_R1, 0, 0, chains.R1, sizeof(float) * 4 * nc, cudaMemcpyDeviceToDevice);
	}
	cudaDeviceSynchronize();
	//correlator binding
	cudaBindSurfaceToArray(s_correlator, corr->gpu_corr.d_correlator);
}

void ensemble_block::deactivate_block() {
	//copies chain conformations to storing memory

	if (!(steps_count & 0x00000001)) {
		cudaMemcpy2DFromArray(chains.QN, sizeof(float) * z_max * 4, d_a_QN, 0, 0, z_max * sizeof(float) * 4, nc, cudaMemcpyDeviceToDevice);
		cudaMemcpy2DFromArray(chains.tau_CD, sizeof(float) * z_max, d_a_tCD, 0, 0, z_max * sizeof(float), nc, cudaMemcpyDeviceToDevice);
		cudaMemcpyFromArray(chains.R1, d_a_R1, 0, 0, sizeof(float) * 4 * nc, cudaMemcpyDeviceToDevice);
	} else {
		cudaMemcpy2DFromArray(chains.QN, sizeof(float) * z_max * 4, d_b_QN, 0, 0, z_max * sizeof(float) * 4, nc, cudaMemcpyDeviceToDevice);
		cudaMemcpy2DFromArray(chains.tau_CD, sizeof(float) * z_max, d_b_tCD, 0, 0, z_max * sizeof(float), nc, cudaMemcpyDeviceToDevice);
		cudaMemcpyFromArray(chains.R1, d_b_R1, 0, 0, sizeof(float) * 4 * nc, cudaMemcpyDeviceToDevice);
	}
	cudaDeviceSynchronize();
}

