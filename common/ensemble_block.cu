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
#include "timer.h"
#define max_sync_interval 1 //Doi-Takimoto requires synchronising every timestep

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
int *d_value_found;
int* d_shift_found;
float* d_add_rand;
int* d_end_list;		// binary list of creations/destructions at the end of each arm in ensemble
int* d_end_counter;		// counter for d_end_list
int* d_destroy_list;	// list of paired chains for destroyed entanglements at the end of each arm in ensemble
int* d_destroy_counter;
int* d_destroy_list_2;	// list of paired chains for destroyed entanglements at the end of each arm in ensemble
int* d_destroy_counter_2;
int* d_create_counter;
int* d_doi_weights;

// these arrays used by time evolution kernels
cudaArray* d_a_QN; //device arrays for vector part of chain conformations
cudaArray* d_a_tCD;
cudaArray* d_a_tcr;
cudaArray* d_a_pair_chains;
cudaArray* d_b_QN;
cudaArray* d_b_tCD;
cudaArray* d_b_tcr;
cudaArray* d_b_pair_chains;
cudaArray* d_a_R1;
cudaArray* d_b_R1;
cudaArray* d_corr_a;
cudaArray* d_corr_b;
cudaArray* d_ft;
cudaArray* d_arm_index;

cudaArray* d_sum_W; // sum of probabilities for each entanglement
cudaArray* d_sum_W_sorted;
cudaArray* d_stress; // stress calculation temp array

// There are two arrays with the vector part of the chain conformations  on device.
// And there is only one array with the scalar part of the chain conformations
// Every timestep the vector part is copied from one array to another.
// The coping is done in entanglement parallel portion of the code
// This allows to use textures/surfaces and speeds up memory access
// The scalar part(chain headers) is updated in the chain parallel portion of the code
// Chain headers occupy less memory,and there are no specific memory access technics for them.

//random shuffle
template< class RandomIt >
void random_s(RandomIt first, RandomIt last, Ran* eran)
{
	typename std::iterator_traits<RandomIt>::difference_type i, n;
	n = last - first;
	for (i = n - 1; i > 0; --i) {
		using std::swap;
		swap(first[i], first[(int)(eran->flt()* RAND_MAX) % (i + 1)]);
	}
}

void ensemble_block::init(int nc_, vector_chains chains_, scalar_chains* chain_heads_, int nsteps_){
	//allocates arrays, copies chain conformations to device
	//ensemble_call_block *cb pointer for call block structure, just ref parameter
	//int nc  is a number of chains in this ensemble_call_block.
	//sstrentp chains, chain_head* chain_heads pointers to array with the chain conformations

	nc = nc_;
	nsteps = nsteps_;
	chains = chains_;
	chain_heads = chain_heads_;
	block_time = universal_time;

	//blank dynamics probabalities
	float *buffer = new float[4 * z_max * nc];
	memset(buffer, 0, sizeof(float4) * z_max * nc);
	cudaMemcpy2DToArray(d_sum_W, 0, 0, buffer, z_max * sizeof(int), z_max * sizeof(int), nc, cudaMemcpyHostToDevice);
	delete[] buffer;

	float *buffer3 = new float[z_max * nc];
	memset(buffer3, 0, sizeof(float) * z_max * nc);
	cudaMemcpy2DToArray(d_sum_W_sorted, 0, 0, buffer3, z_max * sizeof(float), z_max * sizeof(float), nc, cudaMemcpyHostToDevice);
	delete[] buffer3;

	float *buffer2 = new float[nc];
	memset(buffer2, 0, sizeof(float) * nc);
	cudaMemcpyToArray(d_ft, 0, 0, buffer2, nc * sizeof(float), cudaMemcpyHostToDevice);
	delete[] buffer2;

	// allocating device arrays
	cudaMalloc(&d_dt, sizeof(float) * nc);
	cudaMemset(d_dt, 0, sizeof(float) * nc);

	cudaMalloc(&d_offset, sizeof(int) * nc * narms);
	cudaMemset(d_offset, 0xff, sizeof(float) * nc * narms);

	cudaMalloc(&reach_flag, sizeof(float) * nc);
	cudaMalloc(&d_new_strent, sizeof(float) * 4 * nc);
	cudaMalloc(&d_new_tau_CD, sizeof(float) * nc);
	cudaMalloc(&d_new_cr_time, sizeof(float) * nc);
	cudaMalloc(&d_new_pair, sizeof(float) * nc);

	//cudaMemset(d_new_pair, -1.0f, sizeof(float) * nc);

	int s = ceil(log((float)nsteps/correlator_size)/log(correlator_res)) + 1; //number of correlator levels
	cout << "number of correlator levels" << '\t' << s << '\n' << '\n';
	corr = new correlator(nc, s);//Allocate memory for c_correlator structure in cb
	cudaMalloc((void**) &d_write_time, sizeof(int) * nc);//Allocated memory on device for correlator time for every chain in block
	cudaMemset(d_write_time, 0, sizeof(int) * nc);	//Initialize d_correlator_time with zeros

//	cudaMallocManaged((void**)&(stress_average), sizeof(float4) * nsteps * nc); //4vectors for every tau_k for every chain
	CUT_CHECK_ERROR("kernel execution failed");
}

template<int type> int  ensemble_block::time_step(double reach_time, int correlator_type, bool* run_flag, int *progress_bar) {
	//bind textures/surfaces, perform time evolution, unbind textures/surfaces

	//Declare and create streams for parallel correlator update
	cudaStream_t stream_calc1;
	cudaStream_t stream_calc2;
	cudaStream_t stream_calc3;
	cudaStream_t stream_calc4;
	cudaStream_t stream_update;
	cudaStreamCreate(&stream_calc1);
	cudaStreamCreate(&stream_calc2);
	cudaStreamCreate(&stream_calc3);
	cudaStreamCreate(&stream_calc4);
	cudaStreamCreate(&stream_update);

	cudaChannelFormatDesc channelDesc1 = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat); //to read float4
	cudaChannelFormatDesc channelDesc4 = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat); //to read float

	//loop preparing
	dim3 dimBlock(tpb_strent_kernel/4, tpb_strent_kernel);
	dim3 dimGrid((z_max + dimBlock.x - 1) / dimBlock.x, (nc + dimBlock.y - 1) / dimBlock.y);

	dim3 dimBlockFlat(z_max, 1);
	dim3 dimGridFlat((z_max + dimBlockFlat.x - 1) / dimBlockFlat.x, nc);

	steps_count = 0;
	activate_block();

	float time_step_interval = reach_time - block_time;
	int number_of_syncs = int(floor((time_step_interval - 0.5) / max_sync_interval)) + 1;

	float *rtbuffer;
	cudaMallocHost(&rtbuffer, sizeof(float)*nc);

	int *tbuffer;
	cudaMallocHost(&tbuffer, sizeof(int)*nc);

	float *entbuffer;
	cudaMallocHost(&entbuffer, sizeof(float)*nc);

	std::vector<unsigned long long> enttime_bins (20000, 0);

	int Narms_ensemble = nc*narms;

	bool texture_flag = true;

	cudaBindSurfaceToArray(s_arm_index, d_arm_index);

	//random number generator for pairing chains
	Ran eran_2(2);
	//Initialize random
	eran_2.seed(narms * N_cha);

	//Loop begins
	for (int i_sync = 0; i_sync < number_of_syncs; i_sync++) {
		
		float sync_interval = max_sync_interval;
		if ((i_sync + 1) == number_of_syncs)
			sync_interval = time_step_interval - i_sync * max_sync_interval;

		bool reach_flag_all = false;
		cudaMemset(reach_flag, 0, sizeof(float) * nc);

		//update universal_time on device
		float tf = block_time + i_sync * max_sync_interval;

		//printf("\nStep %i, time %f", i_sync, tf);
		CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_universal_time, &tf, sizeof(float)));
		
		while (!reach_flag_all) {
			//cout << "\nSteps_count " << steps_count << " a/b " << !(steps_count & 0x00000001);
			if (!(steps_count & 0x00000001)) { //For odd number of steps

				cudaBindTextureToArray(t_a_QN, d_a_QN, channelDesc4);
				cudaBindSurfaceToArray(s_b_QN, d_b_QN);
				cudaBindTextureToArray(t_a_tCD, d_a_tCD, channelDesc1);
				cudaBindSurfaceToArray(s_b_tCD, d_b_tCD);
				cudaBindTextureToArray(t_a_tcr, d_a_tcr, channelDesc1);
				cudaBindSurfaceToArray(s_b_tcr, d_b_tcr);
				cudaBindTextureToArray(t_a_pair, d_a_pair_chains, channelDesc1);
				cudaBindSurfaceToArray(s_b_pair, d_b_pair_chains);
				cudaBindTextureToArray(t_a_R1, d_a_R1, channelDesc4);
				cudaBindSurfaceToArray(s_b_R1, d_b_R1);
			}
			else { //For even number of steps
				cudaBindTextureToArray(t_a_QN, d_b_QN, channelDesc4);
				cudaBindSurfaceToArray(s_b_QN, d_a_QN);
				cudaBindTextureToArray(t_a_tCD, d_b_tCD, channelDesc1);
				cudaBindSurfaceToArray(s_b_tCD, d_a_tCD);
				cudaBindTextureToArray(t_a_tcr, d_b_tcr, channelDesc1);
				cudaBindSurfaceToArray(s_b_tcr, d_a_tcr);
				cudaBindTextureToArray(t_a_pair, d_b_pair_chains, channelDesc1);
				cudaBindSurfaceToArray(s_b_pair, d_a_pair_chains);
				cudaBindTextureToArray(t_a_R1, d_b_R1, channelDesc4);
				cudaBindSurfaceToArray(s_b_R1, d_a_R1);
			}

			if (texture_flag == true) {
				cudaBindSurfaceToArray(s_corr, d_corr_b);
			}
			else {
				cudaBindSurfaceToArray(s_corr, d_corr_a);
			}

			strent_kernel<type> <<<dimGrid, dimBlock, 0, stream_calc1>>> (chain_heads, d_dt, d_offset, d_new_strent, d_new_tau_CD, d_new_cr_time, d_new_pair);
			CUT_CHECK_ERROR("kernel execution failed");
			boundary2_kernel<3> <<<(Narms_ensemble + tpb_chain_kernel - 1) / tpb_chain_kernel, tpb_chain_kernel, 0, stream_calc2 >>> (chain_heads, d_offset, d_new_strent, d_new_tau_CD);
			CUT_CHECK_ERROR("kernel execution failed");
			boundary1_kernel <<<(Narms_ensemble + tpb_chain_kernel - 1) / tpb_chain_kernel, tpb_chain_kernel, 0, stream_calc3 >>> (chain_heads, d_offset, d_new_strent);
			CUT_CHECK_ERROR("kernel execution failed");
			cudaStreamSynchronize(stream_calc2);
			cudaStreamSynchronize(stream_calc3);

			chain_control_kernel<type> <<<(nc + tpb_chain_kernel - 1) / tpb_chain_kernel, tpb_chain_kernel, 0, stream_calc4 >>> (chain_heads, d_dt, reach_flag, sync_interval, d_offset, d_new_strent, d_write_time, correlator_type, steps_count % stressarray_count);
			CUT_CHECK_ERROR("kernel execution failed");

			scan_kernel <<<dimGridFlat, dimBlockFlat, 2 * z_max * sizeof(int), stream_calc1 >>> (chain_heads, d_rand_used, d_value_found, d_shift_found, d_add_rand);
			CUT_CHECK_ERROR("kernel execution failed");
			
			cudaMemcpyAsync(rtbuffer, reach_flag, sizeof(float) * nc, cudaMemcpyDeviceToHost, stream_calc4);
			cudaStreamSynchronize(stream_calc4);

			chain_kernel<type> <<<(nc + tpb_chain_kernel - 1) / tpb_chain_kernel, tpb_chain_kernel, 0, stream_calc1 >>> (chain_heads, d_dt, reach_flag, sync_interval, d_offset, d_new_strent, d_new_tau_CD, d_new_cr_time, d_write_time, correlator_type, d_rand_used, d_tau_CD_used_CD, d_tau_CD_used_SD, d_value_found, d_shift_found, d_add_rand, d_end_list, d_end_counter, d_destroy_list, d_destroy_counter, d_create_counter, d_new_pair);
			CUT_CHECK_ERROR("kernel execution failed");

			float sumrt = 0;
			for (int i = 0; i < nc; i++)
				sumrt += rtbuffer[i];
			reach_flag_all = (sumrt == nc);

			cudaUnbindTexture(t_a_QN);
			cudaUnbindTexture(t_a_tCD);
			cudaUnbindTexture(t_a_tcr);
			cudaUnbindTexture(t_a_pair);
			cudaUnbindTexture(t_a_R1);

			steps_count++;

//			copy entanglement lifetimes
			cudaMemcpyFromArrayAsync(entbuffer, d_ft, 0, 0, sizeof(float) * nc, cudaMemcpyDeviceToHost, stream_calc1);
			cudaStreamSynchronize(stream_calc1);
			for (int i = 0; i < nc; i++){
				if ((entbuffer[i]>0.0) && (entbuffer[i]<20.0)){
					enttime_bins[floor(entbuffer[i]*1000)]++;
				}
			}

			//Should be shared between blocks...

			// update progress bar
//			if (steps_count % 50 == 0) {
//				cudaStreamSynchronize(stream_calc);
//				cudaMemcpyAsync(tbuffer, d_write_time, sizeof(int) * nc, cudaMemcpyDeviceToHost, stream_calc);
//				cudaStreamSynchronize(stream_calc);
//				int sumt = 0;
//				for (int i = 0; i < nc; i++)
//					sumt += tbuffer[i];
//				*progress_bar = (int)(100.0f * sumt / (nc) / reach_time);
//				//cout << "\r" << *progress_bar << "%\t ";
//			}

			// check for reached time
//			cudaMemcpyAsync(rtbuffer, reach_flag, sizeof(float) * nc, cudaMemcpyDeviceToHost, stream_calc1);

			// check for rand refill
			if (steps_count % uniformrandom_count == 0) {
				random_textures_refill(nc, 0);
				steps_count = 0;
			}

			if (steps_count % stressarray_count == 0) {
				cudaStreamSynchronize(stream_calc1);
				cudaUnbindTexture(t_corr);
				if (texture_flag == true) {
					cudaBindTextureToArray(t_corr, d_corr_b, channelDesc4);
					texture_flag = false;
				}
				else {
					cudaBindTextureToArray(t_corr, d_corr_a, channelDesc4);
					texture_flag = true;
				}
				if (type == 0 && correlator_type == 0) {
					update_correlator <<<(nc + tpb_chain_kernel - 1) / tpb_chain_kernel, tpb_chain_kernel, 0, stream_update >>>((corr)->gpu_corr, stressarray_count, correlator_type);
				}
				if (correlator_type == 1 || correlator_type == 2) {
					flow_stress <<<(nc + tpb_chain_kernel - 1) / tpb_chain_kernel, tpb_chain_kernel, 0, stream_update >>>((corr)->gpu_corr, stressarray_count, stress_average, nc);
				}
			}

			// stop, if run_flag is changed from outside
			if (*run_flag == false)
				return -1;
		}
		//print (for dynamic pairing)
		cudaStreamSynchronize(stream_calc1);
		
		cudaMemset(d_end_list, 0, sizeof(int) * nc * narms);
		cudaMemset(d_end_counter, 0, sizeof(int) * nc * narms);

		cudaMemset(d_destroy_list_2, 0, sizeof(int) * nc * 10);
		cudaMemset(d_destroy_counter_2, 0, sizeof(int) * nc);

		for (int i = 0; i < nc; i++) {
			for (int arm = 0; arm < narms; arm++) {
				for (int k = 0; k < d_destroy_counter[narms * i + arm]; k++) {
					int ch = d_destroy_list[10 * (narms*i + arm) + k];
					d_destroy_list_2[10 * ch + d_destroy_counter_2[ch]] = i;
					d_destroy_counter_2[ch]++;
				}
			}
		}
		
		int n_destroy_iterations = 0;
		for (int i = 0; i < nc; i++) {
			if (d_destroy_counter_2[i] > n_destroy_iterations)
				n_destroy_iterations = d_destroy_counter_2[i];
		}
		n_destroy_iterations = n_destroy_iterations + n_destroy_iterations % 2;

		cudaMemset(d_destroy_list, 0, sizeof(int) * nc * narms * 10);
		cudaMemset(d_destroy_counter, 0, sizeof(int) * nc * narms);

		std::vector<std::pair<int, int> > NewPairs;
		//Number of new entanglements created by every arm in the ensemble
		//cout << "Number of new entanglements created by every arm in the ensemble\n";
		for (int i = 0; i < nc; i++) {
			int create_counter = 0;
			for (int arm = 0; arm < narms; arm++)
				create_counter += d_create_counter[i*narms + arm];
				//cout << "Chain " << i << "\tCounter " << create_counter << "\n";
			for (int c = 0; c < create_counter; c++) {
				NewPairs.push_back(std::make_pair(i, -1));
			}
		}
		cudaMemset(d_create_counter, 0, sizeof(int) * nc * narms);

		random_s(NewPairs.begin(), NewPairs.end(), &eran_2);

		//cout << "Now making " << n_destroy_iterations << " destroy iterations\n";
		// Destroy second parts of entanglement pairs
		int add_steps_count = 0;
		while (add_steps_count < n_destroy_iterations) {
			//cout << "\nIteration " << add_steps_count << "a/b " << !((steps_count + add_steps_count) & 0x00000001) << "\n";
			if (!((steps_count + add_steps_count) & 0x00000001)) { //For odd number of steps

				cudaBindTextureToArray(t_a_QN, d_a_QN, channelDesc4);
				cudaBindSurfaceToArray(s_b_QN, d_b_QN);
				cudaBindTextureToArray(t_a_tCD, d_a_tCD, channelDesc1);
				cudaBindSurfaceToArray(s_b_tCD, d_b_tCD);
				cudaBindTextureToArray(t_a_tcr, d_a_tcr, channelDesc1);
				cudaBindSurfaceToArray(s_b_tcr, d_b_tcr);
				cudaBindTextureToArray(t_a_pair, d_a_pair_chains, channelDesc1);
				cudaBindSurfaceToArray(s_b_pair, d_b_pair_chains);
				cudaBindTextureToArray(t_a_R1, d_a_R1, channelDesc4);
				cudaBindSurfaceToArray(s_b_R1, d_b_R1);
			}
			else { //For even number of steps
				cudaBindTextureToArray(t_a_QN, d_b_QN, channelDesc4);
				cudaBindSurfaceToArray(s_b_QN, d_a_QN);
				cudaBindTextureToArray(t_a_tCD, d_b_tCD, channelDesc1);
				cudaBindSurfaceToArray(s_b_tCD, d_a_tCD);
				cudaBindTextureToArray(t_a_tcr, d_b_tcr, channelDesc1);
				cudaBindSurfaceToArray(s_b_tcr, d_a_tcr);
				cudaBindTextureToArray(t_a_pair, d_b_pair_chains, channelDesc1);
				cudaBindSurfaceToArray(s_b_pair, d_a_pair_chains);
				cudaBindTextureToArray(t_a_R1, d_b_R1, channelDesc4);
				cudaBindSurfaceToArray(s_b_R1, d_a_R1);
			}

			cudaMemset(d_value_found, -1, sizeof(int) * nc);
			strent_doi_sync_kernel<type> <<<dimGrid, dimBlock, 0, stream_calc1 >>> (chain_heads, d_dt, d_offset, d_new_strent, d_new_tau_CD, d_new_cr_time, d_new_pair, d_destroy_list_2, d_destroy_counter_2, d_value_found, add_steps_count);
			CUT_CHECK_ERROR("kernel execution failed");
			chain_doi_destroy_kernel<type> <<<(nc + tpb_chain_kernel - 1) / tpb_chain_kernel, tpb_chain_kernel, 0, stream_calc1 >>> (chain_heads, d_dt, sync_interval, d_offset, d_new_strent, d_new_tau_CD, d_new_cr_time, d_value_found, d_end_list, d_end_counter, d_destroy_list_2, d_destroy_counter_2);
			CUT_CHECK_ERROR("kernel execution failed");
			
			cudaUnbindTexture(t_a_QN);
			cudaUnbindTexture(t_a_tCD);
			cudaUnbindTexture(t_a_tcr);
			cudaUnbindTexture(t_a_pair);
			cudaUnbindTexture(t_a_R1);
			add_steps_count++;
		}
		
		cudaStreamSynchronize(stream_calc1);
		// Search for pairs for new entanglements
		add_steps_count = 0;

		//initialize weights
		chain_doi_initial_weights <<<(nc + tpb_chain_kernel - 1) / tpb_chain_kernel, tpb_chain_kernel, 0, stream_calc1 >>> (chain_heads, d_doi_weights);
		CUT_CHECK_ERROR("kernel execution failed");

		//cout << "\na/b " << !((steps_count + add_steps_count) & 0x00000001);
		if (!((steps_count + add_steps_count) & 0x00000001)) { //For odd number of steps

			cudaBindTextureToArray(t_a_QN, d_a_QN, channelDesc4);
			cudaBindSurfaceToArray(s_b_QN, d_b_QN);
			cudaBindTextureToArray(t_a_tCD, d_a_tCD, channelDesc1);
			cudaBindSurfaceToArray(s_b_tCD, d_b_tCD);
			cudaBindTextureToArray(t_a_tcr, d_a_tcr, channelDesc1);
			cudaBindSurfaceToArray(s_b_tcr, d_b_tcr);
			cudaBindTextureToArray(t_a_pair, d_a_pair_chains, channelDesc1);
			cudaBindSurfaceToArray(s_b_pair, d_b_pair_chains);
			cudaBindTextureToArray(t_a_R1, d_a_R1, channelDesc4);
			cudaBindSurfaceToArray(s_b_R1, d_b_R1);
		}
		else { //For even number of steps
			cudaBindTextureToArray(t_a_QN, d_b_QN, channelDesc4);
			cudaBindSurfaceToArray(s_b_QN, d_a_QN);
			cudaBindTextureToArray(t_a_tCD, d_b_tCD, channelDesc1);
			cudaBindSurfaceToArray(s_b_tCD, d_a_tCD);
			cudaBindTextureToArray(t_a_tcr, d_b_tcr, channelDesc1);
			cudaBindSurfaceToArray(s_b_tcr, d_a_tcr);
			cudaBindTextureToArray(t_a_pair, d_b_pair_chains, channelDesc1);
			cudaBindSurfaceToArray(s_b_pair, d_a_pair_chains);
			cudaBindTextureToArray(t_a_R1, d_b_R1, channelDesc4);
			cudaBindSurfaceToArray(s_b_R1, d_a_R1);
		}

		//restrict double pairing of chains
		strent_doi_sync_2_kernel<type> <<<dimGrid, dimBlock, 0, stream_calc1 >>> (chain_heads, d_dt, d_offset, d_new_strent, d_new_tau_CD, d_new_cr_time, d_new_pair, d_doi_weights);
		CUT_CHECK_ERROR("kernel execution failed");

		cudaStreamSynchronize(stream_calc1);

		//cout << "\nCreating new pairs";

		cudaMemset(d_destroy_list_2, 0, sizeof(int) * nc * 10);
		cudaMemset(d_destroy_counter_2, 0, sizeof(int) * nc);

		for (unsigned pair = 0; pair < NewPairs.size(); pair++) {
			d_destroy_counter_2[NewPairs[pair].first]++;
		}

		int n_create_iterations = 0;
		for (int i = 0; i < nc; i++) {
			if (d_destroy_counter_2[i] > n_create_iterations)
				n_create_iterations = d_destroy_counter_2[i];
			//cout << "\nCreate " << d_destroy_counter_2[i] << " pairs with chain " << i;
		}
		//cout << "\nNeed iterations: " << n_create_iterations;

		cudaStreamSynchronize(stream_calc1);
		//scan_weights_kernel
		for (int counter = 0; counter < n_create_iterations; counter++) {
			chain_doi_scan_weights << <(nc + tpb_chain_kernel - 1) / tpb_chain_kernel, tpb_chain_kernel, 0, stream_calc1 >> > (chain_heads, d_rand_used, d_doi_weights, d_destroy_list_2, d_destroy_counter_2, counter);
		}
		cudaStreamSynchronize(stream_calc1);
	
		//update weights
		//for (int i = 0; i < nc; i++) {
		//	if (d_doi_weights[nc*new_dynamic_pair + i] != 0)
		//		d_doi_weights[nc*new_dynamic_pair + i]--;
		//}
		//d_doi_weights[nc*NewPairs[pair].first + new_dynamic_pair] = 0;
		//d_doi_weights[nc*new_dynamic_pair + NewPairs[pair].first] = 0;

		//apply found pairs to the first part of pair
		chain_doi_label_pairs_kernel<type> <<<(nc + tpb_chain_kernel - 1) / tpb_chain_kernel, tpb_chain_kernel, 0, stream_calc1 >>> (chain_heads, d_dt, sync_interval, d_offset, d_new_strent, d_new_tau_CD, d_new_cr_time, d_new_pair, d_rand_used, d_value_found, d_end_list, d_end_counter, d_destroy_list_2, d_destroy_counter_2, d_doi_weights);
		CUT_CHECK_ERROR("kernel execution failed");

		cudaUnbindTexture(t_a_QN);
		cudaUnbindTexture(t_a_tCD);
		cudaUnbindTexture(t_a_tcr);
		cudaUnbindTexture(t_a_pair);
		cudaUnbindTexture(t_a_R1);

		//cudaDeviceSynchronize();
		cudaStreamSynchronize(stream_calc1);
		//creeate list of pairs to link
		//cout << "\nCreate new entanglements to complete the pair on the following chains:\n";
		cudaMemset(d_destroy_list_2, 0, sizeof(int) * nc * 10);
		cudaMemset(d_destroy_counter_2, 0, sizeof(int) * nc);

		for (unsigned pair = 0; pair < NewPairs.size(); pair++) {
			d_destroy_list_2[10 * NewPairs[pair].second + d_destroy_counter_2[NewPairs[pair].second]]=NewPairs[pair].first;
			d_destroy_counter_2[NewPairs[pair].second]++;
		}

		//int n_create_iterations = 0;
		//for (int i = 0; i < nc; i++) {
		//	//cout << "\nCreate pair on chain " << i << " with ";
		//	//for (int c = 0; c < d_destroy_counter_2[i]; c++) {
		//	//	cout << d_destroy_list_2[10 * i + c] << " ";
		//	//}
		//	//cout << "\n";
		//	if (d_destroy_counter_2[i] > n_create_iterations)
		//		n_create_iterations = d_destroy_counter_2[i];
		//}
		n_create_iterations = n_create_iterations + 3 + (n_create_iterations + 1) % 2;
		add_steps_count++;

		//cudaDeviceSynchronize();
		cudaStreamSynchronize(stream_calc1);
		//start creating second halves of pairs
		//cout << "\nNow making " << n_create_iterations << " create iterations\n";
		while (add_steps_count < n_create_iterations) {
			//cout << "\nIteration " << add_steps_count << " a/b " << !((steps_count + add_steps_count) & 0x00000001) << "\n";
			if (!((steps_count + add_steps_count) & 0x00000001)) { //For odd number of steps

				cudaBindTextureToArray(t_a_QN, d_a_QN, channelDesc4);
				cudaBindSurfaceToArray(s_b_QN, d_b_QN);
				cudaBindTextureToArray(t_a_tcr, d_a_tcr, channelDesc1);
				cudaBindSurfaceToArray(s_b_tcr, d_b_tcr);
				cudaBindTextureToArray(t_a_pair, d_a_pair_chains, channelDesc1);
				cudaBindSurfaceToArray(s_b_pair, d_b_pair_chains);
			}
			else { //For even number of steps
				cudaBindTextureToArray(t_a_QN, d_b_QN, channelDesc4);
				cudaBindSurfaceToArray(s_b_QN, d_a_QN);
				cudaBindTextureToArray(t_a_tcr, d_b_tcr, channelDesc1);
				cudaBindSurfaceToArray(s_b_tcr, d_a_tcr);
				cudaBindTextureToArray(t_a_pair, d_b_pair_chains, channelDesc1);
				cudaBindSurfaceToArray(s_b_pair, d_a_pair_chains);
			}
			strent_doi_sync_3_kernel<type> <<< dimGrid, dimBlock, 0, stream_calc1 >>> (chain_heads, d_dt, d_offset, d_new_strent, d_new_tau_CD, d_new_cr_time, d_new_pair);
			CUT_CHECK_ERROR("kernel execution failed");
			chain_doi_create_kernel<type> <<<(nc + tpb_chain_kernel - 1) / tpb_chain_kernel, tpb_chain_kernel, 0, stream_calc1 >>> (chain_heads, d_dt, sync_interval, d_offset, d_new_strent, d_new_tau_CD, d_new_cr_time, d_rand_used, d_tau_CD_used_CD, d_destroy_list_2, d_destroy_counter_2, add_steps_count-1, d_new_pair);
			CUT_CHECK_ERROR("kernel execution failed");
			//cudaDeviceSynchronize();
			cudaStreamSynchronize(stream_calc1);
			cudaUnbindTexture(t_a_QN);
			cudaUnbindTexture(t_a_tCD);
			cudaUnbindTexture(t_a_tcr);
			cudaUnbindTexture(t_a_pair);
			cudaUnbindTexture(t_a_R1);

			add_steps_count++;
		}

	}	//loop ends


	if (type==0){
		if (steps_count % stressarray_count != 0) {
			cudaStreamSynchronize(stream_calc1);
			cudaStreamSynchronize(stream_update);
			cudaUnbindTexture(t_corr);
			if (texture_flag==true){
				cudaBindTextureToArray(t_corr, d_corr_b, channelDesc4);
				texture_flag = false;
			} else {
				cudaBindTextureToArray(t_corr, d_corr_a, channelDesc4);
				texture_flag = true;
			}
			if (type==0 && correlator_type==0){
				update_correlator<<<(nc + tpb_chain_kernel - 1) / tpb_chain_kernel, tpb_chain_kernel,0,stream_update>>>((corr)->gpu_corr, steps_count, correlator_type);
			}
			if (correlator_type==1 || correlator_type==2){
				flow_stress<<<(nc + tpb_chain_kernel - 1) / tpb_chain_kernel, tpb_chain_kernel,0,stream_update>>>((corr)->gpu_corr, steps_count, stress_average, nc);
			}
		}
	}
	block_time = reach_time;
	cudaHostUnregister(rtbuffer);
	cudaFreeHost(rtbuffer);
	deactivate_block();
	cudaStreamDestroy(stream_update);
	cudaStreamDestroy(stream_calc1);

	//output entanglement lifetime distribution
	unsigned long long enttime_sum = 0;
	for (int it = 0; it < enttime_bins.size(); ++it) {
		enttime_sum += enttime_bins[it];
	}

	ofstream lifetime_file;
	lifetime_file.open(filename_ID("fdt", false));
	//unsigned long long enttime_run_sum = 0;
	for (int it = 0; it < enttime_bins.size(); ++it){
		if (enttime_bins[it] != 0){
			//enttime_run_sum += enttime_bins[it];
			//lifetime_file << powf(10.0f,it/1000.0f-10.0f) << '\t' << 1.0f - (float)enttime_run_sum / (float)enttime_sum << '\n';
			lifetime_file << it << '\t' << enttime_bins[it] << '\n';
		}
	}
	lifetime_file.close();
	return 0;
}

int ensemble_block::equilibrium_calc(double length, int correlator_type, bool* run_flag, int *progress_bar, int np, float* t, float* x){
	transfer_to_device();
	cudaMemset(d_write_time, 0, sizeof(int) * nc);
	if(time_step<0>(length, correlator_type, run_flag, progress_bar)==-1) return -1;
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
	cudaDeviceSynchronize();

	//blank delayed dynamics arrays
	cudaMemset(d_dt, 0, sizeof(float) * nc);
	cudaMemset(d_offset, 0xff, sizeof(float) * nc * narms);

}

void ensemble_block::transfer_from_device() { //copy chains back
	cudaDeviceSynchronize();

	//NOTE accounts for delayed dynamics

	//delayed dynamics
	int *h_offset = new int[nc*narms];
	float4 *h_new_strent = new float4[nc];
	float *h_new_tau_CD = new float[nc];
	cudaMemcpy(h_offset, d_offset, sizeof(int) * nc * narms, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_new_strent, d_new_strent, sizeof(float4) * nc, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_new_tau_CD, d_new_tau_CD, sizeof(float) * nc, cudaMemcpyDeviceToHost);
	int run_sum;
	for (int i = 0; i < nc; i++) {
		run_sum = 0;
		for (int arm=0; arm<narms; arm++){
			if (hoffset_dir(h_offset[i*narms+arm]) == -1) {
				for (int j = NK_arms[arm] - 1; j > 0; j--) {
					chains.QN[i * z_max + run_sum + j] = chains.QN[i * z_max + hmake_offset(j + run_sum, h_offset[i*narms+arm])];
					chains.tau_CD[i * z_max + run_sum + j] = chains.tau_CD[i * z_max + hmake_offset(j + run_sum, h_offset[i*narms+arm])];
				}
				chains.QN[i * z_max + run_sum + hoffset_index(h_offset[i*narms+arm])] = h_new_strent[i];
				chains.tau_CD[i * z_max + run_sum + hoffset_index(h_offset[i*narms+arm])] = h_new_tau_CD[i];
			} else {
				for (int j = 0; j < NK_arms[arm] - 2; j++) {
					chains.QN[i * z_max + run_sum + j] = chains.QN[i * z_max + hmake_offset(j + run_sum, h_offset[i*narms+arm])];
					chains.tau_CD[i * z_max + run_sum + j] = chains.tau_CD[i * z_max + hmake_offset(j + run_sum, h_offset[i*narms+arm])];
				}
			}
			run_sum += NK_arms[arm];
		}
	}

	delete[] h_offset;
	delete[] h_new_strent;
	delete[] h_new_tau_CD;
}

stress_plus ensemble_block::calc_stress(int *r_chain_count) {

	cudaChannelFormatDesc channelDesc4 = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(dn_cha_per_call, &nc, sizeof(int)));
	cudaMemcpy2DToArray(d_a_QN, 0, 0, chains.QN, z_max * sizeof(float) * 4, z_max * sizeof(float) * 4, nc, cudaMemcpyDeviceToDevice);
	cudaDeviceSynchronize();
	cudaBindTextureToArray(t_a_QN, d_a_QN, channelDesc4);
	stress_calc<<<(nc + tpb_chain_kernel - 1) / tpb_chain_kernel, tpb_chain_kernel>>>(chain_heads, d_dt, d_offset, d_new_strent, chains.QN, z_max * 4 * nc);
	CUT_CHECK_ERROR("kernel execution failed");
	cudaUnbindTexture(t_a_QN);
	cudaMemcpy2DFromArray(chains.QN, sizeof(float) * z_max * 4, d_a_QN, 0, 0, z_max * sizeof(float) * 4, nc, cudaMemcpyDeviceToDevice);
	cudaDeviceSynchronize();

	float4 *stress_buf = new float4[nc * 2];
	cudaMemcpyFromArray(stress_buf, d_stress, 0, 0, nc * sizeof(float4) * 2, cudaMemcpyDeviceToHost);
	float4 sum_stress = make_float4(0.0f, 0.0f, 0.0f, 0.0f); //stress: xx,yy,zz,xy
	float4 sum_stress2 = make_float4(0.0f, 0.0f, 0.0f, 0.0f); //stress: yz,xz; Lpp, Ree
	scalar_chains* tchain_heads;
	tchain_heads = new scalar_chains[nc];

	cudaMemcpy(tchain_heads, chain_heads, sizeof(scalar_chains) * nc, cudaMemcpyDeviceToHost);
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
		cudaMemcpy2DToArray(d_a_tcr, 0, 0, chains.tau_cr, z_max * sizeof(float), z_max * sizeof(float), nc, cudaMemcpyDeviceToDevice);
		cudaMemcpy2DToArray(d_a_pair_chains, 0, 0, chains.pair_chain, z_max * sizeof(int), z_max * sizeof(int), nc, cudaMemcpyDeviceToDevice);
		cudaMemcpyToArray(d_a_R1, 0, 0, chains.R1, sizeof(float) * 4 * nc, cudaMemcpyDeviceToDevice);
	} else {
		cudaMemcpy2DToArray(d_b_QN, 0, 0, chains.QN, z_max * sizeof(float) * 4, z_max * sizeof(float) * 4, nc, cudaMemcpyDeviceToDevice);
		cudaMemcpy2DToArray(d_b_tCD, 0, 0, chains.tau_CD, z_max * sizeof(float), z_max * sizeof(float), nc, cudaMemcpyDeviceToDevice);
		cudaMemcpy2DToArray(d_b_tcr, 0, 0, chains.tau_cr, z_max * sizeof(float), z_max * sizeof(float), nc, cudaMemcpyDeviceToDevice);
		cudaMemcpy2DToArray(d_b_pair_chains, 0, 0, chains.pair_chain, z_max * sizeof(int), z_max * sizeof(int), nc, cudaMemcpyDeviceToDevice);
		cudaMemcpyToArray(d_b_R1, 0, 0, chains.R1, sizeof(float) * 4 * nc, cudaMemcpyDeviceToDevice);
	}
	cudaDeviceSynchronize();
}

void ensemble_block::deactivate_block() {
	//copies chain conformations to storing memory

	if (!(steps_count & 0x00000001)) {
		cudaMemcpy2DFromArray(chains.QN, sizeof(float) * z_max * 4, d_a_QN, 0, 0, z_max * sizeof(float) * 4, nc, cudaMemcpyDeviceToDevice);
		cudaMemcpy2DFromArray(chains.tau_CD, sizeof(float) * z_max, d_a_tCD, 0, 0, z_max * sizeof(float), nc, cudaMemcpyDeviceToDevice);
		cudaMemcpy2DFromArray(chains.tau_cr, sizeof(float) * z_max, d_a_tcr, 0, 0, z_max * sizeof(float), nc, cudaMemcpyDeviceToDevice);
		cudaMemcpy2DFromArray(chains.pair_chain, sizeof(int) * z_max, d_a_pair_chains, 0, 0, z_max * sizeof(int), nc, cudaMemcpyDeviceToDevice);
		cudaMemcpyFromArray(chains.R1, d_a_R1, 0, 0, sizeof(float) * 4 * nc, cudaMemcpyDeviceToDevice);
	} else {
		cudaMemcpy2DFromArray(chains.QN, sizeof(float) * z_max * 4, d_b_QN, 0, 0, z_max * sizeof(float) * 4, nc, cudaMemcpyDeviceToDevice);
		cudaMemcpy2DFromArray(chains.tau_CD, sizeof(float) * z_max, d_b_tCD, 0, 0, z_max * sizeof(float), nc, cudaMemcpyDeviceToDevice);
		cudaMemcpy2DFromArray(chains.tau_cr, sizeof(float) * z_max, d_b_tcr, 0, 0, z_max * sizeof(float), nc, cudaMemcpyDeviceToDevice);
		cudaMemcpy2DFromArray(chains.pair_chain, sizeof(int) * z_max, d_b_pair_chains, 0, 0, z_max * sizeof(int), nc, cudaMemcpyDeviceToDevice);
		cudaMemcpyFromArray(chains.R1, d_b_R1, 0, 0, sizeof(float) * 4 * nc, cudaMemcpyDeviceToDevice);
	}
	cudaDeviceSynchronize();
}

ensemble_block::~ensemble_block() {    //free memory
	cudaFree(d_dt);
	cudaFree(reach_flag);
	cudaFree(d_offset);
	cudaFree(d_new_strent);
	cudaFree(d_new_tau_CD);
	cudaFree(d_new_cr_time);
	cudaFree(d_new_pair);

	cudaFree(d_write_time);

	if (corr != NULL) {
		delete corr;
	}
}
