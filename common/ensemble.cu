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

#include "gpu_random.h"
#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <vector>
#include <algorithm>
#include "ensemble.h"
#include "cudautil.h"
#include "cuda_call.h"
#include "textures_surfaces.h"
#include "math.h"
#include <sstream>
#include <iomanip>
#include <set>			// std::set
#include <algorithm>    // std::min

using namespace std;

extern char * filename_ID(string filename, bool temp);
extern float mp,Mk;
extern float step;
extern float* GEX_table;
extern float* GEXd_table;
extern bool PD_flag;
extern cudaArray* d_gamma_table;
extern cudaArray* d_gamma_table_d;

void random_textures_refill(int n_cha, cudaStream_t stream_calc);
void random_textures_fill(int n_cha);

#include "ensemble_block.cu"

#define chains_per_call 10000

vector_chains chains; // host chain conformations
// and only one array with scalar part chain conformation
// every time the vector part is copied from one array to another
// coping is done in entanglement parallel portion of the code
// this allows to use textures/surfaces, which speeds up memory access
// scalar part(chain headers) are update in the chain parallel portion of the code
// chain headers are occupied much smaller memory, no specific memory access technic are used for them.
// depending one odd or even number of time step were performed,
//one of the transfer_from_device_# should be used

scalar_chains* chain_heads; // host device chain headers arrays, store scalar variables of chain conformations

int chain_blocks_number;
ensemble_block *chain_blocks;

//host constants
double universal_time;//since chain_head do not store universal time due to SP issues
		      	  	  //see chain.h chain_head for explanation
int N_cha;
int NK;
int narms;
int* NK_arms;
int* indeces_arms;
int z_max;
float Be;
float kxx, kxy, kxz, kyx, kyy, kyz, kzx, kzy, kzz;
bool PD_flag=0;

bool dbug = false;

//navigation
vector_chains chain_index(const int i) { //absolute navigation i - is a global index of chains i:[0..N_cha-1]
	vector_chains ptr;
	ptr.QN = &(chains.QN[z_max * i]);
	ptr.tau_CD = &(chains.tau_CD[z_max * i]);
	ptr.tau_cr = &(chains.tau_cr[z_max * i]);
	ptr.pair_chain = &(chains.pair_chain[z_max * i]);
	ptr.R1 = &(chains.R1[i]);
	return ptr;
}
vector_chains chain_index_arm(const int i, const int arm) { //absolute navigation i - is a global index of chains i:[0..N_cha-1]
	vector_chains ptr;
	int shift=0;
	for (int k=0; k<arm; k++){
		shift += NK_arms[k];
	}
	ptr.QN = &(chains.QN[z_max * i + shift]);
	ptr.tau_CD = &(chains.tau_CD[z_max * i + shift]);
	ptr.tau_cr = &(chains.tau_cr[z_max * i + shift]);
	ptr.pair_chain = &(chains.pair_chain[z_max * i + shift]);
	ptr.R1 = &(chains.R1[i]);
	return ptr;
}

vector_chains chain_index(const int bi, const int i) {    //block navigation
	//bi is a block index bi :[0..chain_blocks_number]
	//i - is a chain index in the block bi  i:[0..chains_per_call-1]
	vector_chains ptr;
	ptr.QN = &(chains.QN[z_max * (bi * chains_per_call + i)]);
	ptr.tau_CD = &(chains.tau_CD[z_max * (bi * chains_per_call)]);
	ptr.tau_cr = &(chains.tau_cr[z_max * (bi * chains_per_call)]);
	ptr.pair_chain = &(chains.pair_chain[z_max * (bi * chains_per_call)]);
	ptr.R1 = &(chains.R1[bi * chains_per_call + i]);
	return ptr;
}

void chains_malloc() {
	z_max = NK;
	cudaMallocManaged((void**)&chain_heads, sizeof(scalar_chains) * N_cha);

	for (int k=0; k<N_cha; k++){
		chain_heads[k].stall_flag = 0;
		chain_heads[k].time = 0.0f;
		cudaMallocManaged((void**)&(chain_heads[k].Z), sizeof(int) * narms);
	}
	cudaMallocManaged((void**)&(chains.QN), sizeof(float4) * N_cha * z_max);
	cudaMallocManaged((void**)&(chains.tau_CD), sizeof(float4) * N_cha * z_max);
	cudaMallocManaged((void**)&(chains.tau_cr), sizeof(float4) * N_cha * z_max);
	cudaMallocManaged((void**)&(chains.pair_chain), sizeof(int) * N_cha * z_max);
	cudaMallocManaged((void**)&(chains.R1), sizeof(float4) * N_cha);
}

void host_chains_init(Ran* eran) {
	chains_malloc();
	cout << "generating chain conformations on host..";
	universal_time=0.0;
	std::vector<int> Zlist;
	std::vector<int> Entlist;
	std::set<std::pair<int, int>> Pairlist;
	for (int i = 0; i < N_cha; i++) {
		Zlist.push_back(0);

		for (int arm=0; arm<narms; arm++){
			chain_init(&(chain_heads[i].Z[arm]), chain_index_arm(i,arm), NK_arms[arm], NK_arms[arm], false, PD_flag, eran);
			Zlist.back() += chain_heads[i].Z[arm]-1;
		}

		float4 aver_branch_point=make_float4(0.0f, 0.0f, 0.0f, 0.0f);
		float aver_sum = 0.0f;
		for (int arm = 0; arm<narms; arm++) {
			aver_branch_point = aver_branch_point + chain_index_arm(i, arm).QN[0] / chain_index_arm(i, arm).QN[0].w;
			aver_sum += 1 / chain_index_arm(i, arm).QN[0].w;
		}
		aver_branch_point = aver_branch_point / aver_sum;

		for (int arm = 0; arm<narms; arm++) {
			converttoQhat(chain_index_arm(i, arm), aver_branch_point);
		}
	}

	int Z_total = 0;
	//cout << "Total number of entanglements: ";
	for (unsigned i = 0; i<Zlist.size(); i++)
		Z_total += Zlist[i];

	while (Z_total % 2 != 0) {
		Z_total -= Zlist.back();
		Zlist.back() = 0;

		//repeat initialization of the last chain

		for (int arm = 0; arm<narms; arm++) {
			chain_init(&(chain_heads[N_cha-1].Z[arm]), chain_index_arm(N_cha - 1, arm), NK_arms[arm], NK_arms[arm], false, PD_flag, eran);
			Zlist.back() += chain_heads[N_cha - 1].Z[arm] - 1;
		}

		float4 aver_branch_point = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
		float aver_sum = 0.0f;
		for (int arm = 0; arm<narms; arm++) {
			aver_branch_point = aver_branch_point + chain_index_arm(N_cha - 1, arm).QN[0] / chain_index_arm(N_cha - 1, arm).QN[0].w;
			aver_sum += 1 / chain_index_arm(N_cha - 1, arm).QN[0].w;
		}
		aver_branch_point = aver_branch_point / aver_sum;

		for (int arm = 0; arm<narms; arm++) {
			converttoQhat(chain_index_arm(N_cha - 1, arm), aver_branch_point);
		}
		Z_total += Zlist.back();
	}

	//cout << "Number of entanglements in ensemble chains:\n";
	//for (unsigned i = 0; i<Zlist.size(); i++)
	//	cout << ' ' << Zlist[i];
	//cout << '\n';
	//cout << Z_total << '\n';

	for (unsigned i = 0; i < Zlist.size(); i++) {
		for (unsigned j = 0; j < Zlist[i]; j++) {
			Entlist.push_back(i);
		}
	}

	//for (unsigned i = 0; i < Entlist.size(); i++) {
	//	cout << ' ' << Entlist[i];
	//}

	//randomly shuffle the list
	random_s(Entlist.begin(), Entlist.end(), eran);
	//cout << "Randomly shuffled list of entanglements:\n";
	//for (unsigned i = 0; i < Entlist.size(); i++) {
	//	cout << ' ' << Entlist[i];
	//}

	std::pair <int, int> p;

	while (Entlist.size() > 0) {
		int i = 0;
		int j = 1;
		bool found_pair = false;
		while (found_pair == false) {
			p = std::make_pair(std::min(Entlist[i], Entlist[j]), std::max(Entlist[i], Entlist[j]));
			if (Pairlist.count(p) == 0 && p.first!=p.second) {
				Pairlist.insert(p);
				//std::cout << "Pair: " << p.first << " " << p.second << std::endl;
				// erase first and j elements:
				Entlist.erase(Entlist.begin() + j);
				Entlist.erase(Entlist.begin());
				found_pair = true;
			}
			else {
				j++;
			}
		}
	}

	//Save pairing information to chains
	for (int i = 0; i < N_cha; i++) {
		//check pairs with other chains
		std::vector<int> Connectedlist;
		for (int j = 0; j < N_cha; j++) {
			p = std::make_pair(std::min(i,j), std::max(i, j));
			if (Pairlist.count(p) == 1)
				Connectedlist.push_back(j); //save chains (j) connected to this chain (i)
		}

		//randomly shuffle Connectedlist
		random_s(Connectedlist.begin(), Connectedlist.end(), eran);

		//cout << "Chain " << i << " connected to chains:";
		//for (unsigned k = 0; k < Connectedlist.size(); k++) {
		//	cout << ' ' << Connectedlist[k];
		//}
		//cout << "\n";
		
		//divide in N_arms parts
		int run_sum = 0;
		for (int arm = 0; arm < narms; arm++) {
			vector<int> Connectedarmlist(Connectedlist.begin() + run_sum, Connectedlist.begin() + run_sum + chain_heads[i].Z[arm]-1);
			//cout << "Arm " << arm << " connected to chains:";
			//for (unsigned k = 0; k < Connectedarmlist.size(); k++) {
			//	cout << ' ' << Connectedarmlist[k];
			//}
			//cout << "\n";
			
			//initialize pair_chain in vector_chains
			pair_chains(Connectedarmlist.data(), chain_heads[i].Z[arm], chain_index_arm(i,arm));
			
			run_sum += chain_heads[i].Z[arm]-1;
		}
		
	}

	cout << "done\n";
}

//preparation of constants/arrays/etc
void gpu_init(int seed, p_cd* pcd, int nsteps) {
	cout << "preparing GPU chain conformations..\n";

	//Copy host constants from host to device
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(dBe, &Be, sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(dnk, &NK, sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_z_max, &z_max, sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_z_max_arms, NK_arms, sizeof(int)*narms));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_indeces_arms, indeces_arms, sizeof(int)*narms));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(dnk_arms, NK_arms, sizeof(int)*narms));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_narms, &narms, sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_kappa_xx, &kxx, sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_kappa_xy, &kxy, sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_kappa_xz, &kxz, sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_kappa_yx, &kyx, sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_kappa_yy, &kyy, sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_kappa_yz, &kyz, sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_kappa_zx, &kzx, sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_kappa_zy, &kzy, sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_kappa_zz, &kzz, sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_CD_flag, &CD_flag, sizeof(int)));

	float cdtemp;
	if(PD_flag){
		//calculate probability prefactor for polydisperse simulations
		double tem = 0.0f;
		for (int i=0; i+1<1/step; i++){
			p_cd* t_pcd = new p_cd(Be, GEX_table[i]*mp/Mk, NULL);
			tem += (t_pcd->W_CD_destroy_aver());
			delete[] t_pcd;
		}
		cdtemp = step * tem / Be;
	}
	else
		cdtemp = pcd->W_CD_destroy_aver() / Be;

	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_CD_create_prefact, &cdtemp, sizeof(float)));
	cout << " device constants done\n";

	int rsz = chains_per_call;
	if (N_cha < chains_per_call)
		rsz = N_cha;
	cudaChannelFormatDesc channelDesc4 = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
	cudaChannelFormatDesc channelDesc1 = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

	cudaMalloc((void**) &d_value_found, sizeof(int) * rsz);
	cudaMalloc((void**) &d_shift_found, sizeof(int) * rsz);
	cudaMalloc((void**)&d_add_rand, sizeof(float) * rsz);

	cudaMallocArray(&d_a_QN, &channelDesc4, z_max, rsz, cudaArraySurfaceLoadStore);
	cudaMallocArray(&d_a_tCD, &channelDesc1, z_max, rsz, cudaArraySurfaceLoadStore);
	cudaMallocArray(&d_a_tcr, &channelDesc1, z_max, rsz, cudaArraySurfaceLoadStore);
	cudaMallocArray(&d_a_R1, &channelDesc4, rsz, 1, cudaArraySurfaceLoadStore);
	cudaMallocArray(&d_corr_a, &channelDesc4, rsz, stressarray_count, cudaArraySurfaceLoadStore);
	cudaMallocArray(&d_a_pair_chains, &channelDesc1, z_max, rsz, cudaArraySurfaceLoadStore);


	cudaMallocArray(&d_b_QN, &channelDesc4, z_max, rsz, cudaArraySurfaceLoadStore);
	cudaMallocArray(&d_b_tCD, &channelDesc1, z_max, rsz, cudaArraySurfaceLoadStore);
	cudaMallocArray(&d_b_tcr, &channelDesc1, z_max, rsz, cudaArraySurfaceLoadStore);
	cudaMallocArray(&d_b_R1, &channelDesc4, rsz, 1, cudaArraySurfaceLoadStore);
	cudaMallocArray(&d_corr_b, &channelDesc4, rsz, stressarray_count, cudaArraySurfaceLoadStore);
	cudaMallocArray(&d_b_pair_chains, &channelDesc1, z_max, rsz, cudaArraySurfaceLoadStore);

	cudaMallocArray(&d_sum_W, &channelDesc4, z_max, rsz, cudaArraySurfaceLoadStore);
	cudaMallocArray(&d_sum_W_sorted, &channelDesc1, z_max, rsz, cudaArraySurfaceLoadStore);
	cudaMallocArray(&d_stress, &channelDesc4, rsz * 2, 0, cudaArraySurfaceLoadStore);
	cudaMallocArray(&d_ft, &channelDesc1, rsz, 0, cudaArraySurfaceLoadStore);
	cudaBindSurfaceToArray(s_stress, d_stress);
	cudaBindSurfaceToArray(s_probs, d_sum_W);
	cudaBindSurfaceToArray(s_sum_W_sorted, d_sum_W_sorted);
	cudaBindSurfaceToArray(s_ft, d_ft);

	cudaMallocArray(&d_arm_index, &channelDesc1, z_max, rsz, cudaArraySurfaceLoadStore);

	cout << "\n";
	cout << " GPU random generator init: \n";
	cout << "  device random generators 1 seeding..";
	cudaMalloc(&d_random_gens, sizeof(gpu_Ran) * rsz);

	gr_array_seed(d_random_gens, rsz, seed * rsz); //
	cout << ".done\n";
	cout << "  device random generators 2 seeding..";
	cudaMalloc(&d_random_gens2, sizeof(gpu_Ran) * rsz);
	gr_array_seed(d_random_gens2, rsz, (seed + 1) * rsz);
	cout << ".done\n";

	cout << "  preparing random number sequence..";
	random_textures_fill(rsz);

	cout << "  creating arrays for pairing information..";
	cudaMallocManaged((void**)&d_end_list, sizeof(int) * rsz * narms);
	cudaMemset(d_end_list, 0, sizeof(int) * rsz * narms);
	cudaMallocManaged((void**)&d_end_counter, sizeof(int) * rsz * narms);
	cudaMemset(d_end_counter, 0, sizeof(int) * rsz * narms);
	cudaMallocManaged((void**)&d_destroy_list, sizeof(int) * rsz * narms * 10);
	cudaMemset(d_destroy_list, 0, sizeof(int) * rsz * narms * 10);
	cudaMallocManaged((void**)&d_destroy_counter, sizeof(int) * rsz * narms);
	cudaMemset(d_destroy_counter, 0, sizeof(int) * rsz * narms);

	cudaMallocManaged((void**)&d_destroy_list_2, sizeof(int) * rsz * 10);
	cudaMemset(d_destroy_list_2, 0, sizeof(int) * rsz * 10);
	cudaMallocManaged((void**)&d_destroy_counter_2, sizeof(int) * rsz);
	cudaMemset(d_destroy_counter_2, 0, sizeof(int) * rsz);

	cudaMallocManaged((void**)&d_create_list_2, sizeof(int) * rsz * 10);
	cudaMemset(d_create_list_2, 0, sizeof(int) * rsz * 10);
	cudaMallocManaged((void**)&d_create_counter_2, sizeof(int) * rsz);
	cudaMemset(d_create_counter_2, 0, sizeof(int) * rsz);

	cudaMallocManaged((void**)&d_create_counter, sizeof(int) * rsz * narms);
	cudaMemset(d_create_counter, 0, sizeof(int) * rsz * narms);

	cudaMallocManaged((void**)&d_doi_weights, sizeof(int) * rsz * rsz);
	cudaMemset(d_doi_weights, 0, sizeof(int) * rsz * rsz);


	cout << ".done\n";
	cout << " GPU random generator init done.\n";
	cout << "\n";

	//Calculate number of necessary blocks of chains
	chain_blocks_number = (N_cha + chains_per_call - 1) / chains_per_call;
	cout << " Number of ensemble blocks " << chain_blocks_number << '\n';

	chain_blocks = new ensemble_block[chain_blocks_number];
	for (int i = 0; i < chain_blocks_number - 1; i++) {
		chain_blocks[i].init(chains_per_call, chain_index(i, 0), &(chain_heads[i * chains_per_call]), nsteps);
	}
	chain_blocks[chain_blocks_number - 1].init((N_cha - 1) % chains_per_call + 1, chain_index(chain_blocks_number - 1, 0), &(chain_heads[(chain_blocks_number - 1) * chains_per_call]), nsteps);

	//chain_blocks - array of blocks
//	chain_blocks = new ensemble_call_block[chain_blocks_number];
//	for (int i = 0; i < chain_blocks_number - 1; i++) {
//		init_call_block(&(chain_blocks[i]), chains_per_call, chain_index(i, 0), &(chain_heads[i * chains_per_call]),s);
//		cout << "  copying chains to device block " << i + 1 << ". chains in the ensemble block " << chains_per_call << '\n';
//	}
//	init_call_block(&(chain_blocks[chain_blocks_number - 1]), (N_cha - 1) % chains_per_call + 1, chain_index(chain_blocks_number - 1, 0), &(chain_heads[(chain_blocks_number - 1) * chains_per_call]),s);

	cout << "  copying chains to device block " << chain_blocks_number << ". chains in the ensemble block " << (N_cha - 1) % chains_per_call + 1 << '\n';
	cout << " device chains done\n";

	cout << "init done\n";
}

stress_plus calc_stress() {
	stress_plus tmps = make_stress_plus(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
	int total_chains = 0;
	for (int i = 0; i < chain_blocks_number; i++) {
		int cc;
		stress_plus tmp = chain_blocks[i].calc_stress(&cc);
		total_chains += cc;
		double w = double(cc);
		tmps = tmps + tmp * w;
	}
	return tmps / total_chains;
}

void get_chains_from_device()    //Copies chains back to host memory
{
	for (int i = 0; i < chain_blocks_number; i++) {
		chain_blocks[i].transfer_from_device();
	}
}

void random_textures_fill(int n_cha) {
	cudaChannelFormatDesc channelDesc1 = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	cudaChannelFormatDesc channelDesc4 = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);

	cudaMallocArray(&(d_taucd_gauss_rand_CD), &channelDesc4, uniformrandom_count, n_cha, cudaArraySurfaceLoadStore);
	cudaMallocArray(&(d_taucd_gauss_rand_SD), &channelDesc4, uniformrandom_count, n_cha, cudaArraySurfaceLoadStore);
	cudaMallocArray(&(d_uniformrand), &channelDesc1, uniformrandom_count, n_cha, cudaArraySurfaceLoadStore);

	cudaMalloc((void**) &d_rand_used, sizeof(int) * n_cha);
	cudaMemset(d_rand_used, 0, sizeof(int) * n_cha);
	cudaMalloc((void**) &d_tau_CD_used_CD, sizeof(int) * n_cha);
	cudaMalloc((void**) &d_tau_CD_used_SD, sizeof(int) * n_cha);
	cudaMemset(d_tau_CD_used_CD, 0, sizeof(int) * n_cha);
	cudaMemset(d_tau_CD_used_SD, 0, sizeof(int) * n_cha);

	gr_fill_surface_uniformrand(d_random_gens, n_cha, uniformrandom_count, d_uniformrand,0);
	cudaDeviceSynchronize();

	int taucd_gauss_count = uniformrandom_count;
	gr_fill_surface_taucd_gauss_rand(d_random_gens2, n_cha, taucd_gauss_count, false, d_taucd_gauss_rand_CD,0); //Set array with random numbers
	gr_fill_surface_taucd_gauss_rand(d_random_gens2, n_cha, taucd_gauss_count, true,  d_taucd_gauss_rand_SD,0);

	cudaBindTextureToArray(t_uniformrand, d_uniformrand, channelDesc1);
	cudaBindTextureToArray(t_taucd_gauss_rand_CD, d_taucd_gauss_rand_CD, channelDesc4);
	cudaBindTextureToArray(t_taucd_gauss_rand_SD, d_taucd_gauss_rand_SD, channelDesc4);
}

void random_textures_refill(int n_cha, cudaStream_t stream_calc) {
	if (chain_blocks_number != 1)
		n_cha = chains_per_call;

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	cudaChannelFormatDesc channelDesc4 = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);

	cudaUnbindTexture(t_uniformrand);
	gr_fill_surface_uniformrand(d_random_gens, n_cha, uniformrandom_count, d_uniformrand, stream_calc);
	cudaMemsetAsync(d_rand_used, 0, sizeof(int) * n_cha, stream_calc);
	cudaBindTextureToArray(t_uniformrand, d_uniformrand, channelDesc);

	//tau_cd gauss 3d vector
	cudaUnbindTexture(t_taucd_gauss_rand_CD);
	gr_refill_surface_taucd_gauss_rand(d_random_gens2, n_cha, d_tau_CD_used_CD,false, d_taucd_gauss_rand_CD,stream_calc);
	gr_refill_surface_taucd_gauss_rand(d_random_gens2, n_cha, d_tau_CD_used_SD, true, d_taucd_gauss_rand_SD,stream_calc);
	cudaMemsetAsync(d_tau_CD_used_CD, 0, sizeof(int) * n_cha, stream_calc);
	cudaMemsetAsync(d_tau_CD_used_SD, 0, sizeof(int) * n_cha, stream_calc);
	cudaBindTextureToArray(t_taucd_gauss_rand_CD, d_taucd_gauss_rand_CD, channelDesc4);
	cudaBindTextureToArray(t_taucd_gauss_rand_SD, d_taucd_gauss_rand_SD, channelDesc4);
//	cudaDeviceSynchronize();
}

int flow_run(int res, double length, bool* run_flag, int *progress_bar) {
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_correlator_res, &(res), sizeof(int))); //Copy timestep for calculation
	for (int i = 0; i < chain_blocks_number; i++) {
		if(chain_blocks[i].time_step<1>(length,2,run_flag,progress_bar)==-1) return -1;
	}
	universal_time=length;
	return 0;
}

int msd_run(int res, double length, bool* run_flag, int *progress_bar) {
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_correlator_res, &(res), sizeof(int))); //Copy timestep for calculation
	for (int i = 0; i < chain_blocks_number; i++) {
		if(chain_blocks[i].time_step<0>(length,1,run_flag,progress_bar)==-1) return -1;
	}
	universal_time=length;
	return 0;
}

int equilibrium_run(int res, double length, int s, int correlator_type, bool* run_flag, int *progress_bar) {
	//Start simulation
	//Calculate stress with timestep specified
	//Update correlators on the fly for each chain (on GPU)
	//At the end copy results from each chain and average them

	*progress_bar = 1;
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_correlator_res, &(res), sizeof(int))); //Copy timestep for calculation

	int np = correlator_size + (s-1) * (correlator_size - (float)correlator_size/(float)correlator_res);

	float *t = new float[np];
	float *x = new float[np];

	for (int j = 0; j < np; j++) {
		t[j] = 0.0f;
		x[j] = 0.0f;
	}

	for (int i = 0; i < chain_blocks_number; i++){
		chain_blocks[i].equilibrium_calc(length, correlator_type, run_flag, progress_bar, np, t, x);
	}

	ofstream correlator_file;
	if(correlator_type==0)	correlator_file.open(filename_ID("G",false));
	if(correlator_type==1)	correlator_file.open(filename_ID("MSD",false));
	cout << "\n";
	int actual_np = (np - correlator_size + floor(length/((float)res*pow((float)correlator_res,(s-1)))));
	for (int j = 0; j < actual_np; j++) {
		cout << t[j] << '\t' << x[j] << '\n';
		correlator_file << t[j] << '\t' << x[j] << '\n';
	}
	correlator_file.close();
	delete[] t;
	delete[] x;
	return 0;
}

void save_to_file(char *filename) {
//	ofstream file(filename, ios::out | ios::binary);
	ofstream file(filename, ios::out);
	if (file.is_open()) {
		//file.write((char*) &N_cha, sizeof(int));
		file << N_cha << "\n";
		for (int i = 0; i < N_cha; i++) {
			//save_to_file(file, chain_index(i), chain_heads[i]);
			print(file, chain_index_arm(i, 0), chain_heads[i], 0);
			print(file, chain_index_arm(i, 1), chain_heads[i], 1);
			print(file, chain_index_arm(i, 2), chain_heads[i], 2);
		}
		file.close();

	} else
		cout << "file error\n";
}

void save_Z_distribution_to_file(string filename, bool cumulative) {
	//Calculation of Z distribution in ensemble

	for (int arm=0; arm < narms; arm++){
		ofstream file(filename_ID(filename + "_arm_" + static_cast<ostringstream*>( &(ostringstream() << (arm+1) ) )->str(), false), ios::out);
		if (file.is_open()) {
			//Search for maximum and minimum of Z
			int Zmin = chain_heads[0].Z[arm];
			int Zmax = chain_heads[0].Z[arm];
			for (int i = 0; i < N_cha; i++) {
				if (chain_heads[i].Z[arm] > Zmax)
					Zmax = chain_heads[i].Z[arm];
				if (chain_heads[i].Z[arm] < Zmin)
					Zmin = chain_heads[i].Z[arm];
			}

			//Sum up coinciding Ze
			float* P = new float[Zmax - Zmin +1];
			for (int j = Zmin; j <= Zmax; j++) {
				P[j-Zmin]=0.0f;
				int counter=0;
				for (int i = 0; i < N_cha; i++) {
					if (chain_heads[i].Z[arm] == j && !cumulative)
						counter++;
					if (chain_heads[i].Z[arm] <= j && cumulative)
						counter++;
				}
				P[j-Zmin] = (float) counter / N_cha;
			}
			for (int i = Zmin; i <= Zmax; i++){
				file << i << "\t" << P[i-Zmin] << "\n";
			}
			delete[] P;
			file.close();
		}
		else
		cout << "file error\n";
	}

}

void save_N_distribution_to_file(string filename, bool cumulative) {
	int run_sum = 0;
	for (int arm=0; arm < narms; arm++){
		ofstream file(filename_ID(filename + "_arm_" + static_cast<ostringstream*>( &(ostringstream() << (arm+1) ) )->str(), false), ios::out);
		if (file.is_open()) {

			//Search for maximum and minimum of N across all strands in all chains
			int Nmin = chain_index(0).QN[run_sum+1].w;
			int Nmax = chain_index(0).QN[run_sum+1].w;
			int Nstr = 0;
			for (int i = 0; i < N_cha; i++) {
				for (int j = 1; j < chain_heads[i].Z[arm]; j++) {
					//cout << "\nArm " << arm << "\tChain " << i << "\tStrand " << j << "\tN " << chain_index(i).QN[run_sum+j].w;
					if (chain_index(i).QN[run_sum+j].w > Nmax)
						Nmax = chain_index(i).QN[run_sum+j].w;
					if (chain_index(i).QN[run_sum+j].w < Nmin)
						Nmin = chain_index(i).QN[run_sum+j].w;
					if (chain_index(i).QN[run_sum+j].w == 0){
						cout << "Zero length strand in chain " << i << ", #" << j << "\n";
					}
				}
				Nstr += chain_heads[i].Z[arm]-1;
			}

			//Sum up coinciding N
			float* P = new float[Nstr];
			for (int n = Nmin; n <= Nmax; n++) {
				P[n]=0.0f;
				int counter=0;
				for (int i = 0; i < N_cha; i++) {
					for (int j = 1; j < chain_heads[i].Z[arm]; j++) {
						if (chain_index(i).QN[run_sum+j].w == n && !cumulative)
							counter++;
						if (chain_index(i).QN[run_sum+j].w <= n && cumulative)
							counter++;
					}
				}
				P[n] += (float)counter / Nstr;
				file << n << "\t" << P[n] << "\n";
			}
			delete[] P;
			file.close();
		}
		else
			cout << "file error\n";

		ofstream file2(filename_ID(filename + "_arm_" + static_cast<ostringstream*>( &(ostringstream() << (arm+1) ) )->str() + "_bp", false), ios::out);
		if (file2.is_open()) {
			//Near branching point
			int Nmin = chain_index(0).QN[run_sum].w;
			int Nmax = chain_index(0).QN[run_sum].w;
			int Nstr = N_cha;
			for (int i = 0; i < N_cha; i++) {
				if (chain_index(i).QN[run_sum].w > Nmax)
					Nmax = chain_index(i).QN[run_sum].w;
				if (chain_index(i).QN[run_sum].w < Nmin)
					Nmin = chain_index(i).QN[run_sum].w;
				if (chain_index(i).QN[run_sum].w == 0){
					cout << "Zero length strand in chain (near branching point)" << i << "\n";
				}
			}

			//Sum up coinciding N
			float* P2 = new float[Nstr];
			for (int n = Nmin; n <= Nmax; n++) {
				P2[n]=0.0f;
				int counter=0;
				for (int i = 0; i < N_cha; i++) {
					if (chain_index(i).QN[run_sum].w == n && !cumulative)
						counter++;
					if (chain_index(i).QN[run_sum].w <= n && cumulative)
						counter++;
				}
				P2[n] += (float)counter / Nstr;
				file2 << n << "\t" << P2[n] << "\n";
			}
			
			delete[] P2;
			file2.close();
		}
		else
			cout << "file error\n";

		run_sum += NK_arms[arm];
	}
	
	return;
}

void save_Q_distribution_to_file(string filename, bool cumulative) {
	int run_sum = 0;
	for (int arm=0; arm < narms; arm++){
		ofstream file(filename_ID(filename + "_arm_" + static_cast<ostringstream*>( &(ostringstream() << (arm+1) ) )->str(), false), ios::out);
		if (file.is_open()) {
			std::vector<float> Q;
			std::vector<float> P;

			//Calculating strand vector lengths
			for (int i = 0; i < N_cha; i++) {
				for (int j = 1; j < chain_heads[i].Z[arm] - 1; j++) {
					float tt = sqrt(chain_index(i).QN[run_sum+j].x * chain_index(i).QN[run_sum+j].x + chain_index(i).QN[run_sum+j].y * chain_index(i).QN[run_sum+j].y + chain_index(i).QN[run_sum+j].z * chain_index(i).QN[run_sum+j].z);
					Q.push_back(tt);
					if (tt < 0 || tt!=tt)
						cout << "\nProblem detected at strand " << j << " of chain " << i << " length is equal " << tt;
				}
			}
			cout << "\n";

			//Sort vector lenghts
			std::sort(Q.begin(), Q.end());

			for(int i=0; i < Q.size(); i++){
				P.push_back( (float)i / (float)Q.size() );
			}

			int quant = 1;
			for (int i = 0; i < Q.size() / quant; i++) {
				file << Q[i * quant] << "\t" << P[i * quant] << "\n";
			}
			file.close();
		} else
			cout << "\nfile error";

		ofstream file2(filename_ID(filename + "_arm_" + static_cast<ostringstream*>( &(ostringstream() << (arm+1) ) )->str() + "_bp", false), ios::out);
		if (file2.is_open()) {
			//near branching point
			std::vector<float> Q2;
			std::vector<float> P2;

			//Calculating strand vector lengths
			for (int i = 0; i < N_cha; i++) {
				float tt = sqrt(chain_index(i).QN[run_sum].x * chain_index(i).QN[run_sum].x + chain_index(i).QN[run_sum].y * chain_index(i).QN[run_sum].y + chain_index(i).QN[run_sum].z * chain_index(i).QN[run_sum].z);
				Q2.push_back(tt);
				if (tt < 0 || tt!=tt)
					cout << "\nProblem detected at strand " << 0 << " of chain " << i << " length is equal " << tt;
			}

			//Sort vector lenghts
			std::sort(Q2.begin(), Q2.end());

			for(int i=0; i < Q2.size(); i++){
				P2.push_back( (float)i / (float)Q2.size() );
			}

			int quant = 1;
			for (int i = 0; i < Q2.size() / quant; i++) {
				file2 << Q2[i * quant] << "\t" << P2[i * quant] << "\n";
			}
			file2.close();
		} else
			cout << "\nfile error";

		run_sum += NK_arms[arm];
	}
	
}

void load_from_file(char *filename) {

	chains_malloc();
	ifstream file(filename, ios::in | ios::binary);
	if (file.is_open()) {
		int ti;
		file.read((char*) &ti, sizeof(int));
		if (ti != N_cha) {
			cout << "ensemble size mismatch\n";
			exit(2);
		}

		for (int i = 0; i < N_cha; i++) {
			load_from_file(file, chain_index(i), &chain_heads[i]);
		}

	} else
		cout << "file error\n";
}

void gpu_clean() {

	cout << "Memory cleanup.. ";

	delete[] chain_blocks;

	cudaFree(chains.QN);
	cudaFree(chains.tau_CD);
	cudaFree(chains.tau_cr);
	cudaFree(chains.R1);
	cudaFree(chains.pair_chain);

	cudaFree(chain_heads);

	if (PD_flag){
		delete[] GEX_table;
		delete[] GEXd_table;
		cudaFreeArray(d_gamma_table);
		cudaFreeArray(d_gamma_table_d);
	}

	cudaFreeArray(d_a_QN);
	cudaFreeArray(d_a_tCD);
	cudaFreeArray(d_a_tcr);
	cudaFreeArray(d_a_pair_chains);
	cudaFreeArray(d_b_QN);
	cudaFreeArray(d_b_tCD);
	cudaFreeArray(d_b_tcr);
	cudaFreeArray(d_b_pair_chains);
	cudaFreeArray(d_a_R1);
	cudaFreeArray(d_b_R1);
	cudaFreeArray(d_corr_a);
	cudaFreeArray(d_corr_b);
	cudaFreeArray(d_sum_W);
	cudaFreeArray(d_sum_W_sorted);
	cudaFreeArray(d_ft);
	cudaFreeArray(d_stress);
	cudaFreeArray(d_arm_index);

	cudaFree(d_end_list);
	cudaFree(d_end_counter);
	cudaFree(d_destroy_list);
	cudaFree(d_destroy_counter);
	cudaFree(d_destroy_list_2);
	cudaFree(d_destroy_counter_2);
	cudaFree(d_create_list_2);
	cudaFree(d_create_counter_2);
	cudaFree(d_create_counter);
	cudaFree(d_doi_weights);

	cudaFreeArray(d_uniformrand);
	cudaFreeArray(d_taucd_gauss_rand_CD);
	cudaFreeArray(d_taucd_gauss_rand_SD);

	cudaFree(d_tau_CD_used_SD);
	cudaFree(d_tau_CD_used_CD);
	cudaFree(d_rand_used);
	cudaFree(d_value_found);
	cudaFree(d_add_rand);
	cudaFree(d_shift_found);
	cudaFree(d_random_gens);
	cudaFree(d_random_gens2);

	cudaDeviceReset();
	cout << "done.\n";
}
