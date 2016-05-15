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
extern char * filename_ID(string filename, bool temp);

extern float mp,Mk;
extern float step;
extern float GEX_table[200000];
extern bool PD_flag;

void random_textures_refill(int n_cha);
void random_textures_fill(int n_cha);

#include "ensemble_call_block.cu"

using namespace std;

extern cudaArray* d_gamma_table;
extern cudaArray* d_gamma_table_d;

#define chains_per_call 32000
//MAX surface size is 32768

sstrentp chains; // host chain conformations

// and only one array with scalar part chain conformation
// every time the vector part is copied from one array to another
// coping is done in entanglement parallel portion of the code
// this allows to use textures/surfaces, which speeds up memory access
// scalar part(chain headers) are update in the chain parallel portion of the code
// chain headers are occupied much smaller memory, no specific memory access technic are used for them.
// depending one odd or even number of time step were performed,
//one of the get_chains_from_device_# should be used

chain_head* chain_heads; // host device chain headers arrays, store scalar variables of chain conformations

int chain_blocks_number;
ensemble_call_block *chain_blocks;

//host constants
double universal_time;//since chain_head do not store universal time due to SP issues
		      	  	  //see chain.h chain_head for explanation
int N_cha;
int NK;
int z_max;
float Be;
float kxx, kxy, kxz, kyx, kyy, kyz, kzx, kzy, kzz;
bool PD_flag=0;

bool dbug = false;	//true;

//navigation
sstrentp chain_index(const int i) { //absolute navigation i - is a global index of chains i:[0..N_cha-1]
	sstrentp ptr;
	ptr.QN = &(chains.QN[z_max * i]);
	ptr.tau_CD = &(chains.tau_CD[z_max * i]);
	return ptr;
}

sstrentp chain_index(const int bi, const int i) {    //block navigation
	//bi is a block index bi :[0..chain_blocks_number]
	//i - is a chain index in the block bi  i:[0..chains_per_call-1]
	sstrentp ptr;
	ptr.QN = &(chains.QN[z_max * (bi * chains_per_call + i)]);
	ptr.tau_CD = &(chains.tau_CD[z_max * (bi * chains_per_call)]);
	return ptr;
}

void chains_malloc() {
	//setup z_max modification maybe needed for large \beta
	z_max = NK;    // current realization limits z to 2^23
	chain_heads = new chain_head[N_cha];
	chains.QN = new float4[N_cha * z_max];
	chains.tau_CD = new float[N_cha * z_max];
}

void host_chains_init(Ran* eran) {
	chains_malloc();
	cout << "generating chain conformations on host..";
	universal_time=0.0;
	for (int i = 0; i < N_cha; i++) {
		sstrentp ptr = chain_index(i);
		chain_init(&(chain_heads[i]), ptr, NK, z_max, PD_flag, eran);
	}
	cout << "done\n";
}

//preparation of constants/arrays/etc
void gpu_init(int seed, p_cd* pcd, int s) {
	cout << "preparing GPU chain conformations..\n";

	//Copy host constants from host to device
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(dBe, &Be, sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(dnk, &NK, sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_z_max, &z_max, sizeof(int)));
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

	cudaMallocArray(&d_a_QN, &channelDesc4, z_max, rsz, cudaArraySurfaceLoadStore);
	cudaMallocArray(&d_a_tCD, &channelDesc1, z_max, rsz, cudaArraySurfaceLoadStore);

	cudaMallocArray(&d_b_QN, &channelDesc4, z_max, rsz, cudaArraySurfaceLoadStore);
	cudaMallocArray(&d_b_tCD, &channelDesc1, z_max, rsz, cudaArraySurfaceLoadStore);

	cudaMallocArray(&d_sum_W, &channelDesc1, z_max, rsz, cudaArraySurfaceLoadStore);
	cudaMallocArray(&d_stress, &channelDesc4, rsz * 2, 0, cudaArraySurfaceLoadStore);
	cudaBindSurfaceToArray(s_stress, d_stress);
	cudaBindSurfaceToArray(s_sum_W, d_sum_W);

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
	cout << ".done\n";
	cout << " GPU random generator init done.\n";
	cout << "\n";

	//Calculate number of necessary blocks of chains
	chain_blocks_number = (N_cha + chains_per_call - 1) / chains_per_call;
	cout << " Number of ensemble blocks " << chain_blocks_number << '\n';

	//chain_blocks - array of blocks
	chain_blocks = new ensemble_call_block[chain_blocks_number];
	for (int i = 0; i < chain_blocks_number - 1; i++) {
		init_call_block(&(chain_blocks[i]), chains_per_call, chain_index(i, 0), &(chain_heads[i * chains_per_call]),s);
		cout << "  copying chains to device block " << i + 1 << ". chains in the ensemble block " << chains_per_call << '\n';
	}
	init_call_block(&(chain_blocks[chain_blocks_number - 1]), (N_cha - 1) % chains_per_call + 1, chain_index(chain_blocks_number - 1, 0), &(chain_heads[(chain_blocks_number - 1) * chains_per_call]),s);
	cout << "  copying chains to device block " << chain_blocks_number
			<< ". chains in the ensemble block "
			<< (N_cha - 1) % chains_per_call + 1 << '\n';
	cout << " device chains done\n";

	cout << "init done\n";
}

stress_plus calc_stress() {
	stress_plus tmps = make_stress_plus(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
	int total_chains = 0;
	for (int i = 0; i < chain_blocks_number; i++) {
		int cc;
		stress_plus tmp = calc_stress_call_block(&(chain_blocks[i]), &cc);
		total_chains += cc;
		double w = double(cc);
		tmps = tmps + tmp * w;
	}
	return tmps / total_chains;
}

int gpu_time_step(double reach_time, bool* run_flag) {
	for (int i = 0; i < chain_blocks_number; i++) {
		if(time_step_call_block(reach_time, &(chain_blocks[i]), run_flag)==-1) return -1;
	}
	universal_time=reach_time;
	return 0;
}

void get_chains_from_device()    //Copies chains back to host memory
{
	for (int i = 0; i < chain_blocks_number; i++) {
		get_chain_from_device_call_block(&(chain_blocks[i]));
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

	gr_fill_surface_uniformrand(d_random_gens, n_cha, uniformrandom_count, d_uniformrand);
	cudaDeviceSynchronize();

	int taucd_gauss_count = uniformrandom_count;
	gr_fill_surface_taucd_gauss_rand(d_random_gens2, n_cha, taucd_gauss_count, false, d_taucd_gauss_rand_CD); //Set array with random numbers
	gr_fill_surface_taucd_gauss_rand(d_random_gens2, n_cha, taucd_gauss_count, true,  d_taucd_gauss_rand_SD);

	cudaBindTextureToArray(t_uniformrand, d_uniformrand, channelDesc1);
	cudaBindTextureToArray(t_taucd_gauss_rand_CD, d_taucd_gauss_rand_CD, channelDesc4);
	cudaBindTextureToArray(t_taucd_gauss_rand_SD, d_taucd_gauss_rand_SD, channelDesc4);
}

void random_textures_refill(int n_cha) {
	if (chain_blocks_number != 1)
		n_cha = chains_per_call;

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	cudaChannelFormatDesc channelDesc4 = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);

	cudaUnbindTexture(t_uniformrand);
	gr_fill_surface_uniformrand(d_random_gens, n_cha, uniformrandom_count, d_uniformrand);
	cudaMemset(d_rand_used, 0, sizeof(int) * n_cha);
	cudaBindTextureToArray(t_uniformrand, d_uniformrand, channelDesc);
	cudaDeviceSynchronize();

	//tau_cd gauss 3d vector
	cudaUnbindTexture(t_taucd_gauss_rand_CD);
	gr_refill_surface_taucd_gauss_rand(d_random_gens2, n_cha, d_tau_CD_used_CD,false, d_taucd_gauss_rand_CD);
	gr_refill_surface_taucd_gauss_rand(d_random_gens2, n_cha, d_tau_CD_used_SD, true, d_taucd_gauss_rand_SD);
	cudaMemset(d_tau_CD_used_CD, 0, sizeof(int) * n_cha);
	cudaMemset(d_tau_CD_used_SD, 0, sizeof(int) * n_cha);
	cudaBindTextureToArray(t_taucd_gauss_rand_CD, d_taucd_gauss_rand_CD, channelDesc4);
	cudaBindTextureToArray(t_taucd_gauss_rand_SD, d_taucd_gauss_rand_SD, channelDesc4);
	cudaDeviceSynchronize();
}


int gpu_Gt_PCS(int res, double length, float *&t, float *&x, int s, bool* run_flag, int *progress_bar) {
	//Start simulation
	//Calculate stress with timestep specified
	//Update correlators on the fly for each chain (on GPU)
	//At the end copy results from each chain and average them

	*progress_bar = 1;
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_correlator_res, &(res), sizeof(int))); //Copy timestep for calculation

	int np = correlator_size + (s-1) * (correlator_size - (float)correlator_size/(float)correlator_res);

	t = new float[np];
	x = new float[np];

	for (int j = 0; j < np; j++) {
		t[j] = 0.0f;
		x[j] = 0.0f;
	}

	for (int i = 0; i < chain_blocks_number; i++){
		int *tint = new int[np];
		float *x_buf = new float[np];

		get_chain_to_device_call_block(&(chain_blocks[i]));
		cudaMemset(chain_blocks[i].d_correlator_time, 0,sizeof(int) * chain_blocks[i].nc);
		if(EQ_time_step_call_block(length, &(chain_blocks[i]),run_flag, progress_bar)==-1) return -1;
		chain_blocks[i].corr->calc(tint, x_buf);
		get_chain_from_device_call_block(&(chain_blocks[i]));

		for (int j = 0; j < chain_blocks[i].corr->npcorr; j++) {
			t[j] = tint[j];
			x[j] += x_buf[j] / N_cha;
		}
		delete[] x_buf;
		delete[] tint;
	}

	ofstream G_file;
	G_file.open(filename_ID("G",false));
	cout << "\n";
	int actual_np = (np - correlator_size + floor(length/((float)res*pow((float)correlator_res,(s-1)))));
	for (int j = 0; j < actual_np; j++) {
		cout << t[j] << '\t' << x[j] << '\n';
		G_file << t[j] << '\t' << x[j] << '\n';
	}
	G_file.close();
	delete[] t;
	delete[] x;
	return 0;
}


void save_to_file(char *filename) {
	ofstream file(filename, ios::out | ios::binary);
//	ofstream file(filename, ios::out);
	if (file.is_open()) {
		file.write((char*) &N_cha, sizeof(int));

		for (int i = 0; i < N_cha; i++) {
			save_to_file(file, chain_index(i), chain_heads[i]);
		}

	} else
		cout << "file error\n";
}

void save_Z_distribution_to_file(string filename /*char *filename*/, bool cumulative) {

	ofstream file(filename.c_str(), ios::out);
	if (file.is_open()) {
//		file<<"Number of chains: "<<N_cha<<"\n";

		//Calculation of Z distribution in ensemble

		//Search for maximum and minimum of Z
		int Zmin = chain_heads[0].Z;
		int Zmax = chain_heads[0].Z;
		for (int i = 0; i < N_cha; i++) {
			if (chain_heads[i].Z > Zmax)
				Zmax = chain_heads[i].Z;
			if (chain_heads[i].Z < Zmin)
				Zmin = chain_heads[i].Z;
		}

		//Sum up coinciding Z
		float P[N_cha];
		for (int j = Zmin; j <= Zmax; j++) {
			P[j]=0.0f;
			int counter=0;
			for (int i = 0; i < N_cha; i++) {
				if (chain_heads[i].Z == j && !cumulative)
					counter++;
				if (chain_heads[i].Z <= j && cumulative)
					counter++;
			}
			P[j] = (float) counter / N_cha;
		}

		for (int i = Zmin; i <= Zmax; i++)
			file << i << "\t" << P[i] << "\n";
	} else
		cout << "file error\n";
}

void save_N_distribution_to_file(string filename, bool cumulative) {
	ofstream file(filename.c_str(), ios::out);
	if (file.is_open()) {
		//Search for maximum and minimum of N across all strands in all chains
		int Nmin = chain_index(0).QN[0].w;
		int Nmax = chain_index(0).QN[0].w;
		int Nstr = 0;
		for (int i = 0; i < N_cha; i++) {
			for (int j = 0; j < chain_heads[i].Z; j++) {
				if (chain_index(i).QN[j].w > Nmax)
					Nmax = chain_index(i).QN[j].w;
				if (chain_index(i).QN[j].w < Nmin)
					Nmin = chain_index(i).QN[j].w;
				if (chain_index(i).QN[j].w == 0)
					cout << "Zero length strand in chain " << i << ", #" << j << "\n";
			}
			Nstr += chain_heads[i].Z;
		}

		//Sum up coinciding N
		float P[Nstr];
		for (int n = Nmin; n <= Nmax; n++) {
			P[n]=0.0f;
			int counter=0;
			for (int i = 0; i < N_cha; i++) {
				for (int j = 0; j < chain_heads[i].Z; j++) {
					if (chain_index(i).QN[j].w == n && !cumulative)
						counter++;
					if (chain_index(i).QN[j].w <= n && cumulative)
						counter++;
				}
			}
			P[n] += (float)counter / Nstr;
			file << n << "\t" << P[n] << "\n";
		}
	} else
		cout << "file error\n";
}

int compare(const void * a, const void * b) {
	float fa = *(const float*) a;
	float fb = *(const float*) b;
	return (fa > fb) - (fa < fb);
}

void save_Q_distribution_to_file(string filename, bool cumulative) {
	ofstream file(filename.c_str(), ios::out);
	if (file.is_open()) {
		//Search for maximal and minimal value of |Q| across all strands in all chains

		int Nstr = 0;
		int prev[N_cha];

		for (int i = 0; i < N_cha; i++) {
			Nstr += chain_heads[i].Z - 2;
			if (i == 0)
				prev[i] = 0;
			else
				prev[i] = prev[i - 1] + chain_heads[i - 1].Z - 2;
		}

		//Calculating strand vector lengths
		float Q[Nstr];
		for (int i = 0; i < N_cha; i++) {
			for (int j = 1; j < chain_heads[i].Z - 1; j++) {
				Q[prev[i] + j - 1] = sqrt(chain_index(i).QN[j].x * chain_index(i).QN[j].x + chain_index(i).QN[j].y * chain_index(i).QN[j].y + chain_index(i).QN[j].z * chain_index(i).QN[j].z);
				if (Q[prev[i] + j - 1] < 0)
					cout << "NaN detected at strand " << prev[i] + j << "\n";
			}
		}
		cout << "\n";

		//Sort vector lenghts
		qsort(Q, Nstr, sizeof(float), compare);

		//Calculate probabilities
		float P[Nstr];
		for (int i = 0; i < Nstr; i++) {
			if (i != 0) {
				if (Q[i] == Q[i - 1])
					P[i - 1] = (float) i / (float) Nstr;
			}
			P[i] = (float) i / (float) Nstr;
		}
		int quant = 10;
		for (int i = 0; i < Nstr / quant; i++) {
			file << Q[i * quant] << "\t" << P[i * quant] << "\n";
		}
	} else
		cout << "file error\n";
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
	delete[] chain_heads;
	delete[] (chains.QN);
	delete[] (chains.tau_CD);

	cudaFreeArray(d_a_QN);
	cudaFreeArray(d_a_tCD);
	cudaFreeArray(d_b_QN);
	cudaFreeArray(d_b_tCD);
	cudaFreeArray(d_sum_W);
	cudaFreeArray(d_stress);

	cudaFreeArray(d_uniformrand);
	cudaFreeArray(d_taucd_gauss_rand_CD);
	cudaFreeArray(d_taucd_gauss_rand_SD);

	cudaFree(d_tau_CD_used_SD);
	cudaFree(d_tau_CD_used_CD);
	cudaFree(d_rand_used);
	cudaFree(d_random_gens);
	cudaFree(d_random_gens2);

	cudaFree(d_gamma_table);
	cudaFree(d_gamma_table_d);
	cudaDeviceReset();
	cout << "done.\n";
}
