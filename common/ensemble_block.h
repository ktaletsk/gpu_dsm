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

#ifndef ENSEMBLE_BLOCK_H_
#define ENSEMBLE_BLOCK_H_

#include "chain.h"
#include "correlator.h"

class ensemble_block {
	//chain conformations on host (CPU)
	int nc; //number of chains
	vector_chains chains; //vector values
	scalar_chains* chain_heads; //scalar values

	//chain conformation on device (GPU)
	cudaArray* d_QN;  //device arrays for vector part of chain conformations
	cudaArray* d_tCD; //these arrays used to store conformations
	cudaArray* d_R1;

	//regular device arrays
	scalar_chains* gpu_chain_heads;

	float *d_dt;       // time step size from previous time step. used for applying deformation
	float *reach_flag; // flag that chain evolution reached required time
					   //copied to host each times step
	double block_time; //since chain_head do not store universal time due to SP issues
	                   //see chain.h chain_head for explanation

	// delayed dynamics --- see ensemble_kernel
	int *d_offset;        //coded array shifting parameters
	float4 *d_new_strent; //new strent which should be inserted in the middle of the chain//TODO two new_strents will allow do all the updates at once
	float *d_new_tau_CD;  //new life time

	//G(t) calculations
	correlator *corr;     //correlator
	int *d_correlator_time; //index of next cell to fill

public:
	void init(int nc, vector_chains chains, scalar_chains* chain_heads, int s);
	int flow_time_step(double reach_time, bool* run_flag);//void init_call_block(ensemble_call_block *cb, int nc, conformations chains, chain_head* chain_heads, int s)
														  //copies chain conformations from host and prepare block variables
	int equilibrium_time_step(double reach_time, int correlator_type, bool* run_flag, int *progress_bar);
	//int correlator_update_call_block(int n_steps, ensemble_call_block *cb, bool* run_flag);
	//performs time evolution

	int equilibrium_calc(double length, int correlator_type, bool* run_flag, int *progress_bar, int np, float* t, float* x);

	void transfer_to_device();
	//copies chain conformations to device from host

	void transfer_from_device();
	//copies chain conformations from device to host

	stress_plus calc_stress(int *r_chain_count);
	//calculates average stress over chains in the block, also return number of healthy chains

	void free_block();
	// free block memory

	void activate_block(); //prepares block for performing time evolution
	//i.e. copies chain conformations to working memory

	void deactivate_block();//copies chain conformations to storing memory
};

#endif

