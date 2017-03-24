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
#include <vector>

class ensemble_block {
	//chain conformations on host (CPU)
	int nc; //number of chains
	int nsteps;
	vector_chains chains; //vector values
	scalar_chains* chain_heads; //scalar values

	float *d_dt;       // time step size from previous time step. used for applying deformation
	float *reach_flag; // flag that chain evolution reached required time
	double block_time; //since chain_head do not store universal time due to SP issues

	// delayed dynamics --- see ensemble_kernel
	int *d_offset;        //coded array shifting parameters
	float4 *d_new_strent; //new strent which should be inserted in the middle of the chain//TODO two new_strents will allow do all the updates at once
	float *d_new_tau_CD;  //new life time
	float *d_new_cr_time;

	//G(t) calculations
	correlator *corr;     //correlator
	int *d_write_time; //index of next cell to fill

	float4* stress_average;

public:
	void init(int nc, vector_chains chains, scalar_chains* chain_heads, int nsteps);
	template<int type> int time_step(double reach_time, int correlator_type, bool* run_flag, int *progress_bar, vector<float> * jump_times);
	int equilibrium_calc(double length, int correlator_type, bool* run_flag, int *progress_bar, int np, float* t, float* x, vector<float> * jump_times);
	void transfer_to_device(); //copies chain conformations to device from host
	void transfer_from_device(); //copies chain conformations from device to host
	stress_plus calc_stress(int *r_chain_count); //calculates average stress over chains in the block, also return number of healthy chains
	void activate_block(); //prepares block for performing time evolution
	void deactivate_block();//copies chain conformations to storing memory
	~ensemble_block(); //destructor
};

#endif

