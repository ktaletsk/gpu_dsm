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

#ifndef ENSEMBLE_CALL_BLOCK_H_
#define ENSEMBLE_CALL_BLOCK_H_

#include "chain.h"
#include "correlator.h"



    typedef struct ensemble_call_block{
	//host vars
	int nc;
	sstrentp chains;//just a pointer
	chain_head* chain_heads;//just a pointer
	

	//cudaArray vars
	cudaArray* d_QN; //device arrays for vector part of chain conformations
	cudaArray* d_tCD;// these arrays used to store conformations

	//regular device arrays
	chain_head* gpu_chain_heads;
	
	float *d_dt; // time step size from prevous time step. used for appling deformation
	float *reach_flag;// flag that chain evolution reached required time
                     //copied to host each times step

	// delayed dynamics --- see ensemble_kernel

	int *d_offset;//coded array shifting parameters
	float4 *d_new_strent;//new strent which shoud be inserted in the middle of the chain//TODO two new_strents will allow do all the updates at once
	float *d_new_tau_CD;//new life time

	//G(t) calculations
	c_correlator *corr;//correlator
	int *d_correlator_time;//index of next cell to fill
	
	
    }ensemble_call_block;


    void init_call_block(ensemble_call_block *cb,int nc,sstrentp chains, chain_head* chain_heads);
    //copies chain conformations from host and prepare block variables
    
    void init_block_correlator(ensemble_call_block *cb);
    //prepares correlator if G(t) calculations needed
    
    void time_step_call_block(float reach_time,ensemble_call_block *cb);
    void EQ_time_step_call_block(float reach_time,ensemble_call_block *cb);
    //performs time evolution
    
    void get_chain_to_device_call_block(ensemble_call_block *cb);
    //copies chain conformations to device from host

    void get_chain_from_device_call_block(ensemble_call_block *cb);
    //copies chain conformations from device to host
    stress_plus calc_stress_call_block(ensemble_call_block *cb,int *r_chain_count);
    //calculates average stress over chains in the block, also return number of healthy chains
    void free_block(ensemble_call_block *cb);
    // free block memory

    void activate_block(ensemble_call_block *cb);//prepares block for performing time evolution
						 //i.e. copies chain conformations to working memory
    void deactivate_block(ensemble_call_block *cb);//copies chain conformations to storing memory

#endif













