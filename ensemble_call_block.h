
#ifndef ENSEMBLE_CALL_BLOCK_H_
#define ENSEMBLE_CALL_BLOCK_H_

#include "chain.h"


    typedef struct ensemble_call_block{//TODO make it a class?
	//host vars
	int nc;
	sstrentp chains;//just pointer
	chain_head* chain_heads;//just pointer
	
	

	//cudaArray vars
	cudaArray* d_QN; //device arrays for vector part of chain conformations
	cudaArray* d_tCD;
	cudaArray* d_sum_W;// sum of probabalities for each entanglement
	cudaArray* d_stress;// stress calculation temp array

	//regular device arrays
	chain_head* gpu_chain_heads;
// 	gpu_Ran *d_random_gens;// device random number generators//TODO handle random_gen outside this file
	
	float *d_dt; // time step size from prevous time step. used for appling deformation//TODO some of this stuff should be global, not one per call block
	float *reach_flag;// flag that chain evolution reached required time
                     //copied to host each times step

	// delayed dynamics --- see ensemble_kernel

	int *d_offset;//coded shifting parameters
	float4 *d_new_strent;//new strent which shoud be inserted in the middle of the chain//TODO two new_strents will allow do all the updates at once
	float *d_new_tau_CD;//new life time
	
    }ensemble_call_block;


    void init_call_block(ensemble_call_block *cb,int nc,sstrentp chains, chain_head* chain_heads);
    void time_step_call_block(float reach_time,ensemble_call_block *cb);
    void get_chain_from_device_call_block(ensemble_call_block *cb);
    stress_plus calc_stress_call_block(ensemble_call_block *cb,int *r_chain_count);
    void free_block(ensemble_call_block *cb);

    void activate_block(ensemble_call_block *cb);//prepares block for performing time evolution
						 //i.e. copies chain conformations to working memory
    void deactivate_block(ensemble_call_block *cb);//copies chain conformations to storing memory

#endif













