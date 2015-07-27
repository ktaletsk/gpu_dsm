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

#ifndef ENSEMBLE_H_
#define ENSEMBLE_H_

// DSM chain ensemble header
// allows to generate, store and load chain conformations
// also allows to copy conformations to device memory and back and perform time evolution on the device

#include "chain.h"// chain conformations
#include "stress.h"// chain stress
#include "pcd_tau.h"// CD lifetime distribution
#include "ensemble_call_block.h" //GPU can process only 32K chains simulataneously
//we split ensemble in call_blocks and feed them to GPU one by one
//most interesting part of the code is there
using namespace std;

//public variables
extern sstrentp chains;	// arrays with vector part of chain conformations(Q,N,tauCD)
//contains conformations for all the chains. It is 1D array. use chain_index to access chain i

extern chain_head* chain_heads;	//array with scalar part chain conformations header

sstrentp chain_index(const int i);//absolute navigation i - is a global index of chains i:[0..N_cha-1]

sstrentp chain_index(const int bi, const int i);	//block navigation
//bi is a block index bi :[0..chain_blocks_number]
//i - is a chain index in the block bi  i:[0..chains_per_call-1]
extern double universal_time;//since chain_head do not store universal time due to SP issues
	                   //see chain.h chain_head for explanation
extern int N_cha;	// number of chains
extern int NK;	//number of chain segments in each chain
extern int z_max;	// max number of strand in chain (currently same as NK)
extern bool dbug;	//debug flag
extern float kxx, kxy, kxz, kyx, kyy, kyz, kzx, kzy, kzz;	//deformation tensor
extern bool PD_flag;

//public functions
void host_chains_init(Ran* eran);	//prepares chain conformations on host
void gpu_init(int seed, p_cd* pcd);// prepares GPU kernels,random number generators and copies chains to host memory
void get_chains_from_device();	//Copies chains back to host memory
void save_to_file(char *filename);	//saves chain conformations to a file
void save_Z_distribution_to_file(string filename, bool cumulative); //saves Z distrubution to file
void save_N_distribution_to_file(string filename, bool cumulative); //saves N distribution to file
void save_Q_distribution_to_file(string filename, bool cumulative); //saves Q distribution to file
void load_from_file(char *filename);  //loads chain conformations from a file

int gpu_time_step(double reach_time, bool* run_flag);  // performs time evolution of ensemble

int Gt_brutforce(int res, double length, float *&t, float *&x, int &np, bool* run_flag);
int gpu_Gt_calc(int res, double length, float *&t, float *&x, int &np, bool* run_flag); //G(t) relaxation spectrum calculation

void gpu_clean();  //free memory used by ensemble

// random number sequences are generated by separate kernel and stored in the temporary arrays
// this reduces execution time for different conditional branches and optimizes kernel memory use
// we use textures to speed up memory access
// however these arrays need to refilled from time to time
// 	  void random_textures_refill(int n_cha);
// 	  void random_textures_fill(int n_cha);
//actually declared in .cpp file

//stress calculation
stress_plus calc_stress();
#endif /* ENSEMBLE_H_ */
