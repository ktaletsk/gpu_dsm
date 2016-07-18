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
#ifndef CORRELATOR_H_
#define CORRELATOR_H_
#include "textures_surfaces.h"

#define correlator_size 64 // number of points in each correlator
#define correlator_res 8 // number of points to calculate average

texture<float4, 2, cudaReadModeElementType> t_correlator;//to read stress values from d_correlator

typedef struct corr_device {//chain header
	int* d_nc; //device number of chains
	int *d_numcorrelators; //device number of correlator levels
	int *d_dmin;
	int *d_correlator_size;
	int *d_correlator_aver_size;

	//3D arrays
	cudaPitchedPtr d_shift; //storage for incoming stress values (Dij)
	cudaPitchedPtr d_correlation; //storage for correlation results (Cij)
	cudaPitchedPtr d_ncorrelation; //number of values accumulated in correlator (Nij)

	//2D arrays
	float4* d_accumulator; // (Ai)
	size_t d_accumulator_pitch;
	int* d_naccumulator; // (Mi)
	size_t d_naccumulator_pitch;
	int* d_insertindex; //where to insert next data in d_correlator
	size_t d_insertindex_pitch;

	//1D arrays
	int* d_kmax; //maximum attained correlator level during simulation
	float4* d_accval; //accumulated result of incoming variables
} corr_device;

//Multiple tau correlator class
class correlator {
public:
	int nc; //number of chains
	int numcorrelators; //number of correlator levels

	corr_device gpu_corr;

	correlator(int n, int s); //constructor
	~correlator(); //destructor

	void calc(int *t, float *x, int correlator_type); //calculate resulting autocorrelation function

	float* d_lag; //flattened 2D, lag of correlation
	float* d_corr; //flattened 2D, results of correlation


	int npcorr; //actual number of points in correlator
};

#endif
