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

#if !defined gpu_random
#define gpu_random
#include <curand_kernel.h>
#include <math.h>

//Curand random number generators
//creates array of random number generator on device
//fills array with random numbers
//designed to provide texture random number supply for DSM dynamics
void gpu_ran_init();    //initializes device random number

#define gpu_Ran curandState_t
void gr_array_seed(gpu_Ran *gr, int sz, int seed_offest); // seeds array of device random number generators

//Fills surface rand_buffer with uniformaly distributed (0,1] random numbers. They are used to pick jump process.
void gr_fill_surface_uniformrand(gpu_Ran *gr, int sz, int count, cudaArray* d_uniformrand);

//Fills surface rand_buffer with vectors of {1x uniformaly distributed (0,1] random numbers and 3x normally distributed (mean 0, varience 1) random numbers}. They are used for (\tau_CD, \bm{Q_i} generation)
void gr_fill_surface_taucd_gauss_rand(gpu_Ran *gr, int sz, int count, bool SDCD_toggle, cudaArray* d_taucd_gauss_rand);

//Refills surface rand_buffer with vectors of {1x uniformaly distributed (0,1] random numbers and 3x normally distributed (mean 0, varience 1) random numbers}. They are used for (\tau_CD, \bm{Q_i} generation)
//int *count specify how many numbers needs to be refilled for each thread.
void gr_refill_surface_taucd_gauss_rand(gpu_Ran *gr, int sz, int *count,bool SDCD_toggle, cudaArray* d_taucd_gauss_rand);

#endif
