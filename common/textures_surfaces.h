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

#ifndef _TEXTURES_
#define _TEXTURES_


#define uniformrandom_count 250// size of the random arrays
#define stressarray_count 250
#define ran_tpd 256 //thread per block for random_textures_fill()/random_textures_refill()

// see comments in ensemble.h

texture<float, 2, cudaReadModeElementType> t_uniformrand;	// random numbers uniformly distributed
texture<float4, 2, cudaReadModeElementType> t_taucd_gauss_rand_CD; // tauCD lifetimes and normally distributed random numbers (x,y,z), created by CD
texture<float4, 2, cudaReadModeElementType> t_taucd_gauss_rand_SD; // for new entaglments created by SD
surface<void,2> rand_buffer;//temp array for random numbers

//TODO replace a/b with source/dest
texture<float4, 2, cudaReadModeElementType> t_a_QN;		// strents (N,Qx,Qy,Qz)
texture<float, 2, cudaReadModeElementType> t_a_tCD;		// tau_CD
texture<float, 2, cudaReadModeElementType> t_a_tcr;		// creation time
texture<float4, 1, cudaReadModeElementType> t_a_R1;
texture<float4, 2, cudaReadModeElementType> t_corr;

texture<int, 2, cudaReadModeElementType> t_arm_index;

surface<void,2> s_b_QN;// strents (N,Qx,Qy,Qz)
surface<void,2> s_b_tCD;//tau_CD of ent-t
surface<void, 2> s_b_tcr;//tau_CD of ent-t
surface<void,1> s_b_R1;//R_1
surface<void,2> s_corr;

surface<void,2> s_arm_index;// strents (N,Qx,Qy,Qz)
 
surface<void,2> s_W_SD_pm;// SD shift probablities
surface<void,2> s_probs;// SD shift probablities
surface<void,2> s_sum_W_sorted;

surface<void,1> s_ft;//entanglement lifetimes statistics
surface<void,1> s_stress;//float4 xx,yy,zz,xy
#endif
