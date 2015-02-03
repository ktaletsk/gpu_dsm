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

#ifndef _TEXTURES_
#define _TEXTURES_


#define uniformrandom_count 250// size of the random arrays
#define ran_tpd 256 //thread per block for random_textures_fill()/random_textures_refill()

// see comments in ensemble.h

  texture<float, 2, cudaReadModeElementType> t_uniformrand;	// random numbers uniformly distributed
  texture<float4, 2, cudaReadModeElementType> t_taucd_gauss_rand; // tauCD lifetimes and normally distributed random numbers (x,y,z)
  surface<void,2> rand_buffer;//temp array for random numbers

  //TODO replace a/b with source/dest
  texture<float4, 2, cudaReadModeElementType> t_a_QN;		// strents (N,Qx,Qy,Qz)
  texture<float, 2, cudaReadModeElementType> t_a_tCD;		// strents (N,Qx,Qy,Qz)

  //  surface<void,2> s_a_QN;
  //  surface<void,2> s_a_tCD;

  surface<void,2> s_b_QN;  // strents (N,Qx,Qy,Qz)
  surface<void,2> s_b_tCD;//tau_CD of ent-t

 
  surface<void,2> s_W_SD_pm;// SD shift probablities
  surface<void,2> s_sum_W;// SD shift probablities
  surface<void,1> s_stress;//float4 xx,yy,zz,xy
 
#endif
