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

//contains GPU brut correlation function calculator
//to calculate correlation function required
// 1) filled d_correlator array and properly updated counter. s_correlator surface (x(declared textures_surfaces.h)x)
// 2) time ticks for correlation function. passed as array to calc subroutine
// handling of these two things should be done outside this class

#ifndef CORRELATOR_H_
#define CORRELATOR_H_
#include "textures_surfaces.h"

#define correlator_size 1024// size of  correlator page
#define max_corr_function_length 100// size of the random arrays
surface<void, 2> s_correlator; //float4 xy,yz,xz,dummy
texture<float4, 2, cudaReadModeElementType> t_correlator;	// tauxy,tauyz,tauzx
surface<void, 2> s_corr_function;

//in order to calculate correlation function of arbitrary length correlator can be reused
//with different correlator_res
//first run : data is loaded every correlator_res timesteps
//	      correlation function from t=0 to t=correlator_size*res/2 is calculated (G_page_1(t))
//second run : data is loaded every correlator_res*correlator_base timesteps
//	      correlation function from t=correlator_size*res/2 to t=correlator_size*res*correlator_base/2 is calculated (G_page_2(t))
//third run : data is loaded every correlator_res*correlator_base^2 timesteps
//	      correlation function from t=correlator_size*correlator_base*res/2 to t=correlator_size*res*correlator_base^2/2 is calculated (G_page_2(t))
// and so on
// until correlator_size*res*correlator_base^i> simulation length
#define correlator_base 16//
#define correlator_res 1//

class c_correlator {

public:
	cudaArray* d_correlator;		//fill this array with data
	int counter;		// and update this var before calc()
private:
	int nc;
	cudaArray* d_corr_function;		// stress calculation temp array
public:
	c_correlator(int nc);
	~c_correlator();
	void calc(int *t, float *x, int np);
};

void init_correlator();

#endif
