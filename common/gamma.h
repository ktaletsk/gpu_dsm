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

#ifndef GAMMA_
#define GAMMA_

#include <math.h>
#include <stdlib.h>
#include "pcd_tau.h"

#define GAMMATABLESIZE 1000000

float a, b, mp, Mk; //GEX parameters
float step = 0.0f; //equidistant step in y-direction of W(m)
float step_d = 0.0f;
int table_size=0; //size of resulting table of m values
int table_size_d=0;
float gamma_table_cutoff; //calculated MWD cutoff value
void make_gamma_table (float a, float b);
float bisection_root(float a, float b, float lb, float rb, float y, float eps);
float GEX_table[GAMMATABLESIZE];
float GEXd_table[GAMMATABLESIZE];
double LambertW1(const double z);

#endif
