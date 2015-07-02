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

#define GAMMATABLESIZE 200000
#define GAMMATABLECUTOFF 30

float a, b, mp, Mk;
float step = 0;
int table_size=0;

void make_gamma_table (float a, float b);

float gamma_new_table_x[GAMMATABLESIZE];

#endif
