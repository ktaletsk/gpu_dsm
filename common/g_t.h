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

 #if !defined S_G_T_
 #define S_G_T_
#include <iostream>
  using namespace std;

	typedef struct g_t{//structure for storing autocorrelation of stress and number of points
	    float * t;
	    float * x;
	    int np;
	}g_t;

#endif
