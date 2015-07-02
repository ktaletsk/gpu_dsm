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

 #if !defined S_STRESS_
 #define S_STRESS_
#include <iostream>
  using namespace std;

	typedef struct stress_plus{//structure for storing stress tensor + couple extra chain quantities
				    //size 32 byte for proper alignment
	    float xx,yy,zz;
	    float xy,yz,zx;
	    float Lpp;
	    float Z;
	}stress_plus;
	ostream& operator<<(ostream& stream,const stress_plus s);
	stress_plus make_stress_plus(float xx,float yy,float zz,float xy,float yz,float zx,float Lpp,float Z);
	stress_plus operator+(const stress_plus  &s1, const stress_plus &s2);
	stress_plus operator/(const stress_plus  &s1, const double d);
	stress_plus operator*(const stress_plus  &s1, const double m);
	
#endif
