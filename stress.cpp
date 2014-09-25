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

#include "stress.h"

    ostream& operator<<(ostream& stream,const stress_plus s){
	return stream<<s.xx<<' '<<s.yy<<' '<<s.zz<<' '<<s.xy<<' '<<s.yz<<' '<<s.zx<<' '<<s.Lpp<<' '<<s.Z;
    }
    
    stress_plus make_stress_plus(float xx,float yy,float zz,float xy,float yz,float zx,float Lpp,float Z){
	stress_plus t;
	t.xx=xx;t.yy=yy;t.zz=zz;
	t.xy=xy;t.yz=yz;t.zx=zx;
	t.Lpp=Lpp;
	t.Z=Z;
	return t;
    }
    
    stress_plus operator+(const stress_plus  &s1, const stress_plus &s2){
	return make_stress_plus(s1.xx+s2.xx,s1.yy+s2.yy,s1.zz+s2.zz,s1.xy+s2.xy,s1.yz+s2.yz,s1.zx+s2.zx,s1.Lpp+s2.Lpp,s1.Z+s2.Z);
    }
    
    stress_plus operator/(const stress_plus  &s1, const double d){
	return make_stress_plus(s1.xx/d,s1.yy/d,s1.zz/d,s1.xy/d,s1.yz/d,s1.zx/d,s1.Lpp/d,s1.Z/d);
    }
    
    stress_plus operator*(const stress_plus  &s1, const double m){
	return make_stress_plus(s1.xx*m,s1.yy*m,s1.zz*m,s1.xy*m,s1.yz*m,s1.zx*m,s1.Lpp*m,s1.Z*m);
    }

