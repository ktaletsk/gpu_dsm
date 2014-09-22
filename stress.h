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
