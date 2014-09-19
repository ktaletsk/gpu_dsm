 #if !defined random
 #define random
#include <iostream>
#include <math.h>

      
//     double  ISET=0,GSET=0//variables for Gasdev
 using namespace std; 
struct Ran {
	int IDUM,IDUM2,IY,IV[33];
	const int static NTAB=32;
	const int static IN1=2147483563,IK1=40014,IQ1=53668,IR1=12211,IN2=2147483399,IK2=40692,IQ2=52774,IR2=3791;
	const int static INM1=2147483563-1,NDIV=1+(2147483563-1)/32;
//	const double static AN=1.0/2147483563.0;
	double AN;

	int ISET;//variables for Gasdev
	float GSET;//variables for Gasdev
	Ran() {
		//dummy constructor
	}
	void seed(int ISEED) {
	    //subroutine ranils(ISEED)
	    // initiation of the random number generator: from Numerical Recepies
	    //Choice of ISEED: 0 <= ISEED <= 2000000000 (2E+9);
	    AN=1.0/2147483563.0;

	    int J,K;
	    IDUM=ISEED+123456789;
	    IDUM2=IDUM;
	    //Load the shuffle table (after 8 warm-ups)
	    for(J=NTAB+8;J>=1;J--){
	      K=IDUM/IQ1;
	      IDUM=IK1*(IDUM-K*IQ1)-K*IR1;
	      if(IDUM<0) IDUM=IDUM+IN1;
	      if(J<=NTAB) IV[J]=IDUM;
	    }
	    IY=IV[1];
	    ISET=0;
	    GSET=0.0;
	    //end subroutine ranils
	}
	Ran(int ISEED) {seed(ISEED);}

 	float flt() {
        //double precision function ranuls()
	// Random number generator from Numerical Recepies
	int K,J;
	float ranuls;
	for(;;){
       //Linear congruential generator 1
	K=IDUM/IQ1;
	IDUM=IK1*(IDUM-K*IQ1)-K*IR1;
	if(IDUM<0) IDUM=IDUM+IN1;
       //Linear congruential generator 2

	K=IDUM2/IQ2;
	IDUM2=IK2*(IDUM2-K*IQ2)-K*IR2;

	if(IDUM2<0) IDUM2=IDUM2+IN2;

	//Shuffling and subtracting
	J=1+IY/NDIV;
	IY=IV[J]-IDUM2;

	IV[J]=IDUM;
	if(IY<=1) IY=IY+INM1;
	if (IY>=IN1) IY=IY-IN1;
	ranuls=AN*IY;
	if ((ranuls!=1.0)&&(ranuls!=0.0)){
		 return ranuls;
	}
	}
      //end function ranuls
	}
	float gasdev(){
	// function from Numerical Recipies: Gaussian distribution
	float R,V1,V2,FAC;

	if(ISET==0){
		R = 0.0;
		while (R>=1.0 || R==0.0){
			V1 = 2.0*flt()-1.0;
			V2 = 2.0*flt()-1.0;
			R  = pow(V1,2)+pow(V2,2);
		}
		FAC    = sqrt(-2.0*log(R)/R);
		GSET   = V1*FAC;
		ISET=1;
		return V2*FAC;

	}
	else{
		ISET   = 0;
		return GSET;
	}
	}//END function gasdev
	inline float gauss_distr(){return gasdev();}
};

#endif
