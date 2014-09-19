 #include "binomial.h"
 #include <math.h>

 using namespace std;


    double gammln(const double xx)
    {
	int j;
	double x,y,tmp,ser;
	static const double cof[6]={76.18009172947146,-86.50532032941677,
		24.01409824083091,-1.231739572450155,0.1208650973866179e-2,
		-0.5395239384953e-5};

	y=x=xx;
	tmp=x+5.5;
	tmp -= (x+0.5)*log(tmp);
	ser=1.000000000190015;
	for (j=0;j<6;j++) ser += cof[j]/++y;
	return -tmp+log(2.5066282746310005*ser/x);
    }
 
 
    double binomial_distr(const double pp, const int n)
    {
	const double PI=3.141592653589793238;
	int j;
	static int nold=(-1);
	double am,em,g,angle,p,bnl,sq,t,y;
	static double pold=(-1.0),pc,plog,pclog,en,oldg;

	p=(pp <= 0.5 ? pp : 1.0-pp);
	am=n*p;
	if (n < 25) {
	    bnl=0.0;
	    for (j=0;j<n;j++)
		if (eran.flt() < p) ++bnl;
	} else if (am < 1.0) {
	    g=exp(-am);
	    t=1.0;
	    for (j=0;j<=n;j++) {
		t *= eran.flt();
		if (t < g) break;
	    }
	bnl=(j <= n ? j : n);
	} else {
	    if (n != nold) {
		en=n;
		oldg=gammln(en+1.0);
		nold=n;
	    } if (p != pold) {
		pc=1.0-p;
		plog=log(p);
		pclog=log(pc);
		pold=p;
	    }
	    sq=sqrt(2.0*am*pc);
	    do {
		do {
		    angle=PI*eran.flt();
		    y=tan(angle);
		    em=sq*y+am;
		} while (em < 0.0 || em >= (en+1.0));
		em=floor(em);
		t=1.2*sq*(1.0+y*y)*exp(oldg-gammln(em+1.0)-gammln(en-em+1.0)+em*plog+(en-em)*pclog);
	    } while (eran.flt() > t);
	    bnl=em;
	}
	if (p != pp) bnl=n-bnl;
	return bnl;
    }


    float ratio(int A,int n,int i){
    // Calculates ratio of two binomial coefficients:
    // ratio = (i-1)(A-N)!(A-i+1)!/((A-N-i+2)!A!)
	float rat = float(i-1)/float(A-n+1);
	if (n>1) for(int j=0;j<n-1;j++){rat*=(float(A-i+1-j)/float(A-j));}
	return rat;
    }


    double bico(int n,int k){
      #if defined(_MSC_VER) //MSVS 2013 should have support for lgamma TODO add MSVS 2013 support
	return exp(gammln(n+1)-gammln(k+1)-gammln(n-k+1));
      #else
	return exp(lgamma(n+1)-lgamma(k+1)-lgamma(n-k+1));
      #endif    

    }















