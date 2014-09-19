#if !defined __BINOMIAL_DISTR__
#define __BINOMIAL_DISTR__

 #include <iostream>
 #include "random.h"

 using namespace std;
 
 
        extern Ran eran;
	double gammln(const double xx);
	double binomial_distr(const double pp, const int n);
	float ratio(int A,int n,int i);//TODO float or double?
	double bico(int n,int k);

#endif