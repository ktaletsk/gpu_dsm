#ifndef __DETAILED_BALANCE__
#define __DETAILED_BALANCE__
 #include <iostream>
#include <fstream>
#include "chain.h"

//simple bubble sort
//it used for generating cumulative distribution plots
void sort(float* A,int sz)
{
  float t=0;
  for (int  i=1;i<sz;i++){
      for (int j=0;j<sz-i;j++){
	  if (A[j]>A[j+1]){
	   t=A[j+1] ;
	   A[j+1]=A[j];
	   A[j]=t;
	  }
      }
  }
}

//Generates GNU plot for Z distribution from simualtion and analytic one
// used to debug code
void z_plot(chain_head * chead,float Be, int Nk,int N_cha){
  
	float J = powf(1.0f+1.0f/Be,Nk-1);
	ofstream out;
	out.open("z_theor.dat");
	float P = 1.0f/J;
	float tp=1.0f/J;
	for (int  i=1;i<=Nk;i++){
	    out<<i<<' '<<P<<'\n';
	    tp=tp/Be*(Nk-i)/i;
	    P = P + tp;
	}
	out.close();
	float *Z=new float [N_cha];
	for (int  i=0;i<N_cha;i++){
	    Z[i]=chead[i].Z;
	}
	sort(Z, N_cha);
	out.open("z_exp.dat");
	for (int  i=1;i<N_cha;i++){
	    out<<i<<' '<<Z[i]<<'\n';
	}
	out.close();
	delete[] Z;
	out.open("z_distribution.plt");
	out<<" set yrange [0:1.]\n";
	out<<" set xrange [0:"<<float(Nk)<<"]\n";
	out<<"set key bottom right\n";
	out<<"set xlabel 'z, #\n";
	out<<"set ylabel 'cumulative Probability\n";
	out<<"plot 'z_exp.dat' using ($2):($1/"<<N_cha+1<<") title 'simulation' w dots, 'z_theor.dat' title 'analytic' w l, 'z_theor.dat' u ($1+1.):2 title 'analytic' w l\n";
	out.close();
}
#endif