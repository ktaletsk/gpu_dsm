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

#include "gamma.h"

#include <iostream>
#include <fstream>
using namespace std;

#define ITMAX 300 //Maximum allowed number of iterations.
#define EPS 3.0e-7 //Relative accuracy.
#define FPMIN 1.0e-30 //Number near the smallest representable floating-point number.

float temp_M;
float temp_M_1;
float temp_P;
float temp_P_1;

extern float Be;

float gammp(float a, float x) { //Returns the incomplete gamma function P(a, x).
	void gcf(float *gammcf, float a, float x, float *gln);
	void gser(float *gamser, float a, float x, float *gln);
	void nrerror(char error_text[]);
	float gamser, gammcf, gln;
	if (x < 0.0 || a <= 0.0)
		return -1.0;
	if (x < (a + 1.0)) { //Use the series representation.
		gser(&gamser, a, x, &gln);
		return gamser;
	} else { //Use the continued fraction representation
		gcf(&gammcf, a, x, &gln);
		return 1.0 - gammcf; //and take its complement.
	}
}

float gammq(float a, float x) { //Returns the incomplete gamma function Q(a, x) := 1 − P(a, x).
	void gcf(float *gammcf, float a, float x, float *gln);
	void gser(float *gamser, float a, float x, float *gln);
	void nrerror(char error_text[]);
	float gamser, gammcf, gln;
	if (x < 0.0 || a <= 0.0)
		return -1.0;
	if (x < (a + 1.0)) { //Use the series representation
		gser(&gamser, a, x, &gln);
		return 1.0 - gamser; //and take its complement.
	} else {
		//Use the continued fraction representation.
		gcf(&gammcf, a, x, &gln);
		return gammcf;
	}
}

//Returns the incomplete gamma function P(a, x) evaluated by its series representation as gamser.
//Also returns ln Γ(a) as gln.
void gser(float *gamser, float a, float x, float *gln) {
	float gammln(float xx);
	void nrerror(char error_text[]);
	int n;
	float sum, del, ap;
	*gln = lgamma(a);
	if (x <= 0.0) {
		if (x < 0.0)
			return;
		*gamser = 0.0;
		return;
	} else {
		ap = a;
		del = sum = 1.0 / a;
		for (n = 1; n <= ITMAX; n++) {
			++ap;
			del *= x / ap;
			sum += del;
			if (fabs(del) < fabs(sum) * EPS) {
				*gamser = sum * exp(-x + a * log(x) - (*gln));
				return;
			}
		}
		return;
	}
}

//Returns the incomplete gamma function Q(a, x) evaluated by its continued fraction representation
//as gammcf. Also returns ln Γ(a) as gln.
void gcf(float *gammcf, float a, float x, float *gln) {
	float gammln(float xx);
	void nrerror(char error_text[]);
	int i;
	float an, b, c, d, del, h;
	*gln = lgamma(a);
	b = x + 1.0 - a; //Set up for evaluating continued fraction by modified Lentz’s method (§5.2) with b0 = 0.
	c = 1.0 / FPMIN;
	d = 1.0 / b;
	h = d;
	for (i = 1; i <= ITMAX; i++) { //Iterate to convergence.
		an = -i * (i - a);
		b += 2.0;
		d = an * d + b;
		if (fabs(d) < FPMIN)
			d = FPMIN;
		c = b + an / c;
		if (fabs(c) < FPMIN)
			c = FPMIN;
		d = 1.0 / d;
		del = d * c;
		h *= del;
		if (fabs(del - 1.0) < EPS)
			break;
	}
	*gammcf = exp(-x + a * log(x) - (*gln)) * h; //Put factors in front.
}

/* Lambert W function, -1 branch.
   Keith Briggs 2012-01-16.
   */
double LambertW1(const double z) {
  int i;
  const double eps=4.0e-16, em1=0.3678794411714423215955237701614608;
  double p=1.0,e,t,w,l1,l2;
  if (z<-em1 || z>=0.0 || isinf(z) || isnan(z)) {
    fprintf(stderr,"LambertW1: bad argument %g, exiting.\n",z); exit(1);
  }
  /* initial approx for iteration... */
  if (z<-1e-6) { /* series about -1/e */
    p=-sqrt(2.0*(2.7182818284590452353602874713526625*z+1.0));
    w=-1.0+p*(1.0+p*(-0.333333333333333333333+p*0.152777777777777777777777));
  } else { /* asymptotic near zero */
    l1=log(-z);
    l2=log(-l1);
    w=l1-l2+l2/l1;
  }
  if (fabs(p)<1e-4) return w;
  for (i=0; i<10; i++) { /* Halley iteration */
    e=exp(w);
    t=w*e-z;
    p=w+1.0;
    t/=e*p-0.5*(p+1.0)*t/p;
    w-=t;
    if (fabs(t)<eps*(1.0+fabs(w))) return w; /* rel-abs error */
  }
  /* should never get here */
  fprintf(stderr,"LambertW1: No convergence at z=%g, exiting.\n",z);
  exit(1);
}

float bisection_root(float a, float b, float lb, float rb, float y, float eps){
	float delta = (y - gammp((a + 1) / b, powf(lb, b)))/(gammp((a + 1) / b, powf(rb, b))-gammp((a + 1) / b, powf(lb, b)));
	float c =  lb*(1-delta) + rb*delta;
	while (abs(gammp((a+1)/b, powf(c, b))-y)>eps){
		if(gammp((a+1)/b, powf(c, b)) > y)
			rb = c;
		else
			lb = c;
		delta = (y - gammp((a + 1) / b, powf(lb, b)))/(gammp((a + 1) / b, powf(rb, b))-gammp((a + 1) / b, powf(lb, b)));
		c =  lb*(1-delta) + rb*delta;
	}
	return c;
}

void make_gamma_table (float a, float b) {
	//ofstream file("table", ios::out);

	float* GEX_table_P = new float[GAMMATABLESIZE];
	float* GEX_table_M = new float[GAMMATABLESIZE];
	float* GEXd_table_P = new float[GAMMATABLESIZE];
	GEX_table = new float[GAMMATABLESIZE];
	GEXd_table = new float[GAMMATABLESIZE];

	double lgam, g_1;
	step = 0.0f;
//	lgam = lgamma(a/b);
//	g_0  = signgam*exp(lgam);
	lgam = lgamma((a+1)/b);
	g_1  = exp(lgam);
	double m_max = pow(a/b,1/b);
	double W_max=b/g_1*pow(m_max,a)*exp(-pow(m_max,b));//max value of W(m)/m (b/Gamma((a+1)/b) cancel in expression for m_cutoff)
	double c = 0.001*W_max; //Cutoff value for W(m)/m
	gamma_table_cutoff = pow(-a/b*LambertW1(-b/a*pow(c*g_1/b,b/a)),1/b);

	//Generate initial table for cumulative W, equidistant in M; find maximum step in P
	for (int i = 0; i < GAMMATABLESIZE; i++) {
		GEX_table_M[i] = (float)i * gamma_table_cutoff / float(GAMMATABLESIZE - 1);
		GEX_table_P[i] = gammp((a + 1) / b, powf(GEX_table_M[i], b));
		if (i > 0 && abs(GEX_table_P[i] - GEX_table_P[i - 1]) > step) {
			//Choosing maximum step (~in the middle of distribution)
			step = abs(GEX_table_P[i] - GEX_table_P[i - 1]);
		}
	}
	step *= 4;

	//Generate inverse table, equidistant in P, for quick solving: W(M)=rand()
	int j = 0;
	for (int i = 0; i < GAMMATABLESIZE; i++) {
		if (GEX_table_P[i] >= step * j) {
			if (i==0)
				GEX_table[j] = GEX_table_M[i];//initial guess for root Mj;
			else
				GEX_table[j] = bisection_root(a,b,GEX_table_M[i-1],GEX_table_M[i],step*j,0.00001);
			//file << GEX_table[j] << '\n';
			j++;
		}
	}
	temp_M=GEX_table[j-1];
	temp_M_1=GEX_table[j-2];

	while (j<1/step) {
		//Add new points
		temp_M_1=temp_M;
		temp_M+=1.0f * gamma_table_cutoff / float(GAMMATABLESIZE - 1);
		temp_P_1 = temp_P;
		temp_P=gammp((a + 1) / b, powf(temp_M, b));
		if (temp_P >= step * j) {
			GEX_table[j] = bisection_root(a,b,temp_M_1,temp_M,step*j,0.00001);
			//file << GEX_table[j] << '\n';
			j++;
		}
	}
	table_size=j;

	//Generate initial table for Wd(M)
	float norm = 0.0f;
	for(int k=0; k<table_size; k++){
//		p_cd* t_pcd = new p_cd(Be, GEX_table[k]*mp/Mk, NULL);
// 		GEXd_table_P[k] = (t_pcd->W_CD_destroy_aver());
// 		if (k>0)
// 			GEXd_table_P[k] += GEXd_table_P[k-1];
// 		norm += (t_pcd->W_CD_destroy_aver());
// 		delete[] t_pcd;
	}
	for(int k=0; k<table_size; k++){
		GEXd_table_P[k] = GEXd_table_P[k]/norm;
		if (k>0){
			if (GEXd_table_P[k]-GEXd_table_P[k-1] >step_d)
				step_d = GEXd_table_P[k]-GEXd_table_P[k-1];
		}
	}
	////Generate inverse table, equidistant in P, for quick solving: Wd(M)=rand()
	int i = 0;
	j = 0;
	while (j<1/step_d){
		if(GEXd_table_P[i]>=step_d*j){
			GEXd_table[j]=GEX_table[i];
			j++;
		}
		i++;
	}
	table_size_d=j;
}

