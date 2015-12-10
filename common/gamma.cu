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

#define ITMAX 100 //Maximum allowed number of iterations.
#define EPS 3.0e-7 //Relative accuracy.
#define FPMIN 1.0e-30 //Number near the smallest representable floating-point number.

float gamma_table_s[GAMMATABLESIZE];
float gamma_table_x[GAMMATABLESIZE];
float temp_gamma_table_x;
float temp_gamma_table_s;

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


void make_gamma_table (float a, float b) {
	//ofstream file("table", ios::out);
	//TODO: check if we need double or float
	double lgam, g_1, g_0;
	lgam = lgamma(a/b);
	g_0  = signgam*exp(lgam);
	lgam = lgamma((a+1)/b);
	g_1  = signgam*exp(lgam);
	double m_max = pow(a/b,1/b);
	double W_max=b/g_1*pow(m_max,a)*exp(-pow(m_max,b));//max value of W(m)/m (b/Gamma((a+1)/b) cancel in expression for m_cutoff)
	double c = 0.001*W_max; //Cutoff value for W(m)/m
	gamma_table_cutoff = pow(-a/b*LambertW1(-b/a*pow(c*g_1/b,b/a)),1/b);
	for (int i = 0; i < GAMMATABLESIZE; i++) {
		gamma_table_x[i] = i * 1.0f * gamma_table_cutoff / (GAMMATABLESIZE - 1);
		gamma_table_s[i] = gammp((a + 1) / b, pow(gamma_table_x[i], b));
		if (i > 0) {
			if (abs(gamma_table_s[i] - gamma_table_s[i - 1]) > step)
				step = abs(gamma_table_s[i] - gamma_table_s[i - 1]);
				//Choosing maximum step (~in the middle of distribution)
		}
	}
	step *= 4;
	int j = 0;
	for (int i = 0; i < GAMMATABLESIZE; i++) {
		if (gamma_table_s[i] >= step * j) {
			gamma_new_table_x[j] = gamma_table_x[i];
			//file << gamma_new_table_x[j] << '\n';
			j++;
		}
	}
	temp_gamma_table_x=gamma_new_table_x[j-1];
	while (j<1/step) {
		//Add new point
		temp_gamma_table_x+=1.0f * gamma_table_cutoff / (GAMMATABLESIZE - 1);
		temp_gamma_table_s=gammp((a + 1) / b, pow(temp_gamma_table_x, b));

		if (temp_gamma_table_s >= step * j) {
			gamma_new_table_x[j] = temp_gamma_table_x;
			//file << gamma_new_table_x[j] << '\n';
			j++;
		}
	}
	table_size=j;
}
