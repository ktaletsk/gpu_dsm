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

void make_gamma_table (float a, float b) {
	for (int i = 0; i < GAMMATABLESIZE; i++) {
		gamma_table_x[i] = i * 1.0f * GAMMATABLECUTOFF / (GAMMATABLESIZE - 1);
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
		temp_gamma_table_x+=1.0f * GAMMATABLECUTOFF / (GAMMATABLESIZE - 1);
		temp_gamma_table_s=gammp((a + 1) / b, pow(temp_gamma_table_x, b));

		if (temp_gamma_table_s >= step * j) {
			gamma_new_table_x[j] = temp_gamma_table_x;
			//file << gamma_new_table_x[j] << '\n';
			j++;
		}
	}
	table_size=j;
}
