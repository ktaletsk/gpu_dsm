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

#if !defined _P_CD_
#define _P_CD_
#include <iostream>
#include <math.h>
#include "random.h"

using namespace std;

struct p_cd_linear { //Generates \tau_CD lifetimes
//uses analytical approximation to P_cd parameters
	float c1, cm1, rcm1c1, At, Adt, Bdt, normdt;
	float g, alpha, tau_0, tau_max, tau_d;
	float Nk;
	Ran *ran;
	p_cd_linear(float Be, float NK, Ran *tran) {
		//init parameters from analytical fits
		Nk = NK;
		ran = tran;
		double z = (Nk + Be) / (Be + 1.0);
// 	    cout<<Be<<'\t'<<Nk<<'\n';
		g = 0.667f;
		if (Be != 1.0f) {
			//Analytical approximation to P_cd parameters for FSM
			//Unpublished Pilyugina E. (2012)

			alpha = (0.053f * logf(Be) + 0.31f) * powf(z, -0.012f * logf(Be) - 0.024f);
			tau_0 = 0.285f * powf(Be + 2.0f, 0.515f);
			tau_max = ((Nk<2) ? tau_0 : 0.025f * powf(Be + 2.0f, 2.6f) * powf(z, 2.83f));
			tau_d = ((Nk<2) ? tau_0 : 0.036f * powf(Be + 2.0f, 3.07f) * powf(z - 1.0f, 3.02f));
		} else {
			//Analytical approximation to P_cd parameters CFSM
			//Andreev, M., Feng, H., Yang, L., and Schieber, J. D.,J. Rheol. 58, 723 (2014).
			//DOI:10.1122/1.4869252

			alpha = 0.267096f - 0.375571f * expf(-0.0838237f * Nk);
			tau_0 = 0.460277f + 0.298913f * expf(-0.0705314f * Nk);
			tau_max = ((Nk<4) ? tau_0 : 0.0156137f * powf(float(Nk), 3.18849f));
			tau_d = ((Nk<4) ? tau_0 : 0.0740131f * powf(float(Nk), 3.18363f));
		}
		//initialize varibles
		c1 = powf(tau_max, alpha) - powf(tau_0, alpha);
		cm1 = powf(tau_max, alpha - 1.0f) - powf(tau_0, alpha - 1.0f);
		rcm1c1 = ((c1==0.0f) ? (alpha - 1.0f)/alpha/tau_0 : (cm1/c1));//ratio beween c1 and c-1

		At = (1.0f - g);
		Adt = At * alpha / (alpha - 1.0f);
		Bdt = Adt * rcm1c1;
		normdt = Bdt + g / tau_d;
	}
	float tau_CD_f_t() {
		float p = ran->flt();
		if (p < (1.0f - g)) {
			return powf(p * c1 / At + powf(tau_0, alpha), 1.0f / alpha);
		} else {
			return tau_d;
		}
	}
	float tau_CD_f_d_t() {
		float p = ran->flt();
		p = p * normdt;
		if (p < Bdt)
			return powf(p * c1 / Adt + powf(tau_0, alpha - 1.0f), 1.0f / (alpha - 1.0f));
		else
			return tau_d;
	}
	float pcdtauint(float tau){
		if (tau<tau_0){
			return 0;
		}
		if (Nk>4){
			if (tau>=tau_0 && tau<tau_max){
				return (Adt*(powf(tau, alpha-1) - powf(tau_0, alpha-1))/c1);
			}
			else if (tau>=tau_max && tau<tau_d){
				return Bdt;
			}
			else if (tau>=tau_d){
				return normdt;
			}
		}
		else{
			return normdt;
		}
		return -1;
	}
	inline float W_CD_destroy_aver() {
		return normdt;
	}
};

struct p_cd { //Generates \tau_CD lifetimes
			  //uses analytical approximation to P_cd parameters
	float A1, B1, A2, B2;
	float g, alpha_1, alpha_2, tau_0, tau_1, tau_2, tau_d;
	float Nk;
	Ran *ran;
	p_cd(float Be, float NK, Ran *tran) {
		//init parameters from analytical fits
		Nk = NK;
		ran = tran;
		g = 0.0603976f;
		alpha_1 = 0.281379f;
		alpha_2 = -0.188025f;
		tau_0 = 0.271977f;
		tau_1 = 428.864f;
		tau_2 = 450294.0f;
		tau_d = 452485.f;

		A1 = (powf(tau_1, alpha_1) - powf(tau_0, alpha_1)) / alpha_1;
		B1 = (powf(tau_2, alpha_2) - powf(tau_1, alpha_2)) / alpha_2 * powf(tau_1, alpha_1 - alpha_2);
		A2 = (powf(tau_1, alpha_1 - 1) - powf(tau_0, alpha_1 - 1)) / (alpha_1 - 1);
		B2 = (powf(tau_2, alpha_2 - 1) - powf(tau_1, alpha_2 - 1)) / (alpha_2 - 1) * powf(tau_1, alpha_1 - alpha_2);
	}
	float tau_CD_f_t() {
		float p = ran->flt();
		if (p < (1.0f - g)) {
			if (p < (1.0f - g)*A1/(A1+B1)) {
				//solve I
				printf("\n%f", powf(p * alpha_1 * (A1 + B1) / (1 - g) + powf(tau_0, alpha_1), 1.0f / alpha_1));
				return powf(p * alpha_1 * (A1+B1)/(1-g) + powf(tau_0, alpha_1), 1.0f / alpha_1);
			}
			else {
				//solve II
				printf("\n%f", powf(alpha_2 / powf(tau_1, alpha_1 - alpha_2) * (p * (A1 + B1) / (1 - g) - A1) + powf(tau_1, alpha_2), 1.0f / alpha_2));
				return powf(alpha_2/ powf(tau_1, alpha_1 - alpha_2) * (p * (A1 + B1) / (1 - g) - A1) + powf(tau_1, alpha_2), 1.0f / alpha_2);
			}
		}
		else {
			printf("\n%f", tau_d);
			return tau_d;
		}
	}
	float tau_CD_f_d_t() {
		float p = ran->flt();
		if (p < (1.0f - g)) {
			if (p < (1.0f - g)*A1 / (A1 + B1)) {
				//solve I
				return powf(p * alpha_1 * (A1 + B1) / (1 - g) + powf(tau_0, alpha_1), 1.0f / alpha_1);
			}
			else {
				//solve II
				return powf(alpha_2 / powf(tau_1, alpha_1 - alpha_2) * (p * (A1 + B1) / (1 - g) - A1) + powf(tau_1, alpha_2), 1.0f / alpha_2);
			}
		}
		else {
			return tau_d;
		}
	}
	float pcdtauint(float tau) {
		return 0;
	}
	inline float W_CD_destroy_aver() {
		return (1-g)*(A2+B2)/(A1+B1)+g/tau_d;
	}
};


#endif
