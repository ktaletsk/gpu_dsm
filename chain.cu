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

#include <iostream>
#include <stdlib.h>
#include "chain.h"

int CD_flag = 0;

int z_dist(int tNk) {
	//entanglement distribution
// 	return 1 + int(binomial_distr(1.0/(1.0+Be), tNk-1));//binomial_distr see random module
	double t = eran.flt() / (1.0 + Be) * pow(1.0 + 1.0 / Be, tNk);
	int z = 1;
	double sum = 0.0, si = 1.0 / Be;
	while (sum < t) {
		sum += si;
		si = si / Be * (tNk - z) / z;
		z++;
	}
	return z - 1;
}

int z_dist_truncated(int tNk, int z_max) {
	int tz = z_dist(tNk);
	while (tz > z_max)
		tz = z_dist(tNk);
	return tz;
}

float ratio(int A, int n, int i) {
	// Calculates ratio of two binomial coefficients:
	// ratio = (i-1)(A-N)!(A-i+1)!/((A-N-i+2)!A!)
	float rat = float(i - 1) / float(A - n + 1);
	if (n > 1)
		for (int j = 0; j < n - 1; j++) {
			rat *= (float(A - i + 1 - j) / float(A - j));
		}
	return rat;
}

void N_dist(int ztmp, int *&tN, int tNk) {
	tN = new int[ztmp];
	if (ztmp == 1)
		tN[0] = tNk;
	else {
		int A = tNk - 1;
		for (int i = ztmp; i > 1; i--) {
			float p = eran.flt();
			int Ntmp = 0;
			float sum = 0.0;
			while ((p >= sum) && ((++Ntmp) != A - i + 2))
				sum += ratio(A, Ntmp, i);   //ratio of two binomial coefficients
			tN[i - 1] = Ntmp;
			A = A - Ntmp;
		}
		tN[0] = A + 1;
	}
}

void Q_dist(int tz, int *Ntmp, float *&Qxtmp, float *&Qytmp, float *&Qztmp) {

	Qxtmp = new float[tz];
	Qytmp = new float[tz];
	Qztmp = new float[tz];
	if (tz > 2) {    //dangling ends is not part of distribution
		for (int i = 1; i < tz - 1; i++) {
			Qxtmp[i] = eran.gauss_distr() * sqrt(float(Ntmp[i]) / 3.0); //using gaussian distribution
			Qytmp[i] = eran.gauss_distr() * sqrt(float(Ntmp[i]) / 3.0); //gaussian distribution is defined in math_module
			Qztmp[i] = eran.gauss_distr() * sqrt(float(Ntmp[i]) / 3.0);
		}
	}
}

void chain_init(chain_head *chain_head, sstrentp data, int tnk) {
	//chain conformation generated should be identical to CPU program(valid only for 1 chain)

	int tz = z_dist(tnk);   //z distribution
// 	tz=1;
	float *tent_tau = new float[tz - 1];   //temporaly arrays
	if (CD_flag != 0) {
		for (int k = 0; k < tz - 1; tent_tau[k++] = pcd->tau_CD_f_t())
			;   //1-SD entanglement lifetime
	} else
		for (int k = 0; k < tz - 1; tent_tau[k++] = 0.0)
			;

	int *tN;
	float * Qxtmp, *Qytmp, *Qztmp;
	N_dist(tz, tN, tnk);   //N distribution
// 	for(int i=0;i<tz;cout<<tN[i++]<<'\t');cout<<'\n';
	Q_dist(tz, tN, Qxtmp, Qytmp, Qztmp); //Q distributions //realization Free_Energy_module(Gauss)
	memset(data.QN, 0, sizeof(float) * tnk * 4);
	// creating entanglements according to distributions
	for (int k = 0; k < tz; k++) {
// 	    data[k]=new_strent(tN[k],Qxtmp[k],Qytmp[k],Qztmp[k],0.0,0.0,k+1);
		data.QN[k] = make_float4(Qxtmp[k], Qytmp[k], Qztmp[k], float(tN[k]));
		data.tau_CD[k] = 1.0f / tent_tau[k];

	}

	//set_dangling_ends
	data.QN[0] = make_float4(0.0f, 0.0f, 0.0f, float(tN[0]));
	data.QN[tz - 1] = make_float4(0.0f, 0.0f, 0.0f, float(tN[tz - 1]));
	delete[] tN;
	delete[] Qxtmp;
	delete[] Qytmp;
	delete[] Qztmp;
	delete[] tent_tau;
	chain_head->Z = tz;
	chain_head->time = 0.0;
// 	chain_head->rand_used=0;
// 	chain_head->tau_CD_used=0;
	chain_head->stall_flag = 0;

}

void chain_init(chain_head *chain_head, sstrentp data, int tnk, int z_max) {
	int tz = z_dist_truncated(tnk, z_max);   //z distribution
	float *tent_tau = new float[tz - 1];   //temporaly arrays
	if (CD_flag != 0) {
		for (int k = 0; k < tz - 1; tent_tau[k++] = pcd->tau_CD_f_t())
			;   //1-SD entanglement lifetime
	} else
		for (int k = 0; k < tz - 1; tent_tau[k++] = 0.0)
			;

	int *tN;
	float * Qxtmp, *Qytmp, *Qztmp;
	N_dist(tz, tN, tnk);   //N distribution
	Q_dist(tz, tN, Qxtmp, Qytmp, Qztmp); //Q distributions //realization Free_Energy_module(Gauss)
	memset(data.QN, 0, sizeof(float) * z_max * 4);
	// creating entanglements according to distributions
	for (int k = 0; k < tz; k++) {
		data.QN[k] = make_float4(Qxtmp[k], Qytmp[k], Qztmp[k], float(tN[k]));
		data.tau_CD[k] = 1.0f / tent_tau[k];

	}

	//set_dangling_ends
	data.QN[0] = make_float4(0.0f, 0.0f, 0.0f, float(tN[0]));
	data.QN[tz - 1] = make_float4(0.0f, 0.0f, 0.0f, float(tN[tz - 1]));
	delete[] tN;
	delete[] Qxtmp;
	delete[] Qytmp;
	delete[] Qztmp;
	delete[] tent_tau;
	chain_head->Z = tz;
	chain_head->time = 0.0;
// 	chain_head->rand_used=0;
// 	chain_head->tau_CD_used=0;
	chain_head->stall_flag = 0;

}

ostream& operator<<(ostream& stream, const sstrentp c) {

	int end = 10;
	stream << "N:  ";
	for (int j = 0; j != end; j++)
		stream << c.QN[j].w << ' ';
	stream << "\nQx: ";
	for (int j = 0; j != end; j++)
		stream << c.QN[j].x << ' ';
	stream << "\nQy: ";
	for (int j = 0; j != end; j++)
		stream << c.QN[j].y << ' ';
	stream << "\nQz: ";
	for (int j = 0; j != end; j++)
		stream << c.QN[j].z << ' ';
	stream << '\n';
	return stream;

}

void print(ostream& stream, const sstrentp c, const chain_head chead) {
	stream << chead.time << '\n';
	stream << chead.Z << '\n';
	stream << "N:  ";
	for (int j = 0; j < chead.Z; j++)
		stream << c.QN[j].w << ' ';
	stream << "\nQx: ";
	for (int j = 0; j < chead.Z; j++)
		stream << c.QN[j].x << ' ';
	stream << "\nQy: ";
	for (int j = 0; j < chead.Z; j++)
		stream << c.QN[j].y << ' ';
	stream << "\nQz: ";
	for (int j = 0; j < chead.Z; j++)
		stream << c.QN[j].z << ' ';
	stream << '\n';
}

void save_to_file(ostream& stream, const sstrentp c, const chain_head chead) {
	stream.write((char*) &chead, sizeof(chain_head));
	for (int j = 0; j < chead.Z; j++)
		stream.write((char*) &(c.QN[j]), sizeof(float4));
	for (int j = 0; j < chead.Z; j++)
		stream.write((char*) &(c.tau_CD[j]), sizeof(float));
}

void load_from_file(istream& stream, const sstrentp c,
		const chain_head *chead) {
	stream.read((char*) chead, sizeof(chain_head));
	for (int j = 0; j < chead->Z; j++)
		stream.read((char*) &(c.QN[j]), sizeof(float4));
	for (int j = 0; j < chead->Z; j++)
		stream.read((char*) &(c.tau_CD[j]), sizeof(float));
}

