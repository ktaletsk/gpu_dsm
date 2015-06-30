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
#include <fstream>
#include "gpu_random.h"
#include "cuda_call.h"
#include "cudautil.h"
#include "random.h"
#include "ensemble.h"
#include "timer.h"
#include "detailed_balance.h"
#include "job_ID.h"
extern float a,b,mp,Mk;
extern void make_gamma_table (float a, float b);
using namespace std;


Ran eran(1);
p_cd *pcd;

//synonyms:
//GPU - device
//CPU - host
int main(int narg, char** arg) {
	char *savefile = NULL;
	char *loadfile = NULL;
	int device_ID = 0;
	bool distr = 0;
	int G_flag = 0;
	double simulation_time;
	double t_step_size;

	cout << '\n';

	//Processing command line parameters
	int k = 1;
	while (k < narg) {
		if (k == 1) {	//processing job_ID
			job_ID = atoi(arg[k]);
		}
		if ((strcmp(arg[k], "-s") == 0) && (k + 1 < narg)) {	//Save file
			savefile = new char[strlen(arg[k + 1]) + 1];
			strcpy(savefile, arg[k + 1]);
			cout << "final chain conformations will be saved to " << savefile
					<< '\n';
			k++;
		}
		if ((strcmp(arg[k], "-l") == 0) && (k + 1 < narg)) {	//Load file
			loadfile = new char[strlen(arg[k + 1]) + 1];
			strcpy(loadfile, arg[k + 1]);
			k++;
		}
		if ((strcmp(arg[k], "-d") == 0) && (k + 1 < narg)) {//Determine nvidia gpu device_ID
			device_ID = atoi(arg[k + 1]);
			k++;
		}
		if ((strcmp(arg[k], "-distr") == 0) && (k < narg)) {
			distr = 1;
		}
		k++;
	}

	//Print device_ID information
	if (job_ID != 0) {
		cout << "job_ID: " << job_ID << '\n';
		cout << "using " << job_ID
				<< " as a seed for random number generator\n";
		cout << "\"_" << job_ID
				<< "\" will be appended to filename for all files generated\n\n";
	}

	//First checking device
	checkCUDA(device_ID);

	//Read simulation parameters from input file
	ifstream in;
	in.open("input.dat");
	in >> Be;
	in >> NK;
	in >> N_cha;
	in >> kxx >> kxy >> kxz >> kyx >> kyy >> kyz >> kzx >> kzy >> kzz;
	in >> CD_flag;	//Computation mode: 0-only SD / 1-SD+CD
	in >> PD_flag;

	//int int_t;
	//in>>int_t;//TODO SD off not implemented
	in >> G_flag;
	//in>>int_t;//TODO R  not implemented
	//in>>int_t;//TODO f_d not implemented
	in >> t_step_size;
	in >> simulation_time;

	//Print simulation parameters
	cout << "\nsimulation parameters:\n";
	cout << "NK\tBeta\tN_cha" << "\n";
	cout << NK << '\t' << Be << '\t' << N_cha << "\n";
	cout << "deformation tensor:" << "\n";
	cout << kxx << '\t' << kxy << '\t' << kxz << '\n' << kyx << '\t' << kyy << '\t' << kyz << '\n' << kzx << '\t' << kzy << '\t' << kzz << '\n';
	if (G_flag)
		cout << "G(t) calculation is on\n";
	else
		cout << "G(t) calculation is off\n";
	cout << "simulation time, sync time" << "\n";
	cout << simulation_time << '\t' << t_step_size << '\n' << '\n';

	//Toy parameters    
	//     Be=1.0;
	//     NK=46;
	//     N_cha=4000;
	//     kxy=8.16e-05;

	//Determine if there is a flow
	bool flow = (kxx != 0.0) || (kxy != 0.0) || (kxz != 0.0) || (kyx != 0.0) || (kyy != 0.0) || (kyz != 0.0) || (kzx != 0.0) || (kzy != 0.0) || (kzz != 0.0);

	if (PD_flag) {
		ifstream in2;
		in2.open("polydisp.dat");
		in2 >> a;
		in2 >> b;
		in2 >> mp;
		in2 >> Mk;
		make_gamma_table (a, b);
	}

	//Initialize random
	eran.seed(job_ID * N_cha);

	pcd = new p_cd(Be, NK, &eran);

	if (loadfile != NULL) {	//load chain conformations from file
		cout << "loading chain conformations from " << loadfile << "..";
		cout.flush();
		load_from_file(loadfile);
		cout << "done.\n";
	} else
		host_chains_init();	// or generate equilibrium conformations

	gpu_ran_init();	//init gpu random module
	gpu_init(job_ID);	//prepare GPU to run DSM calculation

	ctimer timer;
	if (flow) {	//Flow calculations
		//tau file
		ofstream tau_file;
		tau_file.open(filename_ID("tau"));
		cout << "output file: " << filename_ID("tau") << '\n';
		timer.start();
		//main loop
		cout << "performing time evolution for the ensemble..\n";
		cout << "time\tstress tensor(xx yy zz xy yz xz)\t<Lpp>\t<Z>\n";
		for (double t_step = 0; t_step < simulation_time; t_step +=t_step_size) {
			gpu_time_step(t_step + t_step_size);
			stress_plus stress = calc_stress();
			cout << t_step + t_step_size << '\t' << stress << '\n';
			tau_file << t_step + t_step_size << '\t' << stress << '\n';
			tau_file.flush();
		}
		timer.stop();
		tau_file.close();
		cout << "time evolution done.\n";
	} else {		//Equlibrium calculations
		if (G_flag) {
			cout << "G(t) calc...\n";
			cout.flush();
			ofstream G_file;
			//cout<<filename_ID("tau")<<'\n';
			G_file.open(filename_ID("G"));
			float *t, *x;
			int np;
			timer.start();
			gpu_Gt_calc(t_step_size, simulation_time, t, x, np);
			cout << "G(t) calc done\n";
			for (int j = 0; j < np; j++) {
				cout << t[j] << '\t' << x[j] << '\n';
				G_file << t[j] << '\t' << x[j] << '\n';
			}
			timer.stop();
			G_file.close();
		} else {
			cout
					<< "There are no flow and no equilibrium quantity to calculate. Exiting... \n";
		}
	}

	get_chains_from_device();
	//     z_plot(chain_heads,Be, NK,N_cha);

	if (savefile != NULL) {
		cout << "saving chain conformations to " << savefile << "..";
		save_to_file(savefile);
		cout << "done.\n";
	}

	if (distr) {		//Calculating distributions for Z,N,Q
		cout << "Saving distribution to file...";
		if (CD_flag) {
			save_Z_distribution_to_file("distr_Z.dat", 1);
			save_N_distribution_to_file("distr_N.dat", 1);
			save_Q_distribution_to_file("distr_Q.dat", 1);
		} else {
			save_Z_distribution_to_file("distr_Z_.dat", 1);
			save_N_distribution_to_file("distr_N_.dat", 1);
			save_Q_distribution_to_file("distr_Q_.dat", 1);
		}
	}
	gpu_clean();

	cout << "Calculation time: " << timer.elapsedTime() << " milliseconds\n";
	return 0;
}
