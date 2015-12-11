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

#include "../common/main_cuda.cu"
//#include <mpi.h>

int main(int narg, char** arg) {
//    // Initialize the MPI environment
//    MPI_Init(&argc, &argv);
//
//    // Get the number of processes
//    int world_size;
//    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
//
//    // Get the rank of the process
//    int world_rank;
//    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
//
//    // Get the name of the processor
//    char processor_name[MPI_MAX_PROCESSOR_NAME];
//    int name_len;
//    MPI_Get_processor_name(processor_name, &name_len);
//
//    // Print off a hello world message
//    if(world_rank == 0)
//        printf("\nMaster node");
//    else{
//        printf("Computation node %s, rank %d out of %d processors\n", processor_name, world_rank, world_size);

		bool run_flag = true;
		char *savefile = NULL;
		char *loadfile = NULL;
		bool distr = 0;
		int device_ID = 0;
		int progress_bar = 0;
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
		g_t* dp = new g_t;
		main_cuda(&run_flag, job_ID, savefile, loadfile, device_ID, distr, &progress_bar, dp);
	    cout << "Test cating g_t->void->g_t\t" << dp->np;
		for (int j = 0; j < dp->np; j++) {
			cout << dp->t[j] << '\t' << dp->x[j] << '\n';
		}
		//Send to master node
//    }
//    // Finalize the MPI environment.
//    MPI_Finalize();
	return 0;
}
