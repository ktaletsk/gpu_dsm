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

 #if !defined job_ID_h
 #define job_ID_h
#include <string.h>
#include <stdlib.h>

//this function simplifies running parallel instances of the code in one folder
//every job (indepentent process) receives unique number as a parameter
// this number is added to all filenames for every file generated
// job_ID is variable for the number
// filename_ID generates filename
// example: job_ID=10;
//          filename("log_") returns "log_10.dat"

int job_ID = 0;

char * filename_ID(string filename, bool temp) { //no prior allocation needed, but dealloc is on you
	if (job_ID != 0) {
		char *st1;
		int nn = int(ceil(log10(float(job_ID + 1))) + 1); //number of simbols need for number
		char *stmp2;
		stmp2 = new char[nn + 1];
		sprintf(stmp2, "%d", job_ID);

		if (temp){
			st1 = new char[strlen(9 + filename.c_str()) + 1 + nn + 4 + 1]; //filename + "_" +number +.dat + terminating 0
			strcpy(st1, "/tmp/dsm_");
			strcat(st1, filename.c_str());
		}
		else{
			st1 = new char[strlen(filename.c_str()) + 1 + nn + 4 + 1]; //filename + "_" +number +.dat + terminating 0
			strcpy(st1, filename.c_str());
		}
		strcat(st1, "_");
		strcat(st1, stmp2);
		strcat(st1, ".dat");
		return st1;
	} else {
		char *st1;
		if(temp){
			st1 = new char[5 + strlen(filename.c_str()) + 4 + 1]; //filename +.dat + terminating 0
			strcpy(st1, "/tmp/dsm_");
			strcat(st1, filename.c_str());
		}
		else{
			st1 = new char[strlen(filename.c_str()) + 4 + 1]; //filename +.dat + terminating 0
			strcpy(st1, filename.c_str());
		}
		strcat(st1, ".dat");
		return st1;
	}
};
#endif
