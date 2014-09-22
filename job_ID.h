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


    int job_ID=0;

    char * filename_ID(char *filename){//no prior allocation needed, but dealloc is on you
	if (job_ID!=0){
	    char *st1;
	    int nn=int(ceil(log10(float(job_ID+1)))+1);//number of simbols need for number
	    char *stmp2;
	    stmp2=new char [nn+1];
	    sprintf(stmp2,"%d",job_ID);
	    st1=new char[strlen(filename)+1+nn+4+1];//filename + "_" +number +.dat + terminating 0
	    strcpy(st1,filename);
	    strcat(st1,"_");
	    strcat(st1,stmp2);
	    strcat(st1,".dat");
	    return st1;
	}else{
	    char *st1;
  	    st1=new char[strlen(filename)+4+1];//filename +.dat + terminating 0
	    strcpy(st1,filename);
	    strcat(st1,".dat");
	    return st1;
	}
    };
      
      

#endif