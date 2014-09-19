#include <iostream>
#include <fstream>
 #include "gpu_random.h"
// #include "timer.h"
#include "cuda_call.h"
#include "cudautil.h"
#include "random.h"
#include "ensemble.h"
#include "timer.h"
#include "detailed_balance.h"
// #include "fortran_comment_filter.h"
// #include <boost/iostreams/device/file.hpp>
// #include <boost/iostreams/filtering_stream.hpp>
#include "job_ID.h"

#include "orientation_tensor.h"

    using namespace std;
//     namespace io = boost::iostreams;

// __device__ int temp;
    Ran eran(1);
    p_cd *pcd;

//     int job_ID=1;
 int main(int narg,char** arg)
{
    char *savefile=NULL;
    char *loadfile=NULL;
    int device_ID=0;


//     if (narg>1) cout<<"command line parameters:\n";
    int k=1;
    while (k<narg){
      if (k==1){    //processing job_ID
	job_ID=atoi(arg[k]);

      }
      if((strcmp(arg[k],"-s")==0)&&(k+1<narg)){//save file
	savefile=new char[strlen(arg[k+1])+1];
	strcpy(savefile,arg[k+1]);
	cout<<"final chain conformations will be saved to "<<savefile<<'\n';
	k++;
      }
      if((strcmp(arg[k],"-l")==0)&&(k+1<narg)){//save file
	loadfile=new char[strlen(arg[k+1])+1];
	strcpy(loadfile,arg[k+1]);
	k++;
      }
      if((strcmp(arg[k],"-d")==0)&&(k+1<narg)){//save file
	device_ID=atoi(arg[k+1]);
	k++;
      }
      k++;
    }
    if (job_ID!=0){
	cout<<"\njob_ID: "<<job_ID<<'\n';
	cout<<"using "<<job_ID<< " as a seed for random number generator\n";
	cout<<"\"_"<<job_ID<<"\" will be appended to filename for all files generated\n\n";
    }
    //First checking device
    checkCUDA(device_ID);
    //init random
    eran.seed(job_ID);//TODO seed with job_ID*N_cha

    //using boost filters to remove comments from input file
//     io::filtering_istream in;
//     in.push(fortan_comments_input_filter());
//     in.push(io::file_source("input.dat"));
    ifstream in;
    in.open("input.dat");

    //read parameters;
    in>>Be;
    in>>NK;
    in>>N_cha;
    
    //since non shear flow not implemented dummy variables used
    in>>kxx>>kxy>>kxz>>kyx>>kyy>>kyz>>kzx>>kzy>>kzz;
    gamma_dot=kxy;
    
    in>>CD_flag;//TODO CD off not implemented
    int int_t;
    in>>int_t;//TODO SD off not implemented
    in>>int_t;//TODO G not implemented
    in>>int_t;//TODO R  not implemented
    in>>int_t;//TODO f_d not implemented
    in>>int_t;//TODO D not implemented
    float simulation_time=100000;
    float t_step_size=200;
    in>>t_step_size;
    in>>simulation_time;
    float float_t;
    in>>float_t;;//TODO saving not implemented
    in>>float_t;;//TODO entanglement complexity not implemented

    cout<<"NK Be N_cha gamma_dot"<<"\n";
    cout<<NK<<'\t'<< Be<<'\t'<<N_cha<<'\t'<<gamma_dot<<"\n";
//     Be=1.0;
//     NK=46;
//     N_cha=4000;
//     gamma_dot=8.16e-05;


    CD_flag=1;
    pcd=new p_cd(Be,NK,&eran);

    if (loadfile!=NULL){
      	cout<<"loading chain conformations from "<<loadfile<<"..";
	load_from_file(loadfile);
	cout<<"done.\n";
    }else host_chains_init();
    
    gpu_ran_init();
    gpu_init(job_ID);
    ctimer timer;
    
    //tau file
    ofstream tau_file;
//     cout<<filename_ID("tau")<<'\n';
    tau_file.open(filename_ID("tau"));
    timer.start();
    for (float t_step=0;t_step<simulation_time;t_step+=t_step_size){
  //      cout<<"time_step "<<t_step<<'\n';
      gpu_time_step(t_step+t_step_size);
  //     z_plot(chain_heads,Be, NK,N_cha);
      
      stress_plus stress=calc_stress();
  //     cout<<"stress tensor "<<stress.x<<'\t'<<stress.y<<'\t'<<stress.z<<'\t'<<stress.w<<'\n';
      cout<<t_step+t_step_size<<'\t'<<stress<<'\n';
      tau_file<<t_step+t_step_size<<'\t'<<stress<<'\n';
      tau_file.flush();
    }
      timer.stop();
      get_chains_from_device();
      if (savefile!=NULL){
	  cout<<"saving chain conformations to "<<savefile<<"..";
	  save_to_file(savefile);
	  cout<<"done.\n";
      }
      //TODO remove 
      cout<<"orientation tensor\n";
      calc_o_tensor(Be);

    
      gpu_clean();
      tau_file.close();
  //  test_random();
	  cout<<" running time: "<<timer.elapsedTime()<<" milliseconds\n";
  return 0;
}


 