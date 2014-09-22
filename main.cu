#include <iostream>
#include <fstream>
#include "gpu_random.h"
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

    using namespace std;
//     namespace io = boost::iostreams;

    Ran eran(1);
    p_cd *pcd;

    //synonyms:
    //GPU - device
    //CPU - host
 int main(int narg,char** arg)
{
    char *savefile=NULL;
    char *loadfile=NULL;
    int device_ID=0;
    cout<<'\n';

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
	cout<<"job_ID: "<<job_ID<<'\n';
	cout<<"using "<<job_ID<< " as a seed for random number generator\n";
	cout<<"\"_"<<job_ID<<"\" will be appended to filename for all files generated\n\n";
    }
    //First checking device
    checkCUDA(device_ID);
    //init random
    eran.seed(job_ID*N_cha);

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
    
    in>>kxx>>kxy>>kxz>>kyx>>kyy>>kyz>>kzx>>kzy>>kzz;
    
//     in>>CD_flag;//TODO CD off not implemented
//     int int_t;
//     in>>int_t;//TODO SD off not implemented
//     in>>int_t;//TODO G not implemented
//     in>>int_t;//TODO R  not implemented
//     in>>int_t;//TODO f_d not implemented
    float simulation_time=100000;
    float t_step_size=200;
    in>>t_step_size;
    in>>simulation_time;

    cout <<"\nsimulation parameters:\n";
    cout<<"NK Be N_cha"<<"\n";
    cout<<NK<<'\t'<< Be<<'\t'<<N_cha<<"\n";
    cout<<"deformation tensor:"<<"\n";
    cout<<" "<<kxx<<" "<<kxy<<" "<<kxz<<"\n"<<kyx<<" "<<kyy<<" "<<kyz<<"\n"<<kzx<<" "<<kzy<<" "<<kzz<<'\n';;
    cout<<"simulation time, sync time"<<"\n";
    cout<< simulation_time<<'\t'<< t_step_size<<'\n'<<'\n';

//toy parameters    
//     Be=1.0;
//     NK=46;
//     N_cha=4000;
//     kxy=8.16e-05;


    CD_flag=1;
    pcd=new p_cd(Be,NK,&eran);

    if (loadfile!=NULL){//load chain conformations from file
      	cout<<"loading chain conformations from "<<loadfile<<"..";
	load_from_file(loadfile);
	cout<<"done.\n";
    }else host_chains_init();// or generate equilibrium conformations
    
    gpu_ran_init();//init gpu random module
    gpu_init(job_ID);//prepare GPU to run DSM calculation

    ctimer timer;
    
    //tau file
    ofstream tau_file;
    tau_file.open(filename_ID("tau"));
    cout<<"output file: "<<filename_ID("tau")<<'\n';
    timer.start();
    //main loop
    cout<<"performing time evolution for the ensemble..\n";
    cout<<"time\tstress tensor(xx yy zz xy yz xz)\t<Lpp>\t<Z>\n";
    for (float t_step=0;t_step<simulation_time;t_step+=t_step_size){
	gpu_time_step(t_step+t_step_size);  
	stress_plus stress=calc_stress();
	cout<<t_step+t_step_size<<'\t'<<stress<<'\n';
	tau_file<<t_step+t_step_size<<'\t'<<stress<<'\n';
	tau_file.flush();
    }
    timer.stop();
    cout<<"time evolution done.\n";

    get_chains_from_device();
//     z_plot(chain_heads,Be, NK,N_cha);

    if (savefile!=NULL){
	cout<<"saving chain conformations to "<<savefile<<"..";
	save_to_file(savefile);
	cout<<"done.\n";
    }

    
    gpu_clean();
    tau_file.close();
    cout<<"Calculation time: "<<timer.elapsedTime()<<" milliseconds\n";
    return 0;
}


 