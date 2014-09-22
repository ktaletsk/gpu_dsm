#include "gpu_random.h"
#include <iostream>
#include <stdlib.h>
#include <fstream>
#include "ensemble.h"
#include "cudautil.h"
#include "cuda_call.h"
#include "textures_surfaces.h"

    void random_textures_refill(int n_cha);
    void random_textures_fill(int n_cha);

#include "ensemble_call_block.cu"

    using namespace std;


#define chains_per_call 32000
 //MAX surface size is 32768
 
 
 
    sstrentp chains;// host chain conformations

    	//on device there are two arrays with vector part of chain conformations
	// and only one array with scalar part chain conformation
	// every time the vector part is copied from one array to another
	// coping is done in entanglement parallel portion of the code
	// this allows to use textures/surfaces, which speeds up memory access
	// scalar part(chain headers) are update in the chain parallel portion of the code
	// chain headers are occupied much smaller memory, no specific memory access technic are used for them.
	// depending one odd or even number of time step were performed,
	//one of the get_chains_from_device_# should be used
	
    
    
    chain_head* chain_heads;// host device chain headers arrays, store scalar variables of chain conformations

    int chain_blocks_number;
    ensemble_call_block *chain_blocks;

    //host constants
    int N_cha;
    int NK;
    int z_max;
    float Be;
    float kxx,kxy,kxz,kyx,kyy,kyz,kzx,kzy,kzz;

    bool  dbug=false;//true;
    //bool *reach_time_flag=NULL;
  
    //navigation
    sstrentp chain_index(const int i){//absolute navigation i - is a global index of chains i:[0..N_cha-1]
	sstrentp ptr;
	ptr.QN=&(chains.QN[z_max*i]);
	ptr.tau_CD=&(chains.tau_CD[z_max*i]);
	return ptr;
    }

    sstrentp chain_index(const int bi,const int i){//block navigation
    //bi is a block index bi :[0..chain_blocks_number]
    //i - is a chain index in the block bi  i:[0..chains_per_call-1]
	sstrentp ptr;
	ptr.QN=&(chains.QN[z_max*(bi*chains_per_call +i)]);
	ptr.tau_CD=&(chains.tau_CD[z_max*(bi*chains_per_call)]);
	return ptr;
    }
    
    void chains_malloc(){
      	//setup z_max modification maybe needed for large \beta
	z_max=NK;// current realization limits z to 2^23
	chain_heads=new chain_head[N_cha];
	chains.QN=new float4[N_cha*z_max];
	chains.tau_CD=new float[N_cha*z_max];
    }
    
    void host_chains_init(){
	    chains_malloc();
	    cout<<"generating chain conformations on host..";
	    for(int i=0; i<N_cha;i++){
	      sstrentp ptr=chain_index(i);
	      chain_init(&(chain_heads[i]),ptr,NK,z_max);
	    }
	    cout<<"done\n";
    }
    
 
  //preparation of constants/arrays/etc
    void gpu_init(int seed){
	cout<<"preparing GPU chain conformations..\n";

  	CUDA_SAFE_CALL(cudaMemcpyToSymbol(dBe, &Be,sizeof(float)));
  	CUDA_SAFE_CALL(cudaMemcpyToSymbol(dnk, &NK,sizeof(int)));
	

  	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_z_max, &z_max,sizeof(int)));
	
	
    	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_kappa_xx, &kxx,sizeof(float)));
    	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_kappa_xy, &kxy,sizeof(float)));
    	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_kappa_xz, &kxz,sizeof(float)));
    	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_kappa_yx, &kyx,sizeof(float)));
    	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_kappa_yy, &kyy,sizeof(float)));
    	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_kappa_yz, &kyz,sizeof(float)));
    	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_kappa_zx, &kzx,sizeof(float)));
    	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_kappa_zy, &kzy,sizeof(float)));
    	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_kappa_zz, &kzz,sizeof(float)));
	
	//copy pcd constant to device
	float cdtemp=pcd->W_CD_destroy_aver()/Be;
  	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_CD_create_prefact, &cdtemp,sizeof(float)));
  	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_g, &(pcd->g),sizeof(float)));
  	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_alpha, &(pcd->alpha),sizeof(float)));
  	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_tau_0, &(pcd->tau_0),sizeof(float)));
  	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_tau_max, &(pcd->tau_max),sizeof(float)));
  	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_tau_d, &(pcd->tau_d),sizeof(float)));
	cdtemp=1.0f/pcd->tau_d;
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_tau_d_inv, &(cdtemp),sizeof(float)));

	cdtemp=1.0f/pcd->At;
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_At, &(cdtemp),sizeof(float)));
	cdtemp=powf(pcd->tau_0,pcd->alpha);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_Dt, &(cdtemp),sizeof(float)));
	cdtemp=-1.0f/pcd->alpha;
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_Ct, &(cdtemp),sizeof(float)));
	cdtemp=pcd->normdt/pcd->Adt;
  	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_Adt, &cdtemp,sizeof(float)));
	cdtemp=pcd->Bdt/pcd->normdt;
  	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_Bdt, &cdtemp,sizeof(float)));
	cdtemp=-1.0f/(pcd->alpha-1.0f);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_Cdt, &cdtemp,sizeof(float)));
	cdtemp=powf(pcd->tau_0,pcd->alpha-1.0f);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_Ddt, &(cdtemp),sizeof(float)));

	cout<<" device constants done\n";

	int rsz=chains_per_call;
	if (N_cha<chains_per_call) rsz=N_cha;
	cudaChannelFormatDesc channelDesc4 =cudaCreateChannelDesc(32, 32, 32, 32,cudaChannelFormatKindFloat);
	cudaChannelFormatDesc channelDesc1 =cudaCreateChannelDesc(32, 0, 0, 0,cudaChannelFormatKindFloat);

	cudaMallocArray(&d_a_QN, &channelDesc4, z_max,rsz,cudaArraySurfaceLoadStore);
 	cudaMallocArray(&d_a_tCD, &channelDesc1, z_max,rsz,cudaArraySurfaceLoadStore);
	
 	cudaMallocArray(&d_b_QN, &channelDesc4, z_max,rsz,cudaArraySurfaceLoadStore);
 	cudaMallocArray(&d_b_tCD, &channelDesc1,z_max,rsz,cudaArraySurfaceLoadStore);
	
	cudaMallocArray(&d_sum_W, &channelDesc1, z_max,rsz,cudaArraySurfaceLoadStore);
 	cudaMallocArray(&d_stress, &channelDesc4, rsz*2,0,cudaArraySurfaceLoadStore);
	cudaBindSurfaceToArray(s_stress, d_stress);
	cudaBindSurfaceToArray(s_sum_W, d_sum_W);

	cout<<"\n";
	cout<<" GPU random generator init: \n";
	cout<<"  device random generators 1 seeding..";
	cudaMalloc(&d_random_gens, sizeof(gpu_Ran)*rsz);

	gr_array_seed(d_random_gens,rsz,seed*rsz);
	cout<<".done\n";
	cout<<"  device random generators 2 seeding..";
	cudaMalloc(&d_random_gens2, sizeof(gpu_Ran)*rsz);
	gr_array_seed(d_random_gens2,rsz,(seed+1)*rsz);
	cout<<".done\n";

	cout<<"  preparing random number sequence..";
	random_textures_fill(rsz);
	cout<<".done\n";
	cout<<" GPU random generator init done.\n";
	cout<<"\n";
	
	chain_blocks_number=(N_cha+chains_per_call-1)/chains_per_call;
	cout<<" Number of ensemble blocks "<<chain_blocks_number<<'\n';
	
	chain_blocks=new ensemble_call_block[chain_blocks_number];
	for (int i=0;i<chain_blocks_number-1;i++){
	    init_call_block(&(chain_blocks[i]),chains_per_call,chain_index(i,0), &(chain_heads[i*chains_per_call]));
	    cout<<"  copying chains to device block "<<i+1<<". chains in the ensemble block "<<chains_per_call<<'\n';
	}
	init_call_block(&(chain_blocks[chain_blocks_number-1]),(N_cha-1)%chains_per_call+1,chain_index(chain_blocks_number-1,0), &(chain_heads[(chain_blocks_number-1)*chains_per_call]));
	cout<<"  copying chains to device block "<<chain_blocks_number<<". chains in the ensemble block "<<(N_cha-1)%chains_per_call+1<<'\n';
	cout<<" device chains done\n";
	
	cout<<"init done\n";
    }
    
    stress_plus calc_stress()
    {
	stress_plus tmps=make_stress_plus(0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0);
	int total_chains=0;
	for(int i=0; i<chain_blocks_number;i++){
	    int cc;
	    stress_plus tmp=calc_stress_call_block( &(chain_blocks[i]),&cc);
	    total_chains+=cc;
	    double w=double(cc);
	    tmps=tmps+tmp*w;
	}
	return tmps/total_chains;
    }
    

    void gpu_time_step(float reach_time)
    {
	for(int i=0; i<chain_blocks_number;i++){
	    time_step_call_block(reach_time,&(chain_blocks[i]));
	}
	
    }
    
    void get_chains_from_device()//Copies chains back to host memory
    {
    	for(int i=0; i<chain_blocks_number;i++){
	     get_chain_from_device_call_block(&(chain_blocks[i]));
	}
    }
    
    
    void gpu_clean()
    {
	

		//debug checks
	// 	for(int i=0; i<chain_blocks_number;i++){
	// 	     get_chain_from_device_call_block(&(chain_blocks[i]));
	// 	}
	// 	float *dtbuffer=new float[N_cha];
	// 	cudaMemcpy(dtbuffer,chain_blocks[0].d_dt,sizeof(float)*(chain_blocks[0].nc),cudaMemcpyDeviceToHost);
		

	// 	for(int i=0;i<1+0*N_cha;i++){
	// 	    cout<<"chain "<<'\t'<<i<<" dt "<<dtbuffer[i]<<'\n';
	// 	    cout<<NK<<'\n';
	// 
	// 	    print(cout,chain_index(i),chain_heads[i]);
	// 	    
	// // 	    cout<<"W_sd\n";
	// // 	    cout<<chain_heads[i].W_SD_c_1<<'\t'<<chain_heads[i].W_SD_d_1<<'\n';
	// // 	    cout<<chain_heads[i].W_SD_c_z<<'\t'<<chain_heads[i].W_SD_d_z<<'\n';
	// 
	// 	cout<<"\n";
	// 	cout<<"\n";
	// 	}
	// 	delete[] dtbuffer;
	    
	cout<<"Memory cleanup.. ";
	for(int i=0; i<chain_blocks_number;i++){
	    free_block(&(chain_blocks[i]));
	}
	delete[] chain_blocks;
	//free chains chain_heads?
	
	cudaFreeArray(d_a_QN);
	cudaFreeArray(d_a_tCD);
	cudaFreeArray(d_b_QN);
	cudaFreeArray(d_b_tCD);
	cudaFreeArray(d_sum_W);
	cudaFreeArray(d_stress);
	
	cudaFreeArray(d_uniformrand);
	cudaFreeArray(d_taucd_gauss_rand);
	cudaFree(d_tau_CD_used);
	
	cudaFree(d_random_gens);
	cudaFree(d_random_gens2);
	cout<<"done.\n";

    }
 
 
    void random_textures_fill(int n_cha)
    {
	cudaChannelFormatDesc channelDesc1 =cudaCreateChannelDesc(32, 0, 0, 0,cudaChannelFormatKindFloat);
	cudaChannelFormatDesc channelDesc4 =cudaCreateChannelDesc(32,32,32,32,cudaChannelFormatKindFloat);

	cudaMallocArray(&(d_taucd_gauss_rand),&channelDesc4,uniformrandom_count,n_cha,cudaArraySurfaceLoadStore);
	cudaMallocArray(&(d_uniformrand),&channelDesc1,uniformrandom_count,n_cha,cudaArraySurfaceLoadStore);

	cudaMalloc((void**) &d_rand_used,sizeof(int)*n_cha);
	cudaMemset(d_rand_used,0,sizeof(int)*n_cha);
	cudaMalloc((void**) &d_tau_CD_used,sizeof(int)*n_cha);
	cudaMemset(d_tau_CD_used,0,sizeof(int)*n_cha);
	
	gr_fill_surface_uniformrand(d_random_gens,n_cha,uniformrandom_count, d_uniformrand);
	cudaDeviceSynchronize();
	
	int taucd_gauss_count=uniformrandom_count;
	gr_fill_surface_taucd_gauss_rand(d_random_gens2,n_cha,taucd_gauss_count,d_taucd_gauss_rand);
	
	cudaBindTextureToArray(t_uniformrand, d_uniformrand, channelDesc1);
	cudaBindTextureToArray(t_taucd_gauss_rand, d_taucd_gauss_rand, channelDesc4);
    }
  
  
  
  
    void random_textures_refill(int n_cha)
    {
	if (chain_blocks_number!=1) n_cha=chains_per_call;
	
	cudaChannelFormatDesc channelDesc =cudaCreateChannelDesc(32, 0, 0, 0,cudaChannelFormatKindFloat);
	cudaChannelFormatDesc channelDesc4 =cudaCreateChannelDesc(32, 32,32,32,cudaChannelFormatKindFloat);

	cudaUnbindTexture(t_uniformrand);
	gr_fill_surface_uniformrand(d_random_gens,n_cha,uniformrandom_count, d_uniformrand);
	cudaMemset(d_rand_used,0,sizeof(int)*n_cha);
	cudaBindTextureToArray(t_uniformrand, d_uniformrand, channelDesc);
	cudaDeviceSynchronize();

	//tau_cd gauss 3d vector
	cudaUnbindTexture(t_taucd_gauss_rand);
	gr_refill_surface_taucd_gauss_rand(d_random_gens2,n_cha,d_tau_CD_used,d_taucd_gauss_rand);
	cudaMemset(d_tau_CD_used,0,sizeof(int)*n_cha);
	cudaBindTextureToArray(t_taucd_gauss_rand, d_taucd_gauss_rand, channelDesc4);
	cudaDeviceSynchronize();
    }
  
  
  
  
	void save_to_file(char *filename)
	{  
	    ofstream file (filename, ios::out|ios::binary);
	    if (file.is_open()){
		file.write((char*)&N_cha,sizeof(int));

	   	for(int i=0;i<N_cha;i++){
		    save_to_file(file,chain_index(i),chain_heads[i]);
		}

	    }
	    else cout << "file error\n";
	}

	void load_from_file(char *filename)
	{
	  
	    chains_malloc();
	    ifstream file (filename, ios::in|ios::binary);
	    if (file.is_open()){
		int ti;
		file.read((char*)&ti,sizeof(int));
		if (ti!=N_cha) {cout << "ensemble size mismatch\n";
		  exit(2);
		}

	   	for(int i=0;i<N_cha;i++){
		    cout<<i;
		    load_from_file(file,chain_index(i),&chain_heads[i]);
		}

	    }
	    else cout << "file error\n";
	}

	
  
  
  
  
  
  
  