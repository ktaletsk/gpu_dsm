#include "textures_surfaces.h"
#include "chain.h"
#include "gpu_random.h"
#include "ensemble_kernel.cu"
#include "ensemble_call_block.h"

    //variable arrays, that are common for all the blocks

    gpu_Ran *d_random_gens;// device random number generators
    gpu_Ran *d_random_gens2;//first is used to pick jump process, second is used for creation of new entanglements
    //temporary arrays fpr random numbers sequences
    cudaArray*  d_uniformrand;// uniform random number supply //used to pick jump process
    cudaArray*  d_taucd_gauss_rand;// 1x uniform + 3x normal distributed random number supply// used for creating entanglements
    int steps_count=0;//time step count
    int *d_tau_CD_used;
    int *d_rand_used;
    
    cudaArray* d_a_QN; //device arrays for vector part of chain conformations
    cudaArray* d_a_tCD;// these arrays used by time evolution kernels
    cudaArray* d_b_QN;
    cudaArray* d_b_tCD;
    
    cudaArray* d_sum_W;// sum of probabilities for each entanglement
    cudaArray* d_stress;// stress calculation temp array
    
	// There are two arrays with the vector part of the chain conformations  on device.
	// And there is only one array with the scalar part of the chain conformations
	// Every timestep the vector part is copied from one array to another.
	// The coping is done in entanglement parallel portion of the code
	// This allows to use textures/surfaces and speeds up memory access
	// The scalar part(chain headers) is updated in the chain parallel portion of the code
	// Chain headers occupy less memory,and there are no specific memory access technics for them.
	
    void init_call_block(ensemble_call_block *cb,int nc,sstrentp chains, chain_head* chain_heads){ //allocates arrays, copies chain conformations to device
	//ensemble_call_block *cb pointer for call block structure, just ref parameter
	//int nc  is a number of chains in this ensemble_call_block.
	//sstrentp chains, chain_head* chain_heads pointers to array with the chain conformations
	
	//first take care about nc
	cb->nc=nc;
	cb->chains=chains;
	cb->chain_heads=chain_heads;
	
	
	cudaChannelFormatDesc channelDesc4 =cudaCreateChannelDesc(32, 32, 32, 32,cudaChannelFormatKindFloat);
	cudaChannelFormatDesc channelDesc2 =cudaCreateChannelDesc(32, 32, 0, 0,cudaChannelFormatKindFloat);
	cudaChannelFormatDesc channelDesc1 =cudaCreateChannelDesc(32, 0, 0, 0,cudaChannelFormatKindFloat);

	cudaMallocArray(&(cb->d_QN), &channelDesc4, z_max,cb->nc,cudaArraySurfaceLoadStore);
 	cudaMallocArray(&(cb->d_tCD), &channelDesc1, z_max,cb->nc,cudaArraySurfaceLoadStore);

	//blank dynamics probabalities
	float *buffer=new float[z_max*cb->nc];
	memset(buffer,0,sizeof(float)*z_max*cb->nc);
	cudaMemcpy2DToArray(d_sum_W, 0, 0, buffer, z_max*sizeof(float),z_max*sizeof(float),cb->nc,cudaMemcpyHostToDevice);
	delete[] buffer;
	//copy initial conformations to device
	cudaMemcpy2DToArray(cb->d_QN, 0, 0, cb->chains.QN, z_max*sizeof(float)*4,z_max*sizeof(float)*4,cb->nc,cudaMemcpyHostToDevice);
	cudaMemcpy2DToArray(cb->d_tCD, 0, 0, cb->chains.tau_CD, z_max*sizeof(float),z_max*sizeof(float),cb->nc,cudaMemcpyHostToDevice);
	 cudaMalloc((void**) &cb->gpu_chain_heads,sizeof(chain_head)*cb->nc);
	 cudaMemcpy(cb->gpu_chain_heads,cb->chain_heads, sizeof(chain_head)*cb->nc,  cudaMemcpyHostToDevice);
	 
	 // allocating device arrays
 	cudaMalloc(&(cb->d_dt), sizeof(float)*cb->nc);
	
	cudaMemset(cb->d_dt,0,sizeof(float)*cb->nc);
 	cudaMalloc(&(cb->reach_flag), sizeof(float)*cb->nc);
	cudaMalloc(&(cb->d_offset), sizeof(int)*cb->nc);
	cudaMalloc(&(cb->d_new_strent), sizeof(float)*4*cb->nc);
	cudaMalloc(&(cb->d_new_tau_CD), sizeof(float)*cb->nc);
	
	cudaMemset((cb->d_offset),0xff,sizeof(float)*cb->nc);
	
    }
    
       
    void time_step_call_block(float reach_time,ensemble_call_block *cb){ //bind textures_surfaces perform time evolution unbind textures_surfaces
	//ensemble_call_block *cb pointer for call block structure, just ref parameter
	//sstrentp chains, chain_head* chain_heads needed for debug//TODO not implemented
	
	cudaChannelFormatDesc channelDesc1 =cudaCreateChannelDesc(32, 0,0,0,cudaChannelFormatKindFloat);
	cudaChannelFormatDesc channelDesc4 =cudaCreateChannelDesc(32, 32,32,32,cudaChannelFormatKindFloat);

	//loop preparing
	cudaMemset(cb->reach_flag,0,sizeof(float)*cb->nc);
	dim3 dimBlock(tpb_strent_kernel, tpb_strent_kernel);
	dim3 dimGrid((z_max + dimBlock.x - 1) / dimBlock.x,(cb->nc + dimBlock.y - 1) / dimBlock.y);

	activate_block(cb);
	
	bool reach_flag_all=false;
	float *rtbuffer=new float[cb->nc];

	//Loop begins

	while (!reach_flag_all){
	    //odd time steps
// 	    cout<<steps_count<<'\n';
	    if (!(steps_count&0x00000001)){

		cudaBindTextureToArray(t_a_QN, d_a_QN, channelDesc4);
		cudaBindSurfaceToArray(s_b_QN, d_b_QN);
		cudaBindTextureToArray(t_a_tCD, d_a_tCD, channelDesc1);
		cudaBindSurfaceToArray(s_b_tCD, d_b_tCD);

		strent_kernel<<<dimGrid,dimBlock>>>(cb->gpu_chain_heads,cb->d_dt,cb->d_offset,cb->d_new_strent,cb->d_new_tau_CD);
		CUT_CHECK_ERROR("kernel execution failed");
// 	  	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_steps_count, &steps_count,sizeof(int)));

		chain_CD_kernel<<<(cb->nc+tpb_chain_kernel-1)/tpb_chain_kernel,tpb_chain_kernel>>>(cb->gpu_chain_heads,cb->d_dt,cb->reach_flag,reach_time,cb->d_offset,cb->d_new_strent,cb->d_new_tau_CD,d_rand_used,d_tau_CD_used);
		CUT_CHECK_ERROR("kernel execution failed");

// 		if (dbug)  get_chains_from_device_b();

		cudaUnbindTexture(t_a_QN);
		cudaUnbindTexture(t_a_tCD);
		  
	    }else{
		cudaBindTextureToArray(t_a_QN, d_b_QN, channelDesc4);
		cudaBindSurfaceToArray(s_b_QN, d_a_QN);
		cudaBindTextureToArray(t_a_tCD, d_b_tCD, channelDesc1);
		cudaBindSurfaceToArray(s_b_tCD, d_a_tCD);
		strent_kernel<<<dimGrid,dimBlock>>>(cb->gpu_chain_heads,cb->d_dt,cb->d_offset,cb->d_new_strent,cb->d_new_tau_CD);
		CUT_CHECK_ERROR("kernel execution failed");
// 	  	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_steps_count, &steps_count,sizeof(int)));

		chain_CD_kernel<<<(cb->nc+tpb_chain_kernel-1)/tpb_chain_kernel,tpb_chain_kernel>>>(cb->gpu_chain_heads,cb->d_dt,cb->reach_flag,reach_time,cb->d_offset,cb->d_new_strent,cb->d_new_tau_CD,d_rand_used,d_tau_CD_used);
		CUT_CHECK_ERROR("kernel execution failed");
		
// 		if (dbug) get_chains_from_device_a();

		cudaUnbindTexture(t_a_QN);
		cudaUnbindTexture(t_a_tCD);

	    }
	      
	    steps_count++;
	    // check for rand refill
	    if (steps_count%uniformrandom_count==0){
	      if (dbug) cout<<"steps_count "<<steps_count<<". random_textures_refill()\n";
	      	random_textures_refill(cb->nc);
		steps_count=0;
	    }
	    
	    // check for reached time
	    cudaMemcpy(rtbuffer,cb->reach_flag,sizeof(float)*cb->nc,cudaMemcpyDeviceToHost);
	    float sumrt=0;
	    for(int i=0;i<cb->nc;i++){
		sumrt+=rtbuffer[i];
	    }
	    reach_flag_all=(sumrt==cb->nc);

	    if (dbug){
		float *dtbuffer=new float[cb->nc];
		cudaMemcpy(dtbuffer,cb->d_dt,sizeof(float)*cb->nc,cudaMemcpyDeviceToHost);
		
//  		get_chain_from_device_call_block(cb);//TODO make header and turn it on


		for(int i=0;i<1+0*cb->nc;i++){//just one chain
		    cout<<"dt "<<dtbuffer[i]<<'\n';
// 		    cout<<NK<<'\n';
// 		    print(cout,cb->chain_index(i),cb->chain_heads[i]);
// 		    cout<<"\n";
		    cout<<"\n";
		}
		delete[] dtbuffer;

	    }
	  
	    
	}//loop end
	delete[] rtbuffer;
	deactivate_block(cb);

    }
    
    
    // utility functions
    //h means host(cpu) declarations
    //host copies of gpu inline access functions
    //purpose-- to recreate latest chain conformations from gpu memory (to account for delayed dynamics)
     int hmake_offset(int i,int offset){
    //offset&0xffff00)>>8 offset_index
    //offset&0xff-1; offset_dir
       return i>=((offset&0xffff00)>>8) ? i+((offset&0xff)-1) :i;
    }
    int hoffset_index(int offset){
       return ((offset&0xffff00)>>8) ;
    }
    
     int hoffset_dir(int offset){
       return (offset&0xff)-1; 
    }
    
     bool fetch_hnew_strent(int i, int offset){
       return (i==hoffset_index(offset))&&(hoffset_dir(offset)==-1); 
    } 
    
    
    void get_chain_from_device_call_block(ensemble_call_block *cb){ //copy chains back
    //NOTE accounts for delayed dynamics
    
	cudaMemcpy(cb->chain_heads,cb->gpu_chain_heads, sizeof(chain_head)*cb->nc,  cudaMemcpyDeviceToHost);

	cudaMemcpy2DFromArray(cb->chains.QN,sizeof(float)*z_max*4,cb->d_QN, 0, 0, z_max*sizeof(float)*4,cb->nc,cudaMemcpyDeviceToHost);
	cudaMemcpy2DFromArray(cb->chains.tau_CD,sizeof(float)*z_max,cb->d_tCD, 0, 0, z_max*sizeof(float),cb->nc,cudaMemcpyDeviceToHost);
	
	//delayed dynamics
	int *h_offset=new int[cb->nc];
	float4 *h_new_strent=new float4[cb->nc];
	float *h_new_tau_CD=new float[cb->nc];
	cudaMemcpy(h_offset,cb->d_offset, sizeof(int)*cb->nc,  cudaMemcpyDeviceToHost);
	cudaMemcpy(h_new_strent,cb->d_new_strent, sizeof(float4)*cb->nc,  cudaMemcpyDeviceToHost);
	cudaMemcpy(h_new_tau_CD,cb->d_new_tau_CD, sizeof(float)*cb->nc,  cudaMemcpyDeviceToHost);
	for(int i=0;i<cb->nc;i++){
	    if (hoffset_dir(h_offset[i])==-1){
		for(int j=z_max-1;j>0;j--){
		    chains.QN[i*z_max+j]=chains.QN[i*z_max+hmake_offset(j,h_offset[i])];
		    chains.tau_CD[i*z_max+j]=chains.tau_CD[i*z_max+hmake_offset(j,h_offset[i])];
		}
		chains.QN[i*z_max+hoffset_index(h_offset[i])]=h_new_strent[i];
		chains.tau_CD[i*z_max+hoffset_index(h_offset[i])]=h_new_tau_CD[i];
	    }else{
		for(int j=0;j<z_max-2;j++){
		    chains.QN[i*z_max+j]=chains.QN[i*z_max+hmake_offset(j,h_offset[i])];
		    chains.tau_CD[i*z_max+j]=chains.tau_CD[i*z_max+hmake_offset(j,h_offset[i])];

		}
	    }
	}

	delete[] h_offset;
	delete[] h_new_strent;
	delete[] h_new_tau_CD;
	
	
    }
    
    
    stress_plus calc_stress_call_block(ensemble_call_block *cb,int *r_chain_count){
      
	cudaChannelFormatDesc channelDesc4 =cudaCreateChannelDesc(32, 32, 32, 32,cudaChannelFormatKindFloat);

	CUDA_SAFE_CALL(cudaMemcpyToSymbol(dn_cha_per_call, &cb->nc,sizeof(int)));
	
	cudaBindTextureToArray(t_a_QN, cb->d_QN, channelDesc4);
	stress_calc<<<(cb->nc+tpb_chain_kernel-1)/tpb_chain_kernel,tpb_chain_kernel>>>(cb->gpu_chain_heads,cb->d_dt,cb->d_offset,cb->d_new_strent);
	CUT_CHECK_ERROR("kernel execution failed");
	cudaUnbindTexture(t_a_QN);
	
	float4 *stress_buf=new float4[cb->nc*2];
	cudaMemcpyFromArray(stress_buf,d_stress, 0, 0, cb->nc*sizeof(float4)*2,cudaMemcpyDeviceToHost);
	float4 sum_stress=make_float4(0.0f,0.0f,0.0f,0.0f);//stress: xx,yy,zz,xy
	float4 sum_stress2=make_float4(0.0f,0.0f,0.0f,0.0f);//stress: yz,xz; Lpp, Ree
	chain_head* tchain_heads;
	tchain_heads=new chain_head[cb->nc];
	
	cudaMemcpy(tchain_heads,cb->gpu_chain_heads, sizeof(chain_head)*cb->nc,  cudaMemcpyDeviceToHost);
	int chain_count=cb->nc;
	for (int j=0;j<cb->nc;j++){
	    if (tchain_heads[j].stall_flag==0){
	    sum_stress.x+=stress_buf[j*2].x;
	    sum_stress.y+=stress_buf[j*2].y;
	    sum_stress.z+=stress_buf[j*2].z;
	    sum_stress.w+=stress_buf[j*2].w;
	    sum_stress2.x+=stress_buf[j*2+1].x;
	    sum_stress2.y+=stress_buf[j*2+1].y;
	    sum_stress2.z+=stress_buf[j*2+1].z;
	    sum_stress2.w+=stress_buf[j*2+1].w;

	    // 	     cout<<"stress chain "<<j<<'\t'<<sum_stress.x<<'\t'<<sum_stress.y<<'\t'<<sum_stress.z<<'\t'<<sum_stress.w<<'\n';

	    }
	    else{ chain_count--;
	     cout<<"chain stall "<<j<<'\n';//TODO output gloval index
	    }
	}
	stress_plus rs;
	rs.xx=sum_stress.x/chain_count;
	rs.yy=sum_stress.y/chain_count;
	rs.zz=sum_stress.z/chain_count;
	rs.xy=sum_stress.w/chain_count;
	rs.yz=sum_stress2.x/chain_count;
	rs.zx=sum_stress2.y/chain_count;
	rs.Lpp=sum_stress2.z/chain_count;
	rs.Z=sum_stress2.w/chain_count;
	delete[]stress_buf;
	delete []tchain_heads;
	*r_chain_count=chain_count;
	return rs;
    }
    
    
    void free_block(ensemble_call_block *cb){//free memory
	cudaFree(cb->gpu_chain_heads);
	cudaFreeArray(cb->d_QN);
	cudaFreeArray(cb->d_tCD);

	cudaFree(cb->d_dt);
	cudaFree(cb->reach_flag);
	cudaFree(cb->d_offset);
	cudaFree(cb->d_new_strent);


    }

    void activate_block(ensemble_call_block *cb){//prepares block for performing time evolution
						 //i.e. copies chain conformations to working memory
	cudaChannelFormatDesc channelDesc1 =cudaCreateChannelDesc(32, 0,0,0,cudaChannelFormatKindFloat);
	cudaChannelFormatDesc channelDesc4 =cudaCreateChannelDesc(32, 32,32,32,cudaChannelFormatKindFloat);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(dn_cha_per_call, &cb->nc,sizeof(int)));

	if (!(steps_count&0x00000001)){
	    cudaMemcpyArrayToArray(d_a_QN,0,0,cb->d_QN,0,0,z_max*sizeof(float)*4*cb->nc,cudaMemcpyDeviceToDevice);
	    cudaMemcpyArrayToArray(d_a_tCD,0,0,cb->d_tCD,0,0,z_max*sizeof(float)*cb->nc,cudaMemcpyDeviceToDevice);
	}else{
	    cudaMemcpyArrayToArray(d_b_QN,0,0,cb->d_QN,0,0,z_max*sizeof(float)*4*cb->nc,cudaMemcpyDeviceToDevice);
	    cudaMemcpyArrayToArray(d_b_tCD,0,0,cb->d_tCD,0,0,z_max*sizeof(float)*cb->nc,cudaMemcpyDeviceToDevice);
	}
    }


    void deactivate_block(ensemble_call_block *cb){;//copies chain conformations to storing memory
    
    	if (!(steps_count&0x00000001)){
	    cudaMemcpyArrayToArray(cb->d_QN,0,0,d_a_QN,0,0,z_max*sizeof(float)*4*cb->nc,cudaMemcpyDeviceToDevice);
	    cudaMemcpyArrayToArray(cb->d_tCD,0,0,d_a_tCD,0,0,z_max*sizeof(float)*cb->nc,cudaMemcpyDeviceToDevice);
	}else{
	    cudaMemcpyArrayToArray(cb->d_QN,0,0,d_b_QN,0,0,z_max*sizeof(float)*4*cb->nc,cudaMemcpyDeviceToDevice);
	    cudaMemcpyArrayToArray(cb->d_tCD,0,0,d_b_tCD,0,0,z_max*sizeof(float)*cb->nc,cudaMemcpyDeviceToDevice);
	}  
    }
















