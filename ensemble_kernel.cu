//Short intro
//Cuda devices posses enormous computation capabilities,
//however memory access (especially writing) is relatively slow.
//Unfortinately DSM flow simulation require to update significant part 
//of chain conformations variables every time step, which normally bottlenecks the performance.
// First time conformation updated when flow deformation of strand orientation vectors is applied, second time when jump process is applied.
// If the jump process is SD shift, only two neigbouring N_i must be updated,
// but in case entanglement creation/destruction major portion of chain conformation
// arrays must be moved. On GPU it is a very expensive operation,
// almost as expensive as updating {Q_i} during deformation.
// Thus we combined two conformation updates into one.
// It is done through "delayed dynamics". This means that jump process is not applied
// immediately, but information about it stored  in temporary variables until deformation applied. 
// Next time step shifting of arrays applied simultaneously together with flow deformation.





#ifndef _ENSEMBLE_KERNEL_
#define _ENSEMBLE_KERNEL_

#if defined(_MSC_VER)
#define uint unsigned int
#endif

#define tpb_chain_kernel 256
#define tpb_strent_kernel 32

#include "textures_surfaces.h"
#include "chain.h"




    //d means device variables
    __constant__ float dBe;
    __constant__ int dnk;
    __constant__ int d_z_max;//actuall array size. might come useful for large beta values and for polydisperse systems

    __constant__ int dn_cha_per_call;//number of chains in this call. cannot be bigger than chains_per_call
    
    __constant__ float d_kappa_xx,d_kappa_xy,d_kappa_xz,d_kappa_yx, d_kappa_yy,d_kappa_yz,d_kappa_zx,d_kappa_zy,d_kappa_zz;

    //CD constants
    __constant__ float d_CD_create_prefact;
    __constant__	float d_At,d_Ct,d_Dt,d_Adt,d_Bdt,d_Cdt,d_Ddt;
    __constant__ float d_g, d_alpha ,d_tau_0,d_tau_max,d_tau_d,d_tau_d_inv;
    
    //Next variables actually declared in ensemble_call_block.h
//    float *d_dt; // time step size from prevous time step. used for appling deformation
//    float *reach_flag;// flag that chain evolution reached required time
                     //copied to host each times step

    // delayed dynamics --- how does it work:
    // There are entanglement parallel portion of the code and chain parallel portion.
    // The entanglement parallel part applies flow deformation and calculates jump process probabilities.
    // The chain parallel part picks one of the jump processes, generates a new orientation vector and a tau_CD if needed.
    // It applies only some simpliest chain conformation changes(SD shifting).
    // The Information about complex chain conformation changes(entanglement creation/destruction) is stored in temp arrays d_offset, d_new_strent,d_new_tau_CD.
    // Complex changes are applied next time step by entanglement parallel part.


//    int *d_offset;//coded shifting parameters
//    float4 *d_new_strent;//new strent which shoud be inserted in the middle of the chain//TODO two new_strents will allow do all the updates at once
//    float *d_new_tau_CD;//new life time

    
   
   //offset in 2 component vector {shifting starting index, shifting direction}
   //offset stores both components in the one int variable
   //index in first 3 bytes, direction in last byte
    __device__ __forceinline__ int offset_code(int offset_index,int offset_dir){
      return (offset_dir+1)|(offset_index<<8);
    }

    // returns i or i+/- 1 from offset
    __device__ __forceinline__ int make_offset(int i,int offset){
    //offset&0xffff00)>>8 offset_index
    //offset&0xff-1; offset_dir
       return i>=((offset&0xffff00)>>8) ? i+((offset&0xff)-1) :i;
    }
    
    //returns components of offset
    __device__ __forceinline__ int offset_index(int offset){
       return ((offset&0xffff00)>>8) ;
    }
    
    __device__ __forceinline__ int offset_dir(int offset){
       return (offset&0xff)-1; 
    }
    
    //returns true if d_new_strent should be inserted at index i
    __device__ __forceinline__ bool fetch_new_strent(int i, int offset){
       return (i==offset_index(offset))&&(offset_dir(offset)==-1); 
    }
    
    
    //deformation    
    __device__ __forceinline__ float4 kappa(const float4 QN, const float dt)
    {					//Qx is different for consitency with old version
	    return	make_float4(QN.x+dt*d_kappa_xx*QN.x+dt*d_kappa_xy*QN.y+dt*d_kappa_xz*QN.z,
				   QN.y+dt*(d_kappa_yx*QN.x+d_kappa_yy*QN.y+d_kappa_yz*QN.z),
				   QN.z+dt*(d_kappa_zx*QN.x+d_kappa_zy*QN.y+d_kappa_zz*QN.z),
				   QN.w);
    }
    
    
    
    //lifetime generation from uniform random number p
    __device__ __forceinline__ float d_tau_CD_f_d_t(float p) {
	return p<d_Bdt ? __powf(p*d_Adt+d_Ddt,d_Cdt): d_tau_d_inv;
    }
    __device__ __forceinline__ float d_tau_CD_f_t(float p) {
	return p<1.0f-d_g ? __powf(p*d_At+d_Dt,d_Ct): d_tau_d_inv;
    }

	
    //The entanglement parallel part of the code
    //2D kernel: i- entanglement index j - chain index
    __global__ __launch_bounds__(tpb_strent_kernel*tpb_strent_kernel) void strent_kernel(chain_head* gpu_chain_heads,float *tdt,int *d_offset,float4 *d_new_strent,float *d_new_tau_CD){//TODO add reach flag
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	int j=blockIdx.y*blockDim.y+threadIdx.y;
	if ((j>=dn_cha_per_call)||(i>=d_z_max)) return;
	int tz=gpu_chain_heads[j].Z;
	if (i>=tz) return;
	int oft=d_offset[j];
//using texture fetch
	float4 QN=tex2D(t_a_QN,make_offset(i,oft),j);// all access to strents is done through two operations: first texture fetch
	if (fetch_new_strent(i,oft)) QN=d_new_strent[j];//second check if strent created last time step should go here
	float tcd=tex2D(t_a_tCD,make_offset(i,oft),j);
	if (fetch_new_strent(i,oft)) tcd=d_new_tau_CD[j];
//transform
	float dt=tdt[j];
	QN= kappa(QN,dt);
 	float2 wsh=make_float2(0.0f,0.0f);

//fetch next strent	
	if (i<tz-1){
	    float4 QN2=tex2D(t_a_QN,make_offset(i+1,oft),j);
	    if (fetch_new_strent(i+1,oft)) QN2=d_new_strent[j];

//transform
	    QN2= kappa(QN2,dt);
 //w_shift probability calc	

	    float Q=QN.x*QN.x+QN.y*QN.y+QN.z*QN.z;
	    float Q2=QN2.x*QN2.x+QN2.y*QN2.y+QN2.z*QN2.z;

	    if (QN2.w>1.0f) {//N=1 mean that shift is not possible, also ot will lead to dividing on zero error
	  	//float prefact=__powf( __fdividef(QN.w*QN2.w,(QN.w+1)*(QN2.w-1)),0.75f);
		//TODO try replacing powf with sqrt(x*x*x)

		float 	sig1=__fdividef(0.75f,QN.w*(QN.w+1));
		float 	sig2=__fdividef(0.75f,QN2.w*(QN2.w-1));
		float prefact1=(Q ==0.0f) ? 1.0f : __fdividef(QN.w,(QN.w+1));
		float prefact2=(Q2 ==0.0f) ? 1.0f : __fdividef(QN2.w,(QN2.w-1));
		float f1=(Q ==0.0f) ? 2.0f* QN.w+0.5f: QN.w;
		float f2=(Q2 ==0.0f) ? 2.0f* QN2.w-0.5f: QN2.w;
		float friction=__fdividef(2.0f,f1+f2);
		wsh.x=friction*__powf(prefact1*prefact2,0.75f)*__expf(Q*sig1-Q2*sig2);
	    }
	    if (QN.w>1.0f) {//N=1 mean that shift is not possible, also ot will lead to dividing on zero error

		float 	sig1=__fdividef(0.75f,QN.w*(QN.w-1.0f));
		float 	sig2=__fdividef(0.75f,QN2.w*(QN2.w+1.0f));
		float prefact1=(Q ==0.0f) ? 1.0f : __fdividef(QN.w,(QN.w-1.0f));
		float prefact2=(Q2 ==0.0f) ? 1.0f : __fdividef(QN2.w,(QN2.w+1.0f));
		float f1=(Q ==0.0f) ? 2.0f* QN.w-0.5f: QN.w;
		float f2=(Q2 ==0.0f) ? 2.0f* QN2.w+0.5f: QN2.w;
		float friction=__fdividef(2.0f,f1+f2);
		wsh.y=friction*__powf(prefact1*prefact2,0.75f)*__expf(-Q*sig1+Q2*sig2);
	    }
	    //write probabilities into temp array(surface)
	    surf2Dwrite(wsh.x+wsh.y+tcd+d_CD_create_prefact*(QN.w-1.0f),s_sum_W,4*i,j);
 	 }
    //write updated chain conformation
    surf2Dwrite(QN,s_b_QN,16*i,j);
    surf2Dwrite(tcd,s_b_tCD,4*i,j);
}


__global__ __launch_bounds__(tpb_chain_kernel)
void chain_CD_kernel(chain_head* gpu_chain_heads,float *tdt,float *reach_flag,float reach_time,int *d_offset,float4 *d_new_strent,float *d_new_tau_CD, int *rand_used,int *tau_CD_used){
	int i=blockIdx.x*blockDim.x+threadIdx.x;

	if (i>=dn_cha_per_call) return;
//setup local variables
	int tz=gpu_chain_heads[i].Z;
 	uint oft=d_offset[i];
	d_offset[i]=offset_code(0xffff,+1);

	if ((gpu_chain_heads[i].time>=reach_time)||(gpu_chain_heads[i].stall_flag!=0)){reach_flag[i]=1;tdt[i]=0.0f;return;}
	float olddt=tdt[i];
	float4 new_strent=d_new_strent[i];
	float new_tCD=d_new_tau_CD[i];
// sum W_SD_shifts
	float sum_wshpm=0.0f;
	float tsumw;
	for (int j=0;j<tz-1;j++){
	  surf2Dread(&tsumw,s_sum_W,4*j,i);
	  sum_wshpm+=tsumw;
	}
// W_SD_c/d calc
	float W_SD_c_1=0.0f,W_SD_d_1=0.0f;
	float W_SD_c_z=0.0f,W_SD_d_z=0.0f;
	//declare vars to reuse later
	float4 QNheadn,QNtailp;

	float4 QNhead=tex2D(t_a_QN,make_offset(0,oft),i);// first strent
	if (fetch_new_strent(0,oft)) QNhead=new_strent;
	float4 QNtail=tex2D(t_a_QN,make_offset(tz-1,oft),i);//last strent
	if (fetch_new_strent(tz-1,oft)) QNtail=new_strent;
	float W_CD_c_z =d_CD_create_prefact*(QNtail.w-1.0f);

	if (tz==1){
	  W_SD_c_1=__fdividef(1.0f,(dBe*dnk));
	  W_SD_c_z=W_SD_c_1;
	}else{
	    if (QNhead.w==1.0f){
		//destruction
		QNheadn=tex2D(t_a_QN,make_offset(1,oft),i);
		if (fetch_new_strent(1,oft)) QNheadn=new_strent;
		float f2=(tz ==2) ? QNheadn.w+0.25f: 0.5f*QNheadn.w;
		W_SD_d_1=__fdividef(1.0f,0.75f+f2);
	    }else{
		//creation
		W_SD_c_1=__fdividef(2.0f,dBe*(QNhead.w+0.5f));
	    }



	    if (QNtail.w==1.0f){
		//destruction
		QNtailp=tex2D(t_a_QN,make_offset(tz-2,oft),i);
		if (fetch_new_strent(tz-2,oft)) QNtailp=new_strent;

		float f1=(tz ==2) ?  QNtailp.w+0.25f: 0.5f*QNtailp.w;
		W_SD_d_z=__fdividef(1.0f, f1+0.75f);
	    }else{
		//creation
		W_SD_c_z=__fdividef(2.0f,dBe*(QNtail.w+0.5f));
	    }
	}
        //sum all the probabilities
	float sumW=sum_wshpm+W_SD_c_1+W_SD_c_z+W_SD_d_1+W_SD_d_z+W_CD_c_z;
	tdt[i]=__fdividef(1.0f,sumW);
	// error handling
	if (tdt[i]==0.0f) gpu_chain_heads[i].stall_flag=1;
	if (isnan(tdt[i])) gpu_chain_heads[i].stall_flag=2;
	if (isinf(tdt[i])) gpu_chain_heads[i].stall_flag=3;
	//update time
	gpu_chain_heads[i].time+=tdt[i];
	 //start picking the jump process
	 float pr=(sumW)*tex2D(t_uniformrand,rand_used[i],i);
 	 rand_used[i]++;
	 int j=0;
	 float tpr=0.0f;
	 if (tz!=1) surf2Dread(&tpr,s_sum_W,4*j,i);

	// picking where(which strent) jump process will happen
	// excluding SD creation destruction
	 while((pr>=tpr)&&(j<tz-2)){
	      pr-=tpr;
	      j++;
	      surf2Dread(&tpr,s_sum_W,4*j,i);

	 }
	 
	  if (pr<tpr)
	  {
	    // ok we picked some strent j
	    // now we need to decide which(SD shift or CDd CDc) jump process will happen
    	    // TODO check if the order have an effect on performance

	      float4 QN1=tex2D(t_a_QN,make_offset(j,oft),i);
	      if (fetch_new_strent(j,oft)) QN1=new_strent;
	      float4 QN2=tex2D(t_a_QN,make_offset(j+1,oft),i);
	      if (fetch_new_strent(j+1,oft)) QN2=new_strent;
	      QN1= kappa(QN1,olddt);
	      QN2= kappa(QN2,olddt);

	      
	      // first we check if CDd will happen
	      float wcdd=tex2D(t_a_tCD,make_offset(j,oft),i);
	      if (fetch_new_strent(j,oft)) wcdd=new_tCD;
	      if (pr<wcdd){
	  
		float4 temp =make_float4(QN1.x+QN2.x,QN1.y+QN2.y,QN1.z+QN2.z,QN1.w+QN2.w);
		if ((j==tz-2)||(j==0)){
		    temp=make_float4(0.0f,0.0f,0.0f,QN1.w+QN2.w);
		}
		surf2Dwrite(temp,s_b_QN,16*(j+1),i);
		d_offset[i]=offset_code(j,+1);
		gpu_chain_heads[i].Z--;

		return;
	      }else{
		pr-=wcdd;
	      }
	      
	    // next we check for SD shift
	    // SD shift probs are not saved from entanglement parallel part
	    // so we need to recalculate it
	    float2 twsh=make_float2(0.0f,0.0f); 
	    float Q=QN1.x*QN1.x+QN1.y*QN1.y+QN1.z*QN1.z;
	    float Q2=QN2.x*QN2.x+QN2.y*QN2.y+QN2.z*QN2.z;

	    if (QN2.w>1.0f) {//N=1 mean that shift is not possible, also ot will lead to dividing on zero error
	  	//float prefact=__powf( __fdividef(QN1.w*QN2.w,(QN1.w+1)*(QN2.w-1)),0.75f);
		//TODO replace powf with sqrt(x*x*x)

		float 	sig1=__fdividef(0.75f,QN1.w*(QN1.w+1));
		float 	sig2=__fdividef(0.75f,QN2.w*(QN2.w-1));
		float prefact1=(Q ==0.0f) ? 1.0f : __fdividef(QN1.w,(QN1.w+1));
		float prefact2=(Q2 ==0.0f) ? 1.0f : __fdividef(QN2.w,(QN2.w-1));
		float f1=(Q ==0.0f) ? 2.0f* QN1.w+0.5f: QN1.w;
		float f2=(Q2 ==0.0f) ? 2.0f* QN2.w-0.5f: QN2.w;
		float friction=__fdividef(2.0f,f1+f2);
		twsh.x=friction*__powf(prefact1*prefact2,0.75f)*__expf(Q*sig1-Q2*sig2);
	    }
	    if (QN1.w>1.0f) {//N=1 mean that shift is not possible, also ot will lead to dividing on zero error

		float 	sig1=__fdividef(0.75f,QN1.w*(QN1.w-1.0f));
		float 	sig2=__fdividef(0.75f,QN2.w*(QN2.w+1.0f));
		float prefact1=(Q ==0.0f) ? 1.0f : __fdividef(QN1.w,(QN1.w-1.0f));
		float prefact2=(Q2 ==0.0f) ? 1.0f : __fdividef(QN2.w,(QN2.w+1.0f));
		float f1=(Q ==0.0f) ? 2.0f* QN1.w-0.5f: QN1.w;
		float f2=(Q2 ==0.0f) ? 2.0f* QN2.w+0.5f: QN2.w;
		float friction=__fdividef(2.0f,f1+f2);
		twsh.y=friction*__powf(prefact1*prefact2,0.75f)*__expf(-Q*sig1+Q2*sig2);
	    }

	      if (pr<twsh.x+twsh.y){

		if (pr<twsh.x){
		    QN1.w=QN1.w+1; 
		    QN2.w=QN2.w-1; 
		}else{
		    QN1.w=QN1.w-1; 
		    QN2.w=QN2.w+1; 
		}
	  	surf2Dwrite(QN1,s_b_QN,16*j,i);
		surf2Dwrite(QN2,s_b_QN,16*(j+1),i);
		return;         
	      }else{
		pr-=twsh.x+twsh.y;
	      }
	      //last we check for CDc
	      float wcdc =d_CD_create_prefact*(QN1.w-1.0f);
	      if (pr<wcdc){
		if (tz==d_z_max) return;// possible detail balance issue
		float4 temp=tex2D(t_taucd_gauss_rand,tau_CD_used[i],i);
		tau_CD_used[i]++;
		gpu_chain_heads[i].Z++;
		d_new_tau_CD[i]=d_tau_CD_f_d_t(temp.w);//__fdividef(1.0f,d_tau_d);
		float newn=floorf(__fdividef(pr*(QN1.w-1.0f),wcdc))+1.0f;
		if (j==0){
		    
		    temp.w= QN1.w-newn;
		    float sigma= __fsqrt_rn(__fdividef(temp.w,3.0f));
		    temp.x*=sigma;
		    temp.y*=sigma;
		    temp.z*=sigma;
		    surf2Dwrite(temp,s_b_QN,16*0,i);
		    d_offset[i]=offset_code(0,-1);
		    d_new_strent[i]=make_float4(0.0f,0.0f,0.0f,newn);

		    return;
		}

		temp.w= newn;
		float sigma= __fsqrt_rn(__fdividef(newn*(QN1.w-newn),3.0f*QN1.w));
		float ration=__fdividef(newn,QN1.w);
		temp.x*=sigma;
		temp.y*=sigma;
		temp.z*=sigma;
		temp.x+=QN1.x*ration;
		temp.y+=QN1.y*ration;
		temp.z+=QN1.z*ration;
		surf2Dwrite(make_float4(QN1.x-temp.x,QN1.y-temp.y,QN1.z-temp.z,QN1.w-newn),s_b_QN,16*j,i);
		d_offset[i]=offset_code(j,-1);
		d_new_strent[i]=temp;
		return;
	      }else{
		pr-=wcdc;
	      }
	  }else{
	      pr-=tpr;
	  }
	  
	  //ok none of the jump processes in the middle of the chain were picked
	  // check  what left
	//w_CD_c_z
	if (pr<W_CD_c_z){
	    if (tz==d_z_max) return;// possible detail balance issue

	    float4 temp=tex2D(t_taucd_gauss_rand,tau_CD_used[i],i);
	    tau_CD_used[i]++;
	    gpu_chain_heads[i].Z++;
	    d_new_tau_CD[i]=d_tau_CD_f_d_t(temp.w);//__fdividef(1.0f,d_tau_d);

	    float newn=floorf(__fdividef(pr*(QNtail.w-1.0f),W_CD_c_z))+1.0f;

	    temp.w= newn;
	    float sigma=(tz==1)? 0.0f:__fsqrt_rn(__fdividef(temp.w,3.0f));
	    temp.x*=sigma;
	    temp.y*=sigma;
	    temp.z*=sigma;
	    surf2Dwrite(make_float4(0.0f,0.0f,0.0f,QNtail.w-newn),s_b_QN,16*(tz-1),i);
	    d_offset[i]=offset_code(tz-1,-1);
	    d_new_strent[i]=temp;
	    return;
	}else{
	    pr-=W_CD_c_z;
	}



	
		//w_SD_c/d

 	if (pr<W_SD_c_1+W_SD_c_z){
	    if (tz==d_z_max) return;// possible detail balance issue
	    float4 temp=tex2D(t_taucd_gauss_rand,tau_CD_used[i],i);
	    tau_CD_used[i]++;
	    gpu_chain_heads[i].Z++;
// 	d_new_tau_CD[i]=__fdividef(1.0f,d_tau_d);
	    d_new_tau_CD[i]=d_tau_CD_f_t(temp.w);

	    if (pr<W_SD_c_1){
		temp.w= QNhead.w-1.0f;
		float sigma= (tz==1)? 0.0f:__fsqrt_rn(__fdividef(temp.w,3.0f));
		temp.x*=sigma;
		temp.y*=sigma;
		temp.z*=sigma;
		surf2Dwrite(temp,s_b_QN,16*0,i);//TODO maybe deformation should be applied here
		d_offset[i]=offset_code(0,-1);
		d_new_strent[i]=make_float4(0.0f,0.0f,0.0f,1.0f);
	    }else{
		temp.w= QNtail.w-1.0f;
		float sigma=(tz==1)? 0.0f:__fsqrt_rn(__fdividef(temp.w,3.0f));
		temp.x*=sigma;
		temp.y*=sigma;
		temp.z*=sigma;
		surf2Dwrite(make_float4(0.0f,0.0f,0.0f,1.0f),s_b_QN,16*(tz-1),i);//TODO maybe deformation should be applied here
		d_offset[i]=offset_code(tz-1,-1);
		d_new_strent[i]=temp;
	    }

	    return;
	}else{
  	  pr-=W_SD_c_1+W_SD_c_z;

	}
	if (pr<W_SD_d_1+W_SD_d_z){//to delete entanglement
				     // update cell and neigbours
				     //clear W_sd
				     //
				     //form a list of free cell
		gpu_chain_heads[i].Z--;
	    if (pr<W_SD_d_1){
		surf2Dwrite(make_float4(0.0f,0.0f,0.0f,QNheadn.w+1.0f),s_b_QN,16*1,i);
		d_offset[i]=offset_code(0,+1);
	    }else{
		surf2Dwrite(make_float4(0.0f,0.0f,0.0f,QNtailp.w+1.0f),s_b_QN,16*(tz-2),i);
		d_offset[i]=offset_code(tz,+1);

	    }
	    return;

	}else{
	  pr-=W_SD_d_1+W_SD_d_z;
	}
  
}

    __global__ __launch_bounds__(tpb_chain_kernel) //stress calculation
	void stress_calc(chain_head* gpu_chain_heads,float *tdt,int *d_offset,float4 *d_new_strent ){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	if (i>=dn_cha_per_call) return;
	int tz=gpu_chain_heads[i].Z;
	uint oft=d_offset[i];
	float olddt=tdt[i];
	float4 new_strent=d_new_strent[i];

	float4 sum_stress=make_float4(0.0f,0.0f,0.0f,0.0f);
	float4 sum_stress2=make_float4(0.0f,0.0f,0.0f,0.0f);
	float ree_x=0.0,ree_y=0.0,ree_z=0.0;
	for (int j=0;j<tz;j++){
	      float4 QN1=tex2D(t_a_QN,make_offset(j,oft),i);
	      if (fetch_new_strent(j,oft)) QN1=new_strent;
	      QN1= kappa(QN1,olddt);
	      sum_stress.x-=__fdividef(3.0f*QN1.x*QN1.x,QN1.w);
	      sum_stress.y-=__fdividef(3.0f*QN1.y*QN1.y,QN1.w);
	      sum_stress.z-=__fdividef(3.0f*QN1.z*QN1.z,QN1.w);
	      sum_stress.w-=__fdividef(3.0f*QN1.x*QN1.y,QN1.w);
	      sum_stress2.x-=__fdividef(3.0f*QN1.y*QN1.z,QN1.w);
	      sum_stress2.y-=__fdividef(3.0f*QN1.x*QN1.z,QN1.w);
	      sum_stress2.z+=__fsqrt_rn(QN1.x*QN1.x+QN1.y*QN1.y+QN1.z*QN1.z);
	      ree_x+=QN1.x;
	      ree_y+=QN1.y;
	      ree_z+=QN1.z;
	}
	sum_stress2.w=float(tz);//__fsqrt_rn(ree_x*ree_x+ree_y*ree_y+ree_z*ree_z);
        surf1Dwrite(sum_stress,s_stress,32*i);
        surf1Dwrite(sum_stress2,s_stress,32*i+16);

    }






#endif