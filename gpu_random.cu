 #include "gpu_random.h"
 #include "random.h"
#include <iostream>

#if defined(_MSC_VER)
#define uint unsigned int
#endif


 #include "cudautil.h"
 #include "cuda_call.h"      
#include "textures_surfaces.h"
    __constant__ int  NTAB;
    __constant__ int IN1;
    __constant__ int IK1;
    __constant__ int IQ1;
    __constant__ int IR1;
    __constant__ int IN2;
    __constant__ int IK2;
    __constant__ int IQ2;
    __constant__ int IR2;
    __constant__ int	INM1;
    __constant__ int NDIV;
    __constant__ float AN;
    

    void gpu_ran_init (){//copy constants to device
	int bf=32;
  	CUDA_SAFE_CALL(cudaMemcpyToSymbol(NTAB, &bf,sizeof(int)));
	bf=2147483563;
  	CUDA_SAFE_CALL(cudaMemcpyToSymbol(IN1, &bf,sizeof(int)));
	bf=40014;
  	CUDA_SAFE_CALL(cudaMemcpyToSymbol(IK1, &bf,sizeof(int)));
	bf=53668;
  	CUDA_SAFE_CALL(cudaMemcpyToSymbol(IQ1, &bf,sizeof(int)));
	bf=12211;
  	CUDA_SAFE_CALL(cudaMemcpyToSymbol(IR1, &bf,sizeof(int)));
	bf=2147483399;
  	CUDA_SAFE_CALL(cudaMemcpyToSymbol(IN2, &bf,sizeof(int)));
	bf=40692;
  	CUDA_SAFE_CALL(cudaMemcpyToSymbol(IK2, &bf,sizeof(int)));
	bf=52774;
  	CUDA_SAFE_CALL(cudaMemcpyToSymbol(IQ2, &bf,sizeof(int)));
	bf=3791;
  	CUDA_SAFE_CALL(cudaMemcpyToSymbol(IR2, &bf,sizeof(int)));
 	bf=2147483563-1;
  	CUDA_SAFE_CALL(cudaMemcpyToSymbol(INM1, &bf,sizeof(int)));
	bf=1+(2147483563-1)/32;
  	CUDA_SAFE_CALL(cudaMemcpyToSymbol(NDIV, &bf,sizeof(int)));
	float bff=1.0/2147483563.0;
  	CUDA_SAFE_CALL(cudaMemcpyToSymbol(AN, &bff,sizeof(float)));
    }
      

	
    __device__ void gpu_Ran::seed(int ISEED) {
	//subroutine ranils(ISEED)
	// initiation of the random number generator: from Numerical Recepies
	//Choice of ISEED: 0 <= ISEED <= 2000000000 (2E+9);

	  int J,K;
	IDUM=ISEED+123456789;
	IDUM2=IDUM;
// 	Load the shuffle table (after 8 warm-ups)
	for(J=NTAB+8;J>=1;J--){
	  K=IDUM/IQ1;
	  IDUM=IK1*(IDUM-K*IQ1)-K*IR1;
	  if(IDUM<0) IDUM=IDUM+IN1;
	  if(J<=NTAB) IV[J]=IDUM;
	}
	IY=IV[1];
    }


    __device__ float gpu_Ran::gpu_flt() {
        //double precision function ranuls()
	// Random number generator from Numerical Recepies
	int K,J;
	//Linear congruential generator 1
	K=IDUM/IQ1;
	IDUM=IK1*(IDUM-K*IQ1)-K*IR1;
	if(IDUM<0) IDUM=IDUM+IN1;
	//Linear congruential generator 2

	K=IDUM2/IQ2;
	IDUM2=IK2*(IDUM2-K*IQ2)-K*IR2;

	if(IDUM2<0) IDUM2=IDUM2+IN2;

	//Shuffling and subtracting
	J=1+IY/NDIV;
	IY=IV[J]-IDUM2;

	IV[J]=IDUM;
	if(IY<=1) IY=IY+INM1;
	if (IY>=IN1) IY=IY-IN1;
	return AN*IY;
      //end function ranuls
    }
    __device__ float2 gpu_Ran::gauss_distr(){
	// function from Numerical Recipies: Gaussian distribution
	float V1,V2,FAC;

		float R = 0.0f;
		while (R>=1.0f || R==0.0f){
			V1 = 2.0f*gpu_flt()-1.0f;
			V2 = 2.0f*gpu_flt()-1.0f;
			R  = V1*V1+V2*V2;
		}
// 		FAC    = sqrtf(-2.0f*logf(R)/R);
		FAC    = __fsqrt_rn(-2.0f*__logf(R)/R);
		return make_float2(V2*FAC,V1*FAC);
    }
 

    __global__ __launch_bounds__(ran_tpd) void array_seed (gpu_Ran *gr,int sz,int seed_offset){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	//__syncthreads();
	if (i<sz) gr[i].seed(i+seed_offset);
    }

// surface<void,2> output;
 
 
    __global__ __launch_bounds__(ran_tpd) void fill_surface_rand (gpu_Ran *gr,int n,int count ){
	int i=blockIdx.x*blockDim.x+threadIdx.x;//TODO copy ran to local memory
	float tmp;
	if (i<n){
	  for (int j=0; j<count;j++){
	      tmp=gr[i].gpu_flt();
	      surf2Dwrite(tmp,rand_buffer,4*j,i);
	  }
	  
	}
    }

     __global__ __launch_bounds__(ran_tpd) void fill_surface_taucd_gauss_rand (gpu_Ran *gr,int n,int count ){
	int i=blockIdx.x*blockDim.x+threadIdx.x;//TODO copy ran to local memory
	float4 tmp;
	float g=0.0f;
	float2 g2;
	if (i<n){
	  for (int j=0; j<count;j++){
	      if (g==0.0f){
		  tmp.w=gr[i].gpu_flt();
		  g2=gr[i].gauss_distr();
		  tmp.x=g2.x;
		  tmp.y=g2.y;
		  g2=gr[i].gauss_distr();
		  tmp.z=g2.x;
		  g=g2.y;
	      }else{
		  tmp.w=gr[i].gpu_flt();
		  tmp.x=g;
		  g2=gr[i].gauss_distr();
		  tmp.y=g2.x;
		  tmp.z=g2.y;
		  g=0.0f;
	      }
	      surf2Dwrite(tmp,rand_buffer,16*j,i);
	  }
	  
	}
    }

    __global__ __launch_bounds__(ran_tpd) void refill_surface_taucd_gauss_rand (gpu_Ran *gr,int n,int *count ){
	int i=blockIdx.x*blockDim.x+threadIdx.x;//TODO copy ran to local memory
	float4 tmp;
	float g=0.0f;
	float2 g2;
	if (i<n){
	int cnt=count[i]; 
	  for (int j=0; j<cnt;j++){
	      if (g==0.0f){
		  tmp.w=gr[i].gpu_flt();
		  g2=gr[i].gauss_distr();
		  tmp.x=g2.x;
		  tmp.y=g2.y;
		  g2=gr[i].gauss_distr();
		  tmp.z=g2.x;
		  g=g2.y;
	      }else{
		  tmp.w=gr[i].gpu_flt();
		  tmp.x=g;
		  g2=gr[i].gauss_distr();
		  tmp.y=g2.x;
		  tmp.z=g2.y;
		  g=0.0f;
	      }
	      surf2Dwrite(tmp,rand_buffer,16*j,i);
	  }
	  
	}
    }
 
    __global__ void test_kernel (gpu_Ran *gr,float * output, int n){
  // 		 int i=blockIdx.x*blockDim.x+threadIdx.x;
	//__syncthreads();
	gr->seed(1);
	for (int i=0; i<n;i++){
	  output[i]=gr->gpu_flt();
	}

    }
	 
    __global__ void test_kernel2 (float *out,int n,int count ){
	uint i=blockIdx.x*blockDim.x+threadIdx.x;//TODO copy ran to local memory
	
	if (i<n){
	  for (int j=0; j<count;j++){
	      float tmp;
 	      surf2Dread(&tmp,rand_buffer,4*i,j);  
	      out[i*count+j]=tmp*3;
	  }
	}
    }
    
    
    void gr_array_seed (gpu_Ran *gr,int sz, int seed_offset){
	array_seed<<<(sz+ ran_tpd-1)/ ran_tpd, ran_tpd>>>(gr,sz, seed_offset);
	CUT_CHECK_ERROR("kernel execution failed");
 	cudaDeviceSynchronize();

    }

    void gr_fill_surface_uniformrand(gpu_Ran *gr,int sz,int count , cudaArray*  d_uniformrand){
	cudaBindSurfaceToArray(rand_buffer, d_uniformrand);
	CUT_CHECK_ERROR("kernel execution failed");
        fill_surface_rand<<<(sz+ ran_tpd-1)/ ran_tpd, ran_tpd>>>(gr,sz,count);
	CUT_CHECK_ERROR("kernel execution failed");
 	cudaDeviceSynchronize();

    }

    void gr_fill_surface_taucd_gauss_rand(gpu_Ran *gr,int sz,int count,cudaArray*  d_taucd_gauss_rand ){

	cudaBindSurfaceToArray(rand_buffer, d_taucd_gauss_rand);
        fill_surface_taucd_gauss_rand<<<(sz+ ran_tpd-1)/ ran_tpd, ran_tpd>>>(gr,sz,count);
	CUT_CHECK_ERROR("kernel execution failed");
 	cudaDeviceSynchronize();

    }
    
    
     void gr_refill_surface_taucd_gauss_rand(gpu_Ran *gr,int sz,int *count,cudaArray*  d_taucd_gauss_rand ){

	cudaBindSurfaceToArray(rand_buffer, d_taucd_gauss_rand);
        refill_surface_taucd_gauss_rand<<<(sz+ ran_tpd-1)/ ran_tpd, ran_tpd>>>(gr,sz,count);
	CUT_CHECK_ERROR("kernel execution failed");
 	cudaDeviceSynchronize();

    }
    void  test_random()
    {
     
	int sz=10,count=10;
	gpu_Ran *grs;
	cudaMalloc(&grs, sizeof(gpu_Ran)*sz);
 	array_seed<<<1,sz>>>(grs,sz,0);

	cudaChannelFormatDesc channelDesc =cudaCreateChannelDesc(32, 0, 0, 0,cudaChannelFormatKindFloat);
	cudaArray* cuOutputArray;
	cudaMallocArray(&cuOutputArray, &channelDesc, sz,count,cudaArraySurfaceLoadStore);
	
	
/*	float *data=new float[sz*count];
	for(int i=0;i<sz*count;i++)
	{
	  data[i]=5*i+7;
	  cout<<data[i]<<'\t';
	}
	cout<<"\n";

	cudaMemcpy2DToArray(cuOutputArray, 0, 0, data, sizeof(float)*sz,sz*sizeof(float),count,cudaMemcpyHostToDevice);*/

	
 	cudaBindSurfaceToArray(rand_buffer, cuOutputArray);

   	fill_surface_rand<<<1,sz>>>(grs,sz,/*,outputsurface,*/count);

	float *buffer=new float[sz*count];
	cudaMemcpy2DFromArray(buffer,sizeof(float)*sz,cuOutputArray, 0, 0, sz*sizeof(float),count,cudaMemcpyDeviceToHost);
	cout<<"write to surface back\n";
	for(int i=0;i<sz*count;i++)
	{
	  cout<<buffer[i]<<'\t';
	}
	cout<<"\n";
	 cudaDeviceSynchronize();
	float *tt;
	cudaMalloc(&tt, sizeof(float)*sz*count);
	test_kernel2<<<1,sz>>>(tt,sz,count);

	float *buffer2=new float[sz*count];

 	cudaMemcpy( buffer2,tt, sizeof(float)*sz*count,  cudaMemcpyDeviceToHost);
	cout<<"read surface back\n";
	for(int i=0;i<sz*count;i++)
	{
	  cout<<buffer2[i]<<'\t';
	} 	
	cout<<"\n";
	
    }
//     void  test_random()
//     {
// 	gpu_Ran *gr1;
// 	cudaMalloc(&gr1, sizeof(gpu_Ran));
// 
//       
// 	int sz=100;
// 	float *bf;
// 	float *out=new float[sz];
// 	cudaMalloc(&bf, sizeof(float)*sz);
// 	test_kernel<<<1,1>>>(gr1,bf,sz);
// 	CUT_CHECK_ERROR("kernel execution failed");
// 
// 	cudaMemcpy( out,bf, sizeof(float)*sz,  cudaMemcpyDeviceToHost);
// 
// 	cudaFree(bf);
// 	Ran r1(1);
// 	for(int i=0;i<sz;i++)
// 	{
// 	  std::cout<<i<<'\t'<<out[i]<<'\t'<<r1.flt()<<'\n';
// 	}
//     }
    
    
    
    
    
    
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
