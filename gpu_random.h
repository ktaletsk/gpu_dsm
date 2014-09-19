 #if !defined gpu_random
 #define gpu_random
#include <math.h>
// #include "random.h"

      
      
    //linear congruetor random number generators
    //creates array of random number generator on device
    //fills array with random numbers
    //designed to provide texture random number supply for DSM dynamics
    void gpu_ran_init ();
      
    void test_random();  

      
    struct gpu_Ran {
	int IDUM,IDUM2,IY,IV[33];

	__device__ void seed(int ISEED);
 	__device__ float gpu_flt() ;
	__device__ float2 gauss_distr();
    };
    void gr_array_seed (gpu_Ran *gr,int sz, int seed_offest);
    void gr_fill_surface_uniformrand(gpu_Ran *gr,int sz,int count,cudaArray*  d_uniformrand);
    void gr_fill_surface_taucd_gauss_rand(gpu_Ran *gr,int sz,int count,cudaArray*  d_taucd_gauss_rand);
    void gr_refill_surface_taucd_gauss_rand(gpu_Ran *gr,int sz,int *count,cudaArray*  d_taucd_gauss_rand);

#endif
