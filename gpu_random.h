 #if !defined gpu_random
 #define gpu_random
#include <curand_kernel.h>
 #include <math.h>
      
    //Curand random number generators
    //creates array of random number generator on device
    //fills array with random numbers
    //designed to provide texture random number supply for DSM dynamics
    void gpu_ran_init ();//initializes device random number
    
    #define gpu_Ran curandState_t
//     struct gpu_Ran {    };
    void gr_array_seed(gpu_Ran *gr,int sz, int seed_offest); // seeds array of device random number generators
    void gr_fill_surface_uniformrand(gpu_Ran *gr,int sz,int count,cudaArray*  d_uniformrand);    //Fills surface rand_buffer with uniformaly distributed (0,1] random numbers. They are used to pick jump process.
    
    void gr_fill_surface_taucd_gauss_rand(gpu_Ran *gr,int sz,int count,cudaArray*  d_taucd_gauss_rand); //Fills surface rand_buffer with vectors of {1x uniformaly distributed (0,1] random numbers and 3x normally distributed (mean 0, varience 1) random numbers}. They are used for (\tau_CD, \bm{Q_i} generation)
    
    void gr_refill_surface_taucd_gauss_rand(gpu_Ran *gr,int sz,int *count,cudaArray*  d_taucd_gauss_rand);
    //Refills surface rand_buffer with vectors of {1x uniformaly distributed (0,1] random numbers and 3x normally distributed (mean 0, varience 1) random numbers}. They are used for (\tau_CD, \bm{Q_i} generation)
    //int *count specify how many numbers needs to be refilled for each thread.

#endif
