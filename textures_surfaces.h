#ifndef _TEXTURES_
#define _TEXTURES_


#define uniformrandom_count 250// size of the random arrays
#define ran_tpd 256 //thread per block for random_textures_fill()/random_textures_refill()

// see comments in ensemble.h

  texture<float, 2, cudaReadModeElementType> t_uniformrand;		// random numbers uniformly distributed
  texture<float4, 2, cudaReadModeElementType> t_taucd_gauss_rand;		// tauCD lifetimes and normally distributed random numbers (x,y,z)
  surface<void,2> rand_buffer;//temp array for random numbers

  //TODO replace a/b with source/dest
  texture<float4, 2, cudaReadModeElementType> t_a_QN;		// strents (N,Qx,Qy,Qz)
  texture<float, 2, cudaReadModeElementType> t_a_tCD;		// strents (N,Qx,Qy,Qz)

  //  surface<void,2> s_a_QN;
  //  surface<void,2> s_a_tCD;

  surface<void,2> s_b_QN;  // strents (N,Qx,Qy,Qz)
  surface<void,2> s_b_tCD;//tau_CD of ent-t

 
  surface<void,2> s_W_SD_pm;// SD shift probablities
  surface<void,2> s_sum_W;// SD shift probablities
  surface<void,1> s_stress;//float4 xx,yy,zz,xy
 
#endif
