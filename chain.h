 #if !defined _CHAIN_
 #define _CHAIN_
 #include "random.h"
 #include "binomial.h"
 #include "pcd_tau.h"
 
 //here are functions for generating initial chain conformation on CPU
 
 
	extern Ran eran;
	extern float Be;
	extern int CD_flag;
	extern p_cd *pcd;

	//in DSM chain conformation consist from following variables: {Q_i},{N_i},{tau_CD_i},Z
	//additionally in this code following variables used: time,stall_flag,rand_used,tau_CD_used
	//{*_i} variables are vector(array) variables: simply i:[1,Z] for one chain
	//all other variables are scalar variables, i.e . there is only one variable per chain
	// vector and scalar variables are stored separetely
	
	
	//this structure is used to access array(vector) variables of chain conformation
	//all chain conformations suppose to be stored in one big array
	//this structure contains pointers to elements of the big array where conformation for the chain starts
	typedef struct sstrentp{//hybrid between strand and entanglement
	    float4 *QN;//number of Kuhn steps in strand and coordinates of entanglements
	    float *tau_CD;//lifetime
	}sstrentp;

 
	//this structure contains scalar variables of chain conformation
	typedef struct chain_head{//chain header
	    int Z;//n strands
	    float time;
// 	    int rand_used;//TODO not need actually
// 	    int tau_CD_used;
	    //TODO add dummy field for 16 byte alignment
	   int stall_flag;//ran out of random number or other issues
// 	   float W_SD_c_1,W_SD_c_z,W_SD_d_1,W_SD_d_z;//TODO remove debug
	}chain_head;


	
	//init chain conformation
	void chain_init(chain_head *chain_head,sstrentp data,int tnk);
	void chain_init(chain_head *chain_head,sstrentp data,int tnk,int z_max);//z_max is maximum number of entaglements. purpose - truncate z distribution for optimization.
	
	//debug output. ignores chain head
	ostream& operator<<(ostream& stream,const sstrentp c);
	//outputs chain conformation
	void print(ostream& stream,const sstrentp c,const chain_head chead);
	
	void save_to_file(ostream& stream,const sstrentp c,const chain_head chead);
	void load_from_file(istream& stream,const sstrentp c,const chain_head *chead);


	typedef struct stress_plus{//structure for storinf stress tensor + couple extra vars
				    //size 32 byte for proper alignment
	    float xx,yy,zz;
	    float xy,yz,zx;
	    float Lpp;
	    float Ree;
	}stress_plus;
	ostream& operator<<(ostream& stream,const stress_plus s);
	stress_plus make_stress_plus(float xx,float yy,float zz,float xy,float yz,float zx,float Lpp,float Ree);
	stress_plus operator+(const stress_plus  &s1, const stress_plus &s2);
	stress_plus operator/(const stress_plus  &s1, const double d);
	stress_plus operator*(const stress_plus  &s1, const double m);
	
#endif
