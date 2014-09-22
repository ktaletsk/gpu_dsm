 #if !defined _P_CD_
 #define _P_CD_
#include <iostream>
#include <math.h>
#include "random.h"
      
 using namespace std; 
struct p_cd {//Generates \tau_CD lifetimes 
	      //uses analytical approximation to P_cd parameters
	float At,Adt,Bdt,normdt;
	float g, alpha ,tau_0,tau_max,tau_d;
	Ran *ran;
	p_cd(float Be, int Nk,Ran *tran){
	  //init parameters from analytical fits
	    ran=tran;
	    double z=(Nk+Be)/(Be+1.0);
// 	    cout<<Be<<'\t'<<Nk<<'\n';
	    g=0.667f;
	    if (Be!=1.0f) {//analytical approximation to P_cd parameters for FSM
			  //Unpublished Pilyugina E. (2012)
		alpha=(0.053f*logf(Be)+0.31f)*powf(z,-0.012f*logf(Be)-0.024f);
		tau_0=0.285f*powf(Be+2.0f,0.515f);
		tau_max=0.025f*powf(Be+2.0f,2.6f)*powf(z,2.83f);
		tau_d=0.036f*powf(Be+2.0f,3.07f)*powf(z-1.0f,3.02f);
	    }else{//analytical approximation to P_cd parameters CFSM
		  //Andreev, M., Feng, H., Yang, L., and Schieber, J. D.,J. Rheol. 58, 723 (2014).

		alpha=0.267096f-0.375571f*expf(-0.0838237f*Nk);
		tau_0=0.460277f+0.298913f*expf(-0.0705314f*Nk);
		tau_max=0.0156137f*powf(float(Nk),3.18849f);
		tau_d=0.0740131f*powf(float(Nk),3.18363f);
	    }
//init vars
	    At=(1.0f-g)/(powf(tau_max,alpha)-powf(tau_0,alpha));
	    Adt=(1.0f-g)*alpha/(alpha-1.0f)/(powf(tau_max,alpha)-powf(tau_0,alpha));
	    Bdt=Adt*(powf(tau_max,alpha-1.0f)-powf(tau_0,alpha-1.0f));
	    normdt=Bdt+g/tau_d;
// 	    normdt=1.0f/tau_d;
	}
 	float tau_CD_f_t() {
	    float p=ran->flt();
	    if (p<(1.0f-g)){
		return powf(p/At+powf(tau_0,alpha),1.0f/alpha);
	    }else{
		return tau_d;
 	    }
	}
 	float tau_CD_f_d_t() {
	    float p=ran->flt();

// 	    cout<< " p "<<p<<'\n';
	    p=p*normdt;
	    if (p<Bdt){
//  		cout<<"tau_CD_f_d_t "<<powf(p/Adt+powf(tau_0,alpha-1.0f),1.0f/(alpha-1.0f))<<'\n';

		return powf(p/Adt+powf(tau_0,alpha-1.0f),1.0f/(alpha-1.0f));
// 		write(log_file%handle(),*) "p,normdt,Adt,l_tau_CD_f_d_t",p,normdt,Adt,l_tau_CD_f_d_t
	    }else{
// 		cout<<"tau_CD_f_d_t "<<tau_d<<'\n';
		return tau_d;
	    }
	}
	inline float W_CD_destroy_aver(){ return normdt;
	}
};

#endif