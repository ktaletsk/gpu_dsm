#include <iostream>
#include <stdlib.h>
#include <fstream>

#include "orientation_tensor.h"

// #include "job_ID.h"

 using namespace std;

 
     sstrentp chain_index_l(const int i){//absolute navigation i - is a global index of chains i:[0..N_cha-1]
	sstrentp ptr;
	ptr.QN=&(chains.QN[z_max*i]);
	ptr.tau_CD=&(chains.tau_CD[z_max*i]);
	return ptr;
    }
 
 
 void calc_o_tensor(int resolution){

    //c_i tensor componenents
    //c_i is six(symmetric tensor) 2D arrays
    float **ci_xx,**ci_yy,**ci_zz,**ci_xy,**ci_yz,**ci_xz;
   
    ci_xx=new float*[NK];
    ci_yy=new float*[NK];
    ci_zz=new float*[NK];
    ci_xy=new float*[NK];
    ci_yz=new float*[NK];
    ci_xz=new float*[NK];
   
   
    for(int i=0;i<NK;i++){
	ci_xx[i]=new float[NK];
	ci_yy[i]=new float[NK];
	ci_zz[i]=new float[NK];
	ci_xy[i]=new float[NK];
	ci_yz[i]=new float[NK];
	ci_xz[i]=new float[NK];
    }
    
   //alloc u arrays
    float *ux,*uy,*uz;
    ux=new float[NK];
    uy=new float[NK];
    uz=new float[NK];
    float *Ux,*Uy,*Uz;
    int Un=int(ceil(NK/resolution));
//     cout<<"Un "<<Un<<'\n';
    Ux=new float[Un];
    Uy=new float[Un];
    Uz=new float[Un];

    for(int i=0;i<NK;i++){
	for(int j=0;j<NK;j++){
	    ci_xx[i][j]=0.0f;
	    ci_yy[i][j]=0.0f;
	    ci_zz[i][j]=0.0f;
	    ci_xy[i][j]=0.0f;
	    ci_yz[i][j]=0.0f;
	    ci_xz[i][j]=0.0f;
	}
    }
//     cout<<"N_cha"<<' '<<N_cha<<'\n';

    for(int k=0;k<N_cha;k++){
        //prepare u arrays
        int s=0;
	sstrentp chaink=chain_index_l(k);
// 	cout<<"k chain_heads[k].Z"<<k<<' '<<chain_heads[k].Z<<'\n';
	for(int i=0;i<chain_heads[k].Z;i++){
	    float qx=chaink.QN[i].x;
	    float qy=chaink.QN[i].y;
	    float qz=chaink.QN[i].z;
	    float norm=sqrt(qx*qx+qy*qy+qz*qz);
	    float invnorm=norm!=0.0f ? 1.0f/norm:0.0f;
// 	    cout<<"qx "<<qx<<'\n';
	    for(int ni=0;ni<chaink.QN[i].w;ni++){
		ux[s]=qx*invnorm;
		uy[s]=qy*invnorm;
		uz[s]=qz*invnorm;
		s++;
	    }
	}
	for(int i=0;i<Un;i++){
	    Ux[i]=0.0f;
	    Uy[i]=0.0f;
	    Uz[i]=0.0f;
	}
	s=0;
	for(int i=0;i<NK;i++){
	    Ux[s]+=ux[i];
	    Uy[s]+=uy[i];
	    Uz[s]+=uz[i];
	    if ((i+1)%resolution==0) s++;
	}
	
	for(int i=0;i<Un;i++){
	    for(int j=0;j<Un;j++){
		ci_xx[i][j]+=Ux[i]*Ux[j];
		ci_yy[i][j]+=Uy[i]*Uy[j];
		ci_zz[i][j]+=Uz[i]*Uz[j];
		ci_xy[i][j]+=Ux[i]*Uy[j];
		ci_yz[i][j]+=Uy[i]*Uz[j];
		ci_xz[i][j]+=Ux[i]*Uz[j];
	      
	    }
	}
    }

    for(int i=0;i<Un;i++){
	for(int j=0;j<Un;j++){
	    ci_xx[i][j]/=N_cha;
	    ci_yy[i][j]/=N_cha;
	    ci_zz[i][j]/=N_cha;
	    ci_xy[i][j]/=N_cha;
	    ci_yz[i][j]/=N_cha;
	    ci_xz[i][j]/=N_cha;
	}
    }

    //output
    ofstream c_file;
//     cout<<filename_ID("tau")<<'\n';
    c_file.open("C_ij_xx.dat");

   for(int i=0;i<Un;i++){
      for(int j=0;j<Un;j++){
	    c_file<<ci_xx[i][j]<<'\t';
	}
	c_file<<'\n';
    }
    c_file.close();
    c_file.open("C_ij_yy.dat");
    for(int i=0;i<Un;i++){
	for(int j=0;j<Un;j++){
	      c_file<<ci_yy[i][j]<<'\t';
	  }
	  c_file<<'\n';
    }
    c_file.close();
    c_file.open("C_ij_zz.dat");
    for(int i=0;i<Un;i++){
	for(int j=0;j<Un;j++){
	      c_file<<ci_zz[i][j]<<'\t';
	  }
	  c_file<<'\n';
    }
    c_file.close();
    c_file.open("C_ij_xy.dat");
    for(int i=0;i<Un;i++){
	for(int j=0;j<Un;j++){
	      c_file<<ci_xy[i][j]<<'\t';
	  }
	  c_file<<'\n';
    }
    c_file.close();
    c_file.open("C_ij_yz.dat");
    for(int i=0;i<Un;i++){
	for(int j=0;j<Un;j++){
	      c_file<<ci_yz[i][j]<<'\t';
	  }
	  c_file<<'\n';
    }
    c_file.close();
    c_file.open("C_ij_xz.dat");
    for(int i=0;i<Un;i++){
	for(int j=0;j<Un;j++){
	      c_file<<ci_xz[i][j]<<'\t';
	}
	c_file<<'\n';
    }
    c_file.close();
    
    
    //FREE arrays

    delete[]ux;
    delete[]uy;
    delete[]uz;
    
    delete[]Ux;
    delete[]Uy;
    delete[]Uz;
   
    for(int i=0;i<NK;i++){
	delete []ci_xx[i];
	delete []ci_yy[i];
	delete []ci_zz[i];
	delete []ci_xy[i];
	delete []ci_yz[i];
	delete []ci_xz[i];
    }
    delete []ci_xx;
    delete []ci_yy;
    delete []ci_zz;
    delete []ci_xy;
    delete []ci_yz;
    delete []ci_xz;
 }
