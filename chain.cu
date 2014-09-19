#include <iostream>
#include <stdlib.h>
#include "chain.h"

    int CD_flag=0;

    int z_dist(int tNk){
	//entanglement distribution
	return 1 + int(binomial_distr(1.0/(1.0+Be), tNk-1));//binomial_distr see random module
    }
	
    int z_dist_truncated(int tNk,int z_max){
	int tz=1 + int(binomial_distr(1.0/(1.0+Be), tNk-1));
	while (tz>z_max) tz=1 + int(binomial_distr(1.0/(1.0+Be), tNk-1));
	return tz;
    }
	
    void N_dist(int ztmp,int *&tN,int tNk){
	tN=new int[ztmp];
	if (ztmp==1) tN[0] = tNk;
	else{
		int A = tNk-1;
		for(int i=ztmp;i>1;i--){
			float p = eran.flt();
			int Ntmp = 0;
			float sum = 0.0;
			while ((p>=sum)&&((++Ntmp)!=A-i+2))sum+=ratio(A,Ntmp,i);//ratio of two binomial coefficients:math module
			tN[i-1] = Ntmp;
			A = A - Ntmp;
		}
		tN[0] = A+1;
	}
    }

    void  Q_dist(int tz,int *Ntmp,float *&Qxtmp,float *&Qytmp,float *&Qztmp){

	Qxtmp=new float[tz];
	Qytmp=new float[tz];
	Qztmp=new float[tz];
	if (tz>2) {//dangling ends is not part of distribution 
	    for (int i=1;i<tz-1;i++){
		Qxtmp[i] = eran.gauss_distr()*sqrt(float(Ntmp[i])/3.0);//using gaussian distribution
		Qytmp[i] = eran.gauss_distr()*sqrt(float(Ntmp[i])/3.0);//gaussian distribution is defined in math_module
		Qztmp[i] = eran.gauss_distr()*sqrt(float(Ntmp[i])/3.0);
	    }
	}
    }

    
    
    void chain_init(chain_head *chain_head,sstrentp data,int tnk){
    //chain conformation generated should be identical to CPU program(valid only for 1 chain)
	
	int tz=z_dist(tnk);   //z distribution
// 	tz=1;
	float *tent_tau=new float[tz-1];//temporaly arrays
	if (CD_flag!=0) {
		for(int k=0;k<tz-1;tent_tau[k++]=pcd->tau_CD_f_t());//1-SD entanglement lifetime
	}else for(int k=0;k<tz-1;tent_tau[k++]=0.0);

	int *tN;
	float * Qxtmp, *Qytmp, *Qztmp;
	N_dist(tz,tN,tnk);//N distribution
// 	for(int i=0;i<tz;cout<<tN[i++]<<'\t');cout<<'\n';
	Q_dist(tz,tN,Qxtmp,Qytmp,Qztmp);//Q distributions //realization Free_Energy_module(Gauss)
	memset(data.QN,0,sizeof(float)*tnk*4);
	// creating entanglements according to distributions
	for(int k=0;k<tz;k++){
// 	    data[k]=new_strent(tN[k],Qxtmp[k],Qytmp[k],Qztmp[k],0.0,0.0,k+1);
	    data.QN[k]=make_float4(Qxtmp[k],Qytmp[k],Qztmp[k],float(tN[k]));
	    data.tau_CD[k]=1.0f/tent_tau[k];
  
	}

	//set_dangling_ends
	data.QN[0]=make_float4(0.0f,0.0f,0.0f,float(tN[0]));
	data.QN[tz-1]=make_float4(0.0f,0.0f,0.0f,float(tN[tz-1]));
	delete[] tN;
	delete[] Qxtmp;
	delete[] Qytmp;
	delete[] Qztmp;
	delete[] tent_tau;
	chain_head->Z=tz;
	chain_head->time=0.0;
// 	chain_head->rand_used=0;
// 	chain_head->tau_CD_used=0;
	chain_head->stall_flag=0;

    }


  void chain_init(chain_head *chain_head,sstrentp data,int tnk,int z_max){
	int tz=z_dist_truncated(tnk,z_max);   //z distribution
	float *tent_tau=new float[tz-1];//temporaly arrays
	if (CD_flag!=0) {
		for(int k=0;k<tz-1;tent_tau[k++]=pcd->tau_CD_f_t());//1-SD entanglement lifetime
	}else for(int k=0;k<tz-1;tent_tau[k++]=0.0);

	int *tN;
	float * Qxtmp, *Qytmp, *Qztmp;
	N_dist(tz,tN,tnk);//N distribution
	Q_dist(tz,tN,Qxtmp,Qytmp,Qztmp);//Q distributions //realization Free_Energy_module(Gauss)
	memset(data.QN,0,sizeof(float)*z_max*4);
	// creating entanglements according to distributions
	for(int k=0;k<tz;k++){
	    data.QN[k]=make_float4(Qxtmp[k],Qytmp[k],Qztmp[k],float(tN[k]));
	    data.tau_CD[k]=1.0f/tent_tau[k];
  
	}

	//set_dangling_ends
	data.QN[0]=make_float4(0.0f,0.0f,0.0f,float(tN[0]));
	data.QN[tz-1]=make_float4(0.0f,0.0f,0.0f,float(tN[tz-1]));
	delete[] tN;
	delete[] Qxtmp;
	delete[] Qytmp;
	delete[] Qztmp;
	delete[] tent_tau;
	chain_head->Z=tz;
	chain_head->time=0.0;
// 	chain_head->rand_used=0;
// 	chain_head->tau_CD_used=0;
	chain_head->stall_flag=0;

  }


    ostream& operator<<(ostream& stream,const sstrentp c){
	
	int end=10;
	stream<<"N:  ";
	for(int j=0;j!=end;j++)stream<<c.QN[j].w<<' ';
	stream<<"\nQx: ";
	for(int j=0;j!=end;j++)stream<<c.QN[j].x<<' ';
	stream<<"\nQy: ";
	for(int j=0;j!=end;j++)stream<<c.QN[j].y<<' ';
	stream<<"\nQz: ";
	for(int j=0;j!=end;j++)stream<<c.QN[j].z<<' ';
	stream<<'\n';
	return stream;
      
    }

    void print(ostream& stream,const sstrentp c,const chain_head chead){
	stream<<chead.time<<'\n';
	stream<<chead.Z<<'\n';
 	stream<<"N:  ";
 	for(int j=0;j<chead.Z;j++)stream<<c.QN[j].w<<' ';
 	stream<<"\nQx: ";
 	for(int j=0;j<chead.Z;j++)stream<<c.QN[j].x<<' ';
 	stream<<"\nQy: ";
 	for(int j=0;j<chead.Z;j++)stream<<c.QN[j].y<<' ';
 	stream<<"\nQz: ";
 	for(int j=0;j<chead.Z;j++)stream<<c.QN[j].z<<' ';
 	stream<<'\n';
    }

    void save_to_file(ostream& stream,const sstrentp c,const chain_head chead){
      stream.write((char*)&chead,sizeof(chain_head));
      for(int j=0;j<chead.Z;j++) stream.write((char*)&(c.QN[j]),sizeof(float4));
      for(int j=0;j<chead.Z;j++) stream.write((char*)&(c.tau_CD[j]),sizeof(float));
    }

     void load_from_file(istream& stream,const sstrentp c,const chain_head *chead){
      stream.read((char*)chead,sizeof(chain_head));
      for(int j=0;j<chead->Z;j++) stream.read((char*)&(c.QN[j]),sizeof(float4));
      for(int j=0;j<chead->Z;j++) stream.read((char*)&(c.tau_CD[j]),sizeof(float));
    }   
	ostream& operator<<(ostream& stream,const stress_plus s){
	    return stream<<s.xx<<' '<<s.yy<<' '<<s.zz<<' '<<s.xy<<' '<<s.yz<<' '<<s.zx<<' '<<s.Lpp<<' '<<s.Ree;
	}
	stress_plus make_stress_plus(float xx,float yy,float zz,float xy,float yz,float zx,float Lpp,float Ree){
	    stress_plus t;
	    t.xx=xx;t.yy=yy;t.zz=zz;
	    t.xy=xy;t.yz=yz;t.zx=zx;
	    t.Lpp=Lpp;
	    t.Ree=Ree;
	    return t;
	}
	stress_plus operator+(const stress_plus  &s1, const stress_plus &s2){
	    return make_stress_plus(s1.xx+s2.xx,s1.yy+s2.yy,s1.zz+s2.zz,s1.xy+s2.xy,s1.yz+s2.yz,s1.zx+s2.zx,s1.Lpp+s2.Lpp,s1.Ree+s2.Ree);
	}
	stress_plus operator/(const stress_plus  &s1, const double d){
	    return make_stress_plus(s1.xx/d,s1.yy/d,s1.zz/d,s1.xy/d,s1.yz/d,s1.zx/d,s1.Lpp/d,s1.Ree/d);
	}
	stress_plus operator*(const stress_plus  &s1, const double m){
	    return make_stress_plus(s1.xx*m,s1.yy*m,s1.zz*m,s1.xy*m,s1.yz*m,s1.zx*m,s1.Lpp*m,s1.Ree*m);
	}


