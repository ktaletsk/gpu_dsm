// Copyright 2014 Marat Andreev
// 
// This file is part of gpu_dsm.
// 
// gpu_dsm is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// at your option) any later version.
// 
// gpu_dsm is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with gpu_dsm.  If not, see <http://www.gnu.org/licenses/>.

#if !defined _CHAIN_
#define _CHAIN_
#include "random.h"
#include "pcd_tau.h"
using namespace std;
//here are functions for generating initial chain conformation on CPU

extern Ran eran;
extern float Be;
extern int CD_flag;
extern p_cd *pcd;

//DSM chain conformation consist from a following variables: {Q_i},{N_i},{tau_CD_i},Z
//additionally in this code following chain variables used: time,stall_flag
//{*_i} variables are vectors(arrays): simply i:[1,Z] for one chain
//other variables are scalars, i.e . there is only one variable per chain
// vector and scalar variables are stored separetely

//this structure is used to access arrays(vectors) of chain conformation
//there is one big array with chain conformations for the whole ensemble
//this structure contains pointers beginning of the chain conformation in the big array
typedef struct sstrentp {//actually there are only Z-1 tau_CD and only Z-2 Q_i(i:[2,Z-1])
	//but we are ignoring this fact here
	float4 *QN;	//number of chain segments in the strand and the strand connector vector
	float *tau_CD;				//CD lifetime
} sstrentp;

//this structure contains scalar variables of chain conformation
typedef struct chain_head {//chain header
	int Z;//n strands
	float time;
	float dummy;// dummy field for 16 byte alignment
	int stall_flag;//algorithm crash flag
} chain_head;

//init chain conformation
void chain_init(chain_head *chain_head, sstrentp data, int tnk);
void chain_init(chain_head *chain_head, sstrentp data, int tnk, int z_max);	//z_max is maximum number of entaglements. purpose - truncate z distribution for optimization. NOTE: not tested

//debug output. ignores chain head
ostream& operator<<(ostream& stream, const sstrentp c);
//outputs chain conformation
void print(ostream& stream, const sstrentp c, const chain_head chead);

void save_to_file(ostream& stream, const sstrentp c, const chain_head chead);
void load_from_file(istream& stream, const sstrentp c, const chain_head *chead);

#endif
