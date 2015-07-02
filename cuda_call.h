// Copyright 2015 Marat Andreev, Konstantin Taletskiy, Maria Katzarova
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

#ifndef _CUDA_SAFE_CALL_
#define _CUDA_SAFE_CALL_

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
//Wrapper to catch CUDA errors
//just copy of "CUDA programming guide" code


#define CUDA_SAFE_CALL_NOSYNC(err)	__cudaSafeCallNoSync(err, __FILE__, __LINE__)
#define CUDA_SAFE_CALL(err)			__cudaSafeCall(err, __FILE__, __LINE__)
#define CUT_CHECK_ERROR(err)		__cutilGetLastError(err, __FILE__, __LINE__)

inline void __cudaSafeCallNoSync( cudaError err, const char *file, const int line )
{
    if( cudaSuccess != err) {
        fprintf(stderr, "%s(%i) : cudaSafeCallNoSync() Runtime API error %d : %s.\n",
                file, line, (int)err, cudaGetErrorString( err ));
        exit(-1);
    }
}


inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
    if( cudaSuccess != err) {
		fprintf(stderr, "%s(%i) : cudaSafeCall() Runtime API error %d: %s.\n",
                file, line, (int)err, cudaGetErrorString( err ) );
        exit(-1);
    }
}


inline void __cutilGetLastError( const char *errorMessage, const char *file, const int line )
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) {
        fprintf(stderr, "%s(%i) : cutilCheckMsg() CUTIL CUDA error : %s : (%d) %s.\n",
                file, line, errorMessage, (int)err, cudaGetErrorString( err ) );
        exit(-1);
    }
}

#endif

