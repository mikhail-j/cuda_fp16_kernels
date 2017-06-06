/*
Copyright (C) 2017 Qijia (Michael) Jin

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
*/
#include <cuda.h>
#include <cuda_runtime.h>	//this includes device_functions.h
#include <cuda_fp16.h>

__global__ void hadd_m_kernel(const __half* A, const __half* B, __half* C){
	//m is the number of columns
	//n is the number of rows
	//blockIdx.x is [0, m x n)
	//threadId.x is [0, 1)
	C[blockIdx.x] = __hadd(A[blockIdx.x], B[blockIdx.x]);
}

extern "C" {
	void hadd_m(int m, int n, const __half* A, const __half* B, __half* C) {
		dim3 num_blocks(m * n);			//m x n
		dim3 threads_per_block(1);		//careful: maximum threads per block is 1024
		hadd_m_kernel<<<num_blocks, threads_per_block>>>(A, B, C);
		cudaDeviceSynchronize();
	}
}