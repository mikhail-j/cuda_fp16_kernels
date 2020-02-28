/*
Copyright (C) 2017-2020 Qijia (Michael) Jin

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

#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>	//this includes device_functions.h
#include <cuda_fp16.h>
#include <stdint.h>

void hadd_m(int m, int n, const __half* A, const __half* B, __half* C);

int main(int argc, char* argv[]) {
	__half* gpu_A;
	__half* gpu_B;
	__half* gpu_C;
	unsigned short A[6] = {0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00};
	unsigned short B[6] = {0x0000, 0x3c00, 0x4000, 0x4200, 0x4400, 0x4500};
	__half C[6];
	cudaError_t status;
	if ((status = cudaMalloc(&gpu_A, (sizeof(__half) * 6))) != 0) {
		printf("gpu_A: %s\n", cudaGetErrorString(status));
	}
	if ((status = cudaMalloc(&gpu_B, (sizeof(__half) * 6))) != 0) {
		printf("gpu_B: %s\n", cudaGetErrorString(status));
	}
	if ((status = cudaMalloc(&gpu_C, (sizeof(__half) * 6))) != 0) {
		printf("gpu_C: %s\n", cudaGetErrorString(status));
	}
	if ((status = cudaMemcpy(gpu_A, A, (sizeof(__half) * 6), cudaMemcpyHostToDevice)) != 0) {
		printf("memcpy A -> gpu_A: %s\n", cudaGetErrorString(status));
	}
	if ((status = cudaMemcpy(gpu_B, B, (sizeof(__half) * 6), cudaMemcpyHostToDevice)) != 0) {
		printf("memcpy B -> gpu_B: %s\n", cudaGetErrorString(status));
	}

	hadd_m(2, 3, gpu_A, gpu_B, gpu_C);

	if ((status = cudaMemcpy(C, gpu_C, (sizeof(__half) * 6), cudaMemcpyDeviceToHost)) != 0) {
		printf("memcpy gpu_C -> C: %s\n", cudaGetErrorString(status));
	}
	for (int m = 0; m < 2; m++) {
		for (int n = 0; n < 3; n++) {
			printf("%f ", __half2float(C[n + (3 * m)]));	//print the unsigned short representation
		}
		printf("\n");
	}
	if ((status = cudaFree(gpu_A)) != 0) {
		printf("free gpu_A: %s\n", cudaGetErrorString(status));
	}
	if ((status = cudaFree(gpu_B)) != 0) {
		printf("free gpu_B: %s\n", cudaGetErrorString(status));
	}
	if ((status = cudaFree(gpu_C)) != 0) {
		printf("free gpu_C: %s\n", cudaGetErrorString(status));
	}
	return 0;
}