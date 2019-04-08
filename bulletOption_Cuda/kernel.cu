#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_runtime_api.h"
#include <fstream>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include "rng.h"
#include <device_functions.h>

#include <cuda.h>
#include "kernel.h"


#define L 1000 //time step number
#define M 10 // payment date number
#define N 50 // space step number


// precision to compare with
__constant__  float err_r = 1e-4;

__constant__  float Kgmax = 5.52f;
__constant__  float Kgmin = 2.996f;
__constant__  float K = 100.f;
__constant__  int P1 = 4;
__constant__  int P2 = 7;
__constant__  float B = 4.7874f;

//maturity
__constant__  float TT = 3.f;
__constant__  float delta_T = 3.f / 1000;

__constant__  float delta_X = (5.52f - 2.996f) / 50;

// prefixed dates correspond to maturity T = 3
__constant__  float T[M] = { 0.03, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7 };

__device__  bool is_synchronized[L];
__device__  int synchro_count[L];
__device__  float As[M*(N - 1)];
__device__  float Bs[M*N];
__device__  float Cs[M*(N - 1)];
__device__  float Ys[M*N];
__device__ float Zs[M*N];

bool *is_synchronized_c;
int *synchro_count_c;
float *Asc, *Bsc, *Csc, *Ysc, *Zsc;

// Free all parameters
void FreeVar()
{
	free(is_synchronized_c);
	free(synchro_count_c);
	free(Asc);
	free(Bsc);
	free(Csc);
	free(Ysc);
	free(Zsc);
}

// auxiliary functions
float Maxc(float X, float Y) {
	return X < Y ? Y : X;
}

__device__ float Max(float X, float Y) {
	return X < Y ? Y : X;
}

__device__ bool equal(float x, float y) {
	return abs(x - y) < err_r;
}

// r_g : we do not implement!
__device__ float get_rgc() {
	return 0.f;
}

// returns between which two moments from T tStep is
__device__ int which_slot(int tStep) {
	float t = tStep * delta_T;
	int slot = M;
	for (int i = M-1; i >= 0; i--) {
		if (t < T[i]) {
			slot = i;
		}
	}
	return slot ;
}

// defines the Scema for updation Ys
__device__ int which_Schema(int tStep) {
	float t = tStep * delta_T;
	if (equal(t, T[M - 1])) {
		return 1;
	}
	else {
		for (int k = 2; k <= M - 1; k++) {
			if (equal(t, T[M - k])) {
				return k;
			}
		}
		return 0;// "normal" 
	}
}

__device__ float One_Zero(float S, float B, bool up) {
	if (up) {
		return S < B ? 0.f : 1.f;
	}
	else {
		return S < B ? 1.f : 0.f;
	}
}

// update Ys by Schema0 (which means E[(S_T - K)_+|S_t])
__device__ void Update_by_Schema0(int tStep) {

	int idx = threadIdx.x + (blockIdx.x - 1) * N; 
	int slot = which_slot(tStep);
	int p_inf = P2 < slot ? P2 : slot;
	
	if (blockIdx.x - 1  + (M - slot)   <  P1 || blockIdx.x - 1 > p_inf) {
		Ys[idx] = 0.0;
		return;
	}

	float x = Kgmin + threadIdx.x * delta_X; 

	if (x >= Kgmax - delta_X - err_r) {
		float pmax = exp(Kgmax) - K;
		Ys[idx] = -As[idx - 1] * Zs[idx - 1] + (2 - Bs[idx]) * Zs[idx] + (-Cs[idx]) * pmax;
	}
	else {
		if (x <= Kgmin + err_r) {
			float pmin = 0;
			Ys[idx] = -As[idx] * pmin + (2 - Bs[idx]) * Zs[idx] + (-Cs[idx + 1]) * Zs[idx + 1];
		}
		else {
			Ys[idx] = -As[idx - 1] * Zs[idx - 1] + (2 - Bs[idx]) * Zs[idx] + (-Cs[idx + 1]) * Zs[idx + 1];
		}

	}
	return;
}
__device__ void Update_by_Schema1(bool up) {
	//printf("Hello Schema 1\n");

	int idx = threadIdx.x + (blockIdx.x - 1) * N; // SER_c
	float x = Kgmin + threadIdx.x * delta_X; // SER_c


	// SER_c: x - delta, error
	if (x >= Kgmax - delta_X - err_r) {
		// upper boundary
		float pmax = exp(Kgmax) - K;//M_2
		// when touches upper boundary, As[idx+1] is set to be As[idx]
		// SER_c: As, Bs, Cs // also modified
		Ys[idx] = -As[idx - 1] * Zs[idx - 1] * One_Zero(x - delta_X, B, up)\
			+ (2 - Bs[idx]) * Zs[idx] * One_Zero(x, B, up) + \
			(-Cs[idx]) * pmax * One_Zero(x + delta_X, B, up);
		
	}
	else {
		// SER_c: error
		if (x <= Kgmin + err_r) {
			// lower boundary
			float pmin = 0;
			//SER_c: Bs, Cs
			Ys[idx] = -As[idx] * pmin * One_Zero(x - delta_X, B, up) + \
				Zs[idx] * (2 - Bs[idx]) * One_Zero(x, B, up) + \
				(-Cs[idx + 1]) * Zs[idx + 1] * One_Zero(x + delta_X, B, up);
			//printf("idx of Y: %d,  Ys: %f\n", idx, Ys[idx]);
		}
		else {
			// in the middle
			//SER_c: Bs, Cs
			Ys[idx] = -As[idx - 1] * Zs[idx - 1] * One_Zero(x - delta_X, B, up) + \
				(2 - Bs[idx]) * Zs[idx] * One_Zero(x, B, up) + \
				(-Cs[idx + 1]) * Zs[idx + 1] * One_Zero(x + delta_X, B, up);
			//printf("idx of Y: %d,  Ys: %f\n", idx, Ys[idx]);
		}

	}
	//printf("idx of Y: %d,  Ys: %f\n", idx, Ys[idx]);

	return;
}

__device__ void Update_by_Schema2(int tStep) {

	int idx_this_j = threadIdx.x + (blockIdx.x - 1) * N;
	int idx_next_j = threadIdx.x + (blockIdx.x) * N;


	int slot = which_slot(tStep);
	int p_inf = P2 < slot ? P2 : slot;
	
	if (blockIdx.x - 1 + (M - slot) < P1 || blockIdx.x - 1 > p_inf) {
		Ys[idx_this_j] = 0.0;
		return;
	}

	float x = Kgmin + threadIdx.x * delta_X;

	if (x >= Kgmax - delta_X - err_r) {
		float pmax = exp(Kgmax) - K;

		Ys[idx_this_j] = -As[idx_this_j - 1] * Zs[idx_this_j - 1] * One_Zero(x - delta_X, B, true) + \
			(2 - Bs[idx_this_j]) * Zs[idx_this_j] * One_Zero(x, B, true) + \
			(-Cs[idx_this_j + 1]) * pmax * One_Zero(x + delta_X, B, true) + \

			- As[idx_next_j - 1] * Zs[idx_next_j - 1] * One_Zero(x - delta_X, B, false) + \
			(2 - Bs[idx_next_j]) * Zs[idx_next_j] * One_Zero(x, B, false) + \
			(-Cs[idx_next_j + 1]) * pmax * One_Zero(x + delta_X, B, false);
	}

	else {
		if (x <= Kgmin + err_r) {

			float pmin = 0;

			Ys[idx_this_j] = -As[idx_this_j - 1] * pmin * One_Zero(x - delta_X, B, true) + \
				(2 - Bs[idx_this_j]) * Zs[idx_this_j] * One_Zero(x, B, true) + \
				(-Cs[idx_this_j + 1]) * Zs[idx_this_j + 1] * One_Zero(x + delta_X, B, true) + \

				- As[idx_next_j - 1] * pmin * One_Zero(x - delta_X, B, false) + \
				(2 - Bs[idx_next_j]) * Zs[idx_next_j] * One_Zero(x, B, false) + \
				(-Cs[idx_next_j + 1]) * Zs[idx_next_j + 1] * One_Zero(x + delta_X, B, false);
		}

		else {

			Ys[idx_this_j] = -As[idx_this_j - 1] * Zs[idx_this_j - 1] * One_Zero(x - delta_X, B, true) + \
				(2 - Bs[idx_this_j]) * Zs[idx_this_j] * One_Zero(x, B, true) + \
				(-Cs[idx_this_j + 1]) * Zs[idx_this_j + 1] * One_Zero(x + delta_X, B, true) + \
				
				- As[idx_next_j - 1] * Zs[idx_next_j - 1] * One_Zero(x - delta_X, B, false) + \
				(2 - Bs[idx_next_j]) * Zs[idx_next_j] * One_Zero(x, B, false) + \
				(-Cs[idx_next_j + 1]) * Zs[idx_next_j + 1] * One_Zero(x + delta_X, B, false);
		}
	}

	return;
}

__device__ void Update_Ys(int tStep, int NN) {



	// Ys is the right part of the recursive equation
	// please notice that: we update first Ys for step i-1 using As, Bs, Cs, at step i, then we update As, Bs, Cs for the step i-1
	// by doing that, we don't need to create three extra array to store pd,pm,pd since they can be calculated from As,Bs,Cs as following
	// *********************************************************************************
	// As[ threadIdx.x + N*blockIdx.x ] = qd( threadIdx.x*delta_t, threadIdx.x*delta_x)
	// Bs[ threadIdx.x + N*blockIdx.x ] = qm( threadIdx.x*delta_t, threadIdx.x*delta_x)
	// Cs[ threadIdx.x + N*blockIdx.x ] = qu( threadIdx.x*delta_t, threadIdx.x*delta_x)
	// pd = - qd 
	// pm = 2 - qm
	// pu = - qu
	// *********************************************************************************
	int schema = which_Schema(tStep);

	if (schema == 0) {
		Update_by_Schema0(tStep);
		
	}
	if (schema == 1) {
		if (blockIdx.x - 1 == P2 || blockIdx.x - 1 == (P1 - 1)) {
			bool up = true;
			if (blockIdx.x == (P1 - 1)) {
				up = false;
			}
			Update_by_Schema1(up);
		}
		else {
			Update_by_Schema0(tStep); // SER_c: it was 1 but we use the standard dynamic
		}
	}
	if (schema >= 2) {
		//******************* TO CHECK**************************
		int Pk = Max(P1 - schema, 0);
		if (blockIdx.x - 1 == P2 || blockIdx.x - 1 == (Pk - 1)) {
			bool up = true;
			if (blockIdx.x - 1 == (Pk - 1)) {
				up = false;
			}
			Update_by_Schema1(up);
		}
		else {
			///int Pk = Max(P1 - k + 1, 0);
			Update_by_Schema2(tStep);
		//*******************END TO CHECK************************
		}
	}

}

__device__ void Update_As(int tStep, int NN) {
	int idx = threadIdx.x + (blockIdx.x - 1) * NN;
	if (threadIdx.x < NN - 1) {
		float sigma = 0.2;
		float miu = get_rgc() - sigma * sigma / 2;
		As[idx] = -(sigma * sigma * delta_T) / (4 * delta_X * delta_X) - miu * delta_T / (4 * delta_X);
	}
}

__device__ void Update_Bs(int tStep, int NN) {
	int idx = threadIdx.x + (blockIdx.x - 1) * NN;
	if (threadIdx.x < NN) {
		float sigma = 0.2;
		Bs[idx] = 1 + (sigma * sigma * delta_T) / (2 * delta_X * delta_X);
	}
}

__device__ void Update_Cs(int tStep, int NN) {
	int idx = threadIdx.x + (blockIdx.x - 1) * NN;
	if (threadIdx.x < NN - 1) { 
		float sigma = 0.2;

		float miu = get_rgc() - sigma * sigma / 2;
		Cs[idx] = -(sigma * sigma * delta_T) / (4 * delta_X * delta_X) + miu * delta_T / (4 * delta_X);
	}
}

// Tridiagonal systems solver
__device__ void Thomas_Solver(int tStep) { 

	if (threadIdx.x == 0) {
		int idx = threadIdx.x + N * (blockIdx.x - 1);		
		Cs[idx] = Cs[idx] / Bs[idx];
		Ys[idx] = Ys[idx] / Bs[idx];

		//only the thread 0 of each block solves the system by Thomas, since Thomas is sequential algorithm
		for (int i = 1; i < N - 1; i++) {
			Cs[idx + i] = Cs[idx + i] / (Bs[idx + i] - As[idx + i - 1] * Cs[idx + i - 1]);
		}

		for (int i = 1; i < N; i++) { 
			Ys[idx + i] = (Ys[idx + i] - As[idx + i - 1] * Ys[idx + i - 1]) / (Bs[idx + i] - As[idx + i - 1] * Cs[idx + i - 1]);
		}

		Zs[idx + N - 1] = Ys[idx + N - 1];

		for (int i = N - 2; i > -1; i--) {
			Zs[idx + i] = Ys[idx + i] - Cs[idx + i] * Zs[idx + i + 1];
		}
		
	}
}

// initializer of Bs, Ys, Zs, As, Cs, is_synchronized, synchro_count
__device__ void init() {

	int P1c = 4;
	int P2c = 7;
	float Kgminc = 20.f;
	float Kc = 100.f;
	float delta_Xc = (5.52f - 2.996f) / N;

	for (int i = 0; i < L; i++) {
		is_synchronized[i] = false;
	}
	is_synchronized[L - 1] = true;
	for (int i = 0; i < L; i++) {
		synchro_count[i] = 0;
	}
	for (int i = 0; i < M*N; i++) {
		Bs[i] = 0.f;
		Ys[i] = 0.f;
	}

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < M; j++) {
			if (j < P1c || j > P2c) {
				Zs[i + j * N] = 0.f;
			}
			else {
				float x = log(Kgminc) + i * delta_Xc;
				Zs[i + j * N] = exp(x) - Kc > 0 ? exp(x) - Kc : 0.f;
			}
		}
	}

	for (int i = 0; i < M*(N - 1); i++) {
		As[i] = 0.f;
		Cs[i] = 0.f;
	}

	return;

}

// main kernel solves the whole problem
__global__ void PDE_Solver() { 
	// block zero is responsible for synchronization for other blocks
	
	init();
	__syncthreads();

	if (blockIdx.x == 0) {
		// only the first thread in block zero is used, other threads do nothing
		if (threadIdx.x == 0) {

		// tracking the changes	of Zs		
		for (int i = L - 1; i >= 0; i--) {
			if ((L - 1 - i) % 50 == 0 )  {//&& (L - 1 - i) < 451 && (L - 1 - i) > 299
				printf("step : %d \n\n", L - i - 1);
				for (int ii = 0; ii < M; ii++) {
					for (int jj = 0; jj < N; jj++) {
						printf("%2.0f ", Zs[jj + N * ii]);
					}
					printf("\n");
				}
				printf("\n \n");
			}

				while (synchro_count[i] != M) {
					// wait for all other blocks finishing calculation for time step t_i 
				}
				// once all other blocks have finished calculation, change the i-1'th synchronization state
				// allows all other blocks continuing the t_i-1 time step work
				is_synchronized[i - 1] = true;	
			}
		}
	}
	else {

		// Only block 1 to M do the work, so that we garantie the sum of sychro_count[i] == M
		if ((blockIdx.x >= 1) && (blockIdx.x <= M)) {//(blockIdx.x >= 1) && (blockIdx.x < = M)
		
		// other blocks are responsible for calculation
			for (int i = L - 1; i >= 0; i--) {
				while (is_synchronized[i] != true) {
					// wait for the time step i's state to be changed to true
				}

				if (threadIdx.x < N) {
					Update_Ys(i, N); // SER_c:
				}
				
				// ATTENTION: all threads in the same block locked by
				// __syncthreads until all threads arrive here

				__syncthreads();
				if (threadIdx.x < N) {
					Update_As(i, N);
				}
				
				//__syncthreads();

				if (threadIdx.x < N) {
					Update_Bs(i, N);
				}
				//__syncthreads();

				if (threadIdx.x < N) {
					Update_Cs(i, N);
				}
				__syncthreads();

				if (threadIdx.x < N) {
					Thomas_Solver(i);
				}
				__syncthreads();

				// only the first thread has the right to write in synchro_count
				// since all threads in the block ar synchronized, only the first thread writes
				// can still garantie that all threads in the block have finished the task.
				if (threadIdx.x == 0) {
					atomicAdd(&synchro_count[i], 1);
				}
			}
		}
	}

	
	if (blockIdx.x == 0) {
		if (threadIdx.x == 0)
		{
			printf("result: \n");
			for (int i = 0; i < M; i++) {
				for (int j = 0; j < N; j++) {
					printf("%2.1f ", Zs[j + N * i]);
				}
				printf("\n");
			}
		}
	}
	
}


int main(int argc, char **argv) {

	std::freopen("output.txt", "w", stdout);

	float Tim;							// GPU timer instructions
	cudaEvent_t start, stop;			// GPU timer instructions

	cudaEventCreate(&start);			// GPU timer instructions
	cudaEventCreate(&stop);				// GPU timer instructions
	cudaEventRecord(start, 0);			// GPU timer instructions

	PDE_Solver << <M + 1, N >> > ();

	cudaEventRecord(stop, 0);			// GPU timer instructions
	cudaEventSynchronize(stop);			// GPU timer instructions
	cudaEventElapsedTime(&Tim,			// GPU timer instructions
		start, stop);					// GPU timer instructions
	cudaEventDestroy(start);			// GPU timer instructions
	cudaEventDestroy(stop);				// GPU timer instructions


	printf("Execution time %f ms\n", Tim);
	FreeVar();
	return 0;
}