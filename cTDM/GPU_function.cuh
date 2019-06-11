#include <omp.h>
#include <complex>
#include <cuda.h>
#include <cufft.h>
#include <cuComplex.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "FFT.cuh"
using namespace std;

void cudaFTshift(cufftComplex * input, int sizeX, int sizeY)
{
	int blocksInX = (sizeX+8-1)/8;
	int blocksInY = (sizeY+8-1)/8;
	dim3 grid(blocksInX, blocksInY);
	dim3 block(8, 8);

	cuFFT2Dshift<<<grid,block>>>(input, sizeX, sizeY);
}