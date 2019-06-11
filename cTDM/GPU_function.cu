/*#include "GPU_function.cuh"

void cudaFTshift(cufftComplex * input, int sizeX, int sizeY)
{
	int blocksInX = (sizeX+8-1)/8;
	int blocksInY = (sizeY+8-1)/8;
	dim3 grid(blocksInX, blocksInY);
	dim3 block(8, 8);

	cuFFT2Dshift<<<grid,block>>>(input, sizeX, sizeY);
}*/
