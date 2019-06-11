#define _USE_MATH_DEFINES

#include <Windows.h>
#include <time.h>
#include <direct.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <io.h>
#include <iostream>
#include <fstream>
#include <complex>
#include <cmath>
#include <omp.h>
#include <cuda.h>
#include <cufft.h>
#include <cuComplex.h>
#include <device_functions.h>
#include <helper_cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


//for memory copying
float *h_ZerosArray, *h_OnesArray;	//host
float *d_ZerosArray, *d_OnesArray;	//device

//for calculating operator
int *countNonZeroArray;				//device array for count non-zero element
int *cudaSumAdjoinArray;			//device array for cumlation


using namespace std;

int blocksX = (1024+32-1)/32;
int blocksY = (1024+32-1)/32;
dim3 gridGold(blocksX, blocksY);
dim3 blockGold(32, 32);

//--------------------------------------------------------------------------------------
void cudaGoldsteinUnwrap2D(float *cu_phi, float *cu_UnWrapPhase, int sizeX, int sizeY,int frameNumber);
//--------------------------------------------------------------------------------------
__global__ void cudaPhaseResidues(float *phi,int sizeX, int sizeY, float *ResiduesCharge);
//--------------------------------------------------------------------------------------
void cuBranchCuts(float *ResiduesCharge, float *BranchCut, float *IM_Mask, int sizeX, int sizeY, int MaxBoxRadius);
__global__	void cudaBC(float *BranchCut, float *ResidueBalanced, float *ResiduesChargeMasked, int MaxBoxRadius, int sizeX, int sizeY);
__global__ void changeElement(float *arr, int idx, float val);
__global__ void inverseLogic(float* IM_mask, float *BranchCut, int sizeX, int sizeY);
__device__ void cuPlaceBranchCutInternal(float *BranchCut,int sizeX,int sizeY,
										unsigned int r1, unsigned int c1, unsigned int r2, unsigned int c2);
//--------------------------------------------------------------------------------------
__global__ void cudaCheckIMMask(float *BranchCut, float *IMMask, int sizeX, int sizeY);
void cudaFloodFill(float *BranchCut, float *phimap, float *IM_Unwrapped, float *IM_Mask, int sizeX, int sizeY);
__global__ void cuFF_MatchArray(float* Adjoin, float *AdjoinStuck, float *flagArray, int sizeX, int sizeY);
__global__ void cuFF_SeedPoints(float *BranchCut, float *Adjoin, float *phimap, float *IM_Unwrapped, float *UnwrappedBinary, int sizeX, int sizeY);
__global__ void cuFF_Internal_1(float* Adjoin, float *BranchCut, float *UnwrappedBinary, float *IM_Unwrapped, float *phimap, int sizeX, int sizeY);
__global__ void cuFF_Internal_2(float* Adjoin, float *UnwrappedBinary, float *IM_Unwrapped, float *phimap, int sizeX, int sizeY);
__global__ void cuFFcheckArray(float *BranchCut, float *Adjoin, int sizeX, int sizeY);
void FloodFill(float *BranchCut,int rowSize, int colSize, float *phi, float *IMUnwrapped,float *IMMask);
int sumAdjoin(float *Adjoin,int rowSize,int colSize);
float getCudaVariable(float *arr, int idx);
__global__ void getCudaValue(float *arr, int idx, float *returnValue);


int countNonZero(float *arr, int sizeX, int sizeY);
__global__ void checkZero(float* arr, int* buf, int sizeX, int sizeY);
__global__ void reduce(int* buf, int range, int size);


int cudaSumAdjoin(float *arr, int sizeX, int sizeY);
__global__ void checkValue(float* arr, int* buf, int sizeX, int sizeY);

__device__ float cudaMod(float a, float b);
__device__ float cudaRound(float number);