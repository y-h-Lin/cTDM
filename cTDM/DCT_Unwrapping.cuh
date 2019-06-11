/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/**
**************************************************************************
* \file Common.h
* \brief Common includes header.
*
* This file contains includes of all libraries used by the project.
*/

#pragma once
#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>       // helper functions for CUDA timing and initialization
#include <helper_functions.h>  // helper functions for timing, string parsing

#include "FFT.cuh"


//#include "HilbertTransform.cuh"

extern clock_t s_unwrap, e_unwrap;
extern float unwrap_time;
extern cufftHandle plan_1D_C2C_FORWARD;
extern cufftHandle plan_1D_C2C_INVERSE;

typedef struct
{
    int width;          //!< ROI width
    int height;         //!< ROI height
} ROI;

/**
*  The dimension of pixels block
*/
#define BLOCK_SIZE          8


/**
*  Square of dimension of pixels block
*/
#define BLOCK_SIZE2         64


/**
*  log_2{BLOCK_SIZE), used for quick multiplication or division by the
*  pixels block dimension via shifting
*/
#define BLOCK_SIZE_LOG2     3


/**
*  log_2{BLOCK_SIZE*BLOCK_SIZE), used for quick multiplication or division by the
*  square of pixels block via shifting
*/
#define BLOCK_SIZE2_LOG2    6


/**
*  This macro states that __mul24 operation is performed faster that traditional
*  multiplication for two integers on CUDA. Please undefine if it appears to be
*  wrong on your system
*/
#define __MUL24_FASTER_THAN_ASTERIX


/**
*  Wrapper to the fastest integer multiplication function on CUDA
*/
#ifdef __MUL24_FASTER_THAN_ASTERIX
#define FMUL(x,y)   (__mul24(x,y))
#else
#define FMUL(x,y)   ((x)*(y))
#endif


/**
*  This macro allows using aligned memory management
*/
//#define __ALLOW_ALIGNED_MEMORY_MANAGEMENT


//float *MallocPlaneFloat(int width, int height, int *pStepBytes);
void DCT_UWLS_Unwrapped(float *ImgDst, float *ImgSrc, int sizeX, int sizeY);
__global__ void LaplacianFilter(float *output, float *input, int width, int height);
__global__ void devConstant(float* buf, int sizeX, int sizeY);
__global__ void MultiplyOperator(float *input, float div, int width, int height);
void DCT2D(float *ImgDst, float *ImgSrc, int sizeX, int sizeY, int dir);
__global__ void shift2DArray(float *input, int width, int height);
void DeviceMemOutDCT(char *path, float *arr, int sizeX, int sizeY);
void exportRAW(char * fpath, float* buf, int size);
/////////////////////////////////////////////////////////////////////////


void myDCT(float *ImgDst, float *ImgSrc, int sizeX, int sizeY);
void myIDCT(float *ImgDst, float *ImgSrc, int sizeX, int sizeY);
void DeviceMemOutDCTFFT(char *path, cufftComplex *arr, int sizeX, int sizeY);
__global__ void shift1DFFT(cufftComplex *device_FFT, int Nx, int Ny);
__global__ void float2cufft(cufftComplex *odata, const float *idata, int sizeX, int sizeY);
__global__ void cufft2float(float *odata, const cufftComplex *idata, int sizeX, int sizeY);
__global__ void transposeShared(float *odata, float *idata);
__global__ void countFDCT(float *ImgDst, cufftComplex *ImgSrc, int sizeX, int sizeY, float dTemp);
__global__ void countIDCT(cufftComplex *ImgDst, float *ImgSrc, int sizeX, int sizeY, float dTemp);

__global__ void LaplaceWithPU(float *dst, float *src, int sizeX, int sizeY);

__global__ void NumericDerivative1(float *outX, float *outY, float *input, int sizeX, int sizeY);
__global__ void NumericDerivative2(float *outX, float *outY, float *inX, float *inY, int sizeX, int sizeY);
__global__ void TrigonometricF(float *outX, float *outY, int sizeX, int sizeY);
__global__ void SumDerivative(float *output, float *outX, float *outY, float *src, int sizeX, int sizeY);