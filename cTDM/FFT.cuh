#define _USE_MATH_DEFINES

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <complex>
#include <cuda.h>
#include <cufft.h>
#include <helper_cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;

//Fast fourier transform
int FFT(int,int,double *,double *);
int FFT2D(complex<float> *, int, int, int);
void FFT3D(complex<float> *, int, int, int, int);
void FFT1Dshift(complex<float> *, int);
void FFT2Dshift(complex<float> *, int, int);
void FFT3Dshift(complex<float> *, int, int, int);

void FFT3Dshift_cufftComplex(cufftComplex *, int, int, int);
void cuFFT1D(cufftComplex *ImgArray, int size, int batch, int dir);
void cuFFT2D(cufftComplex *ImgArray, int sizeX, int sizeY, int dir);
void cuFFT2D_Batch(cufftComplex *ImgArray, int sizeX, int sizeY, int sizeZ, int dir);
void cuFFT2Dz(cufftDoubleComplex *ImgArray, int sizeX, int sizeY, int dir);
void cuFFT3D(cufftComplex *ImgArray, int sizeX, int sizeY, int sizeZ, int dir);
__global__ void cuFFT1Dshift(cufftComplex *input, int width);
__global__ void cuFFT2Dshift(cufftComplex *input, int width, int height);
__global__ void cuFFT3Dshift(cufftComplex *input, int width, int height, int slice);
__global__ void scaleFFT1D(cufftComplex *cu_F, int nx, float scale);
__global__ void scaleFFT2D(cufftComplex *cu_F, int nx, int ny, float);
__global__ void scaleFFT2Dz(cufftDoubleComplex *cu_F, int nx, int ny, double);
__global__ void scaleFFT2DReal(float *cu_F, int nx, int ny, float scale);
__global__ void scaleFFT3D(cufftComplex *cu_F, int nx, int ny, int nz, float);

void cuFFT_Real(cufftComplex *freq, float *img, const unsigned int Nx, const unsigned int Ny, int dir);

int DFT(int ,int ,double *,double *);
int Powerof2(int, int *, int *);
void bfilter(complex<float> *, int, int);

void cuFFT1D_test(cufftComplex *d_odata, cufftReal *d_idata, int size, int batch, int dir);