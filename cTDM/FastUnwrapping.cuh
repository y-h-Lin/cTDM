#include "FFT.cuh"
#include <time.h>
using namespace std;

extern clock_t s_unwrap, e_unwrap;
extern float unwrap_time;

extern cufftHandle plan_2D_C2C_FORWARD_FTUP;
extern cufftHandle plan_2D_C2C_INVERSE_FTUP;

//Fast unwrapping
void FastUnwrapping(float *ImgSrc, float *ImgDst, int sizeX, int sizeY);
void FastUnwrapping2(float *ImgSrc, float *ImgDst, int sizeX, int sizeY);
void PhaseNormalize(float *src, int width, int height);
__global__ void RescalePhaseMap(float *src, float min, float max, int width, int height);
__global__ void makeSimmetric(float *sinMap, float *cosMap, float *src, 
							  int Nx2, int Ny2, int Nx1, int Ny1);
__global__ void estimateCosArray(float *dst, float *src, int width, int height);
__global__ void estimateSinArray(float *dst, float *src, int width, int height);
__global__ void Real2Complex(cufftComplex *dst, float *src, int width, int height);
__global__ void Complex2Real(float *dst, cufftComplex *src, int Nx1, int Ny1);
__global__ void transferQuarter  (float *dst, cufftComplex *src, int Nx2, int Ny2, int Nx1, int Ny1);
__global__ void RecoveryComplex    (cufftComplex *src, int width, int height);
__global__ void MultiplyComplex	   (cufftComplex *src, float*mask, int width, int height);
__global__ void MultiplyPosition   (cufftComplex *src, int width, int height);
__global__ void MultiplyInvPosition(cufftComplex *src, int width, int height);
__global__ void MultiplyArray(float *src, float *mask, int width, int height);
__global__ void SubtractComplex(cufftComplex *dst, cufftComplex *src1, cufftComplex *src2, int width, int height);
__global__ void ObtainUnwrap(float *dst, float *diff, float *cont, float *src, int width, int height);
void subtractMin(float *src, int width, int height);
__global__ void PhaseMapSubMin(float *src, float min, int width, int height);

void DeviceMemOut2(char *path, float *arr, int sizeX, int sizeY);


void findMinMax(float &min, float &max, float *arr, int Nx, int Ny);
__global__ void reduce_max_kernel(float *d_out, const float *d_logLum, int size);
__global__ void reduce_max_kernel2(float *d_out, float *d_in);
__global__ void reduce_min_kernel(float *d_out, const float *d_logLum, int size);
__global__ void reduce_min_kernel2(float *d_out, float *d_in);
