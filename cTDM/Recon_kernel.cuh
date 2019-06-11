#define _USE_MATH_DEFINES

#include <omp.h>
#include <complex>
#include <cmath>
#include <cuda.h>
#include <cufft.h>
#include <cuComplex.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
using namespace std;

//#define BLOCK_SIZE	16
#define MEDIAN_DIMENSION  3 // For matrix of 3 x 3. We can Use 5 x 5 , 7 x 7 , 9 x 9......   
#define MEDIAN_LENGTH 9   // Shoul be  MEDIAN_DIMENSION x MEDIAN_DIMENSION = 3 x 3

#define BLOCK_WIDTH 32  // Should be 8 If matrix is of larger then of 5 x 5 elese error occur as " uses too much shared data "  at surround[BLOCK_WIDTH*BLOCK_HEIGHT][MEDIAN_LENGTH]
#define BLOCK_HEIGHT 32// Should be 8 If matrix is of larger then of 5 x 5 elese error occur as " uses too much shared data "  at surround[BLOCK_WIDTH*BLOCK_HEIGHT][MEDIAN_LENGTH]

extern int ReconSave;

//median filter on GPU
void MedianFilter(float *PhaseData, float *AmpData, float *Phase_img, float *Amp_img, int nx, int ny, int nz);
__global__ void Median_Filter_GPU(float *Input_Image, float *Output_Image, int Image_Width, int Image_Height, int Image_Slice);
__global__ void MedianFilter_gpu(float *Device_ImageData, int Image_Width, int Image_Height, int numZ);
__global__ void checkMaps(float *Phase_img, float *Amp_img, int Image_Width, int Image_Height, int numZ);


void CopyDataArray(cufftComplex *u_sp, float *phase, float *amp, int sizeX, int sizeY, int sizeZ, int numZ);
__global__ void cuCopyArray(cufftComplex *u_sp, float *phase, float *amp, int Image_Width, int Image_Height, int numZ);

//estimate the Edward Sphere
void EdwardSphere(cufftComplex *u_sp, cufftComplex *F, float * C, float fm0, float df, float AngX, float AngY
	, int sizeX, int sizeY, int sizeZ);
__global__ void fillEdwardSphere(cufftComplex *u_sp, cufftComplex *F, float * C, float fx0, float fy0, float fz0
	, float fm0, float df, bool Fz_err, int sizeX, int sizeY, int sizeZ);
__device__ int cuMod(int a, int b);
__device__ float cuRound(float num);

//////////
// initial F and F2
void initial_F_and_F2(float *C, cufftComplex *cuF, cufftComplex *cuF2, float recon_dx , int Image_Width, int Image_Height, int Image_Slice);
__global__ void Initial_F_F2(float *cu_C, cufftComplex *cu_F, cufftComplex *cu_F2, float cu_dx,
									int Image_Width, int Image_Height, int Image_Slice);
//////////
//n_3D
void est_n_3D(cufftComplex *n, cufftComplex *F, float recon_k2, float recon_med 
					, int Image_Width, int Image_Height, int Image_Slice);
__global__ void calculate_n3D(cufftComplex * cu_n, cufftComplex *cu_F, float cu_k2, float cu_med,
								int Image_Width, int Image_Height, int Image_Slice);

//////////
//F_3D
void modify_F_3D(cufftComplex *n, cufftComplex *F, float recon_k2, float recon_med 
					, int Image_Width, int Image_Height, int Image_Slice);
__global__ void Modify_F3D(cufftComplex *cu_n, cufftComplex *cu_F, float cu_k2, float cu_med, int Image_Width, int Image_Height, int Image_Slice);


//////////
//check F_3D
void check_F_3D(cufftComplex *F, cufftComplex *F2, int Image_Width, int Image_Height, int Image_Slice);
__global__ void check_F3D(cufftComplex *cu_F, cufftComplex *cu_F2, int Image_Width, int Image_Height, int Image_Slice);












//use for the POCS reconstruciton method and the VOID name should be revised again
//Build the complex array (log(amp), phase)
void Combine2ComplexStack(cufftComplex *u_sp_stack, float *phase_stack, float *amp_stack, int sizeX, int sizeY, int sizeZ);
__global__ void cuCombine2ComplexStack(cufftComplex *u_sp_stack, float *phase_stack, float *amp_stack, int sizeX, int sizeY, int sizeZ);

//do array shift for Batch mode FT
void Shift2DonStack(cufftComplex *U, int sizeX, int sizeY, int sizeZ);
__global__ void cuShift2DonStack(cufftComplex *U, int z, int sizeX, int sizeY, int sizeZ);

//Frequency interpolation
void FrquencyInterpolation(cufftComplex *F, float * C, cufftComplex *U, float *AngleX, float *AngleY
	, float fm0, float df, float dx, float n_med, int frameNum, int FrameSize, int sizeX, int sizeY, int sizeZ);
__global__ void FI_kernel(cufftComplex *F, float * C, cufftComplex *U, float *AngleX, float *AngleY
	, float fx0, float fy0, float fz0, float fm0, float df, float dx, float n_med, int frameNum, int frameSize, int sizeX, int sizeY, int sizeZ);
__device__ void calF(float &Fxp, float &Fyp, float &Fzp, float &fzp, float &anglep,
	float Fx, float Fy, float Fz, float fx, float fy, float fz, float n0, float f0, float angle);
__device__ cufftComplex TrilinearFrequency(cufftComplex a1, cufftComplex a2, cufftComplex a3, cufftComplex a4
	, cufftComplex b1, cufftComplex b2, cufftComplex b3, cufftComplex b4, float Fxp, float Fxp1, float Fxp2
	, float Fyp, float Fyp1, float Fyp2, float angle1, float angle2, float anglep);


__device__ cufftComplex ComplexProduction(cufftComplex V, float F);
__device__ cufftComplex ComplexMultiplication(cufftComplex V1, cufftComplex V2);
__device__ cufftComplex ComplexDivision(cufftComplex V1, cufftComplex V2);
__device__ cufftComplex ComplexAddition(cufftComplex V1, cufftComplex V2);
__device__ cufftComplex ComplexSubtraction(cufftComplex V1, cufftComplex V2);
__device__ cufftComplex ComplexSqrt(cufftComplex V);
__device__ float ComplexABS(cufftComplex V);


void Est_n_3D_POCS(cufftComplex *n, cufftComplex *cu_n2, cufftComplex *cu_F, float recon_k2, float recon_med, int sizeX, int sizeY, int sizeZ);
__global__ void cuEst_n_3D_POCS(cufftComplex * cu_n, cufftComplex *cu_n2, cufftComplex * cu_F, float cu_k2, float cu_med, int sizeX, int sizeY, int sizeZ);


float Est_Dp_Dd_POCS(cufftComplex *cu_F, cufftComplex *cu_F2, int sizeX, int sizeY, int sizeZ);
__global__ void cuEst_Dp_Dd_POCS(float *temp, cufftComplex *cu_F, cufftComplex *cu_F2, int sizeX, int sizeY, int sizeZ);


void ConvertN2F(cufftComplex *cuF, cufftComplex *cuF2, cufftComplex *cuN, cufftComplex *cuN2, float recon_k2, float recon_med
	, int sizeX, int sizeY, int sizeZ);
__global__ void cuConvertN2F(cufftComplex *cuF, cufftComplex *cuF2, cufftComplex *cuN, cufftComplex *cuN2, float cu_k2, float cu_med
	, int sizeX, int sizeY, int sizeZ);


void GradientDescentTV(cufftComplex *out, cufftComplex *cuF2, int sizeX, int sizeY, int sizeZ);
__global__ void cuGradientDescentTV(cufftComplex *out, cufftComplex *cuF2, int sizeX, int sizeY, int sizeZ);


void EstF_TV(cufftComplex *cuF2, cufftComplex *cuN, float dtvg, int sizeX, int sizeY, int sizeZ);
__global__ void cuEstF_TV(cufftComplex *cuF2, cufftComplex *cuN, float dtvg, int sizeX, int sizeY, int sizeZ);


void EstF_Beta(cufftComplex *cuF, cufftComplex *cuF2, float beta, int sizeX, int sizeY, int sizeZ);
__global__ void cuEstF_Beta(cufftComplex *cuF, cufftComplex *cuF2, float beta, int sizeX, int sizeY, int sizeZ);