#define _USE_MATH_DEFINES

#ifdef _WIN32
	#include <Windows.h>
	#include <direct.h>
	#include <io.h>
#elif __linux__ 
	#include <dirent.h>
#endif

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_delete.h>
#include <thrust/device_free.h>
#include <thrust/extrema.h>

//#include "Common.h"

#include "convolutionFFT2D_common.h"
#include "DCT_Unwrapping.cuh"
#include "FFT.cuh"
#include "FastUnwrapping.cuh"
#include "lmmin.cuh"
#include "Recon_kernel.cuh"

using namespace std;

typedef struct microImg
{
	float phase;
	float amp;
}microImg;

/* data structure to transmit arrays and fit model */
typedef struct {
    double *tx;
    double *y;
    double (*f)( double tx, double y, const double *p );
} data_struct;



using namespace std;
clock_t start_time, end_time;
clock_t sF_time, eF_time;
clock_t sE_time, eE_time;
clock_t s_wrap, e_wrap;
clock_t s_unwrap, e_unwrap;
clock_t s_datatransfer, e_datatransfer;
float wrap_time, unwrap_time, dataTransfer_time, extract_time;
float total_time;
int AccumFrame;

//int ResizeFlag, ReconFlag;


#define MEDIAN_DIMENSION  3 // For matrix of 3 x 3. We can Use 5 x 5 , 7 x 7 , 9 x 9......   
#define MEDIAN_LENGTH 9   // Shoul be  MEDIAN_DIMENSION x MEDIAN_DIMENSION = 3 x 3

#define BLOCK_WIDTH 32  // Should be 8 If matrix is of larger then of 5 x 5 elese error occur as " uses too much shared data "  at surround[BLOCK_WIDTH*BLOCK_HEIGHT][MEDIAN_LENGTH]
#define BLOCK_HEIGHT 32// Should be 8 If matrix is of larger then of 5 x 5 elese error occur as " uses too much shared data "  at surround[BLOCK_WIDTH*BLOCK_HEIGHT][MEDIAN_LENGTH]

/**********************************************************************************/
int scriptFileRead(char *);
char *strim(char * str);
char *obtainParameter(char *strLine);
void outputImg(char *,microImg *,bool,double,double,int,int);
void openImage(char *, microImg *, bool &, double &, double &, int &, int &);
int count_file_num(char* target_folder, char *head_name);
void cudaGoldsteinUnwrap2D(float *cu_phi, float *cu_UnWrapPhase, int sizeX, int sizeY,int frameNumber, int SPBG);
void HilbertTransform(char *,char *,char *,char *,int rowPts[], int colPts[],int,int,int,int,int);
#define PHIMAP(R,C) phi[R*colSize+C]
void Combine2Stack(float *Phase3D, float *Amp3D, microImg *Img, int Nx, int Ny, int frameNum);
void LoadDateFromDir(float *Phase3D, float *Amp3D, bool *status_series, float *sampleAngleRadX_Stack, float *sampleAngleRadY_Stack, int &totalFrame, int &deleteCount, char *dir);
void RefreshStack(float *PhaseStack, float *AmpStack, bool *status_series, float *sampleAngleRadX, float *sampleAngleRadY, int Nx, int Ny, int ReconFrame, int deleteCount);
void BatchRecon(float *PhaseStack, float *AmpStack, float *sampleAngleRadX, float *sampleAngleRadY, int ReconFrame);
void BatchReconPOCS(float *PhaseStack, float *AmpStack, float *sampleAngleRadX, float *sampleAngleRadY, int ReconFrame);
void bilinear(float *input, float *output, int M1, int N1, int M2, int N2);
double modulus(double, double);
double myfmod(double,double);
double mymax(double,double);
double mymin(double,double);
double myround(double);
int mod(int,int);
double round_MS(double);
bool is_nan(double);
bool is_inf(double);
void AngleCalculation(float *,double &,double &,double &,double &,int,int);
void bmp_header(char *,int &,int &);
void read_bmp(char *,int,int,unsigned char *);
__global__ void circleImgGenerate(int *circleImg,int Nx,int Ny,int rCircle, int cCircle, int);
void zeros_complex(complex<float> *,int);
void zeros(double *,int);
void ones(double *,int);
void zeros_f(float *,int);
void ones_f(float *,int);
float AngCal(cufftComplex *, int, int, int, int);
float AngCal2(float *src, int sizeX, int sizeY, int frame, int totalFrame);
float AngCal_GPU(float *src, int sizeX, int sizeY, int frame, int totalFrame);
__global__ void cufftComplex2Real(float *d, cufftComplex *s, int sizeX, int sizeY);
void AngCal3(float *src, int sizeX, int sizeY, int frame, int totalFrame, double &AngX, double &AngY);
bool checkArray(float *array2D, float limitSTD, float limitRange, int size);
void PrintProcess(int counter, char *SPDir, char *BGDir, char *AngDir);
void printDevProp(cudaDeviceProp devProp);
void phaseCalibration(float *, int, int);
void ampCalibration(float *, int, int);
void obtainRadius(cufftComplex *, int &, int &, int &, int, int);
void exportRAW_F(char *,  float *, int);
void exportComplex(char * , cufftComplex*, int);


const int TILE_DIM = 32;
const int BLOCK_ROWS = 32;
void extractQPI(float *SP, float *BG, cufftComplex *cuSP_FFT, cufftComplex *cuBG_FFT, int Nx, int Ny);
void sequence1DFFT(float *ResampleArray, cufftComplex *out_array, int Nx, int Ny);
__global__ void real2cufft(cufftComplex *odata, const float *idata);
__global__ void HistogramFT(float *sumFFT_1D, cufftComplex *device_FFT, int Nx, int Ny);
__global__ void CropFTdomain(cufftComplex *device_FFT, cufftComplex *device_crop, int Nx, int Ny, int center);
__global__ void shiftArray(cufftComplex *device_FFT, int Nx, int Ny);
int FindMaxIndex(float *sum, int size);
__global__ void copySharedMem(cufftComplex *odata, const cufftComplex *idata, const float scale);

//LM method
double f1(double,double,double,const double *);
void evaluate_surface(const double *,int,const void *,double *,int *);
double f(double,double,const double *);

//Reconstruction Function

/**********************************************************************************/
__global__ void zeros_cu_int(int *input, int sizeX, int sizeY);
__global__ void zeros_cu_float(float *input, int sizeX, int sizeY);
__global__ void ones_cu_float(float *input, int sizeX, int sizeY);
__global__ void zeros_cufft(cufftComplex *input, int sizeX, int sizeY);
__global__ void copy_cuFFT(cufftComplex *input, cufftComplex *output, int *checkArray, int sizeX, int sizeY);
__global__ void get1stOrder(cufftComplex *out, cufftComplex *in, int radius, int r, int c, int sizeX, int sizeY);
__global__ void estimateWrapPhase(float *SPWrap, float *BGWrap, cufftComplex *SP, cufftComplex *BG, int sizeX, int sizeY);
__global__ void estimatePhase(float *Phase, float *UnSPWrap, float *UnBGWrap, int sizeX, int sizeY);
__global__ void estimateAmp(float *Amp, cufftComplex *SP, cufftComplex *BG, int sizeX, int sizeY);
__global__ void calcWrapPhase(float *Phase, float *Amp, cufftComplex *SP, cufftComplex *BG, int sizeX, int sizeY);
__global__ void bilinear_interpolation_kernel(float * __restrict__ d_result, const float * __restrict__ d_data,
												float * __restrict__ d_result_2, const float * __restrict__ d_data_2,
                                                  const int M1, const int N1, const int M2, const int N2);
__global__ void modifyAngCalArray(cufftComplex *input, int sizeX, int sizeY);

void DeviceMemOut(char *path, float *arr, int sizeX, int sizeY);
void DeviceMemOutFFT(char *path, cufftComplex *arr, int sizeX, int sizeY);

__global__ void MedianFilter_gpu(float *Device_ImageData,int Image_Width,int Image_Height);


__global__ void get1stOrder_new(cufftComplex *out, cufftComplex *in, int radius, int r, int c, int sizeX, int sizeY);